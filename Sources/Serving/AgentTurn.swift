import Foundation

// ── The pure turn-orchestration core (extracted from the MLX serve loop so it's testable end-to-end
// without loading a model). `streamAgentTurn` drives one claude-compatible agent turn over an INJECTED
// backend — it owns the round loop, the four hook seams, and the AgentStreamEvent emission; the backend
// owns only the two effects that need a real model/MCP host: producing a round's gen steps and
// dispatching a tool. MLXBackend.streamAgent wraps this with a ChatSession-backed backend; tests wrap it
// with a scripted stub. No MLX / Process / MCP here.

/// One step of a model's output within a round.
public enum GenStep: Sendable, Equatable {
    case chunk(String)                              // a streamed text fragment
    case toolCall(name: String, argsJSON: String)   // the model invoked a tool
}

/// What the sequencer feeds back to continue a turn: a tool's result, or a user nudge (when a reasoning
/// model trailed off narrating intent without acting). Pure (no MLX) so the sequencer stays testable.
public enum ResumeMessage: Sendable, Equatable {
    case toolResult(String)
    case user(String)
}

/// The two model/host effects the sequencer delegates. `round` streams one assistant turn; `dispatch`
/// runs a tool. Sendable so the sequencer can drive it from a detached task.
public protocol AgentTurnBackend: Sendable {
    /// Stream one round. First round: `resume` empty → answer `prompt`. Later rounds: `resume` carries the
    /// prior tool results (or a user nudge) to continue from. `toolsEnabled` is false on the final round to
    /// force a text answer. `instructions` is the hook-augmented system text (used when the session is built).
    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error>
    /// Dispatch a tool call; return its result text + whether it errored.
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool)
}

public struct AgentTurnConfig: Sendable {
    public var modelLabel: String
    public var sessionID: String
    public var cwd: String
    public var toolNames: [String]
    public var maxRounds: Int
    /// Governs whether this turn's tool calls actually dispatch (plan/approval/auto). Default `.auto` keeps
    /// every existing caller's behavior unchanged.
    public var permission: ToolPermissionPolicy
    /// Resolves an `ExitPlanMode` call: approve → flip the turn to `.auto` and implement; reject → revise.
    /// `nil` → auto-approve (headless/Orbital default). The CLI supplies a stdin prompt.
    public var approvePlan: PlanApprover?
    /// Per-dispatch wall-clock bound so one wedged tool can't stall the turn (`<= 0` disables it).
    public var toolTimeoutSeconds: Double
    public init(modelLabel: String = "swiftlm-mlx", sessionID: String = UUID().uuidString,
                cwd: String = FileManager.default.currentDirectoryPath,
                toolNames: [String] = [], maxRounds: Int = 8,
                permission: ToolPermissionPolicy = .init(),
                approvePlan: PlanApprover? = nil,
                toolTimeoutSeconds: Double = defaultToolDispatchTimeoutSeconds) {
        self.modelLabel = modelLabel; self.sessionID = sessionID; self.cwd = cwd
        self.toolNames = toolNames; self.maxRounds = maxRounds; self.permission = permission
        self.approvePlan = approvePlan; self.toolTimeoutSeconds = toolTimeoutSeconds
    }
}

/// Default number of consecutive ERROR-FREE tool rounds before the loop forces a final answer. Bounds an
/// over-eager model that keeps calling tools after the task is already done (the GLM "ramble into clipboard"
/// behavior), WITHOUT cutting error-repair short — an errored/blocked round resets the count, so the
/// refine-on-error loop keeps its full budget. Generous enough for normal multi-step work, well below maxRounds.
public let defaultMaxQuietToolRounds = 4

/// Identity of a tool call for duplicate detection: name + whitespace-normalized arguments. A repeated key
/// means the model re-issued an identical call (zero new information) — a non-progress signal.
public func toolCallKey(name: String, argsJSON: String) -> String {
    name + "|" + argsJSON.components(separatedBy: .whitespacesAndNewlines).joined()
}

/// Pull the `plan` field out of an `ExitPlanMode` argument JSON (falls back to the raw string).
public func extractPlan(fromArgsJSON json: String) -> String {
    guard let data = json.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let plan = obj["plan"] as? String else { return json }
    return plan
}

/// Drive one agent turn, emitting AgentStreamEvents (which serialize to claude-stream-json for Orbital).
/// Sequence per turn: systemInit → (textDelta* | toolUse → toolResult)* → result, with the four hook
/// seams fired at the same points as the MLX loop.
public func streamAgentTurn(
    prompt: String,
    instructions: String?,
    hooks: HookChain?,
    config: AgentTurnConfig,
    backend: any AgentTurnBackend
) -> AsyncThrowingStream<AgentStreamEvent, Error> {
    AsyncThrowingStream { continuation in
        let task = Task {
            do {
                continuation.yield(.systemInit(model: config.modelLabel, sessionID: config.sessionID,
                                               tools: config.toolNames, cwd: config.cwd))

                // SEAM A — UserPromptSubmit: hooks (memory-hooks advise, etc.) inject grounding/hedge.
                var instr = instructions
                if let hooks {
                    let r = await hooks.fire(.userPromptSubmit(prompt: prompt))
                    if let ctx = r.additionalContext, !ctx.isEmpty {
                        instr = (instr.map { $0 + "\n\n" } ?? "") + ctx
                    }
                }
                // Plan mode: tell the model up front it's planning, so it produces a plan instead of
                // flailing against the write/exec tools the policy will block below.
                if let planNote = config.permission.instructionPrefix {
                    instr = (instr.map { $0 + "\n\n" } ?? "") + planNote
                }
                // Mutable across rounds: an approved ExitPlanMode flips this from .plan to .auto mid-turn.
                var permission = config.permission

                var resume: [ResumeMessage] = []
                var finalText = ""
                var outputChars = 0
                var round = 0
                var consecutiveNudges = 0
                var toolsCalled: [String] = []
                var seenCalls = Set<String>()    // (name+args) already dispatched → a repeat is non-progress
                var quietToolRounds = 0          // consecutive error-free tool rounds → completion pressure
                // Tools persist ACROSS rounds so a room agent can iterate (write→compile→fix); the last round
                // reserves no tools to force an answer; maxRounds bounds any runaway. Matches the live loop.
                let maxToolRounds = max(1, config.maxRounds - 1)
                while round < config.maxRounds {
                    // Tools stay on until the round budget OR the model has called tools error-free for a
                    // while without finishing (over-eager rambling) — then drop tools to force a final answer.
                    let toolsOn = round < maxToolRounds && quietToolRounds < defaultMaxQuietToolRounds
                    let stream = backend.round(instructions: instr, prompt: prompt,
                                               resume: resume, toolsEnabled: toolsOn)
                    resume = []
                    var text = ""
                    var toolCalls: [(name: String, argsJSON: String)] = []
                    var thinkFilter = ReasoningStreamFilter()   // hide <think> content from the streamed deltas
                    for try await step in stream {
                        switch step {
                        case .chunk(let ch):
                            text += ch; outputChars += ch.count
                            let vis = thinkFilter.feed(ch)
                            if !vis.isEmpty { continuation.yield(.textDelta(vis)) }   // live render, no <think>
                        case .toolCall(let name, let argsJSON):
                            toolCalls.append((name, argsJSON))
                        }
                    }
                    let tail = thinkFilter.flush()
                    if !tail.isEmpty { continuation.yield(.textDelta(tail)) }
                    round += 1

                    if toolCalls.isEmpty {
                        let clean = stripReasoning(text)
                        // A real answer, or no tool budget / nudge cap → finish; else nudge the model (it
                        // narrated intent inside <think> without acting). Bounded by ≤2 nudges + maxRounds.
                        if !clean.isEmpty || !toolsOn || consecutiveNudges >= 2 { finalText = text; break }
                        consecutiveNudges += 1
                        resume = [.user("Continue. If you intended to run or check something, call that tool "
                                        + "now. If the task is complete, reply with a brief plain-text summary.")]
                        continue
                    }
                    consecutiveNudges = 0

                    var results: [ResumeMessage] = []
                    var roundHadError = false, roundApproved = false, roundProgressed = false
                    for call in toolCalls {
                        let id = UUID().uuidString
                        continuation.yield(.toolUse(id: id, name: call.name, inputJSON: call.argsJSON))
                        toolsCalled.append(call.name)
                        // ExitPlanMode: the plan is ready → surface it for approval. Approve flips the turn
                        // to .auto (tools go live, the model implements); reject keeps plan mode so it revises.
                        // Intercepted BEFORE the gate so the (non-read-only) ExitPlanMode call is never blocked.
                        if permission.mode == .plan, call.name == exitPlanModeToolName {
                            let plan = extractPlan(fromArgsJSON: call.argsJSON)
                            let verdict = await (config.approvePlan?(plan) ?? .approve)
                            switch verdict {
                            case .approve:
                                permission = permission.approved
                                roundApproved = true   // fresh budget for the implementation that follows
                                let ok = "Plan approved by the user. You are now in execution mode — use the "
                                    + "tools to implement the plan."
                                continuation.yield(.toolResult(id: id, content: ok, isError: false))
                                results.append(.toolResult(ok))
                            case .reject(let reason):
                                roundHadError = true
                                let no = "Plan rejected: \(reason). Revise your plan and call "
                                    + "\(exitPlanModeToolName) again."
                                continuation.yield(.toolResult(id: id, content: no, isError: true))
                                results.append(.toolResult(no))
                            }
                            continue
                        }
                        // SEAM B — Permission ladder: plan mode blocks side-effecting tools before any hook
                        // or dispatch; approval clamps to allow under autopilot. The block becomes a tool
                        // result so the model sees WHY and can adjust (plan the step instead of calling it).
                        if case .deny(let reason) = permission.decide(tool: call.name) {
                            roundHadError = true
                            let blocked = "BLOCKED (\(permission.mode.rawValue)): \(reason)"
                            continuation.yield(.toolResult(id: id, content: blocked, isError: true))
                            results.append(.toolResult(blocked))
                            continue
                        }
                        // SEAM C — PreToolUse: a hook may veto (plan-mode read-only, etc.).
                        if let hooks {
                            let pre = await hooks.fire(.preToolUse(tool: call.name, argumentsJSON: call.argsJSON))
                            if case .deny(let reason) = pre.decision {
                                roundHadError = true
                                let blocked = "BLOCKED by hook: \(reason)"
                                continuation.yield(.toolResult(id: id, content: blocked, isError: true))
                                results.append(.toolResult(blocked))
                                continue
                            }
                        }
                        // Duplicate call → no new information. Don't re-dispatch; push toward finishing.
                        roundProgressed = true
                        if !seenCalls.insert(toolCallKey(name: call.name, argsJSON: call.argsJSON)).inserted {
                            let dup = "You already called `\(call.name)` with these arguments; the result is "
                                + "unchanged. Do not repeat it. If the task is complete, give your final answer."
                            continuation.yield(.toolResult(id: id, content: dup, isError: false))
                            results.append(.toolResult(dup))
                            continue
                        }
                        // Bound the dispatch: a wedged tool (e.g. a hung MCP call) is abandoned and reported
                        // as non-responsive so the turn proceeds instead of hanging forever. A timeout is NOT
                        // an error (don't invite an endless retry) — it counts toward the completion cap.
                        let callName = call.name, callArgs = call.argsJSON
                        var outcome: (result: String, isError: Bool)
                        // `Task` runs a whole nested sub-agent turn (self-bounded) — exempt from the wall-clock bound.
                        let dispatchTimeout = callName == "Task" ? 0 : config.toolTimeoutSeconds
                        if let r = await withToolTimeout(seconds: dispatchTimeout,
                                                         { await backend.dispatch(name: callName, argsJSON: callArgs) }) {
                            outcome = r
                        } else {
                            outcome = ("Tool `\(call.name)` did not respond within \(Int(config.toolTimeoutSeconds))s "
                                       + "and was skipped. Do not retry it; continue or give your final answer.", false)
                        }
                        // SEAM D — PostToolUse: a hook may transform the result (compile-check feedback).
                        if let hooks {
                            let post = await hooks.fire(.postToolUse(tool: call.name, result: outcome.result))
                            if let rep = post.replacementResult { outcome = (rep, outcome.isError) }
                        }
                        if outcome.isError { roundHadError = true }
                        continuation.yield(.toolResult(id: id, content: outcome.result, isError: outcome.isError))
                        results.append(.toolResult(outcome.result))
                    }
                    // Completion pressure: a clean (error-free) tool round with no new direction ticks toward
                    // forcing an answer; an error or a fresh plan-approval resets the budget (real progress).
                    if roundHadError || roundApproved { quietToolRounds = 0 }
                    else if roundProgressed { quietToolRounds += 1 }
                    resume = results
                }

                // Strip reasoning (with the conclusion fallback) so the final result is clean for Orbital.
                let stripped = displayAnswer(finalText)
                let answer = !stripped.isEmpty ? stripped
                    : (finalText.isEmpty ? "(hit round cap without final text)" : "(model emitted only reasoning)")
                // SEAM E — Stop: record the completed turn (memory-write, trace logging).
                if let hooks { _ = await hooks.fire(.stop(answer: answer, toolsCalled: toolsCalled)) }
                // TODO: real token counts from the backend's completion info; chars/4 only feeds Orbital's
                // context-bar display (functionally harmless if off).
                continuation.yield(.result(finalText: answer, inputTokens: 0,
                                           outputTokens: max(0, outputChars / 4), isError: false))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}
