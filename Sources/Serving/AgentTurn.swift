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
    /// VLM variant: `images` are file URLs attached to the FIRST (user) round — the agent's turn carries
    /// pictures for a vision model. The default impl (text backends, test stubs) ignores them and forwards to
    /// the text `round`; only the MLX `ChatSessionTurnBackend` consumes them (attaches to the user `Chat.Message`).
    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool, images: [URL])
        -> AsyncThrowingStream<GenStep, Error>
    /// Dispatch a tool call; return its result text + whether it errored.
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool)
    /// The real running context size (prompt+gen tokens) after the final round, for honest result telemetry
    /// (OPEN ITEM T1). Default nil — a scripted stub that can't measure tokens degrades to the chars/4
    /// estimate, harmless to the test path; the MLX backend returns its CompactingSession's `.info` count.
    func finalContextTokens() -> Int?
}

public extension AgentTurnBackend {
    func finalContextTokens() -> Int? { nil }
    /// Default: drop images and run the text round. Lets non-VLM backends and the test stubs satisfy the new
    /// requirement unchanged; only `ChatSessionTurnBackend` overrides it to feed pixels to the model.
    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool, images: [URL])
        -> AsyncThrowingStream<GenStep, Error> {
        round(instructions: instructions, prompt: prompt, resume: resume, toolsEnabled: toolsEnabled)
    }
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
    /// TERMINAL/ROUTING tools (Orbital: `handoff`/`consult`/`done`) that completion-pressure must NEVER drop.
    /// A room agent's "final answer" IS a routing call — so when the round loop disables WORK tools to force a
    /// conclusion, these stay live, and the backend keeps offering them. Empty for non-room callers (CLI/serve),
    /// which preserves the old "drop all tools on the last round" behavior exactly. See the round loop below.
    public var terminalTools: Set<String>
    public init(modelLabel: String = "swiftlm-mlx", sessionID: String = UUID().uuidString,
                cwd: String = FileManager.default.currentDirectoryPath,
                toolNames: [String] = [], maxRounds: Int = 8,
                permission: ToolPermissionPolicy = .init(),
                approvePlan: PlanApprover? = nil,
                toolTimeoutSeconds: Double = defaultToolDispatchTimeoutSeconds,
                terminalTools: Set<String> = []) {
        self.modelLabel = modelLabel; self.sessionID = sessionID; self.cwd = cwd
        self.toolNames = toolNames; self.maxRounds = maxRounds; self.permission = permission
        self.approvePlan = approvePlan; self.toolTimeoutSeconds = toolTimeoutSeconds
        self.terminalTools = terminalTools
    }
}

/// Default number of consecutive ERROR-FREE tool rounds before the loop forces a final answer. Bounds an
/// over-eager model that keeps calling tools after the task is already done (the GLM "ramble into clipboard"
/// behavior), WITHOUT cutting error-repair short — an errored/blocked round resets the count, so the
/// refine-on-error loop keeps its full budget. Generous enough for normal multi-step work, well below maxRounds.
public let defaultMaxQuietToolRounds = 4

/// MUTATION tools — the work product of a builder. These are PROGRESS, never "rambling": completion pressure
/// must never drop them (dropping a builder's `write_file`/`edit_file` discards the actual fix and loops the
/// room — the chess battletest), and a round that mutates RESETS the quiet-round counter (real new direction).
/// A read-only verifier never calls these, so verifiers are unaffected — they still conclude under pressure.
public let mutationToolNames: Set<String> = ["write_file", "edit_file", "Write", "Edit", "create_file", "apply_patch"]

/// Duplicate-call policy (the build-test-fix loop, made to converge):
///   • A FILE MUTATION (write_file/edit_file/…) CLEARS the duplicate-call set when it runs — so a prior
///     `read_file`/`grep`/`bash` test legitimately RE-RUNS against the now-changed files (edit → re-run the
///     same test → allowed). This is the fix for the original "result is unchanged" false-block that stalled
///     the chess builder after it rewrote engine.py.
///   • `bash` is DEDUP-CHECKED like a read: an identical command with NOTHING changed since is a no-progress
///     spin, not new info — block it. (Observed: the 122B re-ran one debug script 316× without editing,
///     burning the GPU all night. Only an intervening edit lets the same command run again, via the clear above.)
/// So `mutationToolNames` is the set that clears-and-is-exempt; bash is NOT in it.

/// Identity of a tool call for duplicate detection: name + whitespace-normalized arguments. A repeated key
/// means the model re-issued an identical call (zero new information) — a non-progress signal.
public func toolCallKey(name: String, argsJSON: String) -> String {
    name + "|" + argsJSON.components(separatedBy: .whitespacesAndNewlines).joined()
}

/// Per-round turn tracing for live diagnosis (env `ORBITAL_TURN_TRACE=<path>`). Off by default (nil env ⇒
/// no-op, no file). Captures exactly what the round loop decided each round — the signal that distinguishes a
/// model that won't conclude from a harness that won't LET it (e.g. `hasToolTag=true parsed=0` ⇒ the model
/// emitted a tool call the parser/render dropped; the `</tool_call>` residual). Append-per-line (low volume).
private let turnTracePath = ProcessInfo.processInfo.environment["ORBITAL_TURN_TRACE"]
func turnTrace(_ line: @autoclosure () -> String) {
    guard let path = turnTracePath else { return }
    let stamp = ISO8601DateFormatter().string(from: Date())
    let data = Data(("[\(stamp)] " + line() + "\n").utf8)
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile(); fh.write(data); try? fh.close()
    } else {
        try? data.write(to: URL(fileURLWithPath: path))
    }
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
    backend: any AgentTurnBackend,
    images: [URL] = []
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
                // A ROOM agent (terminalTools non-empty) can ALWAYS conclude its turn by routing — its routing
                // tools stay live even under completion pressure (the backend never drops them). So "no work-tool
                // budget" no longer means "must end with whatever text it has (even empty)"; it means "now route
                // or state a verdict." `concludedThisTurn` = the agent already emitted a turn-ending/routing call
                // (handoff/done/consult/ExitPlanMode) ⇒ an empty follow-up round is a clean wind-down, NOT a turn
                // to nudge (the user's ExitPlanMode concern). Both empty for CLI/serve ⇒ old behavior preserved.
                let canRoute = !config.terminalTools.isEmpty
                var concludedThisTurn = false
                // Tools persist ACROSS rounds so a room agent can iterate (write→compile→fix); under completion
                // pressure WORK tools drop to force a conclusion, but routing tools stay (canRoute); maxRounds
                // bounds any runaway. Matches the live loop.
                let maxToolRounds = max(1, config.maxRounds - 1)
                while round < config.maxRounds {
                    // A BUILDER (its permission ALLOWS file mutation) must work freely across many rounds — read,
                    // edit, test, re-edit — so completion pressure must NOT apply to it: dropping its edits, or
                    // route-nudging it mid-write ("conclude, don't call other tools"), truncates a file rewrite to
                    // a broken stub and loops the room (the chess battletest). Read-only agents (critic/referee in
                    // plan/verify mode — write is denied) keep pressure so they conclude. Recomputed per round
                    // because an approved ExitPlanMode flips a planner from .plan (read-only) to .auto (builder).
                    let isBuilder = mutationToolNames.contains { tool in
                        if case .deny = permission.decide(tool: tool) { return false } else { return true }
                    }
                    // Work tools stay on until the round budget OR (non-builders only) the model called tools
                    // error-free for a while without finishing (over-eager rambling) — then drop WORK tools to
                    // force a conclusion. Routing tools ride through (the backend keeps offering them when canRoute).
                    let toolsOn = round < maxToolRounds && (isBuilder || quietToolRounds < defaultMaxQuietToolRounds)
                    // Images ride the FIRST (user) round only — the owned transcript then carries them forward
                    // across rounds (TurnMessage.imageURLs), so later rounds must not re-attach them.
                    let stream = backend.round(instructions: instr, prompt: prompt,
                                               resume: resume, toolsEnabled: toolsOn,
                                               images: resume.isEmpty ? images : [])
                    resume = []
                    var text = ""
                    var toolCalls: [(name: String, argsJSON: String)] = []
                    var thinkFilter = ReasoningStreamFilter()   // split <think> off the visible deltas
                    for try await step in stream {
                        switch step {
                        case .chunk(let ch):
                            text += ch; outputChars += ch.count
                            var rsn = ""
                            let vis = thinkFilter.feed(ch, reasoning: &rsn)
                            // Stream the model's thinking LIVE (per chunk) so the UI's "Reasoning" block fills in
                            // as it reasons — instead of staying blank for the whole (long) round, then dumping
                            // the reasoning at round-end. The reasoning rides the SAME toolIndex as this round.
                            if !rsn.isEmpty { continuation.yield(.reasoningDelta(rsn)) }
                            if !vis.isEmpty { continuation.yield(.textDelta(vis)) }   // live render, no <think>
                        case .toolCall(let name, let argsJSON):
                            toolCalls.append((name, argsJSON))
                        }
                    }
                    let tail = thinkFilter.flush()
                    if !tail.isEmpty { continuation.yield(.textDelta(tail)) }
                    round += 1
                    turnTrace("turn=\(config.sessionID) round=\(round) toolsOn=\(toolsOn) canRoute=\(canRoute) "
                        + "parsedCalls=\(toolCalls.count)[\(toolCalls.map(\.name).joined(separator: ","))] "
                        + "hasToolTag=\(text.contains("tool_call")) textLen=\(text.count) "
                        + "concluded=\(concludedThisTurn) nudges=\(consecutiveNudges)")

                    // (Reasoning is now streamed LIVE in the chunk loop above — no round-end batch, which would
                    // double it. The UI's "Reasoning" block fills in as the model thinks.)

                    // Completion-pressure ENFORCEMENT: rendering terminal-only specs (the backend) tells the model
                    // work tools are paused, but the xmlFunction parser still surfaces any <function=…> by NAME —
                    // so a 122B verifier keeps running perft past the budget and never concludes (it runs to the
                    // round cap → "(hit round cap without final text)"). DROP the non-terminal calls here so the
                    // turn falls into the conclusion branch below; routing calls (terminalTools) still pass, so the
                    // agent can always route. `!toolsOn` ⇒ this agent already spent ≥4 tool rounds (a verifier/
                    // builder), never a consult panelist (which answers in ≤1 round and never reaches pressure).
                    if !toolsOn, canRoute, !toolCalls.isEmpty {
                        // Keep routing tools (so the agent can conclude) AND mutation tools (a builder's edits are
                        // PROGRESS, not rambling — dropping them discards the fix and loops the room). Only the
                        // read-only "rambling" calls (read/bash/grep) are dropped, which is what steers a verifier
                        // to conclude; a builder mid-edit sails through.
                        let kept = toolCalls.filter { config.terminalTools.contains($0.name) || mutationToolNames.contains($0.name) }
                        if kept.count != toolCalls.count {
                            turnTrace("turn=\(config.sessionID) round=\(round) pressure-drop \(toolCalls.count - kept.count) read-only call(s)")
                        }
                        toolCalls = kept
                    }

                    if toolCalls.isEmpty {
                        let clean = stripReasoning(text)
                        if !clean.isEmpty { finalText = text }   // SALVAGE: keep the latest verdict text as the answer
                        // Under completion pressure (`!toolsOn`) a room agent should conclude IN ROLE — state its
                        // verdict AND route (handoff/done) — not just trail off with text. Give it up to the nudge
                        // cap to emit the routing call before we end the turn on text alone. Gated on `!toolsOn`, so
                        // it only fires for an agent that spent a full work budget (verifier/builder), not a panelist.
                        if !toolsOn, canRoute, !concludedThisTurn, consecutiveNudges < 2 {
                            consecutiveNudges += 1
                            turnTrace("turn=\(config.sessionID) round=\(round) route-nudge \(consecutiveNudges)")
                            resume = [.user("Work tools are paused — you've investigated enough. State your verdict "
                                + "in plain text, then call handoff (or done, if you are the referee and the "
                                + "objective is fully met) to pass control. Do not call any other tool.")]
                            continue
                        }
                        // Finish when: a real text answer exists; OR the agent already concluded via a routing call
                        // this turn (empty prose after handoff/done is a clean wind-down — the ExitPlanMode concern);
                        // OR we've nudged twice; OR (non-room callers only) there's no tool budget AND no routing
                        // path to conclude through (the old last-round force-answer). Bounded by ≤2 nudges + maxRounds.
                        if !clean.isEmpty || concludedThisTurn || consecutiveNudges >= 2 || (!toolsOn && !canRoute) {
                            turnTrace("turn=\(config.sessionID) BREAK round=\(round) finalLen=\(clean.count) reason="
                                + (!clean.isEmpty ? "answer" : concludedThisTurn ? "routed"
                                   : consecutiveNudges >= 2 ? "nudgeCap" : "noBudget"))
                            // Don't clobber a salvaged verdict with an empty round: only adopt THIS round's text
                            // when it has visible prose; otherwise keep the best text salvaged from a prior round.
                            if !clean.isEmpty || finalText.isEmpty { finalText = text }
                            break
                        }
                        consecutiveNudges += 1
                        resume = [.user(canRoute
                            ? "You ended your turn without concluding. If your work or verification is finished, "
                              + "state your result/verdict in plain text AND pass control now by calling handoff "
                              + "(or done, if you are the referee and the objective is fully met). If a step "
                              + "remains, call that tool now."
                            : "Continue. If you intended to run or check something, call that tool now. If the "
                              + "task is complete, reply with a brief plain-text summary.")]
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
                        if permission.mode == .plan, permission.allowsPlanExit, call.name == exitPlanModeToolName {
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
                        // Duplicate call → no new information. A FILE MUTATION clears the seen-set (so a prior
                        // read/test re-runs against the changed files: edit → re-run the same test → allowed);
                        // everything else — reads AND bash — is dedup-checked, so an identical command with
                        // nothing changed since is blocked (kills the 316×-debug-script spin).
                        roundProgressed = true
                        if mutationToolNames.contains(call.name) {
                            seenCalls.removeAll(keepingCapacity: true)
                        } else if !seenCalls.insert(toolCallKey(name: call.name, argsJSON: call.argsJSON)).inserted {
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
                        // `consult` is a deliberate fan-out BARRIER that blocks until the whole panel replies; a LOCAL
                        // panel (N×122B serialized on one shared container) legitimately takes MINUTES, far past the
                        // 90s tool timeout — so it must be exempt too, else the consulting agent gives up and the
                        // panel's replies come back orphaned/unused (found in the chess stress test). The panel is
                        // bounded by the panelists' own per-turn limits + the room autopilot, so no-timeout can't hang.
                        let dispatchTimeout = (callName == "Task" || callName == "consult") ? 0 : config.toolTimeoutSeconds
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
                    // A mutation (write_file/edit_file) is real new direction — reset the quiet budget like an
                    // error/plan-approval does, so an actively-building agent never hits completion pressure.
                    let didMutate = toolCalls.contains { mutationToolNames.contains($0.name) }
                    if roundHadError || roundApproved || didMutate { quietToolRounds = 0 }
                    else if roundProgressed { quietToolRounds += 1 }
                    // A routing/terminal call (handoff/done/consult) — or an ExitPlanMode — means the agent has
                    // concluded this turn; an empty follow-up round is then a clean wind-down, not a turn to
                    // nudge (#2's guard against forcing an extra round after a tool that legitimately ends one).
                    if toolCalls.contains(where: { config.terminalTools.contains($0.name) || $0.name == exitPlanModeToolName }) {
                        concludedThisTurn = true
                    }
                    resume = results
                }

                // Strip reasoning (with the conclusion fallback) so the final result is clean for Orbital.
                let stripped = displayAnswer(finalText)
                let answer = !stripped.isEmpty ? stripped
                    : (finalText.isEmpty ? "(hit round cap without final text)" : "(model emitted only reasoning)")
                // SEAM E — Stop: record the completed turn (memory-write, trace logging).
                if let hooks { _ = await hooks.fire(.stop(answer: answer, toolsCalled: toolsCalled)) }
                // T1: real running context size from the backend (CompactingSession's last `.info`) →
                // `inputTokens` is the honest window FILL Orbital maps to `TurnTelemetry.contextTokens`.
                // Falls back to the chars/4 estimate only for a stub backend that can't measure.
                let ctxTokens = backend.finalContextTokens()
                continuation.yield(.result(finalText: answer, inputTokens: ctxTokens ?? 0,
                                           outputTokens: max(0, outputChars / 4), isError: false))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}
