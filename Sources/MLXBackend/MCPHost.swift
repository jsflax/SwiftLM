import Foundation
import MLXLMCommon
import MCP
import System
import Serving
import NativeTools
import MiniBPE

// MCP tool layer for the MLX backend — the validated S6 (toolspike) loop, productized.
// MCPHost spawns/connects MCP servers and aggregates their tools; `runWithTools`
// drives the round-capped agent loop (emit tool_call -> dispatch -> inject -> finish).

/// Convert an MCP `inputSchema` (Value) into a Sendable JSON tree for a ToolSpec.
func sendableJSON(_ v: MCP.Value) -> any Sendable {
    switch v {
    case .null: return "null"
    case .bool(let b): return b
    case .int(let i): return i
    case .double(let d): return d
    case .string(let s): return s
    case .data(_, let d): return d.base64EncodedString()
    case .array(let a): return a.map { sendableJSON($0) }
    case .object(let o): return o.mapValues { sendableJSON($0) }
    }
}

public enum MCPHostError: Error, CustomStringConvertible {
    case unsupportedTransport(String)
    public var description: String {
        switch self {
        case .unsupportedTransport(let t): return "unsupported MCP transport '\(t)' (stdio only)"
        }
    }
}

/// Hosts one or more MCP servers, exposes their tools to the model, and dispatches calls.
public actor MCPHost {
    public struct ServerConfig: Sendable {
        public let name: String          // server label → tools are namespaced `mcp__<name>__<tool>` (Claude-faithful)
        public let command: String
        public let args: [String]
        public let env: [String: String]
        public init(name: String, command: String, args: [String] = [], env: [String: String] = [:]) {
            self.name = name; self.command = command; self.args = args; self.env = env
        }
    }

    private var processes: [Process] = []
    private var dispatchMap: [String: Client] = [:]      // namespaced tool name -> owning client
    private var originalToolName: [String: String] = [:] // namespaced name -> the server's bare tool name
    public private(set) var specs: [ToolSpec] = []       // OpenAI fn-schema for the model
    private var native: NativeToolRegistry?              // built-in in-process tools (priority over MCP)

    public init() {}

    /// Register built-in native tools (Read/Write/Edit/Glob/Grep/Bash). They join the model's tool surface
    /// alongside MCP tools and take dispatch priority by name (so a native `bash` shadows any MCP `bash`).
    public func registerNative(_ registry: NativeToolRegistry) {
        // Idempotent: drop any previously-registered native specs before adding the new set, so re-registering
        // (e.g. to add the `Task` tool once the sub-agent runner exists) doesn't duplicate them.
        let drop = Set((native?.names ?? []) + registry.names)
        specs.removeAll { spec in
            ((spec["function"] as? [String: any Sendable])?["name"] as? String).map(drop.contains) ?? false
        }
        native = registry
        specs.append(contentsOf: registry.specs())   // [[String: any Sendable]] == [ToolSpec]
    }

    /// Spawn an MCP server over stdio, connect, and register its tools — namespaced `mcp__<server>__<tool>`
    /// (Claude-Code-faithful, so the user's `mcp__memory__*` allow-list + agent prompts line up). Native
    /// tools stay bare. Returns the namespaced tool names.
    @discardableResult
    public func connect(_ config: ServerConfig) async throws -> [String] {
        let inPipe = Pipe(), outPipe = Pipe()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: config.command)
        proc.arguments = config.args
        if !config.env.isEmpty {
            var e = ProcessInfo.processInfo.environment
            for (k, v) in config.env { e[k] = v }
            proc.environment = e
        }
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        try proc.run()
        processes.append(proc)

        let transport = StdioTransport(
            input: FileDescriptor(rawValue: outPipe.fileHandleForReading.fileDescriptor),
            output: FileDescriptor(rawValue: inPipe.fileHandleForWriting.fileDescriptor))
        let client = Client(name: "swiftlm-agent", version: "0.1.0")
        _ = try await client.connect(transport: transport)

        let (tools, _) = try await client.listTools()
        var names: [String] = []
        for t in tools {
            let namespaced = "mcp__\(config.name)__\(t.name)"
            names.append(namespaced)
            dispatchMap[namespaced] = client
            originalToolName[namespaced] = t.name
            let params = (sendableJSON(t.inputSchema) as? [String: any Sendable])
                ?? ["type": "object", "properties": [String: any Sendable]()]
            specs.append([
                "type": "function",
                "function": [
                    "name": namespaced,
                    "description": t.description ?? "",
                    "parameters": params,
                ] as [String: any Sendable],
            ])
        }
        return names
    }

    /// Connect every stdio MCP server configured in Claude's `~/.claude.json` (`mcpServers`) — the same
    /// servers `claude -p` sees, so `memory` and friends become real `mcp__<server>__*` tools. `http`
    /// servers are skipped (stdio transport only). `allowlist` (server names) scopes which to connect.
    /// Returns a per-server result so the caller can log what connected / failed / was skipped.
    public func connectClaudeServers(
        configPath: String = ("~/.claude.json" as NSString).expandingTildeInPath,
        allowlist: Set<String>? = nil
    ) async -> [(name: String, outcome: Result<[String], Error>)] {
        guard let data = FileManager.default.contents(atPath: configPath),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let servers = root["mcpServers"] as? [String: Any] else { return [] }
        var results: [(name: String, outcome: Result<[String], Error>)] = []
        for (name, raw) in servers.sorted(by: { $0.key < $1.key }) {
            if let allowlist, !allowlist.contains(name) { continue }
            guard let entry = raw as? [String: Any] else { continue }
            let type = (entry["type"] as? String) ?? "stdio"
            guard type == "stdio", let command = entry["command"] as? String else {
                results.append((name, .failure(MCPHostError.unsupportedTransport(type))))   // e.g. http
                continue
            }
            let args = (entry["args"] as? [String]) ?? []
            let env = (entry["env"] as? [String: String]) ?? [:]
            do {
                let tools = try await connect(.init(name: name, command: command, args: args, env: env))
                results.append((name, .success(tools)))
            } catch {
                results.append((name, .failure(error)))
            }
        }
        return results
    }

    /// Dispatch a model-emitted tool call to the owning MCP server; return result text.
    public func dispatch(name: String, arguments: [String: JSONValue]) async throws -> String {
        if let native, let r = await native.dispatch(name: name, argsJSON: MLXLanguageModel.argsJSON(arguments)) {
            return r   // native tool owns this name
        }
        guard let client = dispatchMap[name] else { return "ERROR: unknown tool \(name)" }
        let argsAny = arguments.mapValues { $0.anyValue }
        let mcpArgs: [String: MCP.Value]? = argsAny.isEmpty
            ? nil
            : try? JSONDecoder().decode([String: MCP.Value].self,
                                        from: JSONSerialization.data(withJSONObject: argsAny))
        return try await call(client: client, name: originalToolName[name] ?? name, mcpArgs: mcpArgs)
    }

    /// Dispatch with arguments as a raw JSON object string — the form the streaming serve loop
    /// (`streamAgentTurn`) carries on the wire. Equivalent to `dispatch(name:arguments:)`; both
    /// round-trip through JSON to MCP.Value.
    public func dispatch(name: String, argumentsJSON: String) async throws -> String {
        if let native, let r = await native.dispatch(name: name, argsJSON: argumentsJSON) {
            return r   // native tool owns this name
        }
        guard let client = dispatchMap[name] else { return "ERROR: unknown tool \(name)" }
        let trimmed = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        let mcpArgs: [String: MCP.Value]? = (trimmed.isEmpty || trimmed == "{}")
            ? nil
            : try? JSONDecoder().decode([String: MCP.Value].self, from: Data(trimmed.utf8))
        return try await call(client: client, name: originalToolName[name] ?? name, mcpArgs: mcpArgs)
    }

    private func call(client: Client, name: String, mcpArgs: [String: MCP.Value]?) async throws -> String {
        let (content, isError) = try await client.callTool(name: name, arguments: mcpArgs)
        let text = content.compactMap { c -> String? in
            if case let .text(t, _, _) = c { return t }
            return nil
        }.joined(separator: "\n")
        return (isError == true ? "ERROR: " : "") + text
    }

    /// The names of every tool currently on the surface (native + MCP) — the runtime set a constrained
    /// tool call's `name` must come from (`ConditionalGrammarProcessor(runtimeEnums: ["name": toolNames])`).
    public var toolNames: [String] {
        specs.compactMap { ($0["function"] as? [String: any Sendable])?["name"] as? String }
    }

    public func shutdown() {
        for p in processes where p.isRunning { p.terminate() }
        processes.removeAll()
    }
}

extension MLXLanguageModel {
    /// Round-capped agent loop (S6): the model emits tool calls, we dispatch them via
    /// the MCP host, inject results, and continue. Once a result is in hand we drop the
    /// tools so the model is forced to produce a final text answer (no infinite re-call).
    @discardableResult
    public func runWithTools(
        _ prompt: String,
        host: MCPHost,
        instructions: String? = nil,
        maxRounds: Int = 5,
        hooks: HookChain? = nil,
        grammarTokenizer: (any GrammarTokenizer)? = nil,
        profile: ModelProfile = .generic,
        permission: ToolPermissionPolicy = .init(),
        approvePlan: PlanApprover? = nil,
        toolTimeoutSeconds: Double = defaultToolDispatchTimeoutSeconds,
        toolAllowlist: Set<String>? = nil,
        batchGenerator: BatchGenerator? = nil
    ) async throws -> String {
        try await runWithToolsTracked(
            prompt, host: host, instructions: instructions,
            maxRounds: maxRounds, hooks: hooks,
            grammarTokenizer: grammarTokenizer, profile: profile,
            permission: permission, approvePlan: approvePlan,
            toolTimeoutSeconds: toolTimeoutSeconds, toolAllowlist: toolAllowlist,
            batchGenerator: batchGenerator).answer
    }

    /// Like `runWithTools`, but also returns the names of every tool the model invoked —
    /// the signal the correctness (tool-pass@1) gate scores against.
    public func runWithToolsTracked(
        _ prompt: String,
        host: MCPHost,
        instructions: String? = nil,
        maxRounds: Int = 5,
        hooks: HookChain? = nil,
        grammarTokenizer: (any GrammarTokenizer)? = nil,
        profile: ModelProfile = .generic,
        permission: ToolPermissionPolicy = .init(),
        approvePlan: PlanApprover? = nil,
        toolTimeoutSeconds: Double = defaultToolDispatchTimeoutSeconds,
        toolAllowlist: Set<String>? = nil,
        batchGenerator: BatchGenerator? = nil
    ) async throws -> (answer: String, toolsCalled: [String]) {
        // Scope the tool surface (sub-agent isolation): only the allow-listed tools are advertised + dispatchable.
        let specs = await { () -> [ToolSpec] in
            let all = await host.specs
            guard let toolAllowlist else { return all }
            return all.filter { spec in
                guard let n = (spec["function"] as? [String: any Sendable])?["name"] as? String else { return false }
                return toolAllowlist.contains(n)
            }
        }()
        // Generation bound = the profile's PHYSICS-DERIVED runaway rail (latency × tok/s, clamped above a
        // reasoning floor), NOT an arbitrary cap. EOS is the real terminator; this only stops a runaway.
        var params = GenerateParameters(maxTokens: profile.budget.maxTokens, temperature: 0.0)
        params.repetitionPenalty = 1.15
        params.repetitionContextSize = 20
        // SEAM A — UserPromptSubmit: hooks (e.g. EngramAdviseHook) inject retrieval grounding / a hedge
        // into the system context before generation.
        var instr = instructions
        if let hooks {
            let r = await hooks.fire(.userPromptSubmit(prompt: prompt))
            if let ctx = r.additionalContext, !ctx.isEmpty {
                instr = (instr.map { $0 + "\n\n" } ?? "") + ctx
            }
        }
        // Plan mode: tell the model it's planning so it produces a plan rather than flailing against the
        // write/exec tools the permission policy blocks at the dispatch seam below.
        if let planNote = permission.instructionPrefix {
            instr = (instr.map { $0 + "\n\n" } ?? "") + planNote
        }
        // Mutable across rounds: an approved ExitPlanMode flips this from .plan to .auto mid-turn.
        var currentPermission = permission
        // A persistent CompactingSession (Claude-Code-style): when context crosses the budget it summarizes
        // the oldest messages, resets, and re-seeds with [summary + recent verbatim] — so a long refine turn
        // never overflows the window. The grammar tokenizer (when present) measures context exactly.
        // SLICE 3: when a batched generator is supplied (sub-agent fan-out), generation routes through the
        // coalescing pool; build the model's own tool-call parser so the round loop still gets `.toolCall`s
        // (the live/Orbital path passes nil → ChatSession streams + parses inline, unchanged).
        let toolCallParser = batchGenerator != nil
            ? (await container.configuration.toolCallFormat)?.createParser() : nil
        let compacting = CompactingSession(model: self, instructions: instr, params: params, specs: specs,
                                           tokenizer: grammarTokenizer, budget: profile.contextBudget,
                                           batchGen: batchGenerator, toolCallParser: toolCallParser)
        var input: [Chat.Message] = [.user(prompt)]
        var answer = ""
        var toolsCalled: [String] = []
        var round = 0
        var consecutiveNudges = 0   // bound on "model narrates intent but doesn't act" nudges
        var seenCalls = Set<String>()    // (name+args) already dispatched → a repeat is non-progress
        var quietToolRounds = 0          // consecutive error-free tool rounds → completion pressure
        // Tools stay available ACROSS rounds so the model can ITERATE: call a tool, read its result/error,
        // then decide the next call (the write → compile → read-error → fix → recompile refine-on-error loop).
        // The final round reserves no tools, forcing a text answer; `maxRounds` bounds any non-terminating
        // re-call loop (the S6 "7B re-calls forever" failure is now bounded, not crudely capped at one round).
        let maxToolRounds = max(1, maxRounds - 1)
        while round < maxRounds {
            // Drop tools at the round budget OR once the model has called tools error-free for a while
            // without finishing (over-eager rambling) — then force a final answer.
            let toolsOn = round < maxToolRounds && quietToolRounds < defaultMaxQuietToolRounds
            var text = ""
            var routed: [(name: String, argsJSON: String)] = []
            var ctxTokens: Int? = nil
            for try await g in await compacting.beginRound(input, toolsEnabled: toolsOn) {
                if let tc = g.toolCall { routed.append((tc.function.name, Self.argsJSON(tc.function.arguments))) }
                else if let ch = g.chunk { text += ch }
                if case .info(let info) = g { ctxTokens = info.promptTokenCount + info.generationTokenCount }
            }
            compacting.finishRound(assistantText: text, contextTokens: ctxTokens)
            // Recover a tool call the backend parser left unsurfaced — profile-driven (a tagged-reasoning
            // family's bare-name `<tool_call>name</tool_call>`; raw-JSON families recover nothing). Only while
            // tools are on, so it never fires on the forced final answer.
            if toolsOn, routed.isEmpty, let tag = profile.recoverMissedToolCall(text) { routed.append(tag) }
            round += 1
            if routed.isEmpty {
                let cleanAnswer = stripReasoning(text)
                // A clean final answer, or no tool budget / nudge cap reached → finish.
                if !cleanAnswer.isEmpty || !toolsOn || consecutiveNudges >= 2 { answer = text; break }
                // Reasoning-only round with budget left: the model TRAILED OFF — narrated an intent (e.g.
                // "let me run it again") WITHOUT emitting the call. Nudge it to act or finalize; bounded by
                // ≤2 consecutive nudges (+ maxRounds), so it can't spin.
                consecutiveNudges += 1
                input = [.user("Continue. If you intended to run or check something, call that tool now. "
                               + "If the task is complete, reply with a brief plain-text summary for the user.")]
                continue
            }
            consecutiveNudges = 0   // a tool call is real progress
            var msgs: [Chat.Message] = []
            var roundHadError = false, roundApproved = false, roundProgressed = false
            for call in routed {
                // CONSTRAINED EMISSION: when a grammar tokenizer is supplied, make the routed call valid by
                // construction — name ∈ the real tools, args ∈ the tool's schema — preserving the model's own
                // values; without one, dispatch the routed call as-is. The decision to call is the model's.
                var callName = call.name
                var argsJSON = call.argsJSON
                let runtimeToolNames: [String]
                if let toolAllowlist { runtimeToolNames = Array(toolAllowlist) }
                else { runtimeToolNames = await host.toolNames }
                if let grammarTokenizer,
                   let c = try? await constrainRoutedToolCall(
                       routedName: callName, routedArgsJSON: call.argsJSON, prompt: prompt, specs: specs,
                       toolNames: runtimeToolNames, grammarTokenizer: grammarTokenizer) {
                    callName = c.name
                    argsJSON = c.argsJSON
                }
                toolsCalled.append(callName)
                if ProcessInfo.processInfo.environment["SWIFTLM_TOOL_DEBUG"] != nil {
                    let r = call.name == callName ? "" : " (routed=\(call.name))"
                    FileHandle.standardError.write(Data("[tool] \(callName)\(r) args=\(argsJSON)\n".utf8))
                }
                // Sub-agent tool scoping: a tool outside this turn's allowlist is not dispatched.
                if let toolAllowlist, !toolAllowlist.contains(callName) {
                    roundHadError = true
                    msgs.append(.tool("Tool `\(callName)` is not available to this agent. Available: "
                                      + "\(toolAllowlist.sorted().prefix(40).joined(separator: ", "))."))
                    continue
                }
                // ExitPlanMode: the plan is ready → surface it for approval. Approve flips this turn to
                // .auto (tools go live, the model implements on the next round); reject keeps plan mode.
                // Intercepted BEFORE the gate so the (non-read-only) ExitPlanMode call is never blocked.
                if currentPermission.mode == .plan, callName == exitPlanModeToolName {
                    let plan = extractPlan(fromArgsJSON: argsJSON)
                    switch await (approvePlan?(plan) ?? .approve) {
                    case .approve:
                        currentPermission = currentPermission.approved
                        roundApproved = true   // fresh budget for the implementation that follows
                        msgs.append(.tool("Plan approved by the user. You are now in execution mode — use "
                                          + "the tools to implement the plan."))
                    case .reject(let reason):
                        roundHadError = true
                        msgs.append(.tool("Plan rejected: \(reason). Revise your plan and call "
                                          + "\(exitPlanModeToolName) again."))
                    }
                    continue
                }
                // SEAM B — Permission ladder: plan mode blocks side-effecting tools before any hook or
                // dispatch (approval clamps to allow under autopilot). The block is fed back as a tool
                // result so the model sees WHY and plans the step instead of calling it.
                if case .deny(let reason) = currentPermission.decide(tool: callName) {
                    roundHadError = true
                    msgs.append(.tool("BLOCKED (\(currentPermission.mode.rawValue)): \(reason)"))
                    continue
                }
                // SEAM C — PreToolUse: a hook may veto the call (e.g. plan-mode read-only).
                if let hooks {
                    let pre = await hooks.fire(.preToolUse(tool: callName, argumentsJSON: argsJSON))
                    if case .deny(let reason) = pre.decision {
                        roundHadError = true
                        msgs.append(.tool("BLOCKED by hook: \(reason)"))
                        continue
                    }
                }
                // Duplicate call → no new information. Don't re-dispatch; push toward finishing.
                roundProgressed = true
                if !seenCalls.insert(toolCallKey(name: callName, argsJSON: argsJSON)).inserted {
                    msgs.append(.tool("You already called `\(callName)` with these arguments; the result is "
                                      + "unchanged. Do not repeat it. If the task is complete, give your final answer."))
                    continue
                }
                // Bound the dispatch so a wedged tool (e.g. a hung MCP call) is abandoned and reported as
                // non-responsive instead of stalling the whole turn. A timeout is NOT flagged as an error
                // (so it doesn't invite an endless retry) — it counts toward the completion cap below.
                // EXCEPTION: `Task` runs a whole nested sub-agent turn (minutes, legitimately) — it self-bounds
                // via the child's own maxRounds + per-dispatch timeouts + completion-stop, so the outer wall-
                // clock bound is DISABLED for it (0), else a real sub-agent gets killed mid-turn.
                let dn = callName, da = argsJSON
                let dispatchTimeout = (callName == "Task") ? 0 : toolTimeoutSeconds
                var result = await withToolTimeout(seconds: dispatchTimeout,
                                                   { (try? await host.dispatch(name: dn, argumentsJSON: da))
                                                       ?? "ERROR: tool \(dn) failed" })
                    ?? "Tool `\(callName)` did not respond within \(Int(toolTimeoutSeconds))s and was skipped. "
                       + "Do not retry it; continue or give your final answer."
                if ProcessInfo.processInfo.environment["SWIFTLM_TOOL_DEBUG"] != nil {
                    FileHandle.standardError.write(Data("[tool] → \(result.prefix(120))\n".utf8))
                }
                // SEAM D — PostToolUse: a hook may transform the tool result (e.g. compile-check feedback).
                if let hooks {
                    let post = await hooks.fire(.postToolUse(tool: callName, result: result))
                    if let rep = post.replacementResult { result = rep }
                }
                // Cap a big tool output (a 10MB read / huge grep) before it enters context — the marker tells
                // the model it was truncated so it can re-query more narrowly.
                if let grammarTokenizer {
                    result = truncateToTokens(result, maxTokens: profile.contextBudget.maxToolOutputTokens,
                                              tokenizer: grammarTokenizer)
                }
                if result.hasPrefix("ERROR") { roundHadError = true }
                msgs.append(.tool(result))
            }
            // Completion pressure: a clean (error-free) tool round with no new direction ticks toward forcing
            // an answer; an error or a fresh plan-approval resets the budget (real progress / refine-in-flight).
            if roundHadError || roundApproved { quietToolRounds = 0 }
            else if roundProgressed { quietToolRounds += 1 }
            input = msgs   // tool results feed the next round; the model iterates on them
        }
        // Strip reasoning (`<think>…</think>`) + stray tool-call tags from what the caller/user sees.
        let stripped = profile.stripForDisplay(answer)
        let finalAnswer = !stripped.isEmpty ? stripped
            : (answer.isEmpty ? "(hit round cap without final text)" : "(model emitted only reasoning)")
        // SEAM E — Stop: a hook may record the completed turn (memory-write, trace logging).
        if let hooks { _ = await hooks.fire(.stop(answer: finalAnswer, toolsCalled: toolsCalled)) }
        return (finalAnswer, toolsCalled)
    }

    /// Encode an MLX tool-call's arguments to a JSON string for PreToolUse hooks (Serving is MLX-free,
    /// so the event carries JSON text, not the MLX JSONValue type).
    static func argsJSON(_ args: [String: JSONValue]) -> String {
        let obj = args.mapValues { $0.anyValue }
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: data, encoding: .utf8) else { return "{}" }
        return s
    }
}
