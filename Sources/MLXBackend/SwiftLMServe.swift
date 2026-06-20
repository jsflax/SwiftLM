import Foundation
import MLXLMCommon
// Re-exported so a downstream linker of THIS product (Orbital's orbital-loop) gets the neutral agent-turn
// surface — AgentStreamEvent / streamAgentTurn / AgentTurnBackend / AgentTurnConfig / ToolPermissionPolicy
// (Serving) and NativeTool / NativeToolRegistry / ToolArguments / JSONSchemaObject (NativeTools) — from a
// single `import MLXBackend`. These ARE MLXBackend's API surface (what `makeAgentBackend`/`streamAgent`
// return and consume), so re-exporting them here is the right seam (no Package product churn).
@_exported import Serving
@_exported import NativeTools
import MiniBPE

// ── The Orbital-facing serve surface: an in-process, claude-compatible streaming agent turn.
//
// `streamAgent` adapts the MLX model into the pure `streamAgentTurn` sequencer (Serving) by supplying the
// two real-model effects — streaming a round (ChatSession) and dispatching a tool (MCPHost) — via
// `ChatSessionTurnBackend`. The sequencer owns the round loop, the four hook seams, and the
// AgentStreamEvent emission; each event serializes (.ndjsonLine()) to the exact claude `stream-json` line
// Orbital's `StreamJsonEvent` decoder consumes. So a local room agent gets the user's ~/.claude hooks
// (Engram grounding, etc.) identically to `claude -p`, and the turn-orchestration is unit-tested without
// loading a model (AgentTurnTests stubs the same backend).
//
// Orbital-agnostic: the CALLER builds the MCPHost (connecting orbital-mcp for handoff/consult/done + any
// other servers) and the HookChain (from ~/.claude/settings.json); streamAgent just runs that surface.
extension MLXLanguageModel {
    public func streamAgent(
        _ prompt: String,
        host: MCPHost,
        instructions: String? = nil,
        sessionID: String = UUID().uuidString,
        cwd: String = FileManager.default.currentDirectoryPath,
        modelLabel: String = "swiftlm-mlx",
        maxRounds: Int = 8,
        hooks: HookChain? = nil,
        permission: ToolPermissionPolicy = .init(),
        approvePlan: PlanApprover? = nil,
        toolTimeoutSeconds: Double = defaultToolDispatchTimeoutSeconds
    ) async -> AsyncThrowingStream<AgentStreamEvent, Error> {
        // One-shot convenience: build a FRESH backend per call (no persistence). Long-lived callers that
        // need per-agent KV + compaction continuity across turns (Orbital's SharedMLXScheduler) hold a
        // `makeAgentBackend(...)` themselves and drive `streamAgentTurn` directly.
        let backend = await makeAgentBackend(host: host)
        let config = AgentTurnConfig(modelLabel: modelLabel, sessionID: sessionID, cwd: cwd,
                                     toolNames: await host.toolNames, maxRounds: maxRounds, permission: permission,
                                     approvePlan: approvePlan, toolTimeoutSeconds: toolTimeoutSeconds)
        return streamAgentTurn(prompt: prompt, instructions: instructions, hooks: hooks,
                               config: config, backend: backend)
    }

    /// Build a PERSISTENT agent-turn backend (ChatSession + MCPHost) the CALLER holds across turns, so a
    /// room agent's KV cache + Claude-Code-style compaction history survive between its turns. N backends
    /// built over ONE shared `MLXLanguageModel` ⇒ ONE `ModelContainer`'s weights — per-agent incremental
    /// cost is only that backend's KV cache + message history (the shared-container floor). Orbital's
    /// SharedMLXScheduler keys one of these per agentId and drives turns via `streamAgentTurn`.
    ///
    /// Profile detection, grammar tokenizer, and generation params are set up HERE (SwiftLM-internal model
    /// knowledge) exactly as the live REPL loop does, so a room agent gets constrained valid tool calls.
    /// `batchGenerator` (B2 co-batching) is forwarded into the round's CompactingSession with the model's
    /// own tool-call parser; nil ⇒ the serial incremental-KV ChatSession path (B1), byte-for-byte the loop.
    public func makeAgentBackend(host: MCPHost, batchGenerator: BatchGenerator? = nil) async -> any AgentTurnBackend {
        let specs = await host.specs
        let toolNames = await host.toolNames
        let profile = await self.profile
        let grammarTok = try? MiniBPE.grammarTokenizer(forModelId: modelId)
        var params = GenerateParameters(maxTokens: profile.budget.maxTokens, temperature: 0.0)
        params.repetitionPenalty = 1.15
        params.repetitionContextSize = 20
        // Only the batched path needs a standalone parser (ChatSession parses inline on the serial path).
        let parser = batchGenerator != nil ? (await container.configuration.toolCallFormat)?.createParser() : nil
        return ChatSessionTurnBackend(model: self, host: host, specs: specs, toolNames: toolNames,
                                      params: params, profile: profile, grammarTokenizer: grammarTok,
                                      batchGenerator: batchGenerator, toolCallParser: parser)
    }
}

/// AgentTurnBackend backed by an MLX ChatSession + MCPHost — the real-model effects the pure sequencer
/// delegates. Holds the session across rounds so tool-result continuations keep context; the sequencer
/// calls `round` strictly sequentially, so the mutable `session` is never touched concurrently. Brings the
/// streaming path to parity with the live loop: GLM bare-name tool-call recovery + grammar-constrained args.
final class ChatSessionTurnBackend: AgentTurnBackend, @unchecked Sendable {
    let model: MLXLanguageModel
    let host: MCPHost
    let specs: [ToolSpec]
    let toolNames: [String]
    let params: GenerateParameters
    let profile: ModelProfile
    let grammarTokenizer: (any GrammarTokenizer)?
    let batchGenerator: BatchGenerator?          // B2 co-batch DI seam (nil ⇒ serial incremental-KV path)
    let toolCallParser: (any ToolCallParser)?    // standalone parser for the batched path (nil on serial)
    private var compacting: CompactingSession?   // persistent conversation + Claude-Code-style compaction
    private var basePrompt = ""   // the original user prompt — used to constrain regenerated tool args

    init(model: MLXLanguageModel, host: MCPHost, specs: [ToolSpec], toolNames: [String],
         params: GenerateParameters, profile: ModelProfile, grammarTokenizer: (any GrammarTokenizer)?,
         batchGenerator: BatchGenerator? = nil, toolCallParser: (any ToolCallParser)? = nil) {
        self.model = model; self.host = host; self.specs = specs; self.toolNames = toolNames
        self.params = params; self.profile = profile; self.grammarTokenizer = grammarTokenizer
        self.batchGenerator = batchGenerator; self.toolCallParser = toolCallParser
    }

    /// The real running context size (last round's prompt+gen tokens, from `Generation.info`) — honest
    /// telemetry the sequencer reads at result-time instead of the hardcoded `0` (OPEN ITEM T1). nil until
    /// the first round measures it.
    func finalContextTokens() -> Int? { compacting?.contextTokens }

    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error> {
        if resume.isEmpty { basePrompt = prompt }   // remember the user prompt for arg constraining
        if compacting == nil {
            compacting = CompactingSession(model: model, instructions: instructions, params: params,
                                           specs: specs, tokenizer: grammarTokenizer, budget: profile.contextBudget,
                                           batchGen: batchGenerator, toolCallParser: toolCallParser)
        }
        let comp = compacting!
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Build the round's input here (not captured) — Chat.Message isn't Sendable; resume/prompt are.
                    let input: [Chat.Message] = resume.isEmpty
                        ? [.user(prompt)]
                        : resume.map { msg -> Chat.Message in
                            switch msg {
                            case .toolResult(let s): return .tool(s)
                            case .user(let s): return .user(s)
                            }
                        }
                    // Collect tool calls during the stream (text streams live); constrain AFTER it finishes so
                    // the arg-regeneration generations don't contend with the in-flight ChatSession.
                    let stream = await comp.beginRound(input, toolsEnabled: toolsEnabled)
                    var text = ""
                    var raw: [(String, String)] = []
                    var ctxTokens: Int? = nil
                    for try await g in stream {
                        if let tc = g.toolCall {
                            raw.append((tc.function.name, MLXLanguageModel.argsJSON(tc.function.arguments)))
                        } else if let ch = g.chunk {
                            text += ch
                            continuation.yield(.chunk(ch))
                        }
                        if case .info(let info) = g { ctxTokens = info.promptTokenCount + info.generationTokenCount }
                    }
                    comp.finishRound(assistantText: text, contextTokens: ctxTokens)
                    // GLM bare-name `<tool_call>` the parser didn't surface (profile-driven; only with tools on).
                    if raw.isEmpty, toolsEnabled, let tag = profile.recoverMissedToolCall(text) {
                        raw.append((tag.name, tag.argsJSON))
                    }
                    for (n0, a0) in raw {
                        let (n, a) = await constrain(n0, a0)
                        continuation.yield(.toolCall(name: n, argsJSON: a))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Make a routed tool call valid by construction (name ∈ tools, args ∈ schema) via the grammar; falls
    /// back to the routed call when there's no grammar tokenizer or the constrain fails.
    private func constrain(_ name: String, _ argsJSON: String) async -> (String, String) {
        guard let grammarTokenizer,
              let c = try? await model.constrainRoutedToolCall(
                  routedName: name, routedArgsJSON: argsJSON, prompt: basePrompt, specs: specs,
                  toolNames: toolNames, grammarTokenizer: grammarTokenizer)
        else { return (name, argsJSON) }
        return (c.name, c.argsJSON)
    }

    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool) {
        do {
            var result = try await host.dispatch(name: name, argumentsJSON: argsJSON)
            // Cap a big tool output before it enters context (the marker tells the model it was truncated).
            if let grammarTokenizer {
                result = truncateToTokens(result, maxTokens: profile.contextBudget.maxToolOutputTokens,
                                          tokenizer: grammarTokenizer)
            }
            return (result, result.hasPrefix("ERROR"))
        } catch {
            return ("ERROR: \(error)", true)
        }
    }
}

// ── A real Foundation.Process runner for Claude-style external hooks (the de-Engram path). Claude hook
// commands are SHELL lines (`/Users/.../memory-hooks advise 2>/dev/null`, redirects + inline args), so
// they run via `/bin/sh -c`. The event JSON goes in on stdin; stdout is parsed by `HookResult.decode`.
// Killed after `timeoutSeconds` (Claude's per-hook `timeout`) so a hung hook never stalls a turn.
public enum ClaudeHookRunner {
    public static let run: ExternalCommandHook.Runner = { command, _, stdinJSON, timeoutSeconds in
        await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            DispatchQueue.global().async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/bin/sh")
                proc.arguments = ["-c", command]
                let inPipe = Pipe(), outPipe = Pipe()
                proc.standardInput = inPipe
                proc.standardOutput = outPipe
                proc.standardError = FileHandle.nullDevice
                do { try proc.run() } catch { cont.resume(returning: "{}"); return }

                inPipe.fileHandleForWriting.write(Data(stdinJSON.utf8))
                try? inPipe.fileHandleForWriting.close()

                let timer = DispatchSource.makeTimerSource(queue: .global())
                let timedOut = TimeoutFlag()
                if let t = timeoutSeconds, t > 0 {
                    timer.schedule(deadline: .now() + .seconds(t))
                    timer.setEventHandler { timedOut.tripped = true; proc.terminate() }
                    timer.resume()
                }
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()  // blocks to EOF
                proc.waitUntilExit()
                timer.cancel()
                let out = String(data: outData, encoding: .utf8) ?? ""
                cont.resume(returning: timedOut.tripped ? "{}" : out)   // a killed hook contributes nothing
            }
        }
    }

    /// Build the hook chain from the user's `~/.claude/settings.json` — the de-Engram path: Engram's
    /// `memory-hooks advise` (and every other `~/.claude` hook) fires exactly as it does for `claude -p`,
    /// with this real Process runner. Returns nil when no settings/hooks are present (run hook-less).
    public static func loadChain(
        settingsPath: String = ("~/.claude/settings.json" as NSString).expandingTildeInPath,
        sessionID: String = UUID().uuidString,
        cwd: String = FileManager.default.currentDirectoryPath
    ) -> HookChain? {
        let ctx = HookContext(sessionID: sessionID, cwd: cwd)
        let hooks = HookConfig.loadClaudeSettings(path: settingsPath).externalHooks(context: ctx, run: run)
        return hooks.isEmpty ? nil : HookChain(hooks)
    }

    /// One-shot box for the watchdog→reader handoff (set on the timer queue, read after waitUntilExit).
    private final class TimeoutFlag: @unchecked Sendable { var tripped = false }
}
