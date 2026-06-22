import Foundation
import MLXLMCommon
import MiniBPE
import Serving

// ── Claude-Code-style context compaction. A persistent conversation that, when it crosses the context
// budget, summarizes the OLDEST messages, "resets" the session, and re-seeds it with [summary + recent
// messages VERBATIM], then continues. The one mlx adaptation: `ChatSession` history is OPAQUE (no mutate),
// so "reset" = build a FRESH `ChatSession` via its `history:` constructor seeded with the summary. Bounds
// context so a local agent survives a long room / a long refine turn without overflowing the window —
// mlx-swift-lm throws no overflow error, it would just silently degrade.
//
// Drives BOTH agent loops: runWithToolsTracked (live REPL) and ChatSessionTurnBackend (the Orbital stream).
// The sequencer calls `stream` strictly sequentially, so the mutable state is never touched concurrently.

/// A batched generation primitive (SLICE 3 DI seam): render → `(promptTokens, maxTokens) -> generated text`.
/// When injected, a round's generation is routed through a COALESCING pool so concurrent sub-agents fuse
/// into one forward pass; when nil, the round streams inline via `ChatSession` (main + Orbital paths, byte
/// for byte unchanged). Sub-agents trade ChatSession's incremental KV (re-prefill per round) for fan-out
/// batching — cheap because sub-agent turns are short.
public typealias BatchGenerator = @Sendable ([Int32], Int) async -> String

final class CompactingSession: @unchecked Sendable {
    private let model: MLXLanguageModel
    private let instructions: String?
    private let params: GenerateParameters
    private let specs: [ToolSpec]
    private let tokenizer: (any GrammarTokenizer)?
    private let budget: ContextBudget
    private let batchGen: BatchGenerator?
    private let toolCallParser: (any ToolCallParser)?
    private let adapter: ModelProfile           // the per-model harness (drives owned render / stops / reasoning)
    private var session: ChatSession
    private var messages: [Chat.Message] = []   // tracked conversation (excludes the system instructions)
    private var structuredTurns: [TurnMessage] = []   // OWNED-RENDER transcript (carries reasoning_content + tool_calls)
    private var lastOwnedToolCalls: [Serving.ToolCall]? = nil // the call parsed this round, recorded into the next assistant turn
    private(set) var contextTokens = 0          // last measured prompt+gen tokens (from Generation.info)
    private(set) var compactions = 0            // count, for the demo / observability

    init(model: MLXLanguageModel, instructions: String?, params: GenerateParameters,
         specs: [ToolSpec], tokenizer: (any GrammarTokenizer)?, budget: ContextBudget,
         adapter: ModelProfile = .generic,
         batchGen: BatchGenerator? = nil, toolCallParser: (any ToolCallParser)? = nil) {
        var p = params
        if let kv = budget.maxKVSize { p.maxKVSize = kv }   // L3: RotatingKVCache memory floor
        self.model = model
        self.instructions = instructions
        self.params = p
        self.specs = specs
        self.tokenizer = tokenizer
        self.budget = budget
        self.adapter = adapter
        self.batchGen = batchGen
        self.toolCallParser = toolCallParser
        self.session = ChatSession(model.container, instructions: instructions,
                                   generateParameters: p, tools: specs)
    }

    /// Prepare a round and return the model's generation stream for the caller to CONSUME directly (so it
    /// streams live). `input` = the NEW messages this round (first: `[.user(prompt)]`; later: tool results or
    /// a `.user` nudge). Compacts BEFORE generating if the running context exceeds the budget. The caller must
    /// call `finishRound` after consuming the stream (passing the assistant text + the `.info` token count).
    /// Split this way (vs a callback) so no closure crosses into an async method — keeps Swift concurrency
    /// happy while preserving live streaming.
    func beginRound(_ input: [Chat.Message], toolsEnabled: Bool) async -> AsyncThrowingStream<Generation, Error> {
        // OWNED RENDER (strict reasoning+tool templates, e.g. the 122B): bypass ChatSession's lossy incremental
        // restart (which drops the user query → "No user query found") and instead re-render the WHOLE
        // structured transcript each round through the model's real template, decoding live via streamFromTokens.
        if adapter.requiresOwnedRender { return await ownedRound(input, toolsEnabled: toolsEnabled) }
        if contextTokens > budget.maxContextTokens { await compact() }
        messages.append(contentsOf: input)
        // SLICE 3 batched path: render the WHOLE conversation (+ tools) to tokens, generate through the
        // coalescing pool (so concurrent sub-agents fuse), parse the tool call from text via the model's own
        // parser, and surface the same `Generation` shape the round loop already consumes. Compaction still
        // works — it edits `messages`, which this render reads. (Re-prefill per round; no ChatSession KV reuse.)
        if let batchGen {
            let convo = (instructions.map { [Chat.Message.system($0)] } ?? []) + messages
            let pairs = convo.map { (role: $0.role.rawValue, content: $0.content) }
            let tokens = (try? await model.renderConversationTokens(pairs, tools: toolsEnabled ? specs : nil)) ?? []
            contextTokens = tokens.count
            let maxTok = params.maxTokens ?? 512
            let text = await batchGen(tokens, maxTok)
            // WITH-arg tool calls parse here (the same parser ChatSession uses). A NO-ARG call (no <arg_key>)
            // returns nil from mlx's tag parser and is recovered by the round loop's
            // profile.recoverMissedToolCall(text) — exactly as on the non-batched path. So this needs no
            // special no-arg handling, only the correct ModelProfile (a GLM .taggedReasoning profile).
            //
            // REASONING-MODEL FIX (GLM co-batch): GLM prefixes its call with `<think>…</think>`; parsing the
            // FULL string makes the standalone parser mis-attribute that reasoning to the call NAME (observed:
            // name = the whole think blob → bogus dispatch → "wrong tool name" retry loop, loop never advances).
            // Strip ONLY `<think>` (keep `<tool_call>`) before parsing, and reject an obviously-garbage name so
            // the round loop's recoverMissedToolCall(text) recovers the real `<tool_call>` call instead.
            var call = toolsEnabled ? toolCallParser?.parse(content: Self.stripThinkSpans(text), tools: specs) : nil
            if let c = call, c.function.name.contains("<") || c.function.name.contains("\n") || c.function.name.count > 64 {
                call = nil
            }
            if ProcessInfo.processInfo.environment["SWIFTLM_BATCH_DEBUG"] != nil {
                FileHandle.standardError.write(Data(("[batch-gen] toolsOn=\(toolsEnabled) "
                    + "parsed=\(call?.function.name ?? "nil") hasToolCall=\(text.contains("<tool_call>")) "
                    + "text=\"\(text.replacingOccurrences(of: "\n", with: " ").prefix(120))\"\n").utf8))
            }
            return AsyncThrowingStream { cont in
                cont.yield(.chunk(text))
                if let call { cont.yield(.toolCall(call)) }
                cont.finish()
            }
        }
        session.tools = toolsEnabled ? specs : nil
        return session.streamDetails(to: input)
    }

    /// Record a round's outcome after the caller consumed the stream: append the assistant's turn + update the
    /// exact context size (from the stream's `.info`). Feeds the next compaction decision.
    func finishRound(assistantText: String, contextTokens: Int?) {
        if adapter.requiresOwnedRender {
            // Record the assistant turn STRUCTURED: split out the `<think>` span as reasoning_content (so it's
            // replayed for the in-progress tool chain), keep the visible content, and attach the tool_calls
            // parsed THIS round (nil ⇒ stored as plain content — never a malformed structured turn).
            let (reasoning, content) = Self.splitReasoning(assistantText, tags: adapter.reasoningTags)
            structuredTurns.append(TurnMessage(role: .assistant, content: content,
                                               reasoningContent: reasoning, toolCalls: lastOwnedToolCalls))
            lastOwnedToolCalls = nil
            if let ct = contextTokens { self.contextTokens = ct }
            return
        }
        if !assistantText.isEmpty { messages.append(.assistant(assistantText)) }
        if let ct = contextTokens { self.contextTokens = ct }
    }

    /// OWNED-RENDER round: append the new input as structured turns, render the WHOLE transcript (user query
    /// preserved by `continuationMessages`) through the model's real template, decode live, and parse the tool
    /// call from the accumulated text (the xmlFunction/json parser the 122B needs; recover handles the rest).
    private func ownedRound(_ input: [Chat.Message], toolsEnabled: Bool) async -> AsyncThrowingStream<Generation, Error> {
        // B1: owned-render re-prefills the WHOLE transcript every round and the cost is SUPERLINEAR (122B bench:
        // 8k→29s, 16k→225s), so bound the transcript before it reaches the catastrophic regime. `contextTokens`
        // is the prior round's rendered size; compact when it crosses the perf cap (NOT the window budget).
        if contextTokens > ownedRenderMaxTokens { await compactOwned() }
        structuredTurns.append(contentsOf: input.map(Self.turnMessage))
        let turns = adapter.continuationMessages(structuredTurns)
        let tokens = (try? await model.renderTurnMessages(turns, tools: toolsEnabled ? specs : nil,
                                                          enableThinking: toolsEnabled)) ?? []
        contextTokens = tokens.count
        let maxTok = params.maxTokens ?? 512
        // The owned-render generation prompt PRIMES `<think>` (enableThinking == toolsEnabled) — the open tag is
        // in the PROMPT, not the output — so a reasoning model's generated text begins INSIDE the think span.
        // Re-insert the open tag on the first chunk so every downstream stripper (the live display filter, the
        // tool-call parse, and finishRound's reasoning split) sees a COMPLETE <think>…</think> span. Without it
        // the whole reasoning block + a dangling </think> leak to the chat, and a tool the model calls WHILE
        // reasoning renders mid-thought.
        let primeThink = toolsEnabled && adapter.emitsReasoning
        let openTag = adapter.reasoningTags.open
        return AsyncThrowingStream { cont in
            let task = Task {
                var text = ""
                var firstChunk = true
                do {
                    for try await g in model.streamFromTokens(tokens, maxTokens: maxTok, adapter: adapter, params: params) {
                        if case .chunk(let c) = g {
                            let piece = (firstChunk && primeThink) ? openTag + c : c
                            firstChunk = false
                            text += piece; cont.yield(.chunk(piece))
                        }
                    }
                } catch { cont.finish(throwing: error); return }
                // Parse the tool call from the full text (think stripped so the parser doesn't read reasoning as
                // the name); reject an obviously-garbage name. A parser miss falls to the round loop's recover.
                if toolsEnabled, let call = toolCallParser?.parse(content: Self.stripThinkSpans(text), tools: specs),
                   !(call.function.name.contains("<") || call.function.name.contains("\n") || call.function.name.count > 64) {
                    self.lastOwnedToolCalls = [Serving.ToolCall(name: call.function.name,
                                                                argsJSON: MLXLanguageModel.argsJSON(call.function.arguments))]
                    cont.yield(.toolCall(call))
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    /// Convert an mlx `Chat.Message` (user/tool/system input) → a Sendable `TurnMessage` for the owned transcript.
    static func turnMessage(_ m: Chat.Message) -> TurnMessage {
        TurnMessage(role: TurnMessage.Role(rawValue: m.role.rawValue) ?? .user, content: m.content)
    }

    /// Split an assistant turn into (reasoning span, visible content). Reasoning = the text inside the model's
    /// `<think>…</think>`; content = the answer with `<think>` AND `<tool_call>` spans removed.
    static func splitReasoning(_ text: String, tags: (open: String, close: String)) -> (reasoning: String?, content: String) {
        var reasoning: String? = nil
        if let o = text.range(of: tags.open),
           let c = text.range(of: tags.close, range: o.upperBound..<text.endIndex) {
            reasoning = String(text[o.upperBound..<c.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (reasoning?.isEmpty == true ? nil : reasoning, stripReasoning(text))
    }

    /// Summarize the oldest messages and re-seed a fresh session with `[summary + recent verbatim]`.
    private func compact() async {
        guard let tokenizer else { return }   // can't measure → degrade to no-compaction (no crash)
        let perMsg = messages.map { tokenizer.tokenize(text: render($0)).count }
        let split = compactionSplitIndex(messageTokens: perMsg, keepTokens: budget.keepRecentTokens)
        guard split > 0 else { return }
        if ProcessInfo.processInfo.environment["SWIFTLM_CTX_DEBUG"] != nil {
            let line = "[compact #\(compactions + 1)] context=\(contextTokens) > budget="
                + "\(budget.maxContextTokens); summarizing \(split) of \(messages.count) msgs, "
                + "keeping \(messages.count - split)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        let transcript = messages[0..<split].map(render).joined(separator: "\n")
        let recent = Array(messages[split...])
        let summary = (try? await model.generate(
            "Summarize the conversation so far for an agent that must CONTINUE the task. Preserve the goal, "
            + "key decisions, facts, file paths, and tool results needed to proceed. Be concise.\n\n\(transcript)",
            maxTokens: 600)) ?? "(summary unavailable)"
        let boundary = Chat.Message.user("[Earlier conversation compacted to save context:\n\(stripReasoning(summary))]")
        messages = [boundary] + recent
        session = ChatSession(model.container, instructions: instructions,
                              history: messages, generateParameters: params, tools: specs)
        contextTokens = 0   // fresh session; recomputed from the next generation's .info
        compactions += 1
    }

    private func render(_ m: Chat.Message) -> String { "\(m.role.rawValue): \(m.content)" }

    /// B1 — owned-render perf cap. Re-prefill is per-round and SUPERLINEAR in transcript length, and owned-render
    /// has no incremental KV yet, so the transcript must be bounded to keep per-round prefill in the fast band.
    /// The window-sized context budget is useless here (70% of the 122B's 262k window is ~183k tokens =
    /// hours/round); THIS is a separate, perf-driven cap. Env-tunable; B2's incremental KV will relax it.
    private var ownedRenderMaxTokens: Int {
        Int(ProcessInfo.processInfo.environment["SWIFTLM_OWNED_RENDER_MAX_TOKENS"] ?? "") ?? 8192
    }

    /// Compact the OWNED-RENDER transcript (`structuredTurns`): summarize the oldest turns into a single user
    /// boundary, keep the recent ones verbatim. The boundary is a real `.user` turn, so it ALSO satisfies the
    /// strict template's surviving-user-query precondition (`continuationMessages`). Mirrors `compact()` but over
    /// the structured transcript instead of the ChatSession `messages`.
    private func compactOwned() async {
        guard let tokenizer, structuredTurns.count > 2 else { return }   // need something worth summarizing
        let perTurn = structuredTurns.map { tokenizer.tokenize(text: Self.renderTurn($0)).count }
        let keep = max(1024, ownedRenderMaxTokens / 2)                   // recent turns kept verbatim
        let split = compactionSplitIndex(messageTokens: perTurn, keepTokens: keep)
        guard split > 0 else { return }
        if ProcessInfo.processInfo.environment["SWIFTLM_CTX_DEBUG"] != nil {
            FileHandle.standardError.write(Data(("[compact-owned #\(compactions + 1)] context=\(contextTokens) > "
                + "\(ownedRenderMaxTokens); summarizing \(split) of \(structuredTurns.count) turns\n").utf8))
        }
        let transcript = structuredTurns[0..<split].map(Self.renderTurn).joined(separator: "\n")
        let recent = Array(structuredTurns[split...])
        let summary = (try? await model.generate(
            "Summarize the conversation so far for an agent that must CONTINUE the task. Preserve the goal, key "
            + "decisions, facts, file paths, and tool results needed to proceed. Be concise.\n\n\(transcript)",
            maxTokens: 600)) ?? "(summary unavailable)"
        let boundary = TurnMessage(role: .user,
                                   content: "[Earlier conversation compacted to save context:\n\(stripReasoning(summary))]")
        structuredTurns = [boundary] + recent
        contextTokens = 0
        compactions += 1
    }

    /// Flatten a structured turn for the compaction summarizer / token measure (role: content).
    static func renderTurn(_ t: TurnMessage) -> String { "\(t.role.rawValue): \(t.content)" }

    /// Strip ONLY `<think>…</think>` spans (KEEP `<tool_call>`) — for the batched tool-call parse of a
    /// reasoning model, whose leading reasoning otherwise confuses the standalone parser into reading the
    /// think text as the call name. (Distinct from `stripReasoning`, which also removes `<tool_call>`.)
    static func stripThinkSpans(_ text: String) -> String {
        var s = text
        while let o = s.range(of: "<think>") {
            if let c = s.range(of: "</think>", range: o.upperBound..<s.endIndex) {
                s.removeSubrange(o.lowerBound..<c.upperBound)
            } else { s.removeSubrange(o.lowerBound..<s.endIndex); break }
        }
        return s
    }
}
