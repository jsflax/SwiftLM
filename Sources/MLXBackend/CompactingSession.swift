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
    private var session: ChatSession
    private var messages: [Chat.Message] = []   // tracked conversation (excludes the system instructions)
    private(set) var contextTokens = 0          // last measured prompt+gen tokens (from Generation.info)
    private(set) var compactions = 0            // count, for the demo / observability

    init(model: MLXLanguageModel, instructions: String?, params: GenerateParameters,
         specs: [ToolSpec], tokenizer: (any GrammarTokenizer)?, budget: ContextBudget,
         batchGen: BatchGenerator? = nil, toolCallParser: (any ToolCallParser)? = nil) {
        var p = params
        if let kv = budget.maxKVSize { p.maxKVSize = kv }   // L3: RotatingKVCache memory floor
        self.model = model
        self.instructions = instructions
        self.params = p
        self.specs = specs
        self.tokenizer = tokenizer
        self.budget = budget
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
                    + "parsed=\(call?.function.name ?? "nil") hasToolCall=\(text.contains("<tool_call>"))\n").utf8))
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
        if !assistantText.isEmpty { messages.append(.assistant(assistantText)) }
        if let ct = contextTokens { self.contextTokens = ct }
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
