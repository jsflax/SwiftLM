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

/// B2 incremental-KV carrier: holds the OWNED-RENDER KV cache (and the exact token row in it) ACROSS rounds, so
/// a round can REUSE the prior round's cache and prefill only the new tail instead of re-prefilling the whole
/// transcript. `@unchecked Sendable` because the non-Sendable `[KVCache]` is only ever read/written INSIDE
/// `streamFromTokens`'s serialized `container.perform` (the sequencer drives a session's rounds strictly
/// sequentially), mirroring how mlx-swift-lm's ChatSession holds its cache in a `SerialAccessContainer<Cache>`.
public final class OwnedKVCacheBox: @unchecked Sendable {
    public var cache: [KVCache]? = nil
    public var cachedRow: [Int32] = []   // EXACTLY the tokens the cache encodes (prefilled row + decoded gen ids)
    public init() {}
    public func reset() { cache = nil; cachedRow = [] }
}

final class CompactingSession: @unchecked Sendable {
    private let kvBox = OwnedKVCacheBox()        // B2: persistent owned-render cache across this session's rounds
    private let model: MLXLanguageModel
    private let instructions: String?
    private let params: GenerateParameters
    private let specs: [ToolSpec]
    private let tokenizer: (any GrammarTokenizer)?
    private let budget: ContextBudget
    private let batchGen: BatchGenerator?
    private let toolCallParser: (any ToolCallParser)?
    private let terminalTools: Set<String>      // routing/terminal tools that survive completion-pressure (never
                                                // dropped) so a room agent can always conclude by routing. Empty ⇒
                                                // unchanged: the last round reserves NO tools (the old behavior).
    private let adapter: ModelProfile           // the per-model harness (drives owned render / stops / reasoning)
    private let activeTraits: Set<TraitID>      // Part C: this agent's role trait-set, bound around the owned-render
                                                // decode (gated to owned-render upstream ⇒ empty on ChatSession/co-batch)
    private var session: ChatSession
    private var messages: [Chat.Message] = []   // tracked conversation (excludes the system instructions)
    private var structuredTurns: [TurnMessage] = []   // OWNED-RENDER transcript (carries reasoning_content + tool_calls)
    private var lastOwnedToolCalls: [Serving.ToolCall]? = nil // the call parsed this round, recorded into the next assistant turn
    private(set) var contextTokens = 0          // last measured prompt+gen tokens (from Generation.info)
    private(set) var compactions = 0            // count, for the demo / observability

    init(model: MLXLanguageModel, instructions: String?, params: GenerateParameters,
         specs: [ToolSpec], tokenizer: (any GrammarTokenizer)?, budget: ContextBudget,
         adapter: ModelProfile = .generic,
         batchGen: BatchGenerator? = nil, toolCallParser: (any ToolCallParser)? = nil,
         activeTraits: Set<TraitID> = [], terminalTools: Set<String> = []) {
        var p = params
        if let kv = budget.maxKVSize { p.maxKVSize = kv }   // L3: RotatingKVCache memory floor
        self.activeTraits = activeTraits
        self.terminalTools = terminalTools
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

    /// Specs to advertise this round. Work tools ON ⇒ ALL specs. Under completion pressure (`toolsEnabled` =
    /// false) ⇒ only the TERMINAL/routing specs, so a room agent can still conclude by routing (the referee can
    /// always `done()`); `nil` when there are none (no terminal tools ⇒ the old "last round reserves no tools").
    /// Parsing is then done against the SAME set, so a dropped work tool can't sneak back in mid-pressure.
    private func renderSpecs(toolsEnabled: Bool) -> [ToolSpec]? {
        if toolsEnabled { return specs }
        // ToolSpec is [String: any Sendable] = {"type":"function","function":{"name":…}} — extract the name the
        // same way the rest of MLXBackend does (MCPHost/GrammarConstraint) to keep only the terminal/routing tools.
        let terminal = specs.filter {
            guard let n = ($0["function"] as? [String: any Sendable])?["name"] as? String else { return false }
            return terminalTools.contains(n)
        }
        return terminal.isEmpty ? nil : terminal
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
        // C1.5: route via owned-render natively (`requiresOwnedRender`) OR whenever the bank is live — owned-render
        // is the ONLY decode path the activeTraits @TaskLocal propagates through, so when traits are in play every
        // local agent (incl. a normally-ChatSession model like the 80B) must use it. Bank OFF ⇒ unchanged.
        if adapter.requiresOwnedRender || LoRARuntime.bankEnabled { return await ownedRound(input, toolsEnabled: toolsEnabled) }
        // Past here = a NON-owned-render decode path (co-batch `batchGen` / ChatSession `streamDetails`), each of
        // which decodes inside nested unstructured Task{}s the activeTraits @TaskLocal cannot cross. Traits are
        // gated to owned-render upstream so this set is normally empty; if one ever leaks here, fail loud rather
        // than silently serve base (Part C "never silently serve base" tripwire).
        LoRARuntime.assertCarriable(activeTraits, path: batchGen != nil ? "co-batch BatchGenerator" : "ChatSession.streamDetails")
        if contextTokens > budget.maxContextTokens { await compact() }
        messages.append(contentsOf: input)
        // SLICE 3 batched path: render the WHOLE conversation (+ tools) to tokens, generate through the
        // coalescing pool (so concurrent sub-agents fuse), parse the tool call from text via the model's own
        // parser, and surface the same `Generation` shape the round loop already consumes. Compaction still
        // works — it edits `messages`, which this render reads. (Re-prefill per round; no ChatSession KV reuse.)
        if let batchGen {
            let convo = (instructions.map { [Chat.Message.system($0)] } ?? []) + messages
            let pairs = convo.map { (role: $0.role.rawValue, content: $0.content) }
            let render = renderSpecs(toolsEnabled: toolsEnabled)   // terminal-only under completion pressure
            let tokens = (try? await model.renderConversationTokens(pairs, tools: render)) ?? []
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
            var call = render != nil ? toolCallParser?.parse(content: Self.stripThinkSpans(text), tools: render!) : nil
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
        session.tools = renderSpecs(toolsEnabled: toolsEnabled)   // terminal/routing tools survive completion pressure
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
        structuredTurns.append(contentsOf: input.map(Self.turnMessage))
        // BUG-2 fix: owned-render must carry the system prompt too. The ChatSession path applies it via
        // `ChatSession(instructions:)`, but the owned-render transcript is built ONLY from `structuredTurns` — so
        // the persona / role instructions / plan-mode note / memory-hook grounding were silently DROPPED (the 122B
        // ran its agents WITHOUT their system prompt). Prepend it as a leading system turn AFTER continuationMessages'
        // user-query scan (which keys on the first `.user` turn), so the system stays first and the scan is
        // unperturbed. The prompt is constant across rounds ⇒ a stable head ⇒ B2 KV prefix-reuse still holds.
        let body = adapter.continuationMessages(structuredTurns)
        let turns = (instructions?.isEmpty == false)
            ? [TurnMessage(role: .system, content: instructions!)] + body
            : body
        // Under completion pressure (`toolsEnabled` false) advertise only the terminal/routing specs, not nil —
        // so the model can STILL conclude by routing (the referee's `done()`/the builder's `handoff()`); `nil`
        // only when there are no terminal tools (the old last-round force-answer). `anyTools` drives thinking +
        // parsing: a routing-only round still reasons (to DECIDE the route) and still parses (to catch the call).
        let render = renderSpecs(toolsEnabled: toolsEnabled)
        let anyTools = render != nil
        let tokens = (try? await model.renderTurnMessages(turns, tools: render,
                                                          enableThinking: anyTools)) ?? []
        contextTokens = tokens.count
        let maxTok = params.maxTokens ?? 512
        // The owned-render generation prompt PRIMES `<think>` (enableThinking == anyTools) — the open tag is
        // in the PROMPT, not the output — so a reasoning model's generated text begins INSIDE the think span.
        // Re-insert the open tag on the first chunk so every downstream stripper (the live display filter, the
        // tool-call parse, and finishRound's reasoning split) sees a COMPLETE <think>…</think> span. Without it
        // the whole reasoning block + a dangling </think> leak to the chat, and a tool the model calls WHILE
        // reasoning renders mid-thought.
        let primeThink = anyTools && adapter.emitsReasoning
        let openTag = adapter.reasoningTags.open
        return AsyncThrowingStream { cont in
            let task = Task {
                var text = ""
                var firstChunk = true
                do {
                    for try await g in model.streamFromTokens(tokens, maxTokens: maxTok, adapter: adapter,
                                                              params: params, kvBox: kvBox, activeTraits: activeTraits) {
                        if case .chunk(let c) = g {
                            let piece = (firstChunk && primeThink) ? openTag + c : c
                            firstChunk = false
                            text += piece; cont.yield(.chunk(piece))
                        }
                    }
                } catch { cont.finish(throwing: error); return }
                // Parse the tool call from the full text (think stripped so the parser doesn't read reasoning as
                // the name); reject an obviously-garbage name. A parser miss falls to the round loop's recover.
                if anyTools, let call = toolCallParser?.parse(content: Self.stripThinkSpans(text), tools: render ?? specs),
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
        kvBox.reset()                         // B2: a rebuilt transcript invalidates the carried KV cache
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
