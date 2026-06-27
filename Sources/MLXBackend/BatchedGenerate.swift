import Foundation
import MLX
import MLXRandom
import MLXLMCommon
import MLXLLM
import Serving   // LocalAgentAdapter (ModelProfile) + TurnMessage — for the owned-render primitives

// Batched best-of-N generation: N rollouts of the SAME prompt + sampler decoded in LOCKSTEP as one
// [N, seq] batch through a single forward pass per step. This is the trace-volume win for the
// flywheel — best-of-N is the easy batching case (identical length at every step), so a stock
// KVCacheSimple with B=N works with no custom cache (its update() reads `let B = keys.dim(0)`).

extension MLXLanguageModel {
    /// Sample one token per stream from `[N, vocab]` logits. Greedy if temperature ≤ 0, else plain
    /// temperature categorical (top-p TODO — temperature alone still gives best-of-N diversity).
    func sampleBatched(_ logits: MLXArray, temperature: Float) -> MLXArray {
        if temperature <= 0 { return argMax(logits, axis: -1) }
        return categorical(logits * (1.0 / temperature))   // [N]
    }

    /// Turn-end stop tokens across chat families. The batched decode has NO ChatSession EOS handling, so it
    /// must stop at these explicitly — else it runs past the assistant turn into the next user turn and
    /// rambles to the token budget (observed: GLM emitting `<|user|>` then re-stating the prompt). Covers
    /// Qwen (`<|im_end|>`), GLM (`<|user|>`/`<|observation|>`), Llama (`<|eot_id|>`), and `<|endoftext|>`.
    func batchStops(_ tok: any Tokenizer, adapter: ModelProfile? = nil) -> Set<Int> {
        var stops = Set<Int>()
        if let e = tok.eosTokenId { stops.insert(e) }
        for t in ["<|im_end|>", "<|user|>", "<|observation|>", "<|eot_id|>", "<|endoftext|>"] {
            if let id = tok.convertTokenToId(t) { stops.insert(id) }
        }
        // The adapter's OWN stops from the model's config — notably the 122B's SECOND eos id (248044), which
        // the universal floor above misses, so a no-`<|im_end|>` finish doesn't run away. (Phase 3c.)
        if let adapter {
            stops.formUnion(adapter.eosTokenIds)
            for s in adapter.stopStrings { if let id = tok.convertTokenToId(s) { stops.insert(id) } }
        }
        return stops
    }

    /// The shared batched decode loop (SLICE 0): take B prompt token-rows of the SAME length, prefill them
    /// as one `[B, L]` forward pass, then decode in lockstep with per-row EVICTION on a stop token (a row
    /// that finishes is dropped from the batch + every layer's KV cache, so the pass SHRINKS). Returns the
    /// generated token ids per row, in input order. Rows are INDEPENDENT — used both for same-prompt
    /// best-of-N and for N DIFFERENT prompts (sub-agent fan-out).
    func batchDecode(model: any LanguageModel, promptRows: [[Int32]], maxTokens: Int,
                     temperature: Float, stops: Set<Int>, stopOnEOS: Bool) -> [[Int]] {
        let n = promptRows.count
        guard n > 0, let L = promptRows.first?.count, L > 0 else { return Array(repeating: [], count: n) }
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(promptRows.flatMap { $0 }).reshaped([n, L])   // [B, L]
        var next = sampleBatched(model(input, cache: cache)[0..., -1, 0...], temperature: temperature)  // [B]
        eval(next)

        var out = Array(repeating: [Int](), count: n)
        var streamOf = Array(0..<n)   // current batch ROW → original stream id; shrinks on eviction
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }   // A1: observe a watchdog/supersession cancel
            let toks = next.asArray(Int.self)
            var keep: [Int] = []
            for r in 0..<streamOf.count {
                if stopOnEOS && stops.contains(toks[r]) { continue }   // evict (don't append the stop)
                out[streamOf[r]].append(toks[r]); keep.append(r)
            }
            if keep.isEmpty { break }
            if keep.count < streamOf.count {                          // shrink batch + KV cache to survivors
                let idx = MLXArray(keep.map { Int32($0) })
                streamOf = keep.map { streamOf[$0] }
                for c in cache { var c = c; let s = c.state; c.state = [s[0][idx], s[1][idx]] }
                next = next[idx]
            }
            next = sampleBatched(model(next.reshaped([streamOf.count, 1]), cache: cache)[0..., -1, 0...],
                                 temperature: temperature)
            eval(next)
        }
        return out
    }

    /// Generate `n` completions of `prompt` in one batched decode (best-of-N — N copies of one prompt).
    public func batchGenerate(
        _ prompt: String, n: Int, maxTokens: Int = 512, temperature: Float = 0.7, topP: Float = 1.0,
        stopOnEOS: Bool = true
    ) async -> [String] {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let ids = (try? tok.applyChatTemplate(messages: [["role": "user", "content": prompt]])) ?? []
            guard !ids.isEmpty else { return Array(repeating: "", count: n) }
            let rows = Array(repeating: ids.map { Int32($0) }, count: n)
            let outIds = self.batchDecode(model: ctx.model, promptRows: rows, maxTokens: maxTokens,
                                          temperature: temperature, stops: self.batchStops(tok),
                                          stopOnEOS: stopOnEOS)
            return outIds.map { tok.decode(tokenIds: $0) }
        }
    }

    /// The SLICE 1b decode (variable-length fan-out), the EFFICIENT/correct way: PREFILL EACH prompt
    /// SEPARATELY at its true length (scalar offset 0, no pad compute, no truncation), MERGE the per-layer
    /// KV caches into one left-padded `BatchedKVCache` (per-row RoPE offsets keep each sequence at its OWN
    /// positions; pad columns are masked), then DECODE in lockstep with per-row eviction. Returns generated
    /// token ids per row, in input order.
    /// SLICE 1b steps 1–2, factored so both the plain and the grammar-constrained decode reuse them:
    /// separately prefill each prompt at its true length, then merge the per-layer KV caches into one
    /// left-padded batched cache. Returns the merged cache, the `[B, vocab]` last-position logits (→ the
    /// first generated token), and the prompt lengths. `nil` if there is nothing to decode.
    func prefillAndMerge(model: any LanguageModel, promptRows: [[Int32]])
        -> (merged: [KVCache], firstLogits: MLXArray, lengths: [Int])? {
        let n = promptRows.count
        let lengths = promptRows.map { $0.count }
        guard n > 0, lengths.allSatisfy({ $0 > 0 }) else { return nil }
        let maxLen = lengths.max()!

        var perRowCaches: [[KVCache]] = []
        var firstLogits: [MLXArray] = []
        perRowCaches.reserveCapacity(n); firstLogits.reserveCapacity(n)
        for row in promptRows {
            let cache = model.newCache(parameters: nil)
            let input = MLXArray(row).reshaped([1, row.count])
            let logits = model(input, cache: cache)[0..., -1, 0...]      // [1, vocab]
            eval(logits)
            for c in cache { eval(c.state) }                            // materialize KV before next iter
            perRowCaches.append(cache)
            firstLogits.append(logits)
        }
        let layerCount = perRowCaches[0].count
        var merged: [KVCache] = []
        merged.reserveCapacity(layerCount)
        for layer in 0..<layerCount {
            var ks: [MLXArray] = []; var vs: [MLXArray] = []
            for r in 0..<n {
                let st = perRowCaches[r][layer].state                   // [keys[1,H,L_r,D], values[...]]
                ks.append(st[0]); vs.append(st[1])
            }
            merged.append(BatchedKVCache(perRowKeys: ks, perRowValues: vs, lengths: lengths, maxLen: maxLen))
        }
        return (merged, concatenated(firstLogits, axis: 0), lengths)    // firstLogits: [B, vocab]
    }

    /// True iff EVERY layer's cache is a standard `[1,H,L,D]` KV that `BatchedKVCache` can left-pad-merge.
    /// HYBRID-attention models are NOT: `qwen3_next`/`qwen3_5` interleave `Qwen3NextGatedDeltaNet` layers whose
    /// `MambaCache` is a conv+recurrent state (not 4-D KV), so the left-pad merge indexes out of range
    /// (`BatchedKVCache` reads `shape[3]`). Probed via a 1-token prefill — the only way to populate state shapes.
    func cachesAreCoBatchable(model: any LanguageModel) -> Bool {
        let cache = model.newCache(parameters: nil)
        _ = model(MLXArray([Int32(0)]).reshaped([1, 1]), cache: cache)
        for c in cache { eval(c.state) }
        return cache.allSatisfy { c in
            let st = c.state
            return st.count == 2 && st[0].ndim == 4 && st[1].ndim == 4
        }
    }

    func batchDecodeDistinct(model: any LanguageModel, promptRows: [[Int32]], maxTokens: Int,
                             temperature: Float, stops: Set<Int>, stopOnEOS: Bool) -> [[Int]] {
        let n = promptRows.count
        guard n > 0 else { return [] }
        // HYBRID models (qwen3_next/qwen3_5 GatedDeltaNet `MambaCache`) can't be KV-merged → decode each row
        // SOLO (the serial floor: correct, just unfused) instead of crashing in the left-pad merge. A single-row
        // `batchDecode` never reaches its own KV-eviction reshape, so it is safe for ANY architecture.
        if n > 1, !cachesAreCoBatchable(model: model) {
            if ProcessInfo.processInfo.environment["SWIFTLM_BATCH_DEBUG"] != nil {
                FileHandle.standardError.write(Data(("[batch-pool] non-mergeable cache (hybrid attention) "
                    + "→ solo fallback for \(n) row(s)\n").utf8))
            }
            return promptRows.map {
                batchDecode(model: model, promptRows: [$0], maxTokens: maxTokens,
                            temperature: temperature, stops: stops, stopOnEOS: stopOnEOS)[0]
            }
        }
        guard let (merged, firstLogits, _) = prefillAndMerge(model: model, promptRows: promptRows) else {
            return Array(repeating: [], count: n)
        }
        var next = sampleBatched(firstLogits, temperature: temperature)  // [B]
        eval(next)

        // JOINT LOCKSTEP DECODE + per-row eviction (rows finish independently → batch SHRINKS).
        var out = Array(repeating: [Int](), count: n)
        var streamOf = Array(0..<n)
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }   // A1: observe a watchdog/supersession cancel
            let toks = next.asArray(Int.self)
            var keep: [Int] = []
            for r in 0..<streamOf.count {
                if stopOnEOS && stops.contains(toks[r]) { continue }    // evict (don't append the stop)
                out[streamOf[r]].append(toks[r]); keep.append(r)
            }
            if keep.isEmpty { break }
            if keep.count < streamOf.count {
                streamOf = keep.map { streamOf[$0] }
                for c in merged { (c as! BatchedKVCache).evict(keep: keep) }
                next = next[MLXArray(keep.map { Int32($0) })]
            }
            next = sampleBatched(model(next.reshaped([streamOf.count, 1]), cache: merged)[0..., -1, 0...],
                                 temperature: temperature)
            eval(next)
        }
        return out
    }

    /// SLICE 1b — batch N DIFFERENT, DIFFERENT-LENGTH prompts (the sub-agent fan-out). Prompts are NOT
    /// truncated or padded-into-one-prefill; each is prefilled at its true length and the caches are merged
    /// (see `batchDecodeDistinct`). Returns one completion per input prompt, in order.
    public func batchGenerateDistinct(
        _ prompts: [String], maxTokens: Int = 512, temperature: Float = 0.0, stopOnEOS: Bool = true
    ) async -> [String] {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let rows: [[Int32]] = prompts.map {
                ((try? tok.applyChatTemplate(messages: [["role": "user", "content": $0]])) ?? []).map { Int32($0) }
            }
            let outIds = self.batchDecodeDistinct(model: ctx.model, promptRows: rows, maxTokens: maxTokens,
                                                  temperature: temperature, stops: self.batchStops(tok),
                                                  stopOnEOS: stopOnEOS)
            return outIds.map { tok.decode(tokenIds: $0) }
        }
    }

    /// Render a full conversation (+ optional tools) to its prompt token row via the model's own processor /
    /// chat template — the sub-agent batched path renders here, then submits the row to the coalescing pool
    /// (so a multi-message turn batches with other sub-agents). Mirrors what `ChatSession` does internally.
    /// `messages` as `(role, content)` pairs (Sendable, unlike `[Chat.Message]`) so they cross into the
    /// `container.perform` @Sendable closure, where they're rebuilt into `Chat.Message`s for the template.
    public func renderConversationTokens(_ messages: [(role: String, content: String)],
                                         tools: [ToolSpec]?) async throws -> [Int32] {
        try await container.perform { ctx in
            let msgs: [Chat.Message] = messages.map { m in
                switch m.role {
                case "system": return .system(m.content)
                case "assistant": return .assistant(m.content)
                case "tool": return .tool(m.content)
                default: return .user(m.content)
                }
            }
            let input = try await ctx.processor.prepare(
                input: tools != nil ? UserInput(chat: msgs, tools: tools!) : UserInput(chat: msgs))
            return input.text.tokens.asType(.int32).asArray(Int32.self)
        }
    }

    /// Render a STRUCTURED conversation to its prompt token row via the model's REAL chat template, using mlx's
    /// raw-dict prompt hatch (`UserInput.messages`) — the dicts pass through `prepare` UNTOUCHED, so a strict
    /// reasoning+tool template (the 122B's) gets the `reasoning_content` + `tool_calls` structure that
    /// `Chat.Message` can't express. This is the OWNED-RENDER input: it preserves the original user query every
    /// round (fixes the 122B "No user query found" throw) and replays prior reasoning for the in-progress tool
    /// chain. `enableThinking` drives the template's generation prompt (`<think>` open). Built INSIDE
    /// `container.perform` (the dicts are Sendable values that cross the boundary).
    public func renderTurnMessages(_ turns: [TurnMessage], tools: [ToolSpec]?, enableThinking: Bool) async throws -> [Int32] {
        try await container.perform { ctx in
            let input = try await ctx.processor.prepare(
                input: UserInput(messages: Self.turnDicts(turns), tools: tools,
                                 additionalContext: ["enable_thinking": enableThinking]))
            return input.text.tokens.asType(.int32).asArray(Int32.self)
        }
    }

    /// VLM variant of `renderTurnMessages`: also carries image inputs and returns the processed pixels the
    /// vision-merge prefill needs. A turn with `imageURLs` renders its content as the model's multimodal content
    /// array (image markers + text — see `turnDicts`), so the chat template emits the `<|vision_start|>…
    /// <|vision_end|>` placeholders; the flat `images` (top-level, in turn order) become the pixel grid those
    /// placeholders bind to. Returns `(tokens, image)` — `image` is nil if the model/processor produced none.
    public func renderTurnMessages(_ turns: [TurnMessage], tools: [ToolSpec]?, enableThinking: Bool,
                                   imageURLs: [URL]) async throws -> (tokens: [Int32], image: OwnedImageBox) {
        // Take `[URL]` (Sendable) not `[UserInput.Image]` (non-Sendable: its `.ciImage`/`.array` cases hold
        // CIImage/MLXArray) so the params can cross into `perform`'s `@Sendable` closure; build the `.url` images
        // INSIDE. The processed pixels are boxed in `OwnedImageBox` (also inside) so the Sendable
        // `([Int32], OwnedImageBox)` tuple satisfies `perform`'s `R: Sendable`.
        try await container.perform { ctx in
            let input = try await ctx.processor.prepare(
                input: UserInput(messages: Self.turnDicts(turns), images: imageURLs.map { .url($0) }, tools: tools,
                                 additionalContext: ["enable_thinking": enableThinking]))
            return (input.text.tokens.asType(.int32).asArray(Int32.self), OwnedImageBox(input.image))
        }
    }

    /// Build the owned-render chat-template dicts from the structured transcript. Shared by both
    /// `renderTurnMessages` overloads. A turn with `imageURLs` becomes the Qwen multimodal content array
    /// (`[{"type":"image"}×N, {"type":"text","text": …}]`) so the template inserts the vision placeholders;
    /// every other turn keeps the plain-string content (the text-only majority — unchanged shape).
    static func turnDicts(_ turns: [TurnMessage]) -> [[String: any Sendable]] {
        turns.map { t in
            var d: [String: any Sendable] = ["role": t.role.rawValue]
            if t.imageURLs.isEmpty {
                d["content"] = t.content
            } else {
                var parts: [[String: any Sendable]] = t.imageURLs.map { _ in ["type": "image"] }
                parts.append(["type": "text", "text": t.content])
                d["content"] = parts
            }
            if let rc = t.reasoningContent, !rc.isEmpty { d["reasoning_content"] = rc }
            if let calls = t.toolCalls, !calls.isEmpty {
                d["tool_calls"] = calls.map { c -> [String: any Sendable] in
                    ["type": "function",
                     "function": ["name": c.name, "arguments": Self.argsDict(c.argsJSON)] as [String: any Sendable]]
                }
            }
            return d
        }
    }

    /// Parse a tool-call args JSON string → the dict the chat template iterates (`tool_call.arguments|items`).
    /// Coerce only the JSON scalar types (tool args are overwhelmingly flat string/number/bool); a nested
    /// object/array is stringified rather than dropped. `{}`/invalid → empty.
    static func argsDict(_ argsJSON: String) -> [String: any Sendable] {
        guard let d = argsJSON.data(using: .utf8),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return [:] }
        var out: [String: any Sendable] = [:]
        for (k, v) in o {
            switch v {
            case let s as String: out[k] = s
            case let b as Bool: out[k] = b
            case let i as Int: out[k] = i
            case let dbl as Double: out[k] = dbl
            case let n as NSNumber: out[k] = n.doubleValue
            default: out[k] = String(describing: v)
            }
        }
        return out
    }

    /// Live-streaming SOLO decode from a PRE-RENDERED prompt row — the OWNED-RENDER generation primitive. It
    /// adapts the single-row `batchDecode` (a fresh single-sequence `KVCache`, NEVER `BatchedKVCache`), so the
    /// hybrid 122B decodes without the merge crash, AND it keeps LIVE streaming: `.chunk`s are emitted
    /// incrementally via decode-then-diff detokenization (the UI sees tokens as they arrive). Stops on the
    /// adapter's stop set (its EOS strings/ids ∪ the universal turn-end floor); `</think>` is never a stop, so a
    /// reasoning model runs through its think block to the action. No `.info` is emitted — the caller sets its
    /// own context-token count (parity with the batched path), which keeps compaction/telemetry working.
    ///
    /// A0 (the whitespace-spiral fix): decoding now runs through mlx's OWN logit processor + sampler
    /// (`params.processor()` / `params.sampler()`), the SAME machinery the ChatSession path uses — so a
    /// strict-template model samples IDENTICALLY here and on the incremental path. This restores the repetition
    /// penalty that the old temperature-only `sampleBatched` silently dropped on the owned-render boundary (the
    /// spiral root cause), plus min-p/top-p when configured. A `DegenerateRunDetector` is the UNCONDITIONAL
    /// backstop the penalty can't promise: a degenerate run aborts in milliseconds instead of running to the
    /// turn watchdog. `params` carries temp + rep-pen + min/top-p (built once in SwiftLMServe from the adapter).
    public func streamFromTokens(_ row: [Int32], maxTokens: Int, adapter: ModelProfile,
                                 params: GenerateParameters, kvBox: OwnedKVCacheBox? = nil,
                                 activeTraits: Set<TraitID> = [],
                                 image imageBox: OwnedImageBox = .init(nil))
        -> AsyncThrowingStream<Generation, Error> {
        AsyncThrowingStream { cont in
            let task = Task {
                await container.perform { ctx in
                    // Part C: bind this agent's role trait-set for THIS decode. `perform`→`read`→`withLock` are
                    // inline continuations in this SAME Task (no hop), so the @TaskLocal reaches every resident
                    // layer's `callAsFunction` below; an EMPTY set ⇒ the resident forward returns base unchanged
                    // (byte-identical). This is the single owned-render bind site verified to propagate.
                    LoRARuntime.$activeTraits.withValue(activeTraits) {
                    let model = ctx.model, tok = ctx.tokenizer
                    let stops = self.batchStops(tok, adapter: adapter)
                    // Pipeline-distributed (M4d): the ring group when this model is sharded across
                    // ≥2 machines, else nil. Used below to broadcast rank 0's sampled token to every
                    // rank each step so a sampling turn stays in lockstep (greedy: a no-op).
                    let ringGroup = (model as? PipelineParallel)?.ringGroup
                    let promptArr = MLXArray(row).reshaped([1, row.count])
                    // B2 incremental KV: reuse the cache carried from the prior round IFF its tokens are an exact
                    // prefix of `row` (the recurrent layers can't rewind, so reuse must be forward-only +
                    // prefix-exact; any divergence ⇒ full re-prefill into a fresh cache — correct, just slow). The
                    // box is `@unchecked Sendable` and read/written ONLY here inside the serialized `perform`, so
                    // the non-Sendable `[KVCache]` never makes an unsafe hop across an isolation boundary.
                    let reuse = (kvBox?.cache != nil) && isReusablePrefix(cached: kvBox!.cachedRow, fresh: row)
                    let cache = reuse ? kvBox!.cache! : model.newCache(parameters: nil)
                    let prefixLen = reuse ? kvBox!.cachedRow.count : 0
                    if ProcessInfo.processInfo.environment["SWIFTLM_CTX_DEBUG"] != nil, kvBox != nil {
                        FileHandle.standardError.write(Data(("[incr-kv] reuse=\(reuse) prefix=\(prefixLen)/"
                            + "\(row.count) (tail=\(row.count - prefixLen) tok)\n").utf8))
                    }
                    // mlx's own decode chain — fresh per round (per-stream rep-pen ring + RNG state). Seed the
                    // rep-pen ring from the FULL row even when reusing the cache, for ChatSession parity.
                    var processor = params.processor()    // PenaltyProcessor? (rep/presence/freq) — nil if none set
                    let sampler = params.sampler()         // ArgMax (temp 0) / TopP (min-p/top-p) / Categorical
                    processor?.prompt(promptArr)
                    var tripwire = DegenerateRunDetector()
                    // Battle-test ONLY: force a pathological distribution to reproduce a spiral through the REAL
                    // path. Read once (inert in production — the env var is unset).
                    let injectId = ProcessInfo.processInfo.environment["SWIFTLM_SPIRAL_INJECT"].flatMap { Int32($0) }
                    // CHUNKED PREFILL of row[prefixLen...] — process the new tail in `prefillStepSize` windows with
                    // an eval between each. A single [1,L] forward's self-attention activation is O(L²): MEASURED
                    // on the 122B, peak working set is ~65GB (weights) at L≤512 but 76.8GB at L=8192 and climbs
                    // with L — a large single-shot prefill on a loaded desktop crosses the memory-pressure
                    // threshold and freezes the machine (the confirmed freeze cause — NOT GPU/compositor
                    // starvation, which the log RCA refuted). Chunking makes each step's attention [step,
                    // totalSoFar], so peak stays ≈ weights+small REGARDLESS of L (measured: 16384 chunked = 66.1GB
                    // vs 8192 single-shot = 76.8GB). The last window's final-position logits ARE the first-
                    // generation logits. With B2 reuse, only the new tail is prefilled (cache.ropeOffset already
                    // equals prefixLen, so the appended tokens get correct absolute RoPE).
                    let stepSize = max(64, params.prefillStepSize)
                    // VLM-COMPATIBLE FORWARD: drive the model through the universal `LMInput.Text` overload, never
                    // the bare `model(MLXArray, cache:)` overload. The bare overload has a `fatalError` DEFAULT in
                    // the LanguageModel protocol (LanguageModel.swift:242) — only text model classes (e.g. the LLM
                    // Qwen35Model) override it; VLM classes (e.g. Qwen35MoE, loaded once MLXVLM is linked) override
                    // the `LMInput.Text` overload instead. For text models the `LMInput.Text` default impl forwards
                    // straight to the bare overload they implement, so this is byte-identical for them — but it is
                    // REQUIRED for the VLM (the bare path would crash every 122B turn, text or image). `fwd` keeps
                    // the chunked prefill below intact (the 122B memory-freeze guard); only the call shape changes.
                    func fwd(_ tokens: MLXArray) -> MLXArray {
                        model(LMInput.Text(tokens: tokens), cache: cache, state: nil).logits[0..., -1, 0...]
                    }
                    var logits: MLXArray
                    if let image = imageBox.image, !reuse {
                        // COLD image round: the vision encoder must run and merge image embeddings at the
                        // `<|image_pad|>` placeholder positions — that happens ONLY through `model.prepare`, not the
                        // bare text forward. The image round's prompt is short (image tokens + a brief user turn), so
                        // a single-shot prepare stays under the 122B freeze threshold. The resulting KV holds the
                        // image-encoded prefix; because the row KEEPS its image markers on later rounds, the existing
                        // `reuse` path below reuses that KV and prefills only the text tail — the vision encoder runs
                        // ONCE per image, never on reuse. (Mirrors MLXLMCommon.TokenIterator.prepare.)
                        let lmInput = LMInput(text: .init(tokens: promptArr), image: image)
                        switch try? model.prepare(lmInput, cache: cache, windowSize: stepSize) {
                        case .some(.logits(let out)):
                            logits = out.logits[0..., -1, 0...]
                        case .some(.tokens(let t)):
                            logits = model(t[text: .newAxis], cache: cache, state: nil).logits[0..., -1, 0...]
                        case .none:
                            cont.yield(.chunk("⚠️ vision prefill failed (model.prepare)")); cont.finish(); return
                        }
                        eval(logits)
                    } else {
                        // TEXT prefill (or image-round REUSE): chunk `row[prefixLen...]` so peak attention stays
                        // bounded regardless of L (the freeze guard). On reuse the image already lives in the reused
                        // KV prefix, so this tail is pure text — no vision work here.
                        logits = fwd(promptArr[0..., prefixLen..<min(prefixLen + stepSize, row.count)])
                        eval(logits)
                        var pStart = prefixLen + stepSize
                        while pStart < row.count {
                            if Task.isCancelled { break }
                            let pEnd = min(pStart + stepSize, row.count)
                            logits = fwd(promptArr[0..., pStart..<pEnd])
                            eval(logits)
                            pStart = pEnd
                        }
                    }
                    var genIds: [Int] = []
                    var emitted = ""
                    var cancelled = false
                    for _ in 0..<max(1, maxTokens) {
                        if Task.isCancelled { cancelled = true; break }   // A1: a watchdog/supersession cancel stops the decode
                        if let injectId { logits = Self.spikeLogits(like: logits, at: injectId) }
                        let processed = processor?.process(logits: logits) ?? logits
                        var nextTok = sampler.sample(logits: processed)          // [1]
                        if let g = ringGroup {
                            // Every rank uses RANK 0's sampled token (all-gather concatenates in rank
                            // order; index 0 is rank 0's). The follower discards its own RNG sample —
                            // this is what keeps a temp>0 turn byte-identical across the ring. CPU stream
                            // (the default GPU stream parks a cross-machine wait inside a Metal command
                            // buffer and trips the watchdog).
                            nextTok = g.allGather(nextTok, stream: .cpu)[0 ..< 1]
                        }
                        eval(nextTok)
                        processor?.didSample(token: nextTok)
                        let t = nextTok.asArray(Int.self)[0]
                        if stops.contains(t) { break }
                        genIds.append(t)
                        let full = tok.decode(tokenIds: genIds)
                        let suffix = full.count > emitted.count ? String(full.dropFirst(emitted.count)) : ""
                        if !suffix.isEmpty {                                  // emit the newly-decoded suffix
                            cont.yield(.chunk(suffix)); emitted = full
                        }
                        if let trip = tripwire.observe(token: t, suffix: suffix) {
                            if ProcessInfo.processInfo.environment["SWIFTLM_CTX_DEBUG"] != nil {
                                FileHandle.standardError.write(Data(("[tripwire] degenerate decode aborted after "
                                    + "\(genIds.count) tokens: \(trip)\n").utf8))
                            }
                            break
                        }
                        logits = fwd(nextTok.reshaped([1, 1]))
                        eval(logits)
                    }
                    // B2: carry the cache for the next round. The cache now contains EXACTLY `row` (prefilled) +
                    // `genIds` (each decoded token was forwarded into it) — a stop token is NOT forwarded. The
                    // NEXT round reuses this iff its render starts with `row + genIds` (guaranteed prefix-stable by
                    // storing the assistant turn VERBATIM in CompactingSession, never re-serialized). A cancelled
                    // decode left the cache in a partial state ⇒ drop it (the box is reset so next round full-prefills).
                    if let kvBox {
                        if cancelled { kvBox.reset() }
                        else { kvBox.cache = cache; kvBox.cachedRow = row + genIds.map { Int32($0) } }
                    }
                    }   // close LoRARuntime.$activeTraits.withValue
                }
                cont.finish()
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }

    /// Battle-test ONLY (`SWIFTLM_SPIRAL_INJECT=<tokenId>`): a `[1, vocab]` logit row peaked far above the rest
    /// at `id`, so the REAL `streamFromTokens` loop deterministically tries to emit that one token forever —
    /// reproducing a decode spiral end-to-end to prove the tripwire bounds it. Inert unless the env var is set.
    static func spikeLogits(like template: MLXArray, at id: Int32) -> MLXArray {
        let vocab = template.dim(-1)
        var base = [Float](repeating: 0, count: vocab)
        let i = Int(id)
        if i >= 0 && i < vocab { base[i] = 60 }
        return MLXArray(base).reshaped([1, vocab])
    }

    /// Token-level batched fan-out (the sub-agent coalescer's entry): decode B PRE-RENDERED token rows
    /// (already chat-templated by the caller, so multi-message sub-agent CONVERSATIONS work) in one merged
    /// pass via SLICE 1b, returning the decoded completion per row. Unlike `batchGenerateDistinct` it does
    /// NOT apply a chat template — the caller owns rendering.
    public func batchGenerateRows(_ rows: [[Int32]], maxTokens: Int = 512, temperature: Float = 0.0,
                                  stopOnEOS: Bool = true) async -> [String] {
        await container.perform { ctx in
            let outIds = self.batchDecodeDistinct(model: ctx.model, promptRows: rows, maxTokens: maxTokens,
                                                  temperature: temperature, stops: self.batchStops(ctx.tokenizer),
                                                  stopOnEOS: stopOnEOS)
            return outIds.map { ctx.tokenizer.decode(tokenIds: $0) }
        }
    }

    /// SLICE 1b correctness proof: decode each (variable-length) prompt (a) as one merged B-row batch via
    /// `batchDecodeDistinct` and (b) ALONE via single-row `batchDecode` (the ground truth — true length,
    /// scalar offset, no pad). A correct merge ⇒ batched ≈ solo (semantically identical; exact token match
    /// up to GPU float non-determinism, which can flip greedy `argMax` at near-ties between batch sizes).
    public func batchEquivalenceCheck(_ prompts: [String], maxTokens: Int = 48)
        async -> [(batchedText: String, soloText: String, firstDiff: Int, batchedIds: [Int], soloIds: [Int])] {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let rows: [[Int32]] = prompts.map {
                ((try? tok.applyChatTemplate(messages: [["role": "user", "content": $0]])) ?? []).map { Int32($0) }
            }
            let stops = self.batchStops(tok)
            let batched = self.batchDecodeDistinct(model: ctx.model, promptRows: rows, maxTokens: maxTokens,
                                                   temperature: 0, stops: stops, stopOnEOS: true)
            let solo = rows.map {
                self.batchDecode(model: ctx.model, promptRows: [$0], maxTokens: maxTokens,
                                 temperature: 0, stops: stops, stopOnEOS: true)[0]
            }
            return (0..<rows.count).map { i in
                let b = batched[i], s = solo[i]
                let diff = (0..<min(b.count, s.count)).first { b[$0] != s[$0] } ?? (b == s ? -1 : min(b.count, s.count))
                return (tok.decode(tokenIds: b), tok.decode(tokenIds: s), diff, b, s)
            }
        }
    }
}
