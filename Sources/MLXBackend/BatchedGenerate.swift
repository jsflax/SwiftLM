import Foundation
import MLX
import MLXRandom
import MLXLMCommon
import MLXLLM

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
    func batchStops(_ tok: any Tokenizer) -> Set<Int> {
        var stops = Set<Int>()
        if let e = tok.eosTokenId { stops.insert(e) }
        for t in ["<|im_end|>", "<|user|>", "<|observation|>", "<|eot_id|>", "<|endoftext|>"] {
            if let id = tok.convertTokenToId(t) { stops.insert(id) }
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
