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
    private func sampleBatched(_ logits: MLXArray, temperature: Float) -> MLXArray {
        if temperature <= 0 { return argMax(logits, axis: -1) }
        return categorical(logits * (1.0 / temperature))   // [N]
    }

    /// Generate `n` completions of `prompt` in one batched decode. Returns the n decoded strings.
    public func batchGenerate(
        _ prompt: String, n: Int, maxTokens: Int = 512, temperature: Float = 0.7, topP: Float = 1.0,
        stopOnEOS: Bool = true
    ) async -> [String] {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let promptIds = (try? tok.applyChatTemplate(messages: [["role": "user", "content": prompt]])) ?? []
            guard !promptIds.isEmpty else { return Array(repeating: "", count: n) }

            // Stop tokens: eos + Qwen/im chat turn-end.
            var stops = Set<Int>()
            if let e = tok.eosTokenId { stops.insert(e) }
            if let im = tok.convertTokenToId("<|im_end|>") { stops.insert(im) }

            let cache = ctx.model.newCache(parameters: nil)
            // [N, L] — N identical copies of the prompt.
            let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])
            var input = repeated(promptArr, count: n, axis: 0)

            // Prefill → last-position logits per stream.
            var logits = ctx.model(input, cache: cache)          // [N, L, vocab]
            var last = logits[0..., -1, 0...]                    // [N, vocab]
            var next = sampleBatched(last, temperature: temperature)
            eval(next)

            var out = Array(repeating: [Int](), count: n)
            var streamOf = Array(0..<n)   // current batch ROW → original stream id; shrinks on eviction
            var steps = 0
            defer { print("  [batchGenerate] \(steps) steps, \(n - streamOf.count)/\(n) finished early (eviction)") }

            // Per-stream EVICTION: when a stream emits a stop token, drop its row from the batch (and
            // its rows from every layer's KV cache) so the forward pass SHRINKS. A lone rambling
            // stream finishes as batch-1 instead of dragging the whole batch to maxTokens — kills the
            // lockstep tail, so maxTokens can stay generous with no truncation risk.
            for _ in 0..<maxTokens {
                steps += 1
                let toks = next.asArray(Int.self)                // [curBatch]
                var keep: [Int] = []                             // local rows that continue
                for r in 0..<streamOf.count {
                    if stopOnEOS && stops.contains(toks[r]) { continue }   // evict (don't append the stop)
                    out[streamOf[r]].append(toks[r])
                    keep.append(r)
                }
                if keep.isEmpty { break }
                if keep.count < streamOf.count {                 // shrink batch + KV cache to survivors
                    let idx = MLXArray(keep.map { Int32($0) })
                    streamOf = keep.map { streamOf[$0] }
                    for c in cache { var c = c; let s = c.state; c.state = [s[0][idx], s[1][idx]] }
                    next = next[idx]
                }
                let logits2 = ctx.model(next.reshaped([streamOf.count, 1]), cache: cache)  // [curBatch,1,vocab]
                next = sampleBatched(logits2[0..., -1, 0...], temperature: temperature)
                eval(next)
            }
            return out.map { tok.decode(tokenIds: $0) }
        }
    }
}
