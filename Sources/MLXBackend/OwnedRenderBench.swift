import Foundation
import MLX
import MLXLMCommon
import Serving

// ── Part B investigation. Owned-render re-prefills the WHOLE transcript every round (streamFromTokens builds a
// fresh KVCache each call). The fix is incremental KV — reuse the prefix, prefill only the new tail. But that
// requires either (a) a TRIMMABLE cache (to drop the decode tokens that don't match the next template re-render)
// or (b) forward-only continuation. The hybrid 122B (qwen3_5_moe) interleaves recurrent GatedDeltaNet layers
// whose state CAN'T be rewound — so the clean "truncate + reprefill tail" approach may be blocked for the exact
// model that needs it. This bench gathers the two deciding facts (trimmability + the real cost curve) before any
// risky cache surgery — measure, don't guess.

extension MLXLanguageModel {
    /// Returns a human-readable report: cache trimmability, fresh-prefill cost vs length (the current per-round
    /// cost), and the incremental-advance potential (prefill only a tail vs a full re-prefill).
    public func ownedRenderBench() async -> String {
        await container.perform { ctx in
            let model = ctx.model
            var out = ["=== OWNED-RENDER BENCH (model: \(self.modelId)) ==="]

            // 1. Cache composition — fully-trimmable ⇒ clean incremental (truncate decode tokens) is possible;
            //    any recurrent layer ⇒ no rewind, needs forward-only/raw-append reconciliation instead.
            let probe = model.newCache(parameters: nil)
            _ = model(MLXArray([Int32(100)]).reshaped([1, 1]), cache: probe)   // 1-tok prefill populates shapes
            for c in probe { eval(c.state) }
            let trimmable = probe.filter { $0.isTrimmable }.count
            out.append("cache layers: \(probe.count), trimmable: \(trimmable)/\(probe.count) → "
                + (trimmable == probe.count
                   ? "fully trimmable (clean incremental OK)"
                   : "HYBRID/recurrent (\(probe.count - trimmable) non-trimmable — can't rewind decode tokens)"))

            // 2. Fresh full prefill cost vs length L + MLX memory counters — shows the per-round cost AND
            //    whether the buffer-cache pool stays bounded by the cacheLimit (the working-set-growth lever).
            func gb(_ bytes: Int) -> Double { Double(bytes) / 1_073_741_824.0 }
            func prefillStep(_ L: Int) -> (secs: Double, active: Double, cache: Double, peak: Double) {
                let row = MLXArray(Array(repeating: Int32(100), count: L)).reshaped([1, L])
                let kvc = model.newCache(parameters: nil)
                MLX.GPU.resetPeakMemory()
                let t0 = Date()
                let logits = model(row, cache: kvc)[0..., -1, 0...]
                eval(logits)
                let secs = Date().timeIntervalSince(t0)
                return (secs, gb(MLX.Memory.activeMemory), gb(MLX.Memory.cacheMemory), gb(MLX.Memory.peakMemory))
            }
            _ = prefillStep(64)   // warm the GPU
            out.append(String(format: "--- fresh full prefill: time + MLX memory (cacheLimit=%.1fGB) ---", gb(MLX.Memory.cacheLimit)))
            for L in [512, 2048, 8192] {
                let r = prefillStep(L)
                out.append(String(format: "  L=%6d : %6.2fs   active=%.2f  cache=%.2f  peak=%.2f GB",
                                  L, r.secs, r.active, r.cache, r.peak))
            }

            // 2b. Single-shot vs CHUNKED prefill PEAK memory. A single [1,L] forward's self-attention activation
            //     is O(L²); chunking into `step` windows makes each step's attention [step, totalSoFar], bounding
            //     the activation spike that drives peak working set (and the freeze). Proves chunking is the
            //     activation-memory fix (not the GPU-preemption story, which the freeze RCA refuted).
            func prefillChunked(_ L: Int, step: Int) -> (secs: Double, peak: Double) {
                let row = MLXArray(Array(repeating: Int32(100), count: L)).reshaped([1, L])
                let kvc = model.newCache(parameters: nil)
                MLX.GPU.resetPeakMemory()
                let t0 = Date()
                var logits = model(row[0..., 0..<min(step, L)], cache: kvc)[0..., -1, 0...]; eval(logits)
                var s = step
                while s < L { let e = min(s + step, L); logits = model(row[0..., s..<e], cache: kvc)[0..., -1, 0...]; eval(logits); s = e }
                return (Date().timeIntervalSince(t0), gb(MLX.Memory.peakMemory))
            }
            out.append("--- single-shot vs CHUNKED(512) prefill PEAK memory (the freeze driver) ---")
            let ss8 = prefillStep(8192)
            let ck8 = prefillChunked(8192, step: 512)
            let ck16 = prefillChunked(16384, step: 512)
            out.append(String(format: "  L= 8192 single : %5.1fs  peak=%.1f GB", ss8.secs, ss8.peak))
            out.append(String(format: "  L= 8192 chunked: %5.1fs  peak=%.1f GB", ck8.secs, ck8.peak))
            out.append(String(format: "  L=16384 chunked: %5.1fs  peak=%.1f GB  (single-shot 16k omitted — it IS the spike)", ck16.secs, ck16.peak))

            // 3. Incremental potential — advance an already-cached prefix by DELTA vs a fresh full re-prefill of
            //    the same total. This is the per-round win IF the rendered prefix can be reused.
            let base = 8192, delta = 512
            let cache = model.newCache(parameters: nil)
            let baseRow = MLXArray(Array(repeating: Int32(100), count: base)).reshaped([1, base])
            let baseLogits = model(baseRow, cache: cache)[0..., -1, 0...]; eval(baseLogits)
            let tAdv = Date()
            let tailRow = MLXArray(Array(repeating: Int32(100), count: delta)).reshaped([1, delta])
            let tailLogits = model(tailRow, cache: cache)[0..., -1, 0...]; eval(tailLogits)
            let advSecs = Date().timeIntervalSince(tAdv)
            let fullSecs = prefillStep(base + delta).secs
            out.append("--- incremental potential (cached \(base) + advance \(delta)) ---")
            out.append(String(format: "  advance %d on cached %d : %.2fs   vs   fresh full %d : %.2fs   → %.1fx faster",
                              delta, base, advSecs, base + delta, fullSecs, fullSecs / max(advSecs, 0.0001)))
            return out.joined(separator: "\n")
        }
    }
}
