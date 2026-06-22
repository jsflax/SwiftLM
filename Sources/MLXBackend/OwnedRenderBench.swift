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

            // 2. Fresh full prefill cost vs length L — the CURRENT per-round cost as the transcript grows.
            func prefillSeconds(_ L: Int) -> Double {
                let row = MLXArray(Array(repeating: Int32(100), count: L)).reshaped([1, L])
                let cache = model.newCache(parameters: nil)
                let t0 = Date()
                let logits = model(row, cache: cache)[0..., -1, 0...]
                eval(logits)
                return Date().timeIntervalSince(t0)
            }
            _ = prefillSeconds(64)   // warm the GPU
            out.append("--- fresh full prefill (the current per-round cost) ---")
            for L in [512, 2048, 8192, 16384] {
                out.append(String(format: "  L=%6d : %6.2fs", L, prefillSeconds(L)))
            }

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
            let fullSecs = prefillSeconds(base + delta)
            out.append("--- incremental potential (cached \(base) + advance \(delta)) ---")
            out.append(String(format: "  advance %d on cached %d : %.2fs   vs   fresh full %d : %.2fs   → %.1fx faster",
                              delta, base, advSecs, base + delta, fullSecs, fullSecs / max(advSecs, 0.0001)))
            return out.joined(separator: "\n")
        }
    }
}
