import Foundation
import CoreML

// Our own sampling primitives — replaces the swift-transformers fork's `TensorUtils`.
// Pure Swift; the only pieces the CoreML decode loop actually used.

/// Top-K logit filter: keep the `k` highest-logit candidates, preserving the
/// (index, logit) pairing. Matches the fork's `TopKLogitsWarper(k:).warp(...)`.
struct TopKLogitsWarper {
    let k: Int
    func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard k > 0, k < logits.count else { return (indices, logits) }
        let top = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(k)
        return (top.map { indices[$0] }, top.map { logits[$0] })
    }
}

/// Nucleus (top-P) filter: keep the smallest set of highest-probability candidates
/// whose cumulative probability exceeds `p` (always keeps at least one). Matches the
/// fork's `TopPLogitsWarper(p:).warp(...)`.
struct TopPLogitsWarper {
    let p: Float
    func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard p < 1.0, logits.count > 1 else { return (indices, logits) }
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { Foundation.exp($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        let probs = exps.map { $0 / sum }
        let order = probs.indices.sorted { probs[$0] > probs[$1] }
        var cumsum: Float = 0
        var cutoff = order.count
        for (rank, idx) in order.enumerated() {
            cumsum += probs[idx]
            if cumsum > p { cutoff = rank + 1; break }
        }
        cutoff = max(cutoff, 1)
        let kept = order.prefix(cutoff)
        return (kept.map { indices[$0] }, kept.map { logits[$0] })
    }
}

/// Categorical sampling helpers (the fork's `Math.sample`).
enum Math {
    /// Sample an index from a (parallel index, probability) distribution on CPU.
    static func sample(indexes: [Int], probs: [Float]) -> Int {
        guard !indexes.isEmpty else { return 0 }
        let total = probs.reduce(0, +)
        guard total > 0 else { return indexes[0] }
        var r = Float.random(in: 0 ..< total)
        for (i, p) in probs.enumerated() {
            r -= p
            if r <= 0 { return indexes[i] }
        }
        return indexes[indexes.count - 1]   // fallback for FP rounding
    }

    /// Sample from GPU tensors: pull probs/indices to CPU, then categorical-sample.
    /// `indexes` are argsort/gather results (Int32); `probs` are a softmax (Float).
    static func sample(indexes: MLTensor, probs: MLTensor) async -> Int {
        let p = await probs.shapedArray(of: Float.self).scalars
        let idx = await indexes.shapedArray(of: Int32.self).scalars.map(Int.init)
        return sample(indexes: idx, probs: p)
    }
}
