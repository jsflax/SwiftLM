import Foundation
import MLX
import MLXLMCommon

// ── SLICE 1b — the "efficient but harder" variable-length batching primitive.
//
// To run N DIFFERENT prompts of DIFFERENT lengths through one LOCKSTEP decode we do NOT
// truncate (loses prompt content) and do NOT left-pad-the-prompt-and-prefill-together
// (wastes prefill compute on pad tokens). Instead:
//
//   1. PREFILL EACH PROMPT SEPARATELY, alone, at its TRUE length L_i (scalar RoPE offset 0)
//      — exact, no pad artifacts, zero wasted prefill compute.
//   2. MERGE the per-sequence KV caches into ONE batched cache: left-pad every row's KV on
//      the sequence axis up to the common L_max so all rows align with their LAST real token
//      at column L_max-1 (decoded tokens then append contiguously, no growing gap).
//   3. DECODE in lockstep, B rows per forward pass.
//
// The merge is correct because mlx-swift-lm models (GLM4 / GLM4MOE / Qwen3 …) read
// `cache.ropeOffset` and feed it to `MLXFast.RoPE(offset:)`, which accepts a PER-SEQUENCE
// `MLXArray` offset. So each row keeps its OWN RoPE positions (`batchOffset[i] = L_i + decoded`)
// even though they share one storage tensor — no re-basing of the prefilled keys is needed.
// The left-pad columns are hidden by `makeMask`, so a padded row attends to exactly its own
// real keys + its own decoded tokens. Storage / update / serialization is delegated to an
// inner `KVCacheSimple` (the proven path); this type adds only the three batched concerns:
// per-row RoPE offsets, the left-pad mask, and row eviction.
final class BatchedKVCache: BatchPositionedKVCache {
    private let inner = KVCacheSimple()
    /// Per-sequence RoPE position of the NEXT token to be written (= true prefill length +
    /// tokens decoded so far). Shape `[B]`, Int32. Read by the model via `ropeOffset`.
    private var _batchOffset: MLXArray
    /// Per-sequence count of left-pad columns to mask out (= L_max - L_i). Shape `[B]`, Int32.
    private var leftPad: MLXArray

    /// Merge ONE layer: build the batched KV from B per-sequence `[1, H, L_i, D]` key/value
    /// tensors (each already trimmed to its true length L_i), left-padding to `maxLen`.
    init(perRowKeys: [MLXArray], perRowValues: [MLXArray], lengths: [Int], maxLen: Int) {
        func leftPadAndStack(_ rows: [MLXArray]) -> MLXArray {
            let padded = zip(rows, lengths).map { (row, L) -> MLXArray in
                let lp = maxLen - L
                guard lp > 0 else { return row }
                let s = row.shape                                  // [1, H, L, D]
                let z = MLXArray.zeros([s[0], s[1], lp, s[3]], dtype: row.dtype)
                return concatenated([z, row], axis: 2)             // [1, H, maxLen, D]
            }
            return concatenated(padded, axis: 0)                   // [B, H, maxLen, D]
        }
        let mergedK = leftPadAndStack(perRowKeys)
        let mergedV = leftPadAndStack(perRowValues)
        inner.state = [mergedK, mergedV]                           // sets inner.offset = maxLen
        self._batchOffset = MLXArray(lengths.map { Int32($0) })
        self.leftPad = MLXArray(lengths.map { Int32(maxLen - $0) })
        eval(mergedK, mergedV, _batchOffset, leftPad)
    }

    private init(adopting inner: [MLXArray], batchOffset: MLXArray, leftPad: MLXArray) {
        self.inner.state = inner
        self._batchOffset = batchOffset
        self.leftPad = leftPad
    }

    // MARK: BatchPositionedKVCache / RoPE

    var batchOffset: MLXArray { _batchOffset }
    /// Declared explicitly (not via the protocol-extension default) so the witness used through a
    /// `KVCache` existential is unambiguously the per-sequence `.batch` form.
    var ropeOffset: RoPEOffset { .batch(_batchOffset) }

    // MARK: KVCache

    var offset: Int { inner.offset }
    var maxSize: Int? { nil }

    func innerState() -> [MLXArray] { inner.innerState() + [_batchOffset, leftPad] }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = inner.update(keys: keys, values: values)
        // Every active row advanced by `n` positions; keep RoPE offsets in lockstep with storage.
        let advanced = _batchOffset + Int32(keys.dim(2))
        eval(advanced)
        _batchOffset = advanced
        return result
    }

    var state: [MLXArray] {
        get { inner.state }
        set { inner.state = newValue }
    }
    var metaState: [String] {
        get { inner.metaState }
        set { inner.metaState = newValue }
    }
    var isTrimmable: Bool { false }
    @discardableResult func trim(_ n: Int) -> Int { 0 }

    /// The left-pad attention mask: row `b` attends to storage column `k` iff `k >= leftPad[b]`
    /// (i.e. `k` is a real or decoded position, not a left-pad slot). For multi-token queries
    /// (`n > 1`) causality among the new rows is also enforced; the decode path uses `n == 1`.
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let kLen = inner.offset + n
        let B = leftPad.shape[0]
        let cols = MLXArray(Int32(0) ..< Int32(kLen)).reshaped([1, 1, 1, kLen])
        var keep = cols .>= leftPad.reshaped([B, 1, 1, 1])                  // [B,1,1,kLen]
        if n > 1 {
            // Query row q (0..<n) is the new token at storage column kLen-n+q; it may attend
            // to columns up to and including its own.
            let qPos = MLXArray(Int32(kLen - n) ..< Int32(kLen)).reshaped([1, 1, n, 1])
            keep = keep & (cols .< (qPos + Int32(1)))                      // [B,1,n,kLen]
        }
        return .array(keep)
    }

    func copy() -> any KVCache {
        BatchedKVCache(
            adopting: inner.state.map { $0[.ellipsis] },
            batchOffset: _batchOffset[.ellipsis],
            leftPad: leftPad[.ellipsis])
    }

    // MARK: Eviction

    /// Drop finished rows, keeping only `keepIndices` (positions in the CURRENT batch order).
    /// Shrinks the storage batch dim and the per-row RoPE/mask metadata together.
    func evict(keep keepIndices: [Int]) {
        let idx = MLXArray(keepIndices.map { Int32($0) })
        let s = inner.state                                // [keys[B,H,off,D], values[B,H,off,D]]
        inner.state = [s[0][idx], s[1][idx]]               // offset preserved (= seq dim, unchanged)
        _batchOffset = _batchOffset[idx]
        leftPad = leftPad[idx]
        eval(_batchOffset, leftPad)
    }
}
