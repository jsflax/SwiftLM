import Testing
@testable import Serving

// B2 incremental-KV gate: the prefix check is the load-bearing correctness guard (reuse the carried KV cache
// ONLY when it's an exact prefix of the new render). These prove the gate, model-free.

struct TokenPrefixTests {
    @Test func commonPrefixLengthBasics() {
        #expect(tokenCommonPrefixLength([], []) == 0)
        #expect(tokenCommonPrefixLength([1, 2, 3], []) == 0)
        #expect(tokenCommonPrefixLength([1, 2, 3], [1, 2, 3]) == 3)
        #expect(tokenCommonPrefixLength([1, 2, 3], [1, 2, 3, 4, 5]) == 3)   // strict prefix
        #expect(tokenCommonPrefixLength([1, 2, 9], [1, 2, 3, 4]) == 2)      // diverge at index 2
        #expect(tokenCommonPrefixLength([9, 2, 3], [1, 2, 3]) == 0)         // diverge immediately
    }

    @Test func reusableOnlyWhenStrictPrefixWithNewTail() {
        // the common case: prior cache + a new tail to append → reuse
        #expect(isReusablePrefix(cached: [1, 2, 3], fresh: [1, 2, 3, 4, 5]) == true)
        // exact equality is NOT reusable (no new tail to prefill, and decoding can't resume from a closed row)
        #expect(isReusablePrefix(cached: [1, 2, 3], fresh: [1, 2, 3]) == false)
        // divergence anywhere in the cached span → NOT reusable (a moved <think> wrapper / re-serialized tool call)
        #expect(isReusablePrefix(cached: [1, 2, 3], fresh: [1, 2, 9, 4]) == false)
        // a shorter fresh row (shouldn't happen, but must not reuse) → false
        #expect(isReusablePrefix(cached: [1, 2, 3, 4], fresh: [1, 2, 3]) == false)
        // empty cache (first round) → false, must full-prefill
        #expect(isReusablePrefix(cached: [], fresh: [1, 2, 3]) == false)
    }
}
