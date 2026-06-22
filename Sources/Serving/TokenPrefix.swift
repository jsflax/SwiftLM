import Foundation

// ── B2 incremental-KV support (MLX-free, pure, so it unit-tests with no model). The owned-render path can reuse
// a KV cache carried from the prior round ONLY when the tokens already in the cache are an EXACT PREFIX of the
// freshly-rendered row for this round — then it prefills just the new tail instead of re-prefilling everything.
// The recurrent (GatedDeltaNet/Mamba) layers can't be rewound, so reuse must be forward-only and prefix-exact:
// any divergence ⇒ discard the cache and full-prefill (correct, just slow). This is the gate that makes the
// reuse provably-never-silently-wrong — it's token equality, not an assumption that the render is stable.

/// Length of the longest common prefix of two token rows (number of leading positions where `a[i] == b[i]`).
public func tokenCommonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
    let n = min(a.count, b.count)
    var i = 0
    while i < n && a[i] == b[i] { i += 1 }
    return i
}

/// True iff `cached` is a STRICT prefix of `fresh` — i.e. every cached token matches and there is at least one
/// new tail token to prefill. This is the exact precondition for incremental KV reuse: the carried cache holds
/// `cached` verbatim, so prefilling `fresh[cached.count...]` reaches the identical end-state as a full prefill
/// of `fresh`. A non-strict-prefix (divergence, or `fresh` not longer) ⇒ caller must full-prefill a fresh cache.
public func isReusablePrefix(cached: [Int32], fresh: [Int32]) -> Bool {
    guard !cached.isEmpty, fresh.count > cached.count else { return false }
    return tokenCommonPrefixLength(cached, fresh) == cached.count
}
