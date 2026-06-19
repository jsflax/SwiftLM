import Foundation
import MiniBPE

// ── Pure context-compaction helpers (MLX-free, unit-testable). The MLX side (CompactingSession) calls these
// to decide WHEN to compact and HOW to split the conversation; the actual summary generation + session
// re-seed is MLX. Token counting uses the model's OWN tokenizer (MiniBPE over the same tokenizer.json), so
// the measurements match what the model actually sees.

/// Cap `text` to `maxTokens` of the model's tokens, appending a clear truncation marker. Keeps a single big
/// tool output (a 10MB read / a 1000-line grep) from blowing context. No-op when already under the cap.
/// The marker tells the model it was truncated so it can re-query more narrowly.
public func truncateToTokens(_ text: String, maxTokens: Int, tokenizer: any GrammarTokenizer) -> String {
    let toks = tokenizer.tokenize(text: text)
    guard toks.count > maxTokens else { return text }
    let kept = toks.prefix(maxTokens).joined()
    return kept + "\n[truncated: \(toks.count - maxTokens) tokens omitted]"
}

/// Decide the compaction split: keep the most-recent messages whose combined token count fits within
/// `keepTokens`; everything older is summarized. Returns the index `i` such that `messages[0..<i]` are
/// summarized and `messages[i...]` are kept VERBATIM. GUARANTEES progress: always summarizes at least the
/// first message and always keeps at least the last one (so compaction shrinks context but the model keeps
/// its immediate context). Call only when the conversation already exceeds the max budget.
public func compactionSplitIndex(messageTokens: [Int], keepTokens: Int) -> Int {
    let n = messageTokens.count
    guard n > 1 else { return 0 }       // 0 or 1 message: nothing to split off
    var sum = 0
    var split = n                       // index of the first KEPT message; n = keep none yet
    var i = n - 1
    while i >= 1 {                       // i >= 1 ⇒ messages[0] is always summarized (progress)
        let t = messageTokens[i]
        if sum + t > keepTokens { break }
        sum += t
        split = i
        i -= 1
    }
    return split == n ? n - 1 : split    // always keep at least the last message
}
