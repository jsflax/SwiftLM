import Foundation

// ── Tool-name repair: map a routed tool name to a REAL tool ONLY when it's an unambiguous near-miss.
//
// The model sometimes emits a tool name that isn't exactly a real tool: a case/separator variation
// (`Write_File`, `read-file`) or a small typo (`read_fil`). Those should be repaired. But it ALSO sometimes
// emits a name that is genuinely NOT a tool at all — e.g. `session-learner`, a Claude-Code/Engram hook
// sub-agent the model read from injected context and tried to "call." Repairing THAT by re-picking the
// "most relevant tool" from the task prompt manufactures an unrelated — possibly destructive — call (it
// re-picked `write_file` on a "write the file" task and fired a spurious second write). So: repair only a
// clear near-miss of one specific tool; otherwise return nil so the caller surfaces an honest
// "unknown tool" error instead of running the wrong tool.

/// The nearest REAL tool to `routed` when it's an unambiguous near-miss (case/separator/≤2-char typo of
/// exactly one tool); `nil` for a semantically-unrelated name (→ honest "unknown tool"). Never guesses from
/// task context — it only compares the EMITTED name against the tool list.
public func closestToolName(_ routed: String, in tools: [String]) -> String? {
    if tools.contains(routed) { return routed }                    // exact (incl. empty tool list → nil)
    let r = normalizedToolName(routed)
    guard !r.isEmpty else { return nil }

    // Normalized-exact: case / separators / spaces only (read-file ≡ Read_File ≡ read_file). Unique or bust.
    let normExact = tools.filter { normalizedToolName($0) == r }
    if normExact.count == 1 { return normExact[0] }
    if normExact.count > 1 { return nil }                          // ambiguous → don't guess

    // Small typo: a UNIQUE nearest tool within edit distance ≤ 2 on the normalized form, and the distance
    // must be small relative to the name length (so short garbage can't match a short tool by coincidence).
    let scored = tools.map { (tool: $0, dist: levenshtein(r, normalizedToolName($0))) }
    guard let minDist = scored.map(\.dist).min() else { return nil }
    let winners = scored.filter { $0.dist == minDist }
    guard winners.count == 1, minDist <= 2, minDist * 2 < r.count else { return nil }
    return winners[0].tool
}

/// Lowercase + drop everything but letters/digits — collapses case and separators (`_`, `-`, space, `.`).
func normalizedToolName(_ s: String) -> String {
    String(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
}

/// Classic two-row Levenshtein edit distance (insert/delete/substitute), pure + allocation-light.
func levenshtein(_ a: String, _ b: String) -> Int {
    let a = Array(a), b = Array(b)
    if a.isEmpty { return b.count }
    if b.isEmpty { return a.count }
    var prev = Array(0...b.count)
    var cur = [Int](repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        cur[0] = i
        for j in 1...b.count {
            cur[j] = a[i - 1] == b[j - 1] ? prev[j - 1]
                : Swift.min(prev[j - 1], prev[j], cur[j - 1]) + 1
        }
        swap(&prev, &cur)
    }
    return prev[b.count]
}
