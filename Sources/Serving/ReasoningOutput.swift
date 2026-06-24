import Foundation

// ── Pure helpers for reasoning-model output. GLM-4 (and R1-distills) emit `<think>…</think>` spans before
// the answer, and emit tool calls as `<tool_call>…</tool_call>` tags. Two needs arise that the MLX backend's
// parser doesn't cover:
//   1. strip the reasoning span from what the user sees (else `…</think>The answer` bleeds through);
//   2. recover a tool call the backend parser left on the floor — notably GLM's BARE-NAME no-arg form
//      `<tool_call>current_time</tool_call>` (args-bearing GLM calls parse fine; the no-arg shape doesn't),
//      so the live loop can still dispatch it.
// MLX-free → unit-tested without loading a model. No-ops for non-reasoning models (no tags present).

/// Remove `<think>…</think>` spans (and a dangling unclosed `<think>` to end-of-string, from truncation) and
/// any `<tool_call>…</tool_call>` tags from text shown to the user. Returns the trimmed remainder.
public func stripReasoning(_ text: String) -> String {
    var s = text
    func removeSpans(open: String, close: String) {
        while let o = s.range(of: open) {
            if let c = s.range(of: close, range: o.upperBound..<s.endIndex) {
                s.removeSubrange(o.lowerBound..<c.upperBound)
            } else {
                s.removeSubrange(o.lowerBound..<s.endIndex)   // unclosed (truncated) → drop to end
                break
            }
        }
    }
    removeSpans(open: "<think>", close: "</think>")
    removeSpans(open: "<tool_call>", close: "</tool_call>")
    s = stripResidualWireTags(s)
    return s.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Remove any ORPHANED tool-call wire tags left in display text. The 122B owned-render parses a tool call out
/// of band (consuming the `<tool_call><function=…>` opening as the call) but can leave the closing `</tool_call>`
/// — or a stray `<function=…>`/`<parameter=…>` — behind, which then renders as junk ("</tool_call>" fragments
/// leaking between tool cards in the transcript). Span removal above only catches MATCHED pairs; this sweeps the
/// standalone remainder. Safe for non-122B models (the tags simply don't occur).
public func stripResidualWireTags(_ text: String) -> String {
    var s = text
    for tag in ["<tool_call>", "</tool_call>", "</function>", "</parameter>"] {
        s = s.replacingOccurrences(of: tag, with: "")
    }
    s = s.replacingOccurrences(of: "<function=[^>]*>", with: "", options: .regularExpression)
    s = s.replacingOccurrences(of: "<parameter=[^>]*>", with: "", options: .regularExpression)
    return s
}

/// Extract the REASONING channel — the concatenated content INSIDE `<think>…</think>` spans (the inverse of
/// `stripReasoning`). Empty for non-reasoning output. Surfaced to Orbital as a separate `reasoningDelta` so a
/// reasoning model's analysis can render as a specially-formatted "thinking" block instead of being dropped.
/// Handles multiple spans and a dangling unclosed `<think>` (truncation → take to end).
public func extractReasoning(_ text: String) -> String {
    var out: [String] = []
    var s = Substring(text)
    while let o = s.range(of: "<think>") {
        let after = s[o.upperBound...]
        if let c = after.range(of: "</think>") {
            out.append(String(after[..<c.lowerBound])); s = after[c.upperBound...]
        } else {
            out.append(String(after)); break   // unclosed (truncated) → to end
        }
    }
    return out.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
}

/// The user-facing answer for (possibly reasoning) output: the text OUTSIDE `<think>` when present, else the
/// CONCLUSION of the reasoning (its last non-empty line) — so a model that finishes its work but ends inside
/// a `<think>` block (GLM does this) still says something useful instead of showing nothing.
public func displayAnswer(_ raw: String) -> String {
    let stripped = stripReasoning(raw)
    if !stripped.isEmpty { return stripped }
    // Reasoning-only turn: surface the last non-empty line of the think content as the conclusion.
    let inner = raw.replacingOccurrences(of: "<think>", with: "")
                   .replacingOccurrences(of: "</think>", with: "")
    return inner.split(whereSeparator: \.isNewline)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .last(where: { !$0.isEmpty }) ?? ""
}

/// Streaming filter that removes whole tagged spans — `<think>…</think>` (reasoning) AND `<tool_call>…
/// </tool_call>` (the raw call, surfaced separately as a tool_use event) — from a sequence of text chunks,
/// handling tags that split ACROSS chunk boundaries. So a reasoning model's live token stream renders only
/// the visible answer (e.g. an Orbital room shows the answer, never the think block or the raw tool tag).
/// Feed each chunk, emit the returned visible text; call `flush()` at end-of-stream for any held-back tail.
public struct ReasoningStreamFilter {
    private let spans: [(open: String, close: String)]
    private var current: Int? = nil   // index of the span we're inside, or nil when outside
    private var buf = ""

    public init(spans: [(open: String, close: String)] =
                [("<think>", "</think>"), ("<tool_call>", "</tool_call>")]) {
        self.spans = spans
    }

    public mutating func feed(_ chunk: String) -> String {
        buf += chunk
        var out = ""
        while true {
            if let cur = current {                        // inside a span: drop until its close tag
                let close = spans[cur].close
                if let r = buf.range(of: close) {
                    buf = String(buf[r.upperBound...]); current = nil
                } else {                                  // hold a possible split close tag
                    buf = String(buf.suffix(partialTailLen(buf, of: close))); break
                }
            } else {                                      // outside a span
                // earliest OPEN across spans
                var bestOpen: (idx: Int, range: Range<String.Index>)?
                for (i, s) in spans.enumerated() {
                    if let r = buf.range(of: s.open),
                       bestOpen == nil || r.lowerBound < bestOpen!.range.lowerBound { bestOpen = (i, r) }
                }
                // earliest CLOSE across spans — a DANGLING close (no matching open in this stream) means the
                // span was opened earlier (the owned-render `<think>` primed in the prompt) or the model emitted
                // a stray close. Emit the genuine text before it and DROP the close tag, so a bare
                // `</think>` / `</tool_call>` never leaks into the chat.
                var bestClose: Range<String.Index>?
                for s in spans {
                    if let r = buf.range(of: s.close),
                       bestClose == nil || r.lowerBound < bestClose!.lowerBound { bestClose = r }
                }
                if let c = bestClose, bestOpen == nil || c.lowerBound < bestOpen!.range.lowerBound {
                    out += String(buf[..<c.lowerBound])
                    buf = String(buf[c.upperBound...]); continue
                }
                if let b = bestOpen {
                    out += String(buf[..<b.range.lowerBound])
                    buf = String(buf[b.range.upperBound...]); current = b.idx
                } else {                                  // emit all but a possible split open/close tag tail
                    let keep = spans.flatMap { [partialTailLen(buf, of: $0.open), partialTailLen(buf, of: $0.close)] }.max() ?? 0
                    out += String(buf.dropLast(keep))
                    buf = String(buf.suffix(keep)); break
                }
            }
        }
        return out
    }

    /// End-of-stream: emit leftover visible text (drop it if we ended mid-span).
    public mutating func flush() -> String {
        let out = current != nil ? "" : buf
        buf = ""; return out
    }

    /// Longest suffix of `s` that is a proper prefix of `tag` (a tag possibly split across chunks).
    private func partialTailLen(_ s: String, of tag: String) -> Int {
        var n = min(s.count, tag.count - 1)
        while n > 0 { if tag.hasPrefix(s.suffix(n)) { return n }; n -= 1 }
        return 0
    }
}

/// Parse a `{"name":…,"arguments":…}` object → (name, argsJSON). nil unless it's a valid NAMED tool call.
/// `requireArguments` demands an `arguments` key too (a stronger signal — used for the bare/fenced forms that
/// lack `<tool_call>` framing, so a stray named JSON object in prose isn't mistaken for a call).
private func parseJSONToolCallObject(_ s: String, requireArguments: Bool = false)
    -> (name: String, argsJSON: String)? {
    let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
    guard t.hasPrefix("{"), let d = t.data(using: .utf8),
          let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
          let n = o["name"] as? String, !n.isEmpty else { return nil }
    if requireArguments, o["arguments"] == nil { return nil }
    let args = o["arguments"].flatMap { try? JSONSerialization.data(withJSONObject: $0) }
        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    return (n, args)
}

/// Recover a tool call the backend parser didn't surface. Handles the three real forms an on-device model
/// emits when its `ToolCallFormat` parser misses: (1) `<tool_call>…</tool_call>` framing — GLM bare-name, or
/// GLM/Qwen JSON-in-tags `{"name":…,"arguments":…}`; (2) a markdown-fenced ```json {"name":…,"arguments":…}```
/// block (a weaker model's wrapper when it omits the tags — observed from Qwen2.5-7B under the batched render).
/// Returns the tool NAME + best-effort args JSON ("{}" when none); the constrained-repair layer fills required
/// args. nil when there's no recoverable call.
public func recoverToolCallTag(_ text: String) -> (name: String, argsJSON: String)? {
    // 1. `<tool_call>…</tool_call>` framing.
    if let open = text.range(of: "<tool_call>") {
        let after = text[open.upperBound...]
        let inner = after.range(of: "</tool_call>").map { String(after[..<$0.lowerBound]) } ?? String(after)
        let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            // JSON-in-tags `{"name":…,"arguments":…}` (GLM/Qwen variants).
            if let jc = parseJSONToolCallObject(trimmed) { return jc }
            // GLM bare-name: the name is the first non-empty line (e.g. `current_time` or `bash\n<arg_key>…`).
            let name = trimmed.split(whereSeparator: \.isNewline).first.map {
                $0.trimmingCharacters(in: .whitespaces) } ?? trimmed
            if !name.isEmpty, !name.contains("<"), !name.contains("{") { return (name, "{}") }
        }
    }
    // 2. Markdown-fenced JSON tool call: ```json\n{"name":…,"arguments":…}\n``` (no <tool_call> tags). Only a
    //    fenced object carrying BOTH name and arguments recovers, so a JSON example in prose is ignored.
    if let fence = text.range(of: "```") {
        let afterFence = String(text[fence.upperBound...])
        let body = afterFence.range(of: "```").map { String(afterFence[..<$0.lowerBound]) } ?? afterFence
        if let brace = body.firstIndex(of: "{"),
           let jc = parseJSONToolCallObject(String(body[brace...]), requireArguments: true) { return jc }
    }
    return nil
}
