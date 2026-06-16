import Foundation

/// Strips PII / secrets from transcript text before it is ever written to disk or counted.
/// Reports redaction TYPES and counts only — never values. On-device, irreversible.
struct Redactor {
    /// One rule = a category label + a compiled regex + a replacement placeholder.
    private struct Rule { let label: String; let re: NSRegularExpression; let repl: String }
    private let rules: [Rule]
    /// counts[label] = number of substitutions made (for type-not-value reporting)
    private(set) var counts: [String: Int] = [:]

    init() {
        func r(_ label: String, _ pattern: String, _ repl: String,
               _ opts: NSRegularExpression.Options = []) -> Rule {
            Rule(label: label,
                 re: try! NSRegularExpression(pattern: pattern, options: opts),
                 repl: repl)
        }
        // Order matters: most-specific secrets first, then generic, then paths last.
        rules = [
            // JWTs (three base64url segments) — before generic hex/token rules.
            r("jwt", #"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"#, "<JWT>"),
            // Provider API keys.
            r("openai_key", #"sk-[A-Za-z0-9_-]{20,}"#, "<OPENAI_KEY>"),
            r("anthropic_key", #"sk-ant-[A-Za-z0-9_-]{20,}"#, "<ANTHROPIC_KEY>"),
            r("github_pat", #"gh[pousr]_[A-Za-z0-9]{20,}"#, "<GITHUB_PAT>"),
            r("github_fine_pat", #"github_pat_[A-Za-z0-9_]{20,}"#, "<GITHUB_PAT>"),
            r("aws_akid", #"AKIA[0-9A-Z]{16}"#, "<AWS_KEY>"),
            r("google_key", #"AIza[0-9A-Za-z_-]{30,}"#, "<GOOGLE_KEY>"),
            r("slack_token", #"xox[baprs]-[A-Za-z0-9-]{10,}"#, "<SLACK_TOKEN>"),
            // Bearer/Authorization headers carrying opaque tokens.
            r("bearer", #"(?i)bearer\s+[A-Za-z0-9._~+/-]{16,}=*"#, "Bearer <TOKEN>"),
            // Emails.
            r("email", #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#, "<EMAIL>"),
            // Long hex blobs (>=32) — sha/secret-ish. After named keys so we don't double-hit.
            r("hex_blob", #"\b[0-9a-fA-F]{32,}\b"#, "<HEX>"),
            // The user's home path — last (so token rules see full strings first).
            r("home_path", #"/Users/jason"#, "/Users/USER"),
        ]
    }

    /// Redact a string in place, accumulating per-category counts.
    mutating func redact(_ s: String) -> String {
        var text = s
        for rule in rules {
            let range = NSRange(text.startIndex..., in: text)
            let n = rule.re.numberOfMatches(in: text, range: range)
            if n > 0 {
                counts[rule.label, default: 0] += n
                text = rule.re.stringByReplacingMatches(
                    in: text, range: NSRange(text.startIndex..., in: text), withTemplate: rule.repl)
            }
        }
        return text
    }

    /// Type-not-value summary suitable for logging.
    var summary: String {
        if counts.isEmpty { return "no PII detected" }
        return counts.sorted { $0.value > $1.value }
            .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
    }
}
