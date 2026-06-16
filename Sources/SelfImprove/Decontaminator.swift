import Foundation

// ── v2a: the LEAKAGE FENCE (step 1 — highest severity).
//
// The eval set is the held-out backward(_:) bodies in the domain repo. The corpus scan found the
// user's own session transcripts contain those exact bodies (5-36 occurrences each). Harvesting
// them into the training mix would turn the flywheel's lift number into a MEMORIZATION score
// (SWE-bench saw 5.4x inflation from exactly this). `Redactor` strips secrets/PII only — it does
// ZERO content decontamination vs the eval set. `Decontaminator` is that missing piece:
//   • CONTENT gate — drop any training text whose (whitespace-insensitive) code contains an
//     eval-target signature. Whitespace-insensitive because transcripts reflow spacing.
//   • PROVENANCE gate — drop any Edit/Write whose `file_path` lives under the eval repo
//     (file-level quarantine; the harvester otherwise has no per-Edit path provenance).
//   • INTEGRITY gate — `assertClean` throws so a train job HARD-FAILS if a leak slips through.
//
// PREVENTIVE, not emergency: the current curriculum is clean; this keeps the NEXT harvest honest.
// It is also self-protecting — a transcript of the session that wrote these signatures (this one)
// contains them and is therefore dropped by the blocklist itself.

public enum DecontamError: Error, CustomStringConvertible {
    case contaminated(signature: String, sample: String)
    public var description: String {
        switch self {
        case let .contaminated(sig, sample):
            return "eval-set leakage: training text matches held-out signature «\(sig)» — sample: \(sample)…"
        }
    }
}

public struct Decontaminator: Sendable {
    /// (whitespace-stripped match key, original line for reporting).
    private let signatures: [(stripped: String, original: String)]
    /// File-path prefixes whose edits are quarantined (the eval repo).
    private let quarantinePrefixes: [String]

    public init(substrings: [String] = DomainEvalSuite.evalSubstrings,
                quarantinePrefixes: [String] = [DomainEvalSuite.llmFromScratch.path]) {
        // ≥12 non-space chars so a signature can't be a trivially-common fragment.
        self.signatures = substrings
            .map { (Self.strip($0), $0) }
            .filter { $0.stripped.count >= 12 }
        self.quarantinePrefixes = quarantinePrefixes
    }

    /// Strip ALL whitespace → whitespace-insensitive code matching.
    static func strip(_ s: String) -> String {
        String(s.unicodeScalars.filter { !CharacterSet.whitespacesAndNewlines.contains($0) })
    }

    /// The first eval signature found in `text`, or nil. Returns the original (spaced) line.
    public func match(in text: String) -> String? {
        let hay = Self.strip(text)
        guard !hay.isEmpty else { return nil }
        for sig in signatures where hay.contains(sig.stripped) { return sig.original }
        return nil
    }

    /// CONTENT gate for a (request/human, answer/assistant) pair.
    public func isContaminatedPair(_ a: String, _ b: String) -> Bool {
        match(in: a) != nil || match(in: b) != nil
    }

    /// PROVENANCE gate: true if a mutating tool call targets a quarantined (eval-repo) file.
    /// Internal — `JSONValue` is module-internal; the in-module `Harvester` is the only caller.
    func isQuarantinedEdit(toolName: String?, input: JSONValue?) -> Bool {
        let mutating: Set<String> = ["Edit", "Write", "MultiEdit", "NotebookEdit"]
        guard let n = toolName, mutating.contains(n),
              let path = input?["file_path"]?.stringValue ?? input?["notebook_path"]?.stringValue
        else { return false }
        return quarantinePrefixes.contains { path.hasPrefix($0) }
    }

    /// INTEGRITY gate — call before training. Throws on the first contaminated text so the train
    /// job fails loudly rather than silently learning the answer.
    public func assertClean(_ texts: [String]) throws {
        for t in texts where !t.isEmpty {
            if let sig = match(in: t) {
                throw DecontamError.contaminated(signature: sig, sample: String(t.prefix(80)))
            }
        }
    }
}

// Field accessors for the heterogeneous tool_use `input` (used for per-Edit provenance).
extension JSONValue {
    var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    subscript(_ key: String) -> JSONValue? { if case .object(let o) = self { return o[key] }; return nil }
}
