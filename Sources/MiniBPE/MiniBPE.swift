import Foundation

/// Pure-Swift byte-level BPE tokenizer that loads a HuggingFace `tokenizer.json`
/// (Qwen / GPT-2 family) and encodes/decodes token-for-token identically. No deps.
public final class MiniBPE: @unchecked Sendable {   // immutable after init (let-only); thread-safe
    private let vocab: [String: Int]                 // token string -> id (BPE base, no specials)
    private let idToToken: [Int: String]             // id -> token string (incl. specials)
    private let ranks: [String: Int]                 // "left right" -> merge rank
    private let specials: [(content: String, id: Int)]   // added tokens, longest-first
    /// Full token↔id map (vocab + specials) the grammar/constrained-decoder consumes.
    public let tokensToIds: [String: Int]
    private let splitRegex: NSRegularExpression
    private let specialRegex: NSRegularExpression?

    public enum LoadError: Error { case notFound(String), badJSON }

    // MARK: tokenizer.json shape (only the fields we need)
    private struct TJSON: Decodable {
        struct Model: Decodable {
            let vocab: [String: Int]
            let merges: [String]   // normalized to "left right"
            enum CodingKeys: String, CodingKey { case vocab, merges }
            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                vocab = try c.decode([String: Int].self, forKey: .vocab)
                // `merges` is either ["Ġ Ġ", ...] (older HF) or [["Ġ","Ġ"], ...] (newer HF).
                if let asStrings = try? c.decode([String].self, forKey: .merges) {
                    merges = asStrings
                } else {
                    merges = try c.decode([[String]].self, forKey: .merges)
                        .map { $0.joined(separator: " ") }
                }
            }
        }
        struct Added: Decodable { let id: Int; let content: String }
        let model: Model
        let added_tokens: [Added]
    }

    public init(tokenizerJSON url: URL) throws {
        guard let data = try? Data(contentsOf: url) else { throw LoadError.notFound(url.path) }
        guard let tj = try? JSONDecoder().decode(TJSON.self, from: data) else { throw LoadError.badJSON }

        self.vocab = tj.model.vocab
        // Merge ranks keyed by "left right" (byte-level pieces never contain a literal space).
        var ranks = [String: Int](minimumCapacity: tj.model.merges.count)
        for (i, m) in tj.model.merges.enumerated() { ranks[m] = i }
        self.ranks = ranks

        // Specials: match longest content first so e.g. "<|im_start|>" wins over "<".
        self.specials = tj.added_tokens
            .map { ($0.content, $0.id) }
            .sorted { $0.content.count > $1.content.count }

        var id2tok = [Int: String](minimumCapacity: tj.model.vocab.count + tj.added_tokens.count)
        for (t, i) in tj.model.vocab { id2tok[i] = t }
        for a in tj.added_tokens { id2tok[a.id] = a.content }
        self.idToToken = id2tok
        self.tokensToIds = Dictionary(uniqueKeysWithValues: id2tok.map { ($0.value, $0.key) })

        // Qwen pre-tokenizer Split pattern (verbatim from tokenizer.json).
        let pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        self.splitRegex = try NSRegularExpression(pattern: pattern)

        if specials.isEmpty {
            self.specialRegex = nil
        } else {
            let alt = specials.map { NSRegularExpression.escapedPattern(for: $0.content) }
                .joined(separator: "|")
            self.specialRegex = try NSRegularExpression(pattern: alt)
        }
    }

    // MARK: - Encode

    /// Encode text → token ids, matching HF byte-level BPE (specials extracted, NFC, digit-split).
    public func encode(_ text: String) -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping   // NFC
        var ids: [Int] = []
        // 1. Carve out special tokens; BPE only the spans between them.
        for piece in splitOnSpecials(normalized) {
            switch piece {
            case .special(let id): ids.append(id)
            case .text(let s): ids.append(contentsOf: encodePlain(s))
            }
        }
        return ids
    }

    private enum Piece { case text(String); case special(Int) }

    private func splitOnSpecials(_ s: String) -> [Piece] {
        guard let re = specialRegex else { return [.text(s)] }
        let ns = s as NSString
        var out: [Piece] = []
        var cursor = 0
        re.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m else { return }
            let r = m.range
            if r.location > cursor {
                out.append(.text(ns.substring(with: NSRange(location: cursor, length: r.location - cursor))))
            }
            let content = ns.substring(with: r)
            if let id = specials.first(where: { $0.content == content })?.id { out.append(.special(id)) }
            cursor = r.location + r.length
        }
        if cursor < ns.length {
            out.append(.text(ns.substring(with: NSRange(location: cursor, length: ns.length - cursor))))
        }
        return out
    }

    /// BPE a special-free string: regex pre-tokenize → byte-level map → merges → ids.
    private func encodePlain(_ s: String) -> [Int] {
        guard !s.isEmpty else { return [] }
        var ids: [Int] = []
        let ns = s as NSString
        splitRegex.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m else { return }
            let piece = ns.substring(with: m.range)        // one pre-token
            let mapped = ByteLevel.encode(Substring(piece)) // bytes → byte-level chars
            for sym in bpe(mapped) {
                if let id = vocab[sym] { ids.append(id) }
            }
        }
        return ids
    }

    /// Canonical GPT-2 BPE: repeatedly merge the lowest-rank adjacent pair (all occurrences).
    private func bpe(_ chars: [Character]) -> [String] {
        var word = chars.map { String($0) }
        if word.count < 2 { return word }
        while true {
            var bestRank = Int.max
            var bestPair = ""
            for i in 0..<(word.count - 1) {
                let pair = word[i] + " " + word[i + 1]
                if let r = ranks[pair], r < bestRank { bestRank = r; bestPair = pair }
            }
            if bestRank == Int.max { break }                 // nothing left to merge
            let sep = bestPair.firstIndex(of: " ")!
            let left = String(bestPair[..<sep])
            let right = String(bestPair[bestPair.index(after: sep)...])
            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == left, word[i + 1] == right {
                    merged.append(left + right); i += 2
                } else {
                    merged.append(word[i]); i += 1
                }
            }
            word = merged
            if word.count == 1 { break }
        }
        return word
    }

    // MARK: - Decode

    public func decode(_ ids: [Int]) -> String {
        var byteChars: [Character] = []
        var out = ""
        for id in ids {
            guard let tok = idToToken[id] else { continue }
            if let sp = specials.first(where: { $0.id == id }) {
                // flush pending byte-level chars, then emit the special literally
                out += ByteLevel.decode(byteChars); byteChars.removeAll()
                out += sp.content
            } else {
                byteChars.append(contentsOf: tok)
            }
        }
        out += ByteLevel.decode(byteChars)
        return out
    }

    public var vocabSize: Int { idToToken.count }
}

/// The minimal vocab surface SwiftLM's grammar / constrained-decoder needs from a
/// tokenizer. MiniBPE satisfies it directly — the grammar runs on OUR tokenizer,
/// not a third-party one (this is what the swift-transformers fork used to provide).
public protocol GrammarTokenizer: Sendable {
    var tokensToIds: [String: Int] { get }
    var idsToTokens: [Int: String] { get }
    var eosTokenId: Int? { get }
    func tokenize(text: String) -> [String]
}

extension MiniBPE: GrammarTokenizer {
    public var idsToTokens: [Int: String] { idToToken }

    public var eosTokenId: Int? {
        specials.first { $0.content == "<|im_end|>" }?.id
            ?? specials.first { $0.content == "<|endoftext|>" }?.id
    }

    /// Byte-level BPE token STRINGS for a text (specials emitted as their literal
    /// content). Same pipeline as `encode`, returning token strings instead of ids.
    public func tokenize(text: String) -> [String] {
        var toks: [String] = []
        for piece in splitOnSpecials(text.precomposedStringWithCanonicalMapping) {
            switch piece {
            case .special(let id):
                if let t = idToToken[id] { toks.append(t) }
            case .text(let s):
                let ns = s as NSString
                splitRegex.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
                    guard let m else { return }
                    toks.append(contentsOf: bpe(ByteLevel.encode(Substring(ns.substring(with: m.range)))))
                }
            }
        }
        return toks
    }
}
