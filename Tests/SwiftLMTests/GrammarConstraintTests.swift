import Testing
import Foundation
import JSONSchema
import MiniBPE
@testable import SwiftLM

// Proves the (now-public) JSON-grammar state machine constrains generation correctly through the MLX-
// facing API — `validTokens()` / `updateState()` / `isComplete`. A synthetic tokenizer with a tiny,
// fully-known vocab makes the constraint deterministic (no model needed). The MLX `GrammarLogitProcessor`
// that masks `MLXArray` logits is the thin wrapper over exactly this; its end-to-end run is the model
// smoke test.

/// Synthetic grammar tokenizer: a whole-string token if known, else per-char.
private struct StubGrammarTokenizer: GrammarTokenizer {
    let tokensToIds: [String: Int]
    var idsToTokens: [Int: String] { Dictionary(uniqueKeysWithValues: tokensToIds.map { ($0.value, $0.key) }) }
    var eosTokenId: Int? { tokensToIds["<eos>"] }
    func tokenize(text: String) -> [String] {
        tokensToIds[text] != nil ? [text] : text.map(String.init)
    }
}

/// Mimics a byte-level BPE tokenizer: the newline byte (0x0A) is the vocab char "Ċ", the space byte (0x20)
/// is "Ġ" — so `decodedToken` reverses that map, exactly as MiniBPE does. The grammar MUST classify on this
/// decoded form, else a control char hides behind "Ċ" and corrupts a JSON string.
private struct ByteLevelStubTokenizer: GrammarTokenizer {
    let tokensToIds: [String: Int]
    var idsToTokens: [Int: String] { Dictionary(uniqueKeysWithValues: tokensToIds.map { ($0.value, $0.key) }) }
    var eosTokenId: Int? { nil }
    func tokenize(text: String) -> [String] { tokensToIds[text] != nil ? [text] : text.map(String.init) }
    func decodedToken(_ token: String) -> String {
        token.replacingOccurrences(of: "Ċ", with: "\n").replacingOccurrences(of: "Ġ", with: " ")
    }
}

@JSONSchema struct GrammarTestShape { let name: String }

struct GrammarConstraintTests {
    private let vocab: [String: Int] =
        ["{": 0, "}": 1, "\"": 2, ":": 3, ",": 4, "name": 5, "hello": 6, "<eos>": 7, "go": 8, "stop": 9]
    private var tok: StubGrammarTokenizer { StubGrammarTokenizer(tokensToIds: vocab) }

    @Test func macroProducesTheExpectedField() {
        #expect(GrammarTestShape.schemaProperties?.map(\.name) == ["name"])
    }

    @Test func firstTokenMustBeOpenBrace() {
        let t = JSONSchemaStateTracker(schema: GrammarTestShape.self, tokenizer: tok)
        let valid = t.validTokens()
        #expect(valid.contains(0))     // "{" allowed
        #expect(!valid.contains(2))    // a quote is NOT allowed before the object opens
        #expect(!valid.contains(5))    // a key token is NOT allowed yet
        #expect(t.isComplete == false)
    }

    @Test func afterOpenBraceMustBeKeyQuote() {
        var t = JSONSchemaStateTracker(schema: GrammarTestShape.self, tokenizer: tok)
        var decoded: [Int] = []
        t.updateState(with: 0, &decoded)   // consume "{"
        let valid = t.validTokens()
        #expect(valid.contains(2))     // now a key-opening quote
        #expect(!valid.contains(0))    // another brace is NOT allowed
    }

    @Test func runtimeEnumConstrainsAStringValue() {
        // `name` is a plain String, but runtimeEnums forces it ∈ {go, stop} — the tool-name mechanism.
        var t = JSONSchemaStateTracker(schema: GrammarTestShape.self, tokenizer: tok,
                                       runtimeEnums: ["name": ["go", "stop"]])
        var d: [Int] = []
        for tid in [0, 2, 5, 2, 3, 2] { t.updateState(with: tid, &d) }   // drive `{ "name" : "`
        let valid = t.validTokens()
        #expect(valid.contains(8))    // "go" allowed (an enum value)
        #expect(valid.contains(9))    // "stop" allowed
        #expect(!valid.contains(6))   // "hello" (arbitrary string content) is NOT allowed
    }

    @Test func runtimeFieldsInitConstrainsAnObject() {
        // no compile-time @JSONSchema type — fields supplied directly (the tool-args mechanism).
        let t = JSONSchemaStateTracker(fields: [SchemaProperty("name", String.self)], tokenizer: tok)
        #expect(t.validTokens().contains(0))   // first token must be "{"
        #expect(t.isComplete == false)
    }

    @Test func runtimeFieldsForceTheFieldKey() {
        var t = JSONSchemaStateTracker(fields: [SchemaProperty("name", String.self)], tokenizer: tok)
        var d: [Int] = []
        t.updateState(with: 0, &d)             // {
        #expect(t.validTokens().contains(2))   // " (open the key)
        t.updateState(with: 2, &d)             // " → key resolves to the runtime field "name"
        #expect(t.validTokens().contains(5))   // forced to emit the "name" key token, not anything else
        #expect(!t.validTokens().contains(6))  // "hello" is NOT a valid key token
    }

    @Test func stringContentExcludesByteLevelControlChars() {
        // Byte-level vocab: "Ċ" is the newline byte (decodes to "\n"); "hello"/"ĠBob" are plain content.
        // A raw control char in a JSON string is invalid (RFC 8259 §7), so "Ċ" must NOT be string content —
        // the classification has to see the DECODED token, not the byte-encoded "Ċ".
        let vocab = ["{": 0, "}": 1, "\"": 2, ":": 3, ",": 4, "name": 5, "hello": 6, "Ċ": 10, "ĠBob": 11]
        let byteTok = ByteLevelStubTokenizer(tokensToIds: vocab)
        var t = JSONSchemaStateTracker(fields: [SchemaProperty("name", String.self)], tokenizer: byteTok)
        var d: [Int] = []
        for tid in [0, 2, 5, 2, 3, 2] { t.updateState(with: tid, &d) }   // drive `{ "name" : "` → in-string
        let valid = t.validTokens()
        #expect(valid.contains(6))     // "hello" — plain content, allowed
        #expect(valid.contains(11))    // "ĠBob" decodes to " Bob"; a space (0x20) is valid string content
        #expect(!valid.contains(10))   // "Ċ" decodes to a raw newline → excluded (the byte-level fix)
        #expect(valid.contains(2))     // the closing quote is always reachable
    }

    @Test func unknownTokenDoesNotAdvancePastBrace() {
        var t = JSONSchemaStateTracker(schema: GrammarTestShape.self, tokenizer: tok)
        var decoded: [Int] = []
        t.updateState(with: 9999, &decoded)   // id not in vocab → ignored, state unchanged
        #expect(t.validTokens().contains(0))  // still expecting "{"
    }
}
