import Foundation
import JSONSchema
import SwiftLM
#if canImport(FoundationModels)
import FoundationModels
#endif

// Makes the MLX backend a first-class `LanguageModel` — the same protocol the
// CoreML and FoundationModels backends conform to. Structured output is produced
// by generate-and-parse for now; grammar-constrained MLX decoding (logit mask on
// the TokenIterator, reusing the MiniBPE-backed JSONSchemaStateTracker) is the
// later valid-by-construction upgrade.

public enum MLXBackendError: Error, CustomStringConvertible {
    case noJSONInOutput(String)
    public var description: String {
        switch self {
        case .noJSONInOutput(let s): "model output contained no JSON object: \(s.prefix(200))"
        }
    }
}

extension JSONSchemaConvertible {
    /// Decode a concrete `JSONSchemaConvertible` from JSON via its existential metatype.
    static func swiftLMDecode(_ data: Data) throws -> Self {
        try JSONDecoder().decode(Self.self, from: data)
    }
}

/// A persistent MLX conversation. Holds the model + system prompt; each `continue`
/// produces a structured response.
public struct MLXConversation: LanguageModelConversation, Sendable {
    let model: MLXLanguageModel
    let systemPrompt: String

    nonisolated(nonsending) public func `continue`<Input: Encodable, Output>(
        input: Input, expecting: Output.Type
    ) async throws -> Output where Output: JSONSchemaConvertible {
        // Encode to a Sendable String locally so we don't send non-Sendable `input`
        // across the async boundary.
        let inputStr: String
        if let s = input as? String { inputStr = s }
        else { inputStr = try String(data: JSONEncoder().encode(input), encoding: .utf8) ?? "" }
        return try await model.structuredAsk(system: systemPrompt, input: inputStr, output: Output.self)
    }
}

extension MLXLanguageModel: LanguageModel {
    public func startConversation(systemPrompt: String) async -> MLXConversation {
        MLXConversation(model: self, systemPrompt: systemPrompt)
    }

    public func structuredAsk<Input, Output>(
        system: String, input: Input, output: Output.Type
    ) async throws -> Output where Input: Encodable {
        let inputStr: String
        if let s = input as? String { inputStr = s }
        else { inputStr = try String(data: JSONEncoder().encode(input), encoding: .utf8) ?? "" }
        let outputType = output as! any (JSONSchemaConvertible & Sendable).Type
        let data = try await generateStructured(system: system, input: inputStr,
                                                schema: outputType.jsonSchema)
        return try outputType.swiftLMDecode(data) as! Output
    }

    /// Build a schema-aware prompt, generate, and extract the JSON object.
    func generateStructured(system: String, input: String, schema: [String: Any]) async throws -> Data {
        let schemaJSON = (try? JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys]))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        let prompt = """
        \(system)

        Respond with ONLY a JSON object matching this schema (no prose, no markdown fences):
        \(schemaJSON)

        Input: \(input)
        """
        let text = try await generate(prompt, maxTokens: 512, temperature: 0.0)
        guard let data = Self.extractJSONObject(text) else { throw MLXBackendError.noJSONInOutput(text) }
        return data
    }

    /// Extract the first balanced top-level `{...}` object from model output.
    static func extractJSONObject(_ s: String) -> Data? {
        guard let start = s.firstIndex(of: "{") else { return nil }
        var depth = 0, inString = false, escaped = false
        var idx = start
        while idx < s.endIndex {
            let c = s[idx]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else if c == "\"" { inString = true }
            else if c == "{" { depth += 1 }
            else if c == "}" {
                depth -= 1
                if depth == 0 { return String(s[start...idx]).data(using: .utf8) }
            }
            idx = s.index(after: idx)
        }
        return nil
    }
}

#if canImport(FoundationModels)
// FoundationModels-gated overloads required by `LanguageModel` on macOS/iOS 26.
@available(macOS 26.0, iOS 26.0, *)
extension MLXLanguageModel {
    public func structuredAsk<Input, Output>(
        system: String, input: Input, output: Output.Type
    ) async throws -> Output where Input: Encodable, Output: JSONSchemaConvertible, Output: Generable {
        let inputStr: String
        if let s = input as? String { inputStr = s }
        else { inputStr = try String(data: JSONEncoder().encode(input), encoding: .utf8) ?? "" }
        let data = try await generateStructured(system: system, input: inputStr, schema: Output.jsonSchema)
        return try Output.swiftLMDecode(data)
    }

    public func structuredAsk<Output>(
        system: String, input: String, output: Output.Type
    ) async throws -> Output where Output: JSONSchemaConvertible, Output: Generable {
        let data = try await generateStructured(system: system, input: input, schema: Output.jsonSchema)
        return try Output.swiftLMDecode(data)
    }
}
#endif
