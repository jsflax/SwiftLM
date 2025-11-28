import Foundation
import JSONSchema
@preconcurrency import Tokenizers
import CoreML

public enum GrammarError: Error, LocalizedError {
    case missingSchemaProperties
    case unsupportedPropertyType(String)
    case invalidParserState(String)

    public var errorDescription: String? {
        switch self {
        case .missingSchemaProperties:
            return "JSON schema is missing required properties"
        case .unsupportedPropertyType(let type):
            return "Unsupported property type in schema: \(type)"
        case .invalidParserState(let state):
            return "Invalid parser state: \(state)"
        }
    }
}

// MARK: - Token Categories
/// Pre-computed token categories for efficient filtering
private struct TokenCategories: Sendable {
    // Single character structural tokens
    let quote: Int              // "
    let colon: Int              // :
    let comma: Int              // ,
    let openBrace: Int          // {
    let closeBrace: Int         // }
    let openBracket: Int        // [
    let closeBracket: Int       // ]

    // Token sets by category
    let pureDigits: Set<Int>           // Tokens that are only digits (0-9, 123, etc.)
    let digitsWithComma: Set<Int>      // Pure digits OR end with comma (for number termination)
    let stringContent: Set<Int>        // Safe inside strings (no unescaped quotes or structural chars that break parsing)
    let quoteOnly: Set<Int>            // Just the quote token
    let colonOnly: Set<Int>            // Just the colon token
    let commaOnly: Set<Int>            // Just the comma token
    let closeBraceOnly: Set<Int>       // Just the } token
    let commaOrCloseBrace: Set<Int>    // , or }

    init(tokenizer: Tokenizer) {
        // Get single-char structural tokens
        self.quote = tokenizer.tokensToIds["\""]!
        self.colon = tokenizer.tokensToIds[":"]!
        self.comma = tokenizer.tokensToIds[","]!
        self.openBrace = tokenizer.tokensToIds["{"]!
        self.closeBrace = tokenizer.tokensToIds["}"]!
        self.openBracket = tokenizer.tokensToIds["["] ?? -1
        self.closeBracket = tokenizer.tokensToIds["]"] ?? -1

        // Build category sets
        var pureDigits = Set<Int>()
        var digitsWithComma = Set<Int>()
        var stringContent = Set<Int>()

        for (token, id) in tokenizer.tokensToIds {
            // Pure digits: only contains 0-9
            if token.allSatisfy({ $0.isNumber }) && !token.isEmpty {
                pureDigits.insert(id)
                digitsWithComma.insert(id)
            }

            // Digits ending with comma (like "123,")
            if token.dropLast().allSatisfy({ $0.isNumber }) && token.last == "," && token.count > 1 {
                digitsWithComma.insert(id)
            }

            // String content: tokens valid inside a JSON string value
            // We need to be careful about quotes:
            // - Single quote `"` is excluded (we don't want model to choose it freely)
            // - Tokens ENDING with quote (like `word"`) are ALLOWED - they naturally end strings
            // - Tokens with quote elsewhere are excluded (would break parsing)
            var isSafeForString = true

            // Exclude the single quote token
            if token == "\"" {
                isSafeForString = false
            }
            // Tokens ending with quote are OK (they terminate the string naturally)
            else if token.hasSuffix("\"") {
                // Allow it - this is how strings end
                isSafeForString = true
            }
            // Tokens with quote NOT at the end are problematic
            else if token.contains("\"") {
                isSafeForString = false
            }

            // Don't allow tokens with newlines (breaks JSON)
            if token.contains("\n") || token.contains("\r") {
                isSafeForString = false
            }

            // Don't allow backslash alone (escape handling is complex)
            if token == "\\" {
                isSafeForString = false
            }

            if isSafeForString {
                stringContent.insert(id)
            }
        }

        self.pureDigits = pureDigits
        self.digitsWithComma = digitsWithComma
        self.stringContent = stringContent
        self.quoteOnly = Set([self.quote])
        self.colonOnly = Set([self.colon])
        self.commaOnly = Set([self.comma])
        self.closeBraceOnly = Set([self.closeBrace])
        self.commaOrCloseBrace = Set([self.comma, self.closeBrace])
    }
}

// MARK: - JSON Generation State Machine
private enum JSONGenState: Sendable {
    case expectingObjectStart                    // Want {
    case expectingKeyStart                       // Want " to start a key
    case inKey(parts: [String], partIndex: Int)  // Generating key tokens
    case expectingKeyEnd                         // Want " to end key
    case expectingColon                          // Want :
    case expectingValueStart(type: String)       // Want value opening based on type
    case inStringValue(charCount: Int)           // Generating string content
    case inIntegerValue(hasDigit: Bool)          // Generating integer
    case inNumberValue(hasDigit: Bool, hasDecimal: Bool) // Generating number
    case expectingCommaOrEnd(hasMoreFields: Bool) // Want , or }
    case expectingCloseBrace                     // Want }
    case complete                                // Done
}

// MARK: - JSON Schema State Tracker (Redesigned)
struct JSONSchemaStateTracker: Sendable {
    private let schema: JSONSchemaConvertible.Type
    private let tokenizer: Tokenizer
    private let categories: TokenCategories
    private var state: JSONGenState = .expectingObjectStart
    private var fields: [(key: String, property: JSONSchema.Property)] = []
    private var currentFieldIndex: Int = 0
    private var currentJSON: String = ""

    // Configuration - maxStringLength is a safety limit, not a target length
    // It's only meant to prevent infinite loops when model refuses to end strings
    private let maxStringLength: Int = 4000

    var isComplete: Bool {
        if case .complete = state { return true }
        return false
    }

    init(schema: JSONSchemaConvertible.Type, tokenizer: Tokenizer) {
        self.schema = schema
        self.tokenizer = tokenizer
        self.categories = TokenCategories(tokenizer: tokenizer)

        // Extract fields from schema
        if let properties = schema.properties {
            self.fields = properties
        }
    }

    // MARK: - Token Filtering

    func applyPenalty(_ tensor: MLTensor) async -> MLTensor {
        let validTokens = getValidTokens()

        // Optimization: if only one valid token, create minimal tensor
        if validTokens.count == 1, let validToken = validTokens.first {
            let vocabSize = await tensor.shapedArray(of: Float.self).scalarCount
            var shaped = MLShapedArray<Float>(repeating: -Float.greatestFiniteMagnitude, shape: [vocabSize])
            shaped[scalarAt: validToken] = Float.greatestFiniteMagnitude
            return MLTensor(shaped)
        }

        // General case: penalize invalid tokens
        let xShaped = await tensor.shapedArray(of: Float.self)
        let disallowedIndices = xShaped.enumerated().compactMap {
            if !validTokens.contains($0.offset) {
                return Int32($0.offset)
            }
            return nil
        }

        return tensor.replacing(
            atIndices: MLTensor(disallowedIndices, scalarType: Int32.self),
            with: -Float.greatestFiniteMagnitude,
            alongAxis: -1
        )
    }

    private func getValidTokens() -> Set<Int> {
        switch state {
        case .expectingObjectStart:
            return Set([categories.openBrace])

        case .expectingKeyStart:
            return categories.quoteOnly

        case .inKey(let parts, let partIndex):
            // Return exact token for current key part
            if partIndex < parts.count {
                if let tokenId = tokenizer.tokensToIds[parts[partIndex]] {
                    return Set([tokenId])
                }
            }
            // Fallback: allow quote to end key
            return categories.quoteOnly

        case .expectingKeyEnd:
            return categories.quoteOnly

        case .expectingColon:
            return categories.colonOnly

        case .expectingValueStart(let type):
            switch type {
            case "string":
                return categories.quoteOnly
            case "integer", "number":
                // Allow digits and optional negative sign
                var valid = categories.pureDigits
                if let minusId = tokenizer.tokensToIds["-"] {
                    valid.insert(minusId)
                }
                return valid
            case "boolean":
                // Allow "true" or "false" starting tokens
                var valid = Set<Int>()
                if let t = tokenizer.tokensToIds["true"] { valid.insert(t) }
                if let f = tokenizer.tokensToIds["false"] { valid.insert(f) }
                // Also allow partial tokens
                if let tr = tokenizer.tokensToIds["tr"] { valid.insert(tr) }
                if let fa = tokenizer.tokensToIds["fa"] { valid.insert(fa) }
                return valid
            default:
                // Unknown type - allow quote for string fallback
                return categories.quoteOnly
            }

        case .inStringValue(let charCount):
            // Force termination if string is too long
            if charCount >= maxStringLength {
                return categories.quoteOnly
            }

            var valid = categories.stringContent
            // Add the single quote token so model can end strings cleanly
            valid.insert(categories.quote)

            // If this is the last field, exclude tokens that end strings with commas
            // to prevent trailing commas in the JSON
            let isLastField = currentFieldIndex >= fields.count - 1
            if isLastField {
                // Remove tokens that would end the string and add a trailing comma
                // Examples: `","`, `",`, `word",`, `word","`
                let tokensToRemove = tokenizer.tokensToIds.filter { (token, _) in
                    // Token ends string AND has comma after the quote
                    token.contains("\"") && (
                        token == "\",\"" ||
                        token == "\"," ||
                        token.hasSuffix("\",") ||
                        token.hasSuffix("\",\"")
                    )
                }.map { $0.value }
                for tokenId in tokensToRemove {
                    valid.remove(tokenId)
                }
            }
            return valid

        case .inIntegerValue:
            // Allow digits or comma to end
            return categories.digitsWithComma.union(categories.commaOrCloseBrace)

        case .inNumberValue(_, let hasDecimal):
            var valid = categories.digitsWithComma.union(categories.commaOrCloseBrace)
            // Allow decimal point if not already used
            if !hasDecimal, let dotId = tokenizer.tokensToIds["."] {
                valid.insert(dotId)
            }
            return valid

        case .expectingCommaOrEnd(let hasMoreFields):
            if hasMoreFields {
                return categories.commaOnly
            } else {
                return categories.commaOrCloseBrace
            }

        case .expectingCloseBrace:
            return categories.closeBraceOnly

        case .complete:
            if let eosId = tokenizer.eosTokenId {
                return Set([eosId])
            }
            return Set()
        }
    }

    // MARK: - State Updates

    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        guard let tokenStr = tokenizer.idsToTokens[token] else { return }
        currentJSON += tokenizer.decode(tokens: [token])

        switch state {
        case .expectingObjectStart:
            if tokenStr == "{" {
                if currentFieldIndex < fields.count {
                    let key = fields[currentFieldIndex].key
                    let parts = tokenizer.tokenize(text: key)
                    state = .expectingKeyStart
                } else {
                    state = .expectingCloseBrace
                }
            }

        case .expectingKeyStart:
            if tokenStr == "\"" {
                let key = fields[currentFieldIndex].key
                let parts = tokenizer.tokenize(text: key)
                if parts.isEmpty {
                    state = .expectingKeyEnd
                } else {
                    state = .inKey(parts: parts, partIndex: 0)
                }
            }

        case .inKey(let parts, let partIndex):
            let nextIndex = partIndex + 1
            if nextIndex >= parts.count {
                state = .expectingKeyEnd
            } else {
                state = .inKey(parts: parts, partIndex: nextIndex)
            }

        case .expectingKeyEnd:
            if tokenStr == "\"" {
                state = .expectingColon
            } else if tokenStr.hasSuffix("\"") {
                // Token ends with quote (e.g., key part + quote)
                state = .expectingColon
            } else if tokenStr == "\":" || tokenStr.hasSuffix("\":") {
                // Token is quote-colon or ends with quote-colon - skip to value
                let valueType = fields[currentFieldIndex].property.type ?? "string"
                state = .expectingValueStart(type: valueType)
            }

        case .expectingColon:
            if tokenStr == ":" {
                let valueType = fields[currentFieldIndex].property.type ?? "string"
                state = .expectingValueStart(type: valueType)
            } else if tokenStr.hasPrefix(":") {
                // Token starts with colon - it's a colon plus something
                let valueType = fields[currentFieldIndex].property.type ?? "string"
                state = .expectingValueStart(type: valueType)
            }

        case .expectingValueStart(let type):
            switch type {
            case "string":
                if tokenStr == "\"" {
                    state = .inStringValue(charCount: 0)
                }
            case "integer":
                state = .inIntegerValue(hasDigit: tokenStr.contains(where: { $0.isNumber }))
                // Check if this token ends the value
                if tokenStr.hasSuffix(",") || tokenStr == "," {
                    advanceToNextField(consumedComma: true)
                } else if tokenStr == "}" {
                    state = .complete
                }
            case "number":
                let hasDecimal = tokenStr.contains(".")
                state = .inNumberValue(hasDigit: tokenStr.contains(where: { $0.isNumber }), hasDecimal: hasDecimal)
                if tokenStr.hasSuffix(",") || tokenStr == "," {
                    advanceToNextField(consumedComma: true)
                } else if tokenStr == "}" {
                    state = .complete
                }
            default:
                // Treat unknown as string
                if tokenStr == "\"" {
                    state = .inStringValue(charCount: 0)
                }
            }

        case .inStringValue(let charCount):
            if tokenStr == "\"" {
                // String ended with single quote
                advanceToNextField(consumedComma: false)
            } else if tokenStr == "\",\"" {
                // Token is `","` - ends string, has comma, AND starts next key quote
                // This is a special compound token that spans two fields
                advanceToNextField(consumedComma: true)
                // The token also provided the opening quote for the next key
                if currentFieldIndex < fields.count {
                    let key = fields[currentFieldIndex].key
                    let parts = tokenizer.tokenize(text: key)
                    if parts.isEmpty {
                        state = .expectingKeyEnd
                    } else {
                        state = .inKey(parts: parts, partIndex: 0)
                    }
                }
            } else if tokenStr == "\"," {
                // Token is `",` - ends string AND includes comma (no next quote)
                advanceToNextField(consumedComma: true)
            } else if tokenStr.hasSuffix("\"") {
                // Token ends with quote (like `word"`) - ends string
                advanceToNextField(consumedComma: false)
            } else {
                // Continue string, track length
                let newCount = charCount + tokenStr.count
                state = .inStringValue(charCount: newCount)
            }

        case .inIntegerValue:
            if tokenStr == "," {
                advanceToNextField(consumedComma: true)
            } else if tokenStr == "}" {
                state = .complete
            } else if tokenStr.hasSuffix(",") {
                advanceToNextField(consumedComma: true)
            }
            // Otherwise stay in integer state

        case .inNumberValue(let hasDigit, let hasDecimal):
            if tokenStr == "," {
                advanceToNextField(consumedComma: true)
            } else if tokenStr == "}" {
                state = .complete
            } else if tokenStr.hasSuffix(",") {
                advanceToNextField(consumedComma: true)
            } else {
                // Update decimal tracking
                let newHasDecimal = hasDecimal || tokenStr.contains(".")
                state = .inNumberValue(hasDigit: true, hasDecimal: newHasDecimal)
            }

        case .expectingCommaOrEnd:
            if tokenStr == "," {
                // Don't call advanceToNextField - the index was already incremented
                // when we entered this state. Just move to expecting the next key.
                state = .expectingKeyStart
            } else if tokenStr == "}" {
                state = .complete
            }

        case .expectingCloseBrace:
            if tokenStr == "}" {
                state = .complete
            }

        case .complete:
            break
        }
    }

    private mutating func advanceToNextField(consumedComma: Bool) {
        currentFieldIndex += 1

        if currentFieldIndex >= fields.count {
            // No more fields
            if consumedComma {
                // We consumed a comma but there are no more fields - need to close
                state = .expectingCloseBrace
            } else {
                state = .expectingCommaOrEnd(hasMoreFields: false)
            }
        } else {
            // More fields to process
            if consumedComma {
                state = .expectingKeyStart
            } else {
                state = .expectingCommaOrEnd(hasMoreFields: true)
            }
        }
    }
}
