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
    let stringContentLastField: Set<Int> // String content for last field (excludes trailing comma tokens)
    let quoteOnly: Set<Int>            // Just the quote token
    let colonOnly: Set<Int>            // Just the colon token
    let commaOnly: Set<Int>            // Just the comma token
    let openBraceOnly: Set<Int>        // Just the { token
    let closeBraceOnly: Set<Int>       // Just the } token
    let openBracketOnly: Set<Int>      // Just the [ token
    let closeBracketOnly: Set<Int>     // Just the ] token
    let commaOrCloseBrace: Set<Int>    // , or }
    let commaOrCloseBracket: Set<Int>  // , or ]

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
            // Tokens ending with quote need careful checking
            else if token.hasSuffix("\"") {
                // Check if there are structural characters before the final quote
                // Tokens like `"},"` or `],"` would corrupt the JSON
                let beforeQuote = String(token.dropLast())
                if beforeQuote.contains(where: { "{}[],".contains($0) }) {
                    isSafeForString = false
                } else if beforeQuote.contains("\"") {
                    // Exclude tokens with quote before the final quote (like `""` or `word""`)
                    // These would result in an extra quote in the output
                    isSafeForString = false
                } else {
                    // Safe - this is how strings end naturally (like `word"`)
                    isSafeForString = true
                }
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

            // Exclude tokens containing structural JSON characters
            // These would corrupt the JSON structure if used inside a string
            if token.contains(",") || token.contains("{") || token.contains("}") ||
               token.contains("[") || token.contains("]") || token.contains(":") {
                isSafeForString = false
            }

            if isSafeForString {
                stringContent.insert(id)
            }
        }

        // Pre-compute string content for last field (excludes trailing comma tokens)
        var stringContentLastField = stringContent
        // Remove tokens that would end the string and add a trailing comma
        for (token, id) in tokenizer.tokensToIds {
            if token.contains("\"") && (
                token == "\",\"" ||
                token == "\"," ||
                token.hasSuffix("\",") ||
                token.hasSuffix("\",\"")
            ) {
                stringContentLastField.remove(id)
            }
        }

        self.pureDigits = pureDigits
        self.digitsWithComma = digitsWithComma
        self.stringContent = stringContent
        self.stringContentLastField = stringContentLastField
        self.quoteOnly = Set([self.quote])
        self.colonOnly = Set([self.colon])
        self.commaOnly = Set([self.comma])
        self.openBraceOnly = Set([self.openBrace])
        self.closeBraceOnly = Set([self.closeBrace])
        self.openBracketOnly = Set([self.openBracket])
        self.closeBracketOnly = Set([self.closeBracket])
        self.commaOrCloseBrace = Set([self.comma, self.closeBrace])
        self.commaOrCloseBracket = Set([self.comma, self.closeBracket])
    }
}

// MARK: - JSON Generation State Machine
private enum JSONGenState: @unchecked Sendable {
    case expectingObjectStart                    // Want {
    case expectingKeyStart                       // Want " to start a key
    case inKey(parts: [String], partIndex: Int)  // Generating key tokens
    case expectingKeyEnd                         // Want " to end key
    case expectingColon                          // Want :
    case expectingValueStart(valueType: any JSONSchemaConvertible.Type)    // Want value opening based on type
    case inStringValue(charCount: Int, maxLength: Int)  // Generating string content with max length
    case inEnumValue(validValues: [String], currentValue: String)  // Generating enum value (constrained string)
    case inIntegerValue(hasDigit: Bool, digitCount: Int)          // Generating integer with digit tracking
    case inNumberValue(hasDigit: Bool, hasDecimal: Bool, digitCount: Int) // Generating number with digit tracking
    case inBooleanValue(partial: String)         // Generating boolean
    case expectingCommaOrEnd(hasMoreFields: Bool) // Want , or }
    case expectingCloseBrace                     // Want }
    // Array states - now include count tracking for min/max constraints
    case expectingArrayStart                     // Want [
    case expectingArrayValueOrEnd(elementType: any JSONSchemaConvertible.Type, currentCount: Int, minCount: Int, maxCount: Int)  // Want value or ] (if count >= min)
    case expectingArrayValue(elementType: any JSONSchemaConvertible.Type, currentCount: Int, minCount: Int, maxCount: Int)       // Want value only (after comma)
    case inArrayValue(elementType: any JSONSchemaConvertible.Type, currentCount: Int, minCount: Int, maxCount: Int)              // Processing array element
    case expectingArrayCommaOrEnd(elementType: any JSONSchemaConvertible.Type, currentCount: Int, minCount: Int, maxCount: Int)  // Want , or ] (if count >= min and count <= max)
    case complete                                // Done
}

// MARK: - Object Context for Nested Structures
private struct ObjectContext: @unchecked Sendable {
    var fields: [SchemaProperty]
    var currentFieldIndex: Int
    var isArray: Bool
    var arrayElementType: (any JSONSchemaConvertible.Type)?
    // Array count tracking
    var arrayCurrentCount: Int = 0
    var arrayMinCount: Int = 0
    var arrayMaxCount: Int = Int.max
}

// MARK: - JSON Schema State Tracker (Redesigned)
struct JSONSchemaStateTracker: Sendable {
    private let schema: JSONSchemaConvertible.Type
    private let tokenizer: Tokenizer
    private let categories: TokenCategories
    private let vocabSize: Int
    private var state: JSONGenState = .expectingObjectStart
    private var fields: [SchemaProperty] = []
    private var currentFieldIndex: Int = 0

    // Stack for nested object/array contexts
    private var contextStack: [ObjectContext] = []

    // Array count tracking (for when inside string/number value in array)
    private var currentArrayCount: Int = 0
    private var currentArrayMinCount: Int = 0
    private var currentArrayMaxCount: Int = Int.max
    private var currentArrayElementType: (any JSONSchemaConvertible.Type)?

    // Configuration - defaultMaxStringLength is a safety limit per field
    // Prevents runaway string generation when model doesn't end strings naturally
    // 200 chars is reasonable for most fields (UUIDs ~36, names ~50, descriptions ~200)
    // This can be overridden per-field using @SchemaGuide(.maxLength(n))
    private let defaultMaxStringLength: Int = 200

    // Max number of digits for numeric values (prevents infinite digit loops)
    // 15 digits covers: latitude/longitude (~10 decimals), large integers, reasonable floats
    // For coordinates like 35.7756661, this allows plenty of precision
    private let maxNumberDigits: Int = 15

    /// Get the max string length for the current field, using constraint if available
    private var currentMaxStringLength: Int {
        currentField?.maxLength ?? defaultMaxStringLength
    }

    var isComplete: Bool {
        if case .complete = state { return true }
        return false
    }

    init(schema: JSONSchemaConvertible.Type, tokenizer: Tokenizer) {
        self.schema = schema
        self.tokenizer = tokenizer
        self.categories = TokenCategories(tokenizer: tokenizer)
        self.vocabSize = tokenizer.tokensToIds.count

        // Extract fields from schema using new schemaProperties
        if let properties = schema.schemaProperties {
            self.fields = properties
        }
    }

    // Helper to get current field
    private var currentField: SchemaProperty? {
        guard currentFieldIndex < fields.count else { return nil }
        return fields[currentFieldIndex]
    }

    // Helper to check if we're at the last field considering the stack
    private var isLastFieldInCurrentContext: Bool {
        currentFieldIndex >= fields.count - 1
    }

    // MARK: - Token Filtering

    func applyPenalty(_ tensor: MLTensor) -> MLTensor {
        let validTokens = getValidTokens()

        if Self.debugEnabled {
            let validTokenStrs = validTokens.compactMap { tokenizer.idsToTokens[$0] }.prefix(10)
            debugLog("State: \(state) -> Valid tokens (\(validTokens.count)): \(validTokenStrs)...")
        }

        // Optimization: if only one valid token, create minimal tensor
        if validTokens.count == 1, let validToken = validTokens.first {
            var shaped = MLShapedArray<Float>(repeating: -Float.greatestFiniteMagnitude, shape: [vocabSize])
            shaped[scalarAt: validToken] = Float.greatestFiniteMagnitude
            return MLTensor(shaped)
        }

        // Build list of disallowed indices (all tokens not in validTokens)
        let disallowedIndices: [Int32] = (0..<vocabSize).compactMap { idx in
            validTokens.contains(idx) ? nil : Int32(idx)
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
            return categories.openBraceOnly

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

        case .expectingValueStart(let valueType):
            let typeName = valueType.type
            switch typeName {
            case "string":
                // Both regular strings and enums start with a quote
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
            case "object":
                // Nested object - expect opening brace
                return categories.openBraceOnly
            case "array":
                // Array - expect opening bracket
                return categories.openBracketOnly
            default:
                // Unknown type - allow quote for string fallback
                return categories.quoteOnly
            }

        case .inEnumValue(let validValues, let currentValue):
            // Only allow tokens that can form valid enum values
            var valid = Set<Int>()

            // Find which enum values are still possible given current prefix
            let possibleValues = validValues.filter { $0.hasPrefix(currentValue) }

            if possibleValues.isEmpty {
                // No valid options - shouldn't happen, but allow quote to end
                return categories.quoteOnly
            }

            // Check if current value exactly matches any enum value
            let exactMatch = possibleValues.contains(currentValue)
            if exactMatch {
                // Allow quote to complete the enum value
                valid.insert(categories.quote)
            }

            // Find all valid next characters/tokens
            for enumValue in possibleValues {
                if enumValue.count > currentValue.count {
                    let remaining = String(enumValue.dropFirst(currentValue.count))
                    // Find tokens that match the start of remaining
                    for (token, id) in tokenizer.tokensToIds {
                        // Skip structural tokens
                        if token == "\"" || token == "," || token == "}" || token == "]" || token == "{" || token == "[" || token == ":" {
                            continue
                        }
                        // Check if this token is a valid continuation
                        if remaining.hasPrefix(token) || token.hasPrefix(remaining) {
                            // Token matches - but verify it doesn't overshoot
                            let newValue = currentValue + token
                            // Check if newValue is a prefix of any valid enum value
                            if validValues.contains(where: { $0.hasPrefix(newValue) || newValue == $0 }) {
                                valid.insert(id)
                            }
                        }
                    }
                }
            }

            return valid.isEmpty ? categories.quoteOnly : valid

        case .inStringValue(let charCount, let maxLength):
            // Force termination if string is too long
            if charCount >= maxLength {
                return categories.quoteOnly
            }

            // Calculate remaining characters allowed
            let remaining = maxLength - charCount

            // If we're close to the limit, filter tokens by length
            // to prevent overshooting the limit
            if remaining <= 50 {
                // Filter tokens to only allow those that won't exceed limit
                var valid = Set<Int>()
                valid.insert(categories.quote) // Always allow quote to end string

                let baseSet = isLastFieldInCurrentContext ? categories.stringContentLastField : categories.stringContent
                for tokenId in baseSet {
                    if let tokenStr = tokenizer.idsToTokens[tokenId] {
                        // For tokens ending with quote, the quote doesn't count toward content
                        let contentLength = tokenStr.hasSuffix("\"") ? tokenStr.count - 1 : tokenStr.count
                        if contentLength <= remaining {
                            valid.insert(tokenId)
                        }
                    }
                }
                return valid
            }

            // Use pre-computed set based on whether this is the last field
            // (avoids re-filtering tokenizer on every call)
            if isLastFieldInCurrentContext {
                var valid = categories.stringContentLastField
                valid.insert(categories.quote)
                return valid
            } else {
                var valid = categories.stringContent
                valid.insert(categories.quote)
                return valid
            }

        case .inIntegerValue(_, let digitCount):
            // If we've reached max digits, only allow terminators (comma, close brace)
            if digitCount >= maxNumberDigits {
                var valid = categories.commaOnly
                if isLastFieldInCurrentContext {
                    valid.insert(categories.closeBrace)
                }
                return valid
            }
            // Allow digits, comma to continue, and close brace only if last field
            var valid = categories.digitsWithComma
            valid.insert(categories.comma)
            // Only allow close brace if this is the last field
            if isLastFieldInCurrentContext {
                valid.insert(categories.closeBrace)
            }
            return valid

        case .inNumberValue(_, let hasDecimal, let digitCount):
            // If we've reached max digits, only allow terminators (comma, close brace)
            if digitCount >= maxNumberDigits {
                var valid = categories.commaOnly
                if isLastFieldInCurrentContext {
                    valid.insert(categories.closeBrace)
                }
                return valid
            }
            var valid = categories.digitsWithComma
            valid.insert(categories.comma)
            // Only allow close brace if this is the last field
            if isLastFieldInCurrentContext {
                valid.insert(categories.closeBrace)
            }
            // Allow decimal point if not already used
            if !hasDecimal, let dotId = tokenizer.tokensToIds["."] {
                valid.insert(dotId)
            }
            return valid

        case .inBooleanValue(let partial):
            // Continue generating the boolean value
            var valid = Set<Int>()
            if partial.hasPrefix("t") {
                // Generating "true"
                let remaining = String("true".dropFirst(partial.count))
                if !remaining.isEmpty {
                    if let tokenId = tokenizer.tokensToIds[remaining] {
                        valid.insert(tokenId)
                    }
                    // Allow partial completions
                    for i in 1...remaining.count {
                        let prefix = String(remaining.prefix(i))
                        if let tokenId = tokenizer.tokensToIds[prefix] {
                            valid.insert(tokenId)
                        }
                    }
                }
            } else if partial.hasPrefix("f") {
                // Generating "false"
                let remaining = String("false".dropFirst(partial.count))
                if !remaining.isEmpty {
                    if let tokenId = tokenizer.tokensToIds[remaining] {
                        valid.insert(tokenId)
                    }
                    for i in 1...remaining.count {
                        let prefix = String(remaining.prefix(i))
                        if let tokenId = tokenizer.tokensToIds[prefix] {
                            valid.insert(tokenId)
                        }
                    }
                }
            }
            // Also allow comma/close brace if boolean is complete
            if partial == "true" || partial == "false" {
                // Only allow close brace if this is the last field
                if isLastFieldInCurrentContext {
                    valid = categories.commaOrCloseBrace
                } else {
                    valid = categories.commaOnly
                }
            }
            return valid

        case .expectingCommaOrEnd(let hasMoreFields):
            if hasMoreFields {
                return categories.commaOnly
            } else {
                // Last field - only allow close brace, not comma
                return categories.closeBraceOnly
            }

        case .expectingCloseBrace:
            return categories.closeBraceOnly

        // Array states
        case .expectingArrayStart:
            return categories.openBracketOnly

        case .expectingArrayValueOrEnd(let elementType, let currentCount, let minCount, let maxCount):
            // Only allow ] if we've met the minimum count
            var valid = Set<Int>()
            if currentCount >= minCount {
                valid.formUnion(categories.closeBracketOnly)
            }
            // Only allow adding more elements if under max count
            if currentCount < maxCount {
                let typeName = elementType.type
                switch typeName {
                case "string":
                    valid.insert(categories.quote)
                case "integer", "number":
                    valid.formUnion(categories.pureDigits)
                    if let minusId = tokenizer.tokensToIds["-"] {
                        valid.insert(minusId)
                    }
                case "boolean":
                    if let t = tokenizer.tokensToIds["true"] { valid.insert(t) }
                    if let f = tokenizer.tokensToIds["false"] { valid.insert(f) }
                    if let tr = tokenizer.tokensToIds["tr"] { valid.insert(tr) }
                    if let fa = tokenizer.tokensToIds["fa"] { valid.insert(fa) }
                case "object":
                    valid.insert(categories.openBrace)
                default:
                    valid.insert(categories.quote)
                }
            }
            return valid

        case .expectingArrayValue(let elementType, _, _, let maxCount):
            // Must have a value - NO ] allowed (prevents trailing commas after comma)
            var valid = Set<Int>()
            let typeName = elementType.type
            switch typeName {
            case "string":
                valid.insert(categories.quote)
            case "integer", "number":
                valid.formUnion(categories.pureDigits)
                if let minusId = tokenizer.tokensToIds["-"] {
                    valid.insert(minusId)
                }
            case "boolean":
                if let t = tokenizer.tokensToIds["true"] { valid.insert(t) }
                if let f = tokenizer.tokensToIds["false"] { valid.insert(f) }
                if let tr = tokenizer.tokensToIds["tr"] { valid.insert(tr) }
                if let fa = tokenizer.tokensToIds["fa"] { valid.insert(fa) }
            case "object":
                valid.insert(categories.openBrace)
            default:
                valid.insert(categories.quote)
            }
            return valid

        case .inArrayValue:
            // Handled by the element type processing - delegate to value states
            return Set()

        case .expectingArrayCommaOrEnd(_, let currentCount, let minCount, let maxCount):
            // After an array element, expect comma (if under max) or ] (if >= min)
            var valid = Set<Int>()
            if currentCount >= minCount {
                valid.insert(categories.closeBracket)
            }
            if currentCount < maxCount {
                valid.insert(categories.comma)
            }
            return valid

        case .complete:
            if let eosId = tokenizer.eosTokenId {
                return Set([eosId])
            }
            return Set()
        }
    }

    // MARK: - Debug

    /// Enable to print state transitions (for debugging grammar issues)
    /// Set JSONSchemaStateTracker.debugEnabled = true before inference
    nonisolated(unsafe) static var debugEnabled = false

    private func debugLog(_ message: @autoclosure () -> String) {
        if Self.debugEnabled {
            print("[Grammar] \(message())")
        }
    }

    // MARK: - State Updates

    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        guard let tokenStr = tokenizer.idsToTokens[token] else { return }

        let oldState = "\(state)"
        defer {
            if Self.debugEnabled && "\(state)" != oldState {
                debugLog("Token '\(tokenStr)' (\(token)): \(oldState) -> \(state)")
            }
        }

        switch state {
        case .expectingObjectStart:
            if tokenStr == "{" {
                if currentFieldIndex < fields.count {
                    state = .expectingKeyStart
                } else {
                    state = .expectingCloseBrace
                }
            }

        case .expectingKeyStart:
            if tokenStr == "\"" {
                let key = fields[currentFieldIndex].name
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
                let valueType = fields[currentFieldIndex].valueType
                state = .expectingValueStart(valueType: valueType)
            }

        case .expectingColon:
            if tokenStr == ":" || tokenStr.hasPrefix(":") {
                let valueType = fields[currentFieldIndex].valueType
                state = .expectingValueStart(valueType: valueType)
            }

        case .expectingValueStart(let valueType):
            let typeName = valueType.type
            switch typeName {
            case "string":
                if tokenStr == "\"" {
                    // Check if this is an enum type with constrained values
                    if let enumValues = valueType.jsonSchema["enum"] as? [Any], !enumValues.isEmpty {
                        let stringValues = enumValues.compactMap { $0 as? String }
                        if !stringValues.isEmpty {
                            state = .inEnumValue(validValues: stringValues, currentValue: "")
                        } else {
                            state = .inStringValue(charCount: 0, maxLength: currentMaxStringLength)
                        }
                    } else {
                        state = .inStringValue(charCount: 0, maxLength: currentMaxStringLength)
                    }
                }
            case "integer":
                let digitCount = tokenStr.filter { $0.isNumber }.count
                state = .inIntegerValue(hasDigit: digitCount > 0, digitCount: digitCount)
                // Check if this token ends the value
                if tokenStr.hasSuffix(",") || tokenStr == "," {
                    advanceToNextField(consumedComma: true)
                } else if tokenStr == "}" {
                    finishCurrentObject()
                }
            case "number":
                let hasDecimal = tokenStr.contains(".")
                let digitCount = tokenStr.filter { $0.isNumber }.count
                state = .inNumberValue(hasDigit: digitCount > 0, hasDecimal: hasDecimal, digitCount: digitCount)
                if tokenStr.hasSuffix(",") || tokenStr == "," {
                    advanceToNextField(consumedComma: true)
                } else if tokenStr == "}" {
                    finishCurrentObject()
                }
            case "boolean":
                // Start tracking the boolean value
                state = .inBooleanValue(partial: tokenStr)
                // Check if complete
                if tokenStr == "true" || tokenStr == "false" {
                    advanceToNextField(consumedComma: false)
                }
            case "object":
                if tokenStr == "{" {
                    // Push current context onto stack and start nested object
                    pushContext()
                    // Load nested object's fields using the metatype's schemaProperties
                    if let nestedProps = valueType.schemaProperties {
                        fields = nestedProps
                    } else {
                        fields = []
                    }
                    currentFieldIndex = 0
                    if fields.isEmpty {
                        state = .expectingCloseBrace
                    } else {
                        state = .expectingKeyStart
                    }
                }
            case "array":
                if tokenStr == "[" {
                    // Get count constraints from the current field
                    let minCount = currentField?.countRange?.lowerBound ?? 0
                    let maxCount = currentField?.countRange?.upperBound ?? Int.max
                    // Start array processing using the metatype's elementType
                    if let elemType = valueType.elementType {
                        state = .expectingArrayValueOrEnd(elementType: elemType, currentCount: 0, minCount: minCount, maxCount: maxCount)
                    } else {
                        // Fallback to string if no element type
                        state = .expectingArrayValueOrEnd(elementType: String.self, currentCount: 0, minCount: minCount, maxCount: maxCount)
                    }
                }
            default:
                // Treat unknown as string
                if tokenStr == "\"" {
                    state = .inStringValue(charCount: 0, maxLength: currentMaxStringLength)
                }
            }

        case .inStringValue(let charCount, let maxLength):
            // Check if we're inside an array (tracked separately)
            if let arrayElemType = currentArrayElementType {
                // Inside array - handle string element completion
                if tokenStr == "\"" || tokenStr.hasSuffix("\"") {
                    // String element complete - go to array comma or end
                    state = .expectingArrayCommaOrEnd(elementType: arrayElemType, currentCount: currentArrayCount, minCount: currentArrayMinCount, maxCount: currentArrayMaxCount)
                    currentArrayElementType = nil
                } else if tokenStr == "\"," {
                    // String ends AND comma - expect next array element
                    state = .expectingArrayValue(elementType: arrayElemType, currentCount: currentArrayCount, minCount: currentArrayMinCount, maxCount: currentArrayMaxCount)
                    currentArrayElementType = nil
                } else {
                    // Continue string
                    let newCount = charCount + tokenStr.count
                    state = .inStringValue(charCount: newCount, maxLength: maxLength)
                }
            } else {
                // Inside object - normal string field handling
                if tokenStr == "\"" {
                    // String ended with single quote
                    advanceToNextField(consumedComma: false)
                } else if tokenStr == "\",\"" {
                    // Token is `","` - ends string, has comma, AND starts next key quote
                    // This is a special compound token that spans two fields
                    advanceToNextField(consumedComma: true)
                    // The token also provided the opening quote for the next key
                    if currentFieldIndex < fields.count {
                        let key = fields[currentFieldIndex].name
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
                    state = .inStringValue(charCount: newCount, maxLength: maxLength)
                }
            }

        case .inEnumValue(let validValues, let currentValue):
            if tokenStr == "\"" {
                // Closing quote - enum value complete
                advanceToNextField(consumedComma: false)
            } else if tokenStr == "\"," {
                // Token is `",` - ends enum value AND includes comma
                advanceToNextField(consumedComma: true)
            } else if tokenStr.hasSuffix("\"") {
                // Token ends with quote (like `value"`) - ends enum value
                advanceToNextField(consumedComma: false)
            } else {
                // Continue building enum value
                let newValue = currentValue + tokenStr
                state = .inEnumValue(validValues: validValues, currentValue: newValue)
            }

        case .inIntegerValue(_, let digitCount):
            if tokenStr == "," {
                advanceToNextField(consumedComma: true)
            } else if tokenStr == "}" {
                finishCurrentObject()
            } else if tokenStr.hasSuffix(",") {
                advanceToNextField(consumedComma: true)
            } else {
                // Update digit count
                let newDigits = tokenStr.filter { $0.isNumber }.count
                state = .inIntegerValue(hasDigit: true, digitCount: digitCount + newDigits)
            }

        case .inNumberValue(_, let hasDecimal, let digitCount):
            if tokenStr == "," {
                advanceToNextField(consumedComma: true)
            } else if tokenStr == "}" {
                finishCurrentObject()
            } else if tokenStr.hasSuffix(",") {
                advanceToNextField(consumedComma: true)
            } else {
                // Update decimal and digit tracking
                let newHasDecimal = hasDecimal || tokenStr.contains(".")
                let newDigits = tokenStr.filter { $0.isNumber }.count
                state = .inNumberValue(hasDigit: true, hasDecimal: newHasDecimal, digitCount: digitCount + newDigits)
            }

        case .inBooleanValue(let partial):
            let newPartial = partial + tokenStr
            if newPartial == "true" || newPartial == "false" {
                advanceToNextField(consumedComma: false)
            } else if tokenStr == "," {
                advanceToNextField(consumedComma: true)
            } else if tokenStr == "}" {
                finishCurrentObject()
            } else {
                state = .inBooleanValue(partial: newPartial)
            }

        case .expectingCommaOrEnd:
            if tokenStr == "," {
                // Don't call advanceToNextField - the index was already incremented
                // when we entered this state. Just move to expecting the next key.
                state = .expectingKeyStart
            } else if tokenStr == "}" {
                finishCurrentObject()
            }

        case .expectingCloseBrace:
            if tokenStr == "}" {
                finishCurrentObject()
            }

        // Array states
        case .expectingArrayStart:
            if tokenStr == "[" {
                // Already handled in expectingValueStart
                break
            }

        case .expectingArrayValueOrEnd(let elementType, let currentCount, let minCount, let maxCount):
            let typeName = elementType.type
            if tokenStr == "]" {
                // End array (only allowed if currentCount >= minCount, enforced in getValidTokens)
                advanceToNextField(consumedComma: false)
            } else if tokenStr == "{" && typeName == "object" {
                // Start nested object in array - increment count
                pushArrayContext(elementType: elementType, currentCount: currentCount + 1, minCount: minCount, maxCount: maxCount)
                // Load nested object's fields using metatype
                if let nestedProps = elementType.schemaProperties {
                    fields = nestedProps
                } else {
                    fields = []
                }
                currentFieldIndex = 0
                if fields.isEmpty {
                    state = .expectingCloseBrace
                } else {
                    state = .expectingKeyStart
                }
            } else if tokenStr == "\"" && typeName == "string" {
                // Start string value in array - track count for when string ends
                currentArrayCount = currentCount + 1
                currentArrayMinCount = minCount
                currentArrayMaxCount = maxCount
                currentArrayElementType = elementType
                state = .inStringValue(charCount: 0, maxLength: defaultMaxStringLength)
            } else if typeName == "integer" || typeName == "number" {
                // Numeric value in array - track count
                currentArrayCount = currentCount + 1
                currentArrayMinCount = minCount
                currentArrayMaxCount = maxCount
                currentArrayElementType = elementType
                let digitCount = tokenStr.filter { $0.isNumber }.count
                if typeName == "integer" {
                    state = .inIntegerValue(hasDigit: digitCount > 0, digitCount: digitCount)
                } else {
                    state = .inNumberValue(hasDigit: digitCount > 0, hasDecimal: tokenStr.contains("."), digitCount: digitCount)
                }
            }

        case .expectingArrayValue(let elementType, let currentCount, let minCount, let maxCount):
            // Same as expectingArrayValueOrEnd but NO ] handling (prevents trailing commas)
            let typeName = elementType.type
            if tokenStr == "{" && typeName == "object" {
                // Start nested object in array - increment count
                pushArrayContext(elementType: elementType, currentCount: currentCount + 1, minCount: minCount, maxCount: maxCount)
                if let nestedProps = elementType.schemaProperties {
                    fields = nestedProps
                } else {
                    fields = []
                }
                currentFieldIndex = 0
                if fields.isEmpty {
                    state = .expectingCloseBrace
                } else {
                    state = .expectingKeyStart
                }
            } else if tokenStr == "\"" && typeName == "string" {
                currentArrayCount = currentCount + 1
                currentArrayMinCount = minCount
                currentArrayMaxCount = maxCount
                currentArrayElementType = elementType
                state = .inStringValue(charCount: 0, maxLength: defaultMaxStringLength)
            } else if typeName == "integer" || typeName == "number" {
                currentArrayCount = currentCount + 1
                currentArrayMinCount = minCount
                currentArrayMaxCount = maxCount
                currentArrayElementType = elementType
                let digitCount = tokenStr.filter { $0.isNumber }.count
                if typeName == "integer" {
                    state = .inIntegerValue(hasDigit: digitCount > 0, digitCount: digitCount)
                } else {
                    state = .inNumberValue(hasDigit: digitCount > 0, hasDecimal: tokenStr.contains("."), digitCount: digitCount)
                }
            }

        case .inArrayValue:
            // This state is mostly a placeholder - actual value handling goes to specific states
            break

        case .expectingArrayCommaOrEnd(let elementType, let currentCount, let minCount, let maxCount):
            if tokenStr == "]" {
                // Array ended (only allowed if currentCount >= minCount, enforced in getValidTokens)
                advanceToNextField(consumedComma: false)
            } else if tokenStr == "," {
                // More elements - use expectingArrayValue to prevent trailing commas
                state = .expectingArrayValue(elementType: elementType, currentCount: currentCount, minCount: minCount, maxCount: maxCount)
            }

        case .complete:
            break
        }
    }

    // Push current context onto stack (for entering nested objects)
    private mutating func pushContext() {
        let context = ObjectContext(
            fields: fields,
            currentFieldIndex: currentFieldIndex,
            isArray: false,
            arrayElementType: nil
        )
        contextStack.append(context)
    }

    // Push current context with array info
    private mutating func pushArrayContext(elementType: any JSONSchemaConvertible.Type, currentCount: Int = 0, minCount: Int = 0, maxCount: Int = Int.max) {
        let context = ObjectContext(
            fields: fields,
            currentFieldIndex: currentFieldIndex,
            isArray: true,
            arrayElementType: elementType,
            arrayCurrentCount: currentCount,
            arrayMinCount: minCount,
            arrayMaxCount: maxCount
        )
        contextStack.append(context)
    }

    // Pop context and restore state (for exiting nested objects)
    private mutating func popContext() {
        guard let context = contextStack.popLast() else {
            state = .complete
            return
        }

        fields = context.fields
        currentFieldIndex = context.currentFieldIndex

        if context.isArray {
            // We were in an array - expect comma or end of array (using saved count info from context)
            if let elemType = context.arrayElementType {
                state = .expectingArrayCommaOrEnd(
                    elementType: elemType,
                    currentCount: context.arrayCurrentCount,
                    minCount: context.arrayMinCount,
                    maxCount: context.arrayMaxCount
                )
            } else {
                advanceToNextField(consumedComma: false)
            }
        } else {
            // We were in a nested object - advance to next field
            advanceToNextField(consumedComma: false)
        }
    }

    // Called when we see a closing brace
    private mutating func finishCurrentObject() {
        if contextStack.isEmpty {
            state = .complete
        } else {
            popContext()
        }
    }

    private mutating func advanceToNextField(consumedComma: Bool) {
        currentFieldIndex += 1

        if currentFieldIndex >= fields.count {
            // No more fields in current object
            if contextStack.isEmpty {
                // At root level
                if consumedComma {
                    state = .expectingCloseBrace
                } else {
                    state = .expectingCommaOrEnd(hasMoreFields: false)
                }
            } else {
                // In nested context - need to close this object
                if consumedComma {
                    state = .expectingCloseBrace
                } else {
                    state = .expectingCommaOrEnd(hasMoreFields: false)
                }
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
