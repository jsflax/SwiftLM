import Foundation
import JSONSchema
import Tokenizers
import CoreML
import NaturalLanguage
import Accelerate

private protocol Parser: Sendable {
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int])
    func getValidTokens() -> Set<Int>
    var isComplete: Bool { get }
}

// MARK: Int Parser
private struct IntParser: Parser {
    enum State {
        case start
        case inInt
        case escape
    }
    
    let tokenizer: Tokenizer
    private let validTokens: Set<Int>
    
    init(tokenizer: Tokenizer, cachedDigitKeys: Set<Int>) {
        self.tokenizer = tokenizer
        var validTokens = cachedDigitKeys
//        let digitTokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
//        for token in digitTokens {
//            validTokens.insert(tokenizer.tokensToIds[token]!)
//        }
        validTokens.insert(tokenizer.tokensToIds["-"]!)
        validTokens.insert(tokenizer.tokensToIds[","]!)
        self.validTokens = validTokens
    }
    
    private var state = State.start
    var isComplete: Bool = false
    
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        let token = tokenizer.idsToTokens[token]!
        switch state {
        case .start:
            state = .inInt
        case .inInt:
            if token == "\"" || token == "," {
                isComplete = true
            }
        default: fatalError()
        }
    }
    
    func getValidTokens() -> Set<Int> {
        return self.validTokens
    }
}

private struct NumberParser: Parser {
    enum State {
        case start
        case inInt
        case overflow
    }
    
    let tokenizer: Tokenizer
    private var validTokens: Set<Int>
    private let decimalIndex: Int
    
    init(tokenizer: Tokenizer, cachedDigitKeys: Set<Int>) {
        self.tokenizer = tokenizer
        var validTokens = cachedDigitKeys
//        let digitTokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
//        for token in digitTokens {
//            validTokens.insert(tokenizer.tokensToIds[token]!)
//        }
        validTokens.insert(tokenizer.tokensToIds["-"]!)
        validTokens.insert(tokenizer.tokensToIds["Ġ-"]!)
//        validTokens.insert(tokenizer.tokensToIds["-."]!)
        self.decimalIndex = tokenizer.tokensToIds["."]!
        validTokens.insert(decimalIndex)
        validTokens.insert(tokenizer.tokensToIds[","]!)
        self.validTokens = validTokens
    }
    
    private var state = State.start
    var isComplete: Bool = false
    var inDecimal = false
    
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        if token == decimalIndex {
            // if a decimal is used, it is no longer valid
            validTokens.remove(decimalIndex)
            inDecimal = true
        }

        // remove negative symbols since they are disallowed after the first token
        validTokens.remove(tokenizer.tokensToIds["-"]!)
        validTokens.remove(tokenizer.tokensToIds["Ġ-"]!)
        
        
        let token = tokenizer.idsToTokens[token]!
        switch state {
        case .start:
            state = .inInt
        case .inInt:
            if token == "\"" || token == "," {
                isComplete = true
            }
        default: fatalError()
        }
    }
    
    func getValidTokens() -> Set<Int> {
        return self.validTokens
    }
}

// MARK: StringParser
private struct StringParser: Parser {
    enum State {
        case start
        case inString
        case escape
    }
    
    let tokenizer: Tokenizer
    let disallowedTokens: Set<Int>
    init(tokenizer: Tokenizer, disallowedtokens: Set<Int> = .init()) {
        self.tokenizer = tokenizer
        self.disallowedTokens = disallowedtokens
    }
    private var state = State.start
    var isComplete: Bool = false
    
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        let token = tokenizer.idsToTokens[token]!
        switch state {
        case .start:
            if token != "\"" {
                state = .inString
            }
        case .inString:
            if token == "\"" || token == "\"," || token.hasSuffix("\"") {
                isComplete = true
            }
        default: fatalError()
        }
    }
    
    func getValidTokens() -> Set<Int> {
        var validTokens = Set<Int>()
        validTokens.formUnion(tokenizer.tokensToIds.values)
        // Exclude unescaped quotes and backslashes
        if let quoteID = tokenizer.tokensToIds["\""], let backslashID = tokenizer.tokensToIds["\\"] {
            validTokens.remove(quoteID)
            validTokens.remove(backslashID)
            // To allow escaped quotes and backslashes, handle separately
            // For simplicity, we can allow them and manage escaping in the state
        }
        validTokens.formSymmetricDifference(disallowedTokens)
        return validTokens
    }
}


private extension String {
    func splitCamelOrSnakeCase() -> [String] {
        // Split snake_case
        let snakeParts = self.split(separator: "_").map { String($0) }

        // Handle camelCase
        let camelParts = self.unicodeScalars.reduce(into: [""]) { result, scalar in
            let character = Character(scalar)
            if CharacterSet.uppercaseLetters.contains(scalar), !result.last!.isEmpty {
                result.append(String(character))
            } else {
                result[result.count - 1].append(character)
            }
        }

        // Determine which split is relevant (snake case has "_", camel case does not)
        return self.contains("_") ? snakeParts : camelParts
    }
}

// MARK: KeyParser
private struct KeyParser: Parser {
    enum State {
        case start
        case inString
        case escape
    }
    var nextKeyPart: String?
    let tokenizer: Tokenizer
    var keyParts: [String]

    init(key: String, tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
        self.keyParts = tokenizer.tokenize(text: key)
        self.nextKeyPart = self.keyParts.removeFirst()
    }
    
    private var state = State.start
    var isComplete: Bool = false
    
    func getValidTokens() -> Set<Int> {
        switch state {
        case .start:
            Set([tokenizer.tokensToIds[self.nextKeyPart!]!])
        case .inString:
            if nextKeyPart != nil {
                Set([tokenizer.tokensToIds[self.nextKeyPart!]!])
            } else {
                Set([tokenizer.tokensToIds["\""]!])
            }
        default: fatalError()
        }
    }
    
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        let token = tokenizer.idsToTokens[token]!
        switch state {
        case .start:
            if token != "\"" {
                state = .inString
            }
            if !keyParts.isEmpty {
                nextKeyPart = keyParts.removeFirst()
            } else {
                nextKeyPart = nil
            }
        case .inString:
            if token == "\""  || token == "\"," {
                isComplete = true
            } else {
                if keyParts.isEmpty {
                    nextKeyPart = nil
                } else {
                    nextKeyPart = keyParts.removeFirst()
                }
            }
        default: fatalError()
        }
    }
}
private struct ObjectParser: Parser {
    enum State {
        case start, key, value, colon, comma, end
    }
    private var state = State.start
    private var parser: Parser?
    var keyValuePairs: [(String, JSONSchema.Property)]
    var nextKey: String?
    var nextValue: JSONSchema.Property?
    let tokenizer: Tokenizer
    var isComplete = false
    private let cachedDigitKeys: Set<Int>
    
    init(tokenizer: Tokenizer, keyValuePairs: [(String, JSONSchema.Property)],
         cachedDigitKeys: Set<Int>) {
        self.tokenizer = tokenizer
        self.keyValuePairs = keyValuePairs
        self.cachedDigitKeys = cachedDigitKeys
    }
    
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        let decoded = tokenizer.idsToTokens[token]!
        if decoded == "}" {
            isComplete = true
        } else if decoded == "\"" && (state == .start || state == .comma) {
            // startKey
            if keyValuePairs.isEmpty {
                state = .end
            } else {
                (nextKey, nextValue) = keyValuePairs.removeFirst()
                parser = KeyParser(key: nextKey!, tokenizer: tokenizer)
                state = .key
            }
        } else if state == .key {
            if parser!.isComplete {
                nextKey = nil
                state = .colon
            } else {
                parser!.updateState(with: token, &totalDecoded)
                if parser!.isComplete {
                    state = .colon
                }
            }
        } else if state == .colon && (decoded == ":" || decoded.contains(":")) {
            state = .value
            switch nextValue!.type {
            case "string":
                parser = StringParser(tokenizer: tokenizer)
            case "integer":
                parser = IntParser(tokenizer: tokenizer, cachedDigitKeys: cachedDigitKeys)
            case "number":
                parser = NumberParser(tokenizer: tokenizer, cachedDigitKeys: cachedDigitKeys)
            default: fatalError()
            }
        } else if state == .value {
            if parser!.isComplete {
                state = .comma
            } else {
                parser!.updateState(with: token, &totalDecoded)
                if parser!.isComplete {
                    if decoded.hasSuffix(",") {
                        if keyValuePairs.isEmpty { // TODO: Add token removal
                            if totalDecoded.last == tokenizer.tokensToIds[","] {
                                totalDecoded.removeLast()
                            }
                            if totalDecoded.last == tokenizer.tokensToIds["\","] {
                                totalDecoded.removeLast()
                                totalDecoded.append(tokenizer.tokensToIds["\""]!)
                            }
                            state = .end
                        } else {
                            state = .start
                        }
                    } else {
                        state = .comma
                    }
                    if keyValuePairs.isEmpty { // TODO: Add token removal
                        if totalDecoded.last == tokenizer.tokensToIds[","] {
                            totalDecoded.removeLast()
                        }
                        if totalDecoded.last == tokenizer.tokensToIds["\","] {
                            totalDecoded.removeLast()
                            totalDecoded.append(tokenizer.tokensToIds["\""]!)
                        }
                        state = .end
                    }
                }
            }
        } else if state == .comma && decoded == "," {
            state = .key
            (nextKey, nextValue) = keyValuePairs.removeFirst()
            parser = KeyParser(key: nextKey!, tokenizer: tokenizer)
        }
    }
    
    func getValidTokens() -> Set<Int> {
        switch state {
        case .start:
            return Set([tokenizer.tokensToIds["\""]!])
        case .key, .value:
            return parser!.getValidTokens()
        case .colon:
            return Set([tokenizer.tokensToIds[":"]!])
            
        case .comma:
            return Set([tokenizer.tokensToIds[","]!])
        case .end:
            return Set([tokenizer.tokensToIds["}"]!])
        }
    }
}

struct JSONSchemaStateTracker: Sendable {
    enum State {
        case root
        case object
        case array
    }
    
    private var currentJSON: String = ""
    private let schema: JSONSchemaConvertible.Type // Define or import your JSONSchema structure
    private let tokenizer: Tokenizer
    private let openBracketIndex: Int
    private var state = State.root
    private var expectedKeysStack: [(String, JSONSchema.Property)] = []
    private var requiredKeysStack: [Set<String>] = []
    private var nextKey: String?
    private var nextValue: JSONSchema.Property?
    private var parser: Parser?
    private let cachedDigitKeys: Set<Int>
    var isComplete: Bool = false
    
    init(schema: JSONSchemaConvertible.Type, tokenizer: Tokenizer) {
        self.schema = schema
        self.tokenizer = tokenizer
        openBracketIndex = tokenizer.tokensToIds["{"]!
        
//        cachedDigitKeys = Set([tokenizer.tokensToIds["0"]!,
//        tokenizer.tokensToIds["1"]!,
//        tokenizer.tokensToIds["2"]!,
//        tokenizer.tokensToIds["3"]!,
//        tokenizer.tokensToIds["4"]!,
//        tokenizer.tokensToIds["5"]!,
//        tokenizer.tokensToIds["6"]!,
//        tokenizer.tokensToIds["7"]!,
//        tokenizer.tokensToIds["8"]!,
//        tokenizer.tokensToIds["9"]!])
        let digitKeys = tokenizer.tokensToIds.compactMap({
            if $0.key.contains(/^\d+$/) {
                return $0.value
            } else {
                return nil
            }
        })
        cachedDigitKeys = Set(digitKeys)
    }
    
    func applyPenalty(_ tensor: MLTensor) async -> MLTensor {
        let validTokens = getValidTokens()
        if validTokens.count == 1, let validToken = validTokens.first {
            var shaped = MLShapedArray<Float>(repeating: 0, shape: [validToken + 1])
//            let count = xShaped.count
            shaped[scalarAt: validToken] = Float.greatestFiniteMagnitude
            return MLTensor(shaped)
        }
        let xShaped = await tensor.shapedArray(of: Float.self)
        let disallowedIndices = xShaped.enumerated().compactMap {
            if !validTokens.contains($0.offset) {
                return Int32($0.offset)
            } else {
                return nil
            }
        }
        return tensor.replacing(atIndices: MLTensor(disallowedIndices, scalarType: Int32.self),
                                with: -Float.greatestFiniteMagnitude,
                                alongAxis: -1)
    }
    
    // Update the state based on the latest token
    mutating func updateState(with token: Int, _ totalDecoded: inout [Int]) {
        let decodedToken = tokenizer.decode(tokens: [token])
        currentJSON += decodedToken
        
        if decodedToken == "{" && state == .root {
            guard let properties = schema.properties else { fatalError() }
            state = .object
            parser = ObjectParser(tokenizer: tokenizer,
                                  keyValuePairs: properties,
                                  cachedDigitKeys: cachedDigitKeys)
        } else {
            parser!.updateState(with: token, &totalDecoded)
            if parser!.isComplete {
                state = .root
            }
        }
    }
    
    // Determine valid tokens based on the current state and schema
    private func getValidTokens() -> Set<Int> {
        var validTokens = Set<Int>()
        
        switch state {
        case .root:
            // Typically expects an object or array
            if self.currentJSON.isEmpty {
                validTokens.insert(tokenizer.tokensToIds["{"] ?? -1)
            } else {
                validTokens.insert(tokenizer.eosTokenId!)
            }
        case .object, .array:
            return parser!.getValidTokens()
        }
        return validTokens
    }
    
    // Helper method to extract the current key being processed
    private func extractCurrentKey() -> String? {
        // Implement logic to extract the current key from the context
        // This might involve parsing the current JSON string or maintaining additional state
        return nil // Placeholder: Replace with actual implementation
    }
}
