import Foundation
import JSONSchema

public typealias Tools = [String: (DynamicCallable, _JSONFunctionSchema)]

public protocol Llama32Tools: Actor {
    static func tools(_ self: Self) -> [String: (DynamicCallable, _JSONFunctionSchema)]
    var tools: Tools! { get }
}

public struct FunctionCall: @unchecked Sendable {
    let name: String
    let parameters: [String: Any]
}
public struct FunctionResponse: Sendable {
    let result: String
    let functionCall: FunctionCall
}

enum ToolError: Error {
    case toolDoesNotExist(String)
}

private let functionCallPattern = #"(\w+)(?:\((.*?)\))?"#
private let nameAndParamsPattern = #"(\w+)\(([^()]*)\)"#
private let parameterPattern = #"(\w+)\s*=\s*([^,]+)"#

public extension Llama32Tools {
    var jsonDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        decoder.dateDecodingStrategy = .formatted(formatter)
        return decoder
    }

    
    func callTool(_ call: FunctionCall) async throws -> String {
        guard let tool = tools[call.name] else {
            throw ToolError.toolDoesNotExist(call.name)
        }
        
        let callable = tool.0
        
        do {
            return try await callable.dynamicallyCall(withKeywordArguments: call.parameters)
        } catch {
            return error.localizedDescription
        }
    }

    nonisolated func parseFunctionCalls(_ input: String) -> [FunctionCall] {
        var functionCalls = [FunctionCall]()
        guard input.trimmingCharacters(in: .whitespacesAndNewlines).starts(with: "[") else {
            return []
        }
        // Remove the outer brackets and whitespace
        let trimmedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        
        // Regex pattern to match function calls considering nested parentheses
//        let functionCallPattern = #"(\w+)(\(([^()]*)\))?"#
        
        let regex = try! NSRegularExpression(pattern: functionCallPattern, options: [])
        let matches = regex.matches(in: trimmedInput, options: [], range: NSRange(location: 0, length: trimmedInput.utf16.count))
        
        for match in matches {
            if let nameRange = Range(match.range(at: 1), in: trimmedInput) {
                
                let paramsString = if let paramsRange = Range(match.range(at: 2), in: trimmedInput) {
                    String(trimmedInput[paramsRange])
                } else {
                    "()"
                }
                let name = String(trimmedInput[nameRange])
                
                if let functionCall = Self.parseFunctionCall(name: name, paramsString: paramsString) {
                    functionCalls.append(functionCall)
                }
            }
        }
        
        return functionCalls
    }

    private static func parseFunctionCall(name: String, paramsString: String) -> FunctionCall? {
        let parameters = parseParameters(paramsString)
        return FunctionCall(name: name, parameters: parameters)
    }

    private static func parseParameters(_ input: String) -> [String: Any] {
        var parameters = [String: Any]()
        
        let parameterStrings = splitParameters(input)
        
        for param in parameterStrings {
            let components = param.components(separatedBy: "=")
            if components.count == 2 {
                let key = components[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let valueString = components[1].trimmingCharacters(in: .whitespacesAndNewlines)
                let value = parseValue(valueString)
                parameters[key] = value
            }
        }
        
        return parameters
    }

    private static func splitParameters(_ input: String) -> [String] {
        var parameters = [String]()
        var currentParam = ""
        var bracketStack = [Character]()
        
        for char in input {
            if char == "," && bracketStack.isEmpty {
                parameters.append(currentParam)
                currentParam = ""
            } else {
                if "([{".contains(char) {
                    bracketStack.append(char)
                } else if let last = bracketStack.last, ")]}".contains(char) {
                    if matches(open: last, close: char) {
                        bracketStack.removeLast()
                    }
                }
                currentParam.append(char)
            }
        }
        
        if !currentParam.isEmpty {
            parameters.append(currentParam)
        }
        
        return parameters
    }

    private static func matches(open: Character, close: Character) -> Bool {
        return (open == "(" && close == ")") ||
               (open == "[" && close == "]") ||
               (open == "{" && close == "}")
    }

    private static func parseValue(_ input: String) -> Any {
        let trimmedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Check for quoted strings
        if (trimmedInput.hasPrefix("\"") && trimmedInput.hasSuffix("\"")) ||
           (trimmedInput.hasPrefix("'") && trimmedInput.hasSuffix("'")) {
            return String(trimmedInput.dropFirst().dropLast())
        }
        
        // Check for numbers
        if let intValue = Int(trimmedInput) {
            return intValue
        } else if let doubleValue = Double(trimmedInput) {
            return doubleValue
        }
        
        // Check for boolean
        if trimmedInput.lowercased() == "true" {
            return true
        } else if trimmedInput.lowercased() == "false" {
            return false
        }
        
        // Check for array
        if trimmedInput.hasPrefix("[") && trimmedInput.hasSuffix("]") {
            let arrayString = String(trimmedInput.dropFirst().dropLast())
            let elements = splitParameters(arrayString)
            return elements.map { parseValue($0) }
        }
        
        // Check for dictionary
        if trimmedInput.hasPrefix("{") && trimmedInput.hasSuffix("}") {
            let dictString = String(trimmedInput.dropFirst().dropLast())
            var dict = [String: Any]()
            let pairs = splitParameters(dictString)
            for pair in pairs {
                let keyValue = pair.components(separatedBy: ":")
                if keyValue.count == 2 {
                    let key = keyValue[0].trimmingCharacters(in: .whitespacesAndNewlines)
                        .replacingOccurrences(of: "\"", with: "")
                        .replacingOccurrences(of: "'", with: "")
                    let value = parseValue(keyValue[1])
                    dict[key] = value
                }
            }
            return dict
        }
        
        // Return as string if none of the above
        return trimmedInput
    }
}

// MARK: @llamaActor Macro
/// A macro that transforms an actor into a Llama tool actor, automatically integrating with `LlamaToolSession`.
///
/// The `@llamaActor` macro simplifies the process of creating an actor with tools that can be used by the Llama language model.
/// It automatically generates the necessary code to initialize a `LlamaToolSession`, register the tools, and conform to the `LlamaActor` protocol.
///
/// ### Usage:
/// ```swift
/// @llamaActor actor MyLlama {
///     /// Gets the user's favorite season.
///     @Tool public func getFavoriteSeason() async throws -> String {
///         return "autumn"
///     }
///
///     /// Gets the user's favorite animal.
///     @Tool public func getFavoriteAnimal() async throws -> String {
///         return "cat"
///     }
/// }
/// ```
///
/// ### Macro Details:
/// - **Attached To**: Actor declarations.
/// - **Produces**:
///   - Member variables and initializers required for tool integration.
///   - An extension conforming the actor to `LlamaActor`, providing necessary functionalities.
/// - **Parameters**: None.
///
/// ### Notes:
/// - The macro processes functions marked with `@Tool` within the actor to generate dynamic callable tools.
/// - It collects the tools and their schemas to register them with `LlamaToolSession`.
///
/// ### See Also:
/// - `@Tool` Macro
/// - `LlamaToolSession`
@attached(member, names: arbitrary)
@attached(extension, conformances: Llama32Tools, names: arbitrary)
public macro llamaActor(_ llamaVersion: LlamaVersion = .v3_2) =
    #externalMacro(module: "LlamaKitMacros",
                   type: "LlamaActorMacro")

/// A macro that marks a function within an actor as a tool callable by the Llama language model.
///
/// The `@Tool` macro indicates that a function should be exposed as a tool to the language model.
/// It processes the function to generate a dynamically callable structure, registers it with `LlamaToolSession`,
/// and includes the tool's metadata in the model's prompt.
///
/// ### Usage:
/// ```swift
/// @llamaActor actor MyLlama {
///     /// Gets the user's favorite animal.
///     @Tool public func getFavoriteAnimal() async throws -> String {
///         return "cat"
///     }
/// }
/// ```
///
/// ### Macro Details:
/// - **Attached To**: Function declarations within an actor marked with `@llamaActor`.
/// - **Produces**:
///   - A dynamically callable structure that wraps the function.
///   - Registers the tool with its name, description, and parameters.
/// - **Parameters**: None.
///
/// ### Notes:
/// - The function's documentation comment is used as the tool's description.
/// - Parameter comments are used to describe the tool's parameters.
/// - Supports functions with parameters and return values that conform to `Codable`.
///
/// ### See Also:
/// - `@llamaActor` Macro
/// - `LlamaToolSession`
@attached(body)
public macro Tool() = #externalMacro(module: "LlamaKitMacros",
                                     type: "ToolMacro")
