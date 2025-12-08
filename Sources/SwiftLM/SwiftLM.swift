import Foundation
//import Models
@preconcurrency import CoreML
import Hub
import Tokenizers
import Generation
import Accelerate
@_exported import JSONSchema
import TensorUtils
import GameKit
import OSLog

// MARK: - Error Types

public enum SwiftLMError: Error, LocalizedError {
    case modelLoadFailed(underlying: Error)
    case missingShapeConstraint(inputName: String)
    case missingModelName
    case tokenizerLoadFailed(underlying: Error)
    case tokenizerConfigInvalid
    case resourceNotFound(name: String, extension: String)
    case predictionFailed(underlying: Error)
    case contextLengthExceeded(current: Int, maximum: Int)
    case invalidLogitsOutput
    case unsupportedModelArchitecture(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let error):
            return "Failed to load model: \(error.localizedDescription)"
        case .missingShapeConstraint(let name):
            return "Cannot obtain shape information for input '\(name)'"
        case .missingModelName:
            return "Model must have a name that identifies it"
        case .tokenizerLoadFailed(let error):
            return "Failed to load tokenizer: \(error.localizedDescription)"
        case .tokenizerConfigInvalid:
            return "Tokenizer configuration is invalid"
        case .resourceNotFound(let name, let ext):
            return "Resource '\(name).\(ext)' not found in bundle"
        case .predictionFailed(let error):
            return "Model prediction failed: \(error.localizedDescription)"
        case .contextLengthExceeded(let current, let maximum):
            return "Context length exceeded: \(current) > \(maximum)"
        case .invalidLogitsOutput:
            return "Model returned invalid logits output"
        case .unsupportedModelArchitecture(let arch):
            return "Unsupported model architecture: \(arch)"
        }
    }
}

// MARK: - Model Configuration

public struct ModelConfiguration: Sendable {
    public let modelId: String?
    public let modelType: String?
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let hiddenSize: Int
    public let headDim: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int

    private static let metadataPrefix = "co.swiftlm."
    private static let tokenizerKey = "co.huggingface.exporters.name"

    public init(from model: MLModel) {
        let metadata = (model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]) ?? [:]

        self.modelId = metadata[Self.tokenizerKey]
        self.modelType = metadata["\(Self.metadataPrefix)model_type"]
        self.numHiddenLayers = Int(metadata["\(Self.metadataPrefix)num_hidden_layers"] ?? "") ?? 28
        self.numAttentionHeads = Int(metadata["\(Self.metadataPrefix)num_attention_heads"] ?? "") ?? 32
        self.numKeyValueHeads = Int(metadata["\(Self.metadataPrefix)num_key_value_heads"] ?? "") ?? 8
        self.hiddenSize = Int(metadata["\(Self.metadataPrefix)hidden_size"] ?? "") ?? 3072
        self.headDim = Int(metadata["\(Self.metadataPrefix)head_dim"] ?? "") ?? 128
        self.vocabSize = Int(metadata["\(Self.metadataPrefix)vocab_size"] ?? "") ?? 128256
        self.maxPositionEmbeddings = Int(metadata["\(Self.metadataPrefix)max_position_embeddings"] ?? "") ?? 8192
    }

    /// Detect the model family from modelType or modelId
    public var modelFamily: ModelFamily {
        if let type = modelType?.lowercased() {
            if type.contains("llama") { return .llama }
            if type.contains("mistral") { return .mistral }
            if type.contains("qwen") { return .qwen }
            if type.contains("deepseek") { return .deepseek }
        }
        if let id = modelId?.lowercased() {
            if id.contains("llama") { return .llama }
            if id.contains("mistral") { return .mistral }
            if id.contains("qwen") { return .qwen }
            if id.contains("deepseek") { return .deepseek }
        }
        return .unknown
    }
}

public enum ModelFamily: String, Sendable {
    case llama
    case mistral
    case qwen
    case deepseek
    case unknown
}

// MARK: - Chat Template

public protocol ChatTemplate: Sendable {
    func formatSystemPrompt(_ prompt: String) -> String
    func formatUserMessage(_ message: String) -> String
    func formatAssistantPrefix() -> String
    func formatFullPrompt(system: String, userMessages: [(role: String, content: String)]) -> String
    var beginOfText: String { get }
    var endOfTurn: String { get }
}

public struct Llama3ChatTemplate: ChatTemplate, Sendable {
    public init() {}

    public var beginOfText: String { "<|begin_of_text|>" }
    public var endOfTurn: String { "<|eot_id|>" }

    public func formatSystemPrompt(_ prompt: String) -> String {
        "<|start_header_id|>system<|end_header_id|>\n\(prompt)\(endOfTurn)"
    }

    public func formatUserMessage(_ message: String) -> String {
        "<|start_header_id|>user<|end_header_id|>\n\(message)\(endOfTurn)"
    }

    public func formatAssistantPrefix() -> String {
        "<|start_header_id|>assistant<|end_header_id|>\n"
    }

    public func formatFullPrompt(system: String, userMessages: [(role: String, content: String)]) -> String {
        var result = beginOfText + formatSystemPrompt(system)
        for message in userMessages {
            if message.role == "user" {
                result += formatUserMessage(message.content)
            } else if message.role == "assistant" {
                result += formatAssistantPrefix() + message.content + endOfTurn
            }
        }
        result += formatAssistantPrefix()
        return result
    }
}

public struct MistralChatTemplate: ChatTemplate, Sendable {
    public init() {}

    public var beginOfText: String { "<s>" }
    public var endOfTurn: String { "</s>" }

    public func formatSystemPrompt(_ prompt: String) -> String {
        "[SYSTEM_PROMPT]\(prompt)[/SYSTEM_PROMPT]"
    }

    public func formatUserMessage(_ message: String) -> String {
        "[INST]\(message)[/INST]"
    }

    public func formatAssistantPrefix() -> String {
        ""
    }

    public func formatFullPrompt(system: String, userMessages: [(role: String, content: String)]) -> String {
        var result = beginOfText + formatSystemPrompt(system)
        for message in userMessages {
            if message.role == "user" {
                result += formatUserMessage(message.content)
            } else if message.role == "assistant" {
                result += message.content + endOfTurn
            }
        }
        return result
    }
}

public struct QwenChatTemplate: ChatTemplate, Sendable {
    public init() {}

    public var beginOfText: String { "" }
    public var endOfTurn: String { "<|im_end|>" }

    public func formatSystemPrompt(_ prompt: String) -> String {
        "<|im_start|>system\n\(prompt)\(endOfTurn)\n"
    }

    public func formatUserMessage(_ message: String) -> String {
        "<|im_start|>user\n\(message)\(endOfTurn)\n"
    }

    public func formatAssistantPrefix() -> String {
        "<|im_start|>assistant\n"
    }

    public func formatFullPrompt(system: String, userMessages: [(role: String, content: String)]) -> String {
        var result = formatSystemPrompt(system)
        for message in userMessages {
            if message.role == "user" {
                result += formatUserMessage(message.content)
            } else if message.role == "assistant" {
                result += formatAssistantPrefix() + message.content + endOfTurn + "\n"
            }
        }
        result += formatAssistantPrefix()
        return result
    }
}

public struct DeepSeekChatTemplate: ChatTemplate, Sendable {
    public init() {}

    // DeepSeek R1 Distill uses Qwen's ChatML format
    public var beginOfText: String { "" }
    public var endOfTurn: String { "<|im_end|>" }

    public func formatSystemPrompt(_ prompt: String) -> String {
        "<|im_start|>system\n\(prompt)\(endOfTurn)\n"
    }

    public func formatUserMessage(_ message: String) -> String {
        "<|im_start|>user\n\(message)\(endOfTurn)\n"
    }

    public func formatAssistantPrefix() -> String {
        "<|im_start|>assistant\n"
    }

    public func formatFullPrompt(system: String, userMessages: [(role: String, content: String)]) -> String {
        var result = formatSystemPrompt(system)
        for message in userMessages {
            if message.role == "user" {
                result += formatUserMessage(message.content)
            } else if message.role == "assistant" {
                result += formatAssistantPrefix() + message.content + endOfTurn + "\n"
            }
        }
        result += formatAssistantPrefix()
        return result
    }
}

public extension ModelFamily {
    var chatTemplate: any ChatTemplate {
        switch self {
        case .llama:
            return Llama3ChatTemplate()
        case .mistral:
            return MistralChatTemplate()
        case .qwen:
            return QwenChatTemplate()
        case .deepseek:
            return DeepSeekChatTemplate()
        case .unknown:
            return Llama3ChatTemplate() // Default fallback
        }
    }
}

// MARK: - Logger Extension

extension Logger {
    /// Using your bundle identifier is a great way to ensure a unique identifier.
    private static let subsystem = Bundle.module.bundleIdentifier ?? "com.swiftlm"

    /// Logs the view cycles like a view that appeared.
    static let languageModel = Logger(subsystem: subsystem, category: "llama")

    /// All logs related to tracking and analytics.
    static let llamaSession = Logger(subsystem: subsystem, category: "llama.session")
}

extension MLMultiArray {
    func toFloatArray() -> [Float] {
        return (0..<count).map { Float(truncating: self[$0]) }
    }
    func toInt32Array() -> [Int32] {
        return (0..<count).map { Int32(truncating: self[$0]) }
    }
}

import CoreML
import Tokenizers
import Generation
import Hub

public protocol LanguageModelProtocol: AnyObject {
    var doSample: Bool { get set }
    
    var topK: Int { get set }
    
    var topP: Float { get set }
    
    var repeatPenalty: Float { get set }
    
    var temperature: Float { get set}
}

public typealias _JSONSchemaConvertible = JSONSchemaConvertible
public typealias _JSONSchema = JSONSchema
public typealias _SchemaProperty = SchemaProperty

