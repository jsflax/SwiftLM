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

public enum LlamaANEError: Error, LocalizedError {
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

    private static let metadataPrefix = "co.llamaane."
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
    private static let subsystem = Bundle.module.bundleIdentifier ?? "com.llamaane"

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

public final class LanguageModel: @unchecked Sendable, LanguageModelProtocol {
    public typealias JSONSchemaConvertible = _JSONSchemaConvertible
    public typealias JSONSchema = _JSONSchema
    public typealias SchemaProperty = _SchemaProperty
    
    public let model: MLModel
    public let modelConfig: ModelConfiguration

    public let minContextLength: Int
    public let maxContextLength: Int

    static let inputIdsKey = "inputIds"
    static let attentionMaskKey = "attentionMask"

    public let tokenizer: Tokenizer

    lazy var requiresAttention: Bool =
        model.modelDescription.inputDescriptionsByName[Self.attentionMaskKey] != nil

    lazy var requiresCausal: Bool =
        model.modelDescription.inputDescriptionsByName["causalMask"] != nil

    public var chatTemplate: any ChatTemplate {
        modelConfig.modelFamily.chatTemplate
    }

    private let tools: (any Llama32Tools)?

    lazy var config = GenerationConfig(maxLength: maxContextLength,
                                       maxNewTokens: 2000)

    /// Initialize with a CoreML model, automatically selecting the appropriate tokenizer.
    /// The tokenizer is selected based on the model's metadata (model_type field).
    public convenience init(model: MLModel,
                            temperature: Double = 1.0,
                            topK: Int = 0,
                            topP: Double = 1.0,
                            repetitionPenalty: Double = 1.0,
                            tools: (any Llama32Tools)? = nil) throws {
        // Detect model family from metadata and load appropriate tokenizer
        let config = ModelConfiguration(from: model)
        let tokenizer = try Self.loadBundledTokenizer(for: config.modelFamily)
        try self.init(
            model: model,
            tokenizer: tokenizer,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            tools: tools
        )
    }

    /// Initialize with a CoreML model and custom tokenizer.
    /// Use this when your model requires a different tokenizer (e.g., Qwen, Mistral).
    public init(model: MLModel,
                tokenizer: Tokenizer,
                temperature: Double = 1.0,
                topK: Int = 0,
                topP: Double = 1.0,
                repetitionPenalty: Double = 1.0,
                tools: (any Llama32Tools)? = nil) throws {
        self.model = model
        self.modelConfig = ModelConfiguration(from: model)
        self.tokenizer = tokenizer

        // Parse shape constraints from model description
        let inputDescription = model.modelDescription.inputDescriptionsByName[Self.inputIdsKey]
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            throw LlamaANEError.missingShapeConstraint(inputName: Self.inputIdsKey)
        }

        switch shapeConstraint.type {
        case .enumerated:
            // Support enumerated shapes (use first shape)
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            let range = inputDescription?.multiArrayConstraint?.shapeConstraint.sizeRangeForDimension[1] as? NSRange
            minContextLength = range?.location ?? 1
            maxContextLength = range?.length ?? 128
        case .unspecified:
            minContextLength = 128
            maxContextLength = 128
        @unknown default:
            minContextLength = 128
            maxContextLength = 128
        }

        self.tools = tools

        self.config.topK = topK
        self.config.topP = topP
        self.config.repetitionPenalty = repetitionPenalty
        self.config.temperature = temperature
        if config.temperature > 0 && config.temperature != 1 ||
            config.topP < 1 || config.topK > 0 || config.repetitionPenalty != 1.0 {
            config.doSample = true
        } else {
            config.doSample = false
        }
    }

    /// Load a tokenizer from a local directory containing tokenizer.json and tokenizer_config.json.
    /// This is useful for loading tokenizers saved by the export script.
    public static func loadTokenizer(from directory: URL) throws -> Tokenizer {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let dataURL = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LlamaANEError.resourceNotFound(name: "tokenizer_config", extension: "json")
        }
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw LlamaANEError.resourceNotFound(name: "tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw LlamaANEError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as LlamaANEError {
            throw error
        } catch {
            throw LlamaANEError.tokenizerLoadFailed(underlying: error)
        }
    }

    /// Download and load a tokenizer from Hugging Face Hub.
    /// This downloads the tokenizer files to a local cache and loads them.
    public static func downloadTokenizer(from modelId: String) async throws -> Tokenizer {
        do {
            return try await AutoTokenizer.from(pretrained: modelId)
        } catch {
            throw LlamaANEError.tokenizerLoadFailed(underlying: error)
        }
    }

    /// Load a bundled tokenizer for a specific model family.
    public static func loadBundledTokenizer(for family: ModelFamily) throws -> Tokenizer {
        let prefix: String
        switch family {
        case .llama:
            prefix = "llama"
        case .qwen:
            prefix = "qwen"
        case .mistral:
            // Mistral uses the same tokenizer format as Llama
            prefix = "llama"
        case .deepseek:
            // DeepSeek uses Llama tokenizer as base
            prefix = "llama"
        case .unknown:
            // Default to Llama
            prefix = "llama"
        }

        guard let configURL = Bundle.module.url(forResource: "\(prefix)_tokenizer_config", withExtension: "json") else {
            throw LlamaANEError.resourceNotFound(name: "\(prefix)_tokenizer_config", extension: "json")
        }
        guard let dataURL = Bundle.module.url(forResource: "\(prefix)_tokenizer", withExtension: "json") else {
            throw LlamaANEError.resourceNotFound(name: "\(prefix)_tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw LlamaANEError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as LlamaANEError {
            throw error
        } catch {
            throw LlamaANEError.tokenizerLoadFailed(underlying: error)
        }
    }

    private static func loadBundledTokenizer() throws -> Tokenizer {
        // Default to Llama tokenizer for backwards compatibility
        guard let configURL = Bundle.module.url(forResource: "llama_tokenizer_config", withExtension: "json") else {
            throw LlamaANEError.resourceNotFound(name: "llama_tokenizer_config", extension: "json")
        }
        guard let dataURL = Bundle.module.url(forResource: "llama_tokenizer", withExtension: "json") else {
            throw LlamaANEError.resourceNotFound(name: "llama_tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw LlamaANEError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as LlamaANEError {
            throw error
        } catch {
            throw LlamaANEError.tokenizerLoadFailed(underlying: error)
        }
    }
    
    public var doSample: Bool {
        get {
            config.doSample
        }
        set {
            config.doSample = newValue
        }
    }
    
    public var topK: Int {
        get {
            config.topK
        }
        set {
            config.topK = newValue
        }
    }
    
    public var topP: Float {
        get {
            Float(config.topP)
        }
        set {
            config.topP = Double(newValue)
        }
    }
    
    public var repeatPenalty: Float {
        get {
            Float(config.repetitionPenalty)
        }
        set {
            config.repetitionPenalty = Double(newValue)
        }
    }
    
    public var temperature: Float {
        get {
            Float(config.temperature)
        }
        set {
            config.temperature = Double(newValue)
        }
    }
}

public extension LanguageModel {
    static func loadCompiled(url: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws -> LanguageModel {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        do {
            let model = try MLModel(contentsOf: url, configuration: mlConfig)
            return try LanguageModel(model: model)
        } catch let error as LlamaANEError {
            throw error
        } catch {
            throw LlamaANEError.modelLoadFailed(underlying: error)
        }
    }
}

public extension LanguageModel {
    var description: String {
        if let description = model.modelDescription.metadata[MLModelMetadataKey.description] as? String,
           !description.isEmpty {
            return description
        }
        return model.configuration.modelDisplayName ?? ""
    }
    
    /// `name_or_path` in the Python world
    var modelName: String? {
        modelConfig.modelId ?? model.configuration.modelDisplayName
    }

    var inputIdsDescription: MLFeatureDescription? {
        model.modelDescription.inputDescriptionsByName[Self.inputIdsKey]
    }

    var inputIdsName: String {
        inputIdsDescription?.name ?? Self.inputIdsKey
    }

    /// The expected shape of the model's latent sample input
    var inputIdsShape: [Int]? {
        inputIdsDescription?.multiArrayConstraint?.shape.map { $0.intValue }
    }

    var requiresCache: Bool {
        model.modelDescription.inputDescriptionsByName["keyCache"] != nil
    }
    
    public func tokenize(text: String) throws -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Warm up the model by running a dummy inference.
    /// Call this before first real inference to avoid cold-start latency.
    /// The ANE/GPU needs to initialize on first use, which can add noticeable delay.
    public func warmup() async throws {
        // Use BOS token or token ID 1 as dummy input
        let dummyToken = tokenizer.bosTokenId ?? 1
        let inputIds = MLShapedArray<Int32>(scalars: [Int32(dummyToken)], shape: [1, 1])

        var inputDict: [String: MLFeatureValue] = [
            Self.inputIdsKey: MLFeatureValue(shapedArray: inputIds)
        ]

        // Add causal mask if required
        if requiresCausal {
            let causalMask = MLShapedArray<Float16>(repeating: 1.0, shape: [1, 1, 1, 1])
            inputDict["causalMask"] = MLFeatureValue(shapedArray: causalMask)
        }

        // Add attention mask if required
        if requiresAttention {
            let attentionMask = MLShapedArray<Int32>(scalars: [1], shape: [1, 1])
            inputDict[Self.attentionMaskKey] = MLFeatureValue(shapedArray: attentionMask)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)

        // Run a single prediction to warm up the compute units
        if !model.modelDescription.stateDescriptionsByName.isEmpty {
            let state = model.makeState()
            _ = try await model.prediction(from: input, using: state)
        } else {
            _ = try await model.prediction(from: input)
        }
    }

    /// Create a new session for inference.
    ///
    /// The session supports both free-form text generation and JSON-constrained generation.
    /// - Parameters:
    ///   - systemPrompt: The system prompt to use for the session
    ///   - tools: Optional tools for function calling
    ///   - temperature: Sampling temperature (1.0 = no change)
    ///   - topK: Top-K sampling parameter (0 = disabled)
    ///   - topP: Top-P (nucleus) sampling parameter (1.0 = disabled)
    ///   - repetitionPenalty: Penalty for repeated tokens (1.0 = no penalty, 1.1 recommended for grammar)
    ///   - doSample: Whether to use sampling (false = greedy decoding)
    ///   - logging: Whether to enable logging
    /// - Returns: A new Session actor
    public func makeSession(
        systemPrompt: String = "You are an AI Assistant.",
        tools: (any Llama32Tools)? = nil,
        temperature: Double = 1.0,
        topK: Int = 0,
        topP: Double = 1.0,
        repetitionPenalty: Double = 1.1,
        doSample: Bool = false,
        logging: Bool = false
    ) async -> Session {
        var config = GenerationConfig(maxNewTokens: maxContextLength)
        config.topK = topK
        config.topP = topP
        config.repetitionPenalty = repetitionPenalty
        config.temperature = temperature
        config.doSample = doSample

        return await Session(
            model: model,
            tokenizer: tokenizer,
            systemPrompt: systemPrompt,
            contextSize: maxContextLength,
            config: config,
            chatTemplate: chatTemplate,
            tools: tools,
            logging: logging
        )
    }

    /// One-shot inference without maintaining session state.
    public func oneShot(prompt: String, input: String) async throws -> String {
        let session = await makeSession(systemPrompt: prompt)
        return await session.infer(prompt: input).reduce("", +)
    }
    
    public func oneShot<Output: JSONSchemaConvertible & Sendable>(prompt: String,
                                                       input: String,
                                                       output: Output.Type) async throws -> Output {
        let session = await makeSession(systemPrompt: prompt)
        return try await session.infer(prompt: input, as: output)
    }
}

public extension LanguageModel {
    fileprivate static func logitsWarpers(config: GenerationConfig) -> [any LogitsWarper] {
        var logitsWarpers = [any LogitsWarper]()
        if config.temperature > 0 && config.temperature != 1 {
            logitsWarpers.append(TemperatureLogitsWarper(temperature: Float(config.temperature)))
        }
        if config.topK > 0 {
            logitsWarpers.append(TopKLogitsWarper(k: config.topK))
        }
        if config.topP < 1.0 {
            logitsWarpers.append(TopPLogitsWarper(p: Float(config.topP)))
        }
        if config.repetitionPenalty != 1.0 {
            logitsWarpers.append(RepetitionPenaltyWarper(penalty: config.repetitionPenalty))
        }
        return logitsWarpers
    }
}

extension LanguageModel {
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 30)
        if let name = modelName?.lowercased(), name.contains("gpt") {
            config.doSample = true
            config.topK = 50
        }
        return config
    }
}

