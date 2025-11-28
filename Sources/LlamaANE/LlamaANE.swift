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

public final class LanguageModel: @unchecked Sendable, LanguageModelProtocol {
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

    public func makeSession(systemPrompt: String = """
                    You are an AI Assistant.
                    """,
                     tools: (any Llama32Tools)? = nil,
                     temperature: Double = 1.0,
                     topK: Int = 0,
                     topP: Double = 1.0,
                     repetitionPenalty: Double = 1.0,
                     isLogginEnabled: Bool = false) async throws -> Session {
        var config = GenerationConfig(maxNewTokens: maxContextLength)
        config.topK = topK
        config.topP = topP
        config.repetitionPenalty = repetitionPenalty
        config.temperature = temperature
        if config.temperature > 0 && config.temperature != 1 ||
            config.topP < 1 || config.topK > 0 || config.repetitionPenalty != 1.0 {
            config.doSample = true
        } else {
            config.doSample = false
        }
        
        return try await Session(model: model, systemPrompt: systemPrompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: Self.inputIdsKey, tokenizer: tokenizer, contextSize: maxContextLength, maxContextLength: maxContextLength, config: config, tools: tools, chatTemplate: chatTemplate)
    }

    public func makeSession<J: JSONSchemaConvertible>(_ grammarType: J.Type,
        systemPrompt: String = """
                    You are an AI Assistant.
                    """,
        tools: (any Llama32Tools)? = nil,
        temperature: Double = 1.0,
        topK: Int = 0,
        topP: Double = 1.0,
        repetitionPenalty: Double = 1.1,  // Slight repetition penalty to prevent loops
        doSample: Bool = false,           // Greedy by default (faster), repetition penalty still applies
        isLogginEnabled: Bool = false) async throws -> GrammarSession<J> {
        var config = GenerationConfig(maxNewTokens: maxContextLength)
        config.topK = topK
        config.topP = topP
        config.repetitionPenalty = repetitionPenalty
        config.temperature = temperature
        config.doSample = doSample

        return await GrammarSession(systemPrompt: systemPrompt,
                                    requiresCausal: requiresCausal,
                                    requiresAttention: requiresAttention,
                                    config: config,
                                    model: model,
                                    tokenizer: tokenizer,
                                    contextSize: maxContextLength,
                                    chatTemplate: chatTemplate)
    }

    public func oneShot(prompt: String) async throws -> String {
        let stream = try await Session.oneShot(model: model, systemPrompt: prompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: Self.inputIdsKey, tokenizer: tokenizer, contextSize: maxContextLength, maxContextLength: maxContextLength, config: config, tools: tools, chatTemplate: chatTemplate)
        return await stream.reduce("", +)
    }
    
    // MARK: Session
    actor Session {
        let model: MLModel
        let requiresAttention: Bool
        let requiresCausal: Bool
        let inputIdsKey: String
        let kvCache: MLState?
        let tokenizer: Tokenizer
        let contextSize: Int
        let maxContextLength: Int
        let tools: (any Llama32Tools)?
        let chatTemplate: any ChatTemplate
        var config: GenerationConfig
        var logger: Logger = .llamaSession
        public let systemPrompt: String
        var hasShownInitialPrompt: Bool = false

        init(model: MLModel, systemPrompt: String, requiresAttention: Bool, requiresCausal: Bool, inputIdsKey: String, tokenizer: Tokenizer, contextSize: Int, maxContextLength: Int,
             config: GenerationConfig,
             tools: (any Llama32Tools)?,
             chatTemplate: any ChatTemplate,
             flush: Bool = true,
             logging: Bool = false) async throws {
            self.model = model
            self.requiresAttention = requiresAttention
            self.requiresCausal = requiresCausal
            self.inputIdsKey = inputIdsKey
            self.chatTemplate = chatTemplate
            self.config = config
            self.config.eosTokenId = tokenizer.eosTokenId
            if !logging {
                self.logger = .init(OSLog.disabled)
            }
            if !model.modelDescription.stateDescriptionsByName.isEmpty {
                kvCache = model.makeState()
            } else {
                kvCache = nil
            }
            self.systemPrompt = systemPrompt
            self.tokenizer = tokenizer
            self.contextSize = contextSize
            self.maxContextLength = maxContextLength
            self.tools = tools
        }
        
        
        
        fileprivate static func oneShot(model: MLModel, systemPrompt: String, requiresAttention: Bool,
                                        requiresCausal: Bool, inputIdsKey: String,
                                        tokenizer: Tokenizer,
                                        contextSize: Int, maxContextLength: Int,
                                        config: GenerationConfig,
                                        tools: (any Llama32Tools)?,
                                        chatTemplate: any ChatTemplate) async throws -> AsyncStream<String> {
            let session = try await Session(model: model, systemPrompt: systemPrompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: inputIdsKey, tokenizer: tokenizer, contextSize: contextSize, maxContextLength: maxContextLength, config: config, tools: tools, chatTemplate: chatTemplate, flush: false)

            return AsyncStream { stream in
                Task {
                    let prompt = chatTemplate.beginOfText + chatTemplate.formatSystemPrompt(systemPrompt) + chatTemplate.formatAssistantPrefix()
                    try await session.infer(prompt: prompt, stream: stream)
                }
            }
        }
        
        private func synchronousPredict(input: MLDictionaryFeatureProvider) throws -> any MLFeatureProvider {
            do {
                if let kvCache {
                    return try self.model.prediction(from: input, using: kvCache)
                } else {
                    return try self.model.prediction(from: input)
                }
            } catch {
                throw LlamaANEError.predictionFailed(underlying: error)
            }
        }

        private func asynchronousPredict(input: MLDictionaryFeatureProvider) async throws -> any MLFeatureProvider {
            do {
                if let kvCache {
                    return try await self.model.prediction(from: input, using: kvCache)
                } else {
                    return try await self.model.prediction(from: input)
                }
            } catch {
                throw LlamaANEError.predictionFailed(underlying: error)
            }
        }
        
        var outputBuffer: [Int] = []
        
        @discardableResult private func infer(prompt: String,
                                              stream: AsyncStream<String>.Continuation? = nil) async throws -> String {
            let addSpecialTokens = outputBuffer.isEmpty
            let tokens = tokenizer.encode(text: prompt,
                                          addSpecialTokens: addSpecialTokens)
            outputBuffer.append(contentsOf: tokens)
            var input: MLDictionaryFeatureProvider = try await input(from: outputBuffer, totalBuffer: outputBuffer)
            let currentLength = input["inputIds"]?.multiArrayValue?.count ?? 1
            if currentLength > maxContextLength {
                logger.warning("Context length exceeded: \(currentLength) > \(self.maxContextLength)")
                throw LlamaANEError.contextLengthExceeded(current: currentLength, maximum: maxContextLength)
            }

            var totalDecoded = ""
            var isExpectedToolCall = false

            while outputBuffer.count < self.contextSize {
                var time = Date.now
                let output = try await asynchronousPredict(input: input)
                var elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
                logger.debug("\(Thread.current) Prediction took \(elapsed)s")
                time = Date.now
                guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float16.self) else {
                    throw LlamaANEError.invalidLogitsOutput
                }

                var mlTensor =  MLTensor(logitsValue).cast(to: Float.self)
                
                if config.generationMode == .greedy {
                    let shaped = await mlTensor.argmax(alongAxis: 2)
                        .shapedArray(of: Int32.self)
                    let nextToken = Int(shaped[0, shaped.shape[1] - 1].scalar!)
                    outputBuffer.append(nextToken)
                    if nextToken == config.eosTokenId {
                        logger.debug("Found end of service token.")
                        break
                    }
                } else {
//                    let (indices, logits) = logitsProcessor.callAsFunction(await mlTensor.cast(to: Float.self).shapedArray(of: Float.self).scalars)
//                    let tokenID = Math.sample(indexes: indices, probs: logits)
//                    mlTensor = mlTensor.cast(to: Float.self)
                    // MARK: Apply temperature
//                    mlTensor = mlTensor / Float(config.temperature)
                    // Get topK
//                    var otherIndices = MLTensor(await mlTensor.shapedArray(of: Float16.self).indices.map(Int32.init),
                    var otherIndices: MLTensor
                    if mlTensor.shape[1] > 1 {
                        var logits = await mlTensor.shapedArray(of: Float.self).scalars
                        var indices = Array(logits.indices)
                        //                    (logits, indices) = customTopK(array: logits, K: config.topK)
                        (indices, logits) = TopKLogitsWarper(k: config.topK).warp(indices: indices, logits: logits)
                        //                    var (mlTensor, otherIndices) = mlTensor.topK(Int(config.topK))
                        //                    var logits = await mlTensor.shapedArray(of: Float.self).scalars
                        //                    (mlTensor, otherIndices) = mlTensor.flattened().topK(Int(config.topK))
                        mlTensor = MLTensor(shape: [1, config.topK], scalars: logits)
                        otherIndices = MLTensor(shape: [1, config.topK], scalars: indices.map(Int32.init))
                    } else {
                        (mlTensor, otherIndices) = mlTensor.topK(Int(config.topK))
                    }
                    (mlTensor, otherIndices) = await mlTensor.topP(Float16(config.topP), indices: otherIndices)
                    (mlTensor, otherIndices) = await mlTensor.penalizeRepetition(Float16(config.repetitionPenalty),
                                                                                 atIndices: otherIndices)
                    
                    let otherPossibleTokenId = await Math.sample(
                        indexes: otherIndices,
                        probs: mlTensor.softmax()
                    )
                    
                    outputBuffer.append(otherPossibleTokenId)
                    if otherPossibleTokenId == config.eosTokenId {
                        logger.debug("Found end of service token.")
                        break
                    }
                }
                let nextToken = outputBuffer.last!
                time = Date.now
                input = try await self.input(from: [nextToken], totalBuffer: outputBuffer)
                elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
                logger.debug("Next Input gen took \(elapsed)")
//                maxTokens = min(input["inputIds"]?.multiArrayValue?.count ?? 1, self.maxContextLength)
                let decoded = tokenizer.decode(tokens: [nextToken])
//                print("Decoded: \(decoded)", nextToken)
//                guard !isExpectedToolCall else {
//                    totalDecoded += decoded
//                    continue
//                }
                if tools != nil && decoded.starts(with: "[") {
                    // we have entered a possible function call
                    totalDecoded += decoded
                    isExpectedToolCall = true
                } else {
                    if totalDecoded.isEmpty && decoded.filter(\.isNewline).count > 0 {
                        // do nothing
                    } else {
                        totalDecoded += decoded
                        stream?.yield(decoded)
                    }
                }
            }
            
            if outputBuffer.count >= self.contextSize {
                logger.warning("Context size exceeded")
            }
            if let tools {
                var responses: [FunctionResponse] = []
                let functionCalls = tools.parseFunctionCalls(totalDecoded)
                
                if !functionCalls.isEmpty {
                    for call in functionCalls {
                        do {
                            responses.append(
                                FunctionResponse(result: try await tools.callTool(call),
                                                 functionCall: call)
                            )
                        } catch {}
                    }
                    
                    try await infer(prompt: """
                <|start_header_id|>user<|end_header_id|>
                
                \(responses.map({
                    """
                    The response for \($0.functionCall.name) is: \($0.result)
                    Please report the result for the user and do not call the function again.
                    """
                }).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """, stream: stream)
                }
            }
            
            stream?.finish()
            
            return totalDecoded
        }
        
        public func infer(prompt: String) -> AsyncStream<String> {
            var formattedPrompt = chatTemplate.formatUserMessage(prompt) + chatTemplate.formatAssistantPrefix()
            if !hasShownInitialPrompt {
                formattedPrompt = chatTemplate.beginOfText + chatTemplate.formatSystemPrompt(systemPrompt) + formattedPrompt
                hasShownInitialPrompt = true
            }
            return AsyncStream { stream in
                Task {
                    try await infer(prompt: formattedPrompt, stream: stream)
                }
            }
        }
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

