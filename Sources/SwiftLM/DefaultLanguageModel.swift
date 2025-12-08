import Foundation
import CoreML
import Tokenizers
import Hub
import Generation
import TensorUtils


public final class CoreMLLanguageModel: @unchecked Sendable, LanguageModelProtocol {
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
            throw SwiftLMError.missingShapeConstraint(inputName: Self.inputIdsKey)
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
            throw SwiftLMError.resourceNotFound(name: "tokenizer_config", extension: "json")
        }
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw SwiftLMError.resourceNotFound(name: "tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw SwiftLMError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as SwiftLMError {
            throw error
        } catch {
            throw SwiftLMError.tokenizerLoadFailed(underlying: error)
        }
    }

    /// Download and load a tokenizer from Hugging Face Hub.
    /// This downloads the tokenizer files to a local cache and loads them.
    public static func downloadTokenizer(from modelId: String) async throws -> Tokenizer {
        do {
            return try await AutoTokenizer.from(pretrained: modelId)
        } catch {
            throw SwiftLMError.tokenizerLoadFailed(underlying: error)
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
            throw SwiftLMError.resourceNotFound(name: "\(prefix)_tokenizer_config", extension: "json")
        }
        guard let dataURL = Bundle.module.url(forResource: "\(prefix)_tokenizer", withExtension: "json") else {
            throw SwiftLMError.resourceNotFound(name: "\(prefix)_tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw SwiftLMError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as SwiftLMError {
            throw error
        } catch {
            throw SwiftLMError.tokenizerLoadFailed(underlying: error)
        }
    }

    private static func loadBundledTokenizer() throws -> Tokenizer {
        // Default to Llama tokenizer for backwards compatibility
        guard let configURL = Bundle.module.url(forResource: "llama_tokenizer_config", withExtension: "json") else {
            throw SwiftLMError.resourceNotFound(name: "llama_tokenizer_config", extension: "json")
        }
        guard let dataURL = Bundle.module.url(forResource: "llama_tokenizer", withExtension: "json") else {
            throw SwiftLMError.resourceNotFound(name: "llama_tokenizer", extension: "json")
        }

        do {
            let configData = try Data(contentsOf: configURL)
            let tokenizerDataContent = try Data(contentsOf: dataURL)

            let tokenizerConfig = try JSONSerialization.jsonObject(with: configData)
            let tokenizerData = try JSONSerialization.jsonObject(with: tokenizerDataContent)

            guard let configDict = tokenizerConfig as? [NSString: Any],
                  let dataDict = tokenizerData as? [NSString: Any] else {
                throw SwiftLMError.tokenizerConfigInvalid
            }

            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        } catch let error as SwiftLMError {
            throw error
        } catch {
            throw SwiftLMError.tokenizerLoadFailed(underlying: error)
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

public extension CoreMLLanguageModel {
    static func loadCompiled(url: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws -> CoreMLLanguageModel {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        do {
            let model = try MLModel(contentsOf: url, configuration: mlConfig)
            return try CoreMLLanguageModel(model: model)
        } catch let error as SwiftLMError {
            throw error
        } catch {
            throw SwiftLMError.modelLoadFailed(underlying: error)
        }
    }
}

public extension CoreMLLanguageModel {
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

public extension CoreMLLanguageModel {
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

extension CoreMLLanguageModel {
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 30)
        if let name = modelName?.lowercased(), name.contains("gpt") {
            config.doSample = true
            config.topK = 50
        }
        return config
    }
}

