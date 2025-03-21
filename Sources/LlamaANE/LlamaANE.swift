import Foundation
import Models
@preconcurrency import CoreML
import Hub
import Tokenizers
import Generation
import Accelerate
@_exported import JSONSchema
import TensorUtils
import GameKit
import OSLog

extension Logger {
    /// Using your bundle identifier is a great way to ensure a unique identifier.
    private static let subsystem = Bundle.module.bundleIdentifier!

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
    
    public let minContextLength: Int
    public let maxContextLength: Int
    
    let input_ids = "inputIds"
    let attention_mask = "attentionMask"
    
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }
    
//    private var configuration: LanguageModelConfigurationFromHub? = nil
    private var _tokenizer: Tokenizer? = nil
    let numHiddenLayers = 26  // Example for Llama-3B
    let batchSize = 1
    let numKeyValueHeads = 16
    let headDim = 128

    public let tokenizer: Tokenizer
    
    lazy var requiresAttention: Bool =
        model.modelDescription.inputDescriptionsByName[attention_mask] != nil
    
    lazy var requiresCausal: Bool
        = model.modelDescription.inputDescriptionsByName["causalMask"] != nil
    
    private let tools: (any Llama32Tools)?
    
    
    lazy var config = GenerationConfig(maxLength: maxContextLength,
                                       maxNewTokens: 2000)
    
    public init(model: MLModel,
                temperature: Double = 1.0,
                topK: Int = 0,
                topP: Double = 1.0,
                repetitionPenalty: Double = 1.0,
                tools: (any Llama32Tools)? = nil) {
        self.model = model
        // We assume inputs named "input_ids" with shape (1, seq_length)
        // Perhaps we should convert to vectors of shape (seq_length) and use sequenceConstraint instead of shapeConstraint
        let inputDescription = model.modelDescription.inputDescriptionsByName[input_ids]
        // print(inputDescription)
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            fatalError("Cannot obtain shape information")
        }
        
        switch shapeConstraint.type {
        case .enumerated:
            // TODO: support a set of fixed shapes (keeping the first one here)
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
                
//        kvCacheShape = [numHiddenLayers, batchSize, numKeyValueHeads, contextSize, headDim]
//        keyCache = try! MLMultiArray(shape: kvCacheShape.map { NSNumber(value: $0) }, dataType: .float16)
//        valueCache = try! MLMultiArray(shape: kvCacheShape.map { NSNumber(value: $0) }, dataType: .float16)
        let tokenizerConfig = try! JSONSerialization.jsonObject(with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer_config", withExtension: "json")!))
        let tokenizerData = try! JSONSerialization.jsonObject(with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer", withExtension: "json")!))
        self.tokenizer = try! AutoTokenizer.from(tokenizerConfig: Config(tokenizerConfig as! [NSString : Any]),
                                            tokenizerData: Config(tokenizerData as! [NSString : Any]))
//        if !model.modelDescription.stateDescriptionsByName.isEmpty {
//            kvCache = model.makeState()
//        }
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
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: config)
        return LanguageModel(model: model)
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
    var modelName: String {
        if let userFields = model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String : String],
           let name = userFields["co.huggingface.exporters.name"] {
            return name
        }
        // This is usually the basename of the file, that's our best bet if no metadata exists
        guard let modelName = model.configuration.modelDisplayName else { fatalError("Models must have a name that identifies them") }
        return modelName
    }
        
    var inputIdsDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName[input_ids]!
    }
    
    var inputIdsName: String {
        inputIdsDescription.name
    }
    
    /// The expected shape of the models latent sample input
    var inputIdsShape: [Int] {
        inputIdsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    var requiresCache: Bool {
        model.modelDescription.inputDescriptionsByName["keyCache"] != nil
    }
    
    func tokenize(text: String) throws -> [Int] {
        tokenizer.encode(text: text)
    }
    
    func makeSession(systemPrompt: String = """
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
        
        return try await Session(model: model, systemPrompt: systemPrompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: input_ids, tokenizer: tokenizer, contextSize: maxContextLength, maxContextLength: maxContextLength, config: config, tools: tools)
    }
    
    func makeSession<J: JSONSchemaConvertible>(_ grammarType: J.Type,
        systemPrompt: String = """
                    You are an AI Assistant.
                    """,
        tools: (any Llama32Tools)? = nil,
        temperature: Double = 1.0,
        topK: Int = 0,
        topP: Double = 1.0,
        repetitionPenalty: Double = 1.0,
        isLogginEnabled: Bool = false) async throws -> GrammarSession<J> {
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
        
        return await GrammarSession(systemPrompt: systemPrompt,
                                    requiresCausal: requiresCausal,
                                    requiresAttention: requiresAttention,
                                    config: config,
                                    model: model,
                                    tokenizer: tokenizer,
                                    contextSize: maxContextLength)
    }
    
    func oneShot(prompt: String) async throws -> String {
        let stream = try await Session.oneShot(model: model, systemPrompt: prompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: input_ids, tokenizer: tokenizer, contextSize: maxContextLength, maxContextLength: maxContextLength, config: config, tools: tools)
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
        var config: GenerationConfig
        var logger: Logger = .llamaSession
        public let systemPrompt: String
        var hasShownInitialPrompt: Bool = false
        
        init(model: MLModel, systemPrompt: String, requiresAttention: Bool, requiresCausal: Bool, inputIdsKey: String, tokenizer: Tokenizer, contextSize: Int, maxContextLength: Int,
             config: GenerationConfig,
             tools: (any Llama32Tools)?,
             flush: Bool = true,
             logging: Bool = false) async throws {
            self.model = model
            self.requiresAttention = requiresAttention
            self.requiresCausal = requiresCausal
            self.inputIdsKey = inputIdsKey
            self.config = config
            self.config.eosTokenId = tokenizer.eosTokenId//("<|eot_id|>", text: "")
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
            
            guard flush else {
                return
            }
//            try await infer(prompt: """
//            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
//            \(systemPrompt)
//            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
//            """, stream: nil) // specifically call non encoding infererence
//            hasShownInitialPrompt = true
        }
        
        
        
        fileprivate static func oneShot(model: MLModel, systemPrompt: String, requiresAttention: Bool,
                                        requiresCausal: Bool, inputIdsKey: String,
                                        tokenizer: Tokenizer,
                                        contextSize: Int, maxContextLength: Int,
                                        config: GenerationConfig,
                                        tools: (any Llama32Tools)?) async throws -> AsyncStream<String> {
            let session = try await Session(model: model, systemPrompt: systemPrompt, requiresAttention: requiresAttention, requiresCausal: requiresCausal, inputIdsKey: inputIdsKey, tokenizer: tokenizer, contextSize: contextSize, maxContextLength: maxContextLength, config: config, tools: tools, flush: false)
            
            return AsyncStream { stream in
                Task {
                    // Maybe pad or truncate
                    
                    try await session.infer(prompt: """
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    \(systemPrompt)
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    """, stream: stream)
                }
            }
        }
        
        private func synchronousPredict(input: MLDictionaryFeatureProvider) throws -> any MLFeatureProvider {
            if let kvCache {
                try! self.model.prediction(from: input,
                                           using: kvCache)
            } else {
                try! self.model.prediction(from: input)
            }
        }
        
        private func asynchronousPredict(input: MLDictionaryFeatureProvider) async throws -> any MLFeatureProvider {
            if let kvCache {
                try! await self.model.prediction(from: input,
                                                 using: kvCache)
            } else {
                try! await self.model.prediction(from: input)
            }
        }
        
        var outputBuffer: [Int] = []
        
        @discardableResult private func infer(prompt: String,
                                              stream: AsyncStream<String>.Continuation? = nil) async throws -> String {
            let addSpecialTokens = outputBuffer.isEmpty
            let tokens = tokenizer.encode(text: prompt,
                                          addSpecialTokens: addSpecialTokens)
            outputBuffer.append(contentsOf: tokens)
            var input: MLDictionaryFeatureProvider = await input(from: outputBuffer, totalBuffer: outputBuffer)
            if (input["inputIds"]?.multiArrayValue?.count ?? 1) > maxContextLength {
                print("!! CONTEXT LENGTH EXCEEDED !!")
            }
//            var maxTokens = min(input["inputIds"]?.multiArrayValue?.count ?? 1, maxContextLength)
            var totalDecoded = ""
            var isExpectedToolCall = false
//            let logitsProcessor = LogitsProcessor(logitsWarpers: logitsWarpers(config: config))
            while outputBuffer.count < self.contextSize {
                var time = Date.now
                let output = try await asynchronousPredict(input: input)
                var elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
                logger.debug("\(Thread.current) Prediction took \(elapsed)s")
                time = Date.now
                guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float16.self) else {
                    fatalError()
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
                input = await self.input(from: [nextToken], totalBuffer: outputBuffer)
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
            var prompt = """
            <|start_header_id|>user<|end_header_id|>
            \(prompt)<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
            if !hasShownInitialPrompt {
                prompt = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                \(systemPrompt)
                <|eot_id|>
                """.appending(prompt)
                hasShownInitialPrompt = true
            }
            return AsyncStream { stream in
                Task {
                    // Maybe pad or truncate
                    
                    try await infer(prompt: prompt,
                                    stream: stream)
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
    //TODO: retrieve from the json: https://huggingface.co/nlpcloud/instruct-gpt-j-fp16/blob/main/config.json#L26
    public var defaultGenerationConfig: GenerationConfig {
        var config = GenerationConfig(maxNewTokens: 30)
        switch modelName.lowercased() {
        case let x where x.contains("gpt"):
            config.doSample = true
            config.topK = 50
        default: break
        }
        return config
    }
}

