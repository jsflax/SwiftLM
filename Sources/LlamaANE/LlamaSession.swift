import Foundation
import Generation
import JSONSchema
@preconcurrency import CoreML
import TensorUtils
import Tokenizers
import OSLog

internal protocol _LlamaSession: Actor {
    var contextSize: Int { get }
    var requiresCausal: Bool { get }
    var requiresAttention: Bool { get }
    var config: GenerationConfig { get }
    var model: MLModel { get }
    var tokenizer: Tokenizer { get }
    var kvCache: MLState? { get }
}

public protocol LlamaSession: Actor {
    associatedtype Grammar: JSONSchemaConvertible
    
    
    func infer(prompt: String) -> AsyncStream<String>
}

extension LlamaSession where Self: _LlamaSession {
    func synchronousPredict(input: MLDictionaryFeatureProvider) throws -> any MLFeatureProvider {
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

    func asynchronousPredict(input: MLDictionaryFeatureProvider) async throws -> any MLFeatureProvider {
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
    
    internal func input(from tokens: [Int], totalBuffer: borrowing [Int] = []) async throws -> MLDictionaryFeatureProvider {
        let maxContextSize = contextSize
        let inputDescription = model.modelDescription.inputDescriptionsByName["inputIds"]
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            throw LlamaANEError.missingShapeConstraint(inputName: "inputIds")
        }
        var inputDictionary: [String: MLFeatureValue] = [:]
        if shapeConstraint.type == .range {
            // Determine the actual length we'll process
            var queryLen = min(tokens.count, maxContextSize)
            // Take the required portion of tokens (no additional padding shown here)
            let inputTokens = Array(tokens.prefix(queryLen))
            
            // Create inputIds shaped as (1, query_length)
            let inputIds = MLShapedArray<Int32>(
                scalars: inputTokens.map { Int32($0) },
                shape: [1, queryLen]
            )
            
            inputDictionary["inputIds"] = MLFeatureValue(shapedArray: inputIds)

            if requiresCausal {
                let inputQueryLen = queryLen  // Number of new tokens (1 for incremental)
                let kvLen = min(totalBuffer.count, maxContextSize)  // Total sequence length

                // Construct causal mask
                // For incremental decoding (inputQueryLen=1): just a row of 1s, shape [1,1,1,kvLen]
                // For prefill (inputQueryLen>1): full lower triangular, shape [1,1,kvLen,kvLen]
                let causalMask: MLShapedArray<Float16>
                if inputQueryLen == 1 {
                    // Incremental: single new token can attend to all previous tokens
                    // Simple row of ones - no expensive bandPart operation needed
                    causalMask = MLShapedArray<Float16>(
                        repeating: 1.0,
                        shape: [1, 1, 1, kvLen]
                    )
                } else {
                    // Prefill: need full lower triangular mask
                    let shape = [1, 1, kvLen, kvLen]
                    let onesTensor = MLTensor(ones: shape, scalarType: Float16.self)
                    causalMask = await onesTensor.bandPart(lowerBandCount: -1, upperBandCount: 0)
                        .shapedArray(of: Float16.self)
                }

                inputDictionary["causalMask"] = MLFeatureValue(shapedArray: causalMask)
            }
            
            if requiresAttention {
                let attentionMask = MLShapedArray<Int32>(
                    scalars: Array(repeating: 1, count: queryLen),
                    shape: [1, queryLen]
                )
                inputDictionary["attentionMask"] = MLFeatureValue(shapedArray: attentionMask)
            }
        } else if shapeConstraint.type == .enumerated {
            guard let fixed_seq_len = shapeConstraint.enumeratedShapes.first(where: {
                $0[1].intValue > tokens.count
            })?[1].intValue else {
                throw LlamaANEError.contextLengthExceeded(current: tokens.count, maximum: maxContextSize)
            }
            
            let maxTokens = min(tokens.count, fixed_seq_len)
            let padLength = maxTokens < fixed_seq_len ? fixed_seq_len - maxTokens : 0
            
            // Right-pad: tokens first, then padding
            let paddedTokens = Array(tokens.prefix(maxTokens))
            + Array(repeating: config.padTokenId ?? 0, count: padLength)
            
            // Ensure we have exactly fixed_seq_len tokens
            let currentCount = paddedTokens.count
            let finalTokens: [Int]
            if currentCount > fixed_seq_len {
                finalTokens = Array(paddedTokens.prefix(fixed_seq_len))
            } else if currentCount < fixed_seq_len {
                let extraPad = fixed_seq_len - currentCount
                finalTokens = paddedTokens + Array(repeating: config.padTokenId ?? 0, count: extraPad)
            } else {
                finalTokens = paddedTokens
            }
            
            let inputIds = MLShapedArray<Int32>(
                scalars: finalTokens.map { Int32($0) },
                shape: [1, finalTokens.count]
            )
            let attentionMaskValues = Array(repeating: Int32(1), count: maxTokens)
            + Array(repeating: Int32(0), count: fixed_seq_len - maxTokens)
            
            let attentionMask = MLShapedArray<Int32>(
                scalars: attentionMaskValues,
                shape: [1, fixed_seq_len]
            )
            inputDictionary["inputIds"] = MLFeatureValue(shapedArray: inputIds)
            inputDictionary["attentionMask"] = MLFeatureValue(shapedArray: attentionMask)
        }
        return try MLDictionaryFeatureProvider(dictionary: inputDictionary)
    }
}

extension LanguageModel.Session: LlamaSession, _LlamaSession {
    public typealias Grammar = String
}

public actor GrammarSession<Grammar: JSONSchemaConvertible>: LlamaSession, _LlamaSession {
    var kvCache: MLState?
    var requiresCausal: Bool
    var requiresAttention: Bool
    var config: GenerationConfig
    var model: MLModel
    var outputBuffer: [Int] = []
    var tokenizer: any Tokenizer
    var contextSize: Int
    var logger: Logger = .llamaSession
    var jsonSchemaStateTracker: JSONSchemaStateTracker
    let systemPrompt: String
    let chatTemplate: any ChatTemplate

    init(systemPrompt: String,
         requiresCausal: Bool,
         requiresAttention: Bool,
         config: GenerationConfig,
         model: MLModel,
         tokenizer: any Tokenizer,
         contextSize: Int,
         chatTemplate: any ChatTemplate) async {
        self.systemPrompt = systemPrompt
        self.requiresCausal = requiresCausal
        self.requiresAttention = requiresAttention
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.contextSize = contextSize
        self.chatTemplate = chatTemplate
        self.config.eosTokenId = tokenizer.eosTokenId
        self.logger = .init(OSLog.disabled)
        if !model.modelDescription.stateDescriptionsByName.isEmpty {
            kvCache = model.makeState()
        } else {
            kvCache = nil
        }
        jsonSchemaStateTracker = JSONSchemaStateTracker(schema: Grammar.self,
                                                        tokenizer: tokenizer)
    }
    
    @discardableResult private func infer(prompt: String,
                                          stream: AsyncStream<String>.Continuation? = nil) async throws -> String {
        let addSpecialTokens = outputBuffer.isEmpty
        let tokens = tokenizer.encode(text: prompt,
                                      addSpecialTokens: addSpecialTokens)
        outputBuffer.append(contentsOf: tokens)
        var input: MLDictionaryFeatureProvider = try await input(from: outputBuffer, totalBuffer: outputBuffer)
        let currentLength = input["inputIds"]?.multiArrayValue?.count ?? 1
        if currentLength > contextSize {
            logger.warning("Context length exceeded: \(currentLength) > \(self.contextSize)")
            throw LlamaANEError.contextLengthExceeded(current: currentLength, maximum: contextSize)
        }

        var totalDecoded = ""

        // If this is the first iteration, we should run the prediction to generate
        // the correct internal state, but then skip the processing steps
        // for the first token
        let topK = config.topK

        while outputBuffer.count < self.contextSize {
            var time = Date.now
            let output = try await asynchronousPredict(input: input)
            var elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
            logger.debug("\(Thread.current) Prediction took \(elapsed)s")
            time = Date.now
            guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float16.self) else {
                throw LlamaANEError.invalidLogitsOutput
            }

            var mlTensor =  MLTensor(logitsValue).cast(to: Float.self).flattened()
            mlTensor = await jsonSchemaStateTracker.applyPenalty(mlTensor)
            if config.generationMode == .greedy {
                let shaped = await mlTensor.argmax()
                    .shapedArray(of: Int32.self)
                let nextToken = Int(shaped.scalars[0])
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
//                if mlTensor.scalarCount > 400 {
                    let topK = min((mlTensor.scalarCount) / 2, Int(topK))
                    var logits = await mlTensor.shapedArray(of: Float.self).scalars
                    var indices = Array(logits.indices)
                    //                    (logits, indices) = customTopK(array: logits, K: config.topK)
                    (indices, logits) = TopKLogitsWarper(k: topK).warp(indices: indices, logits: logits)
                    //                    var (mlTensor, otherIndices) = mlTensor.topK(Int(config.topK))
                    //                    var logits = await mlTensor.shapedArray(of: Float.self).scalars
                    //                    (mlTensor, otherIndices) = mlTensor.flattened().topK(Int(config.topK))
                    mlTensor = MLTensor(shape: [1, topK], scalars: logits)
                    otherIndices = MLTensor(shape: [1, topK], scalars: indices.map(Int32.init))
//                } else {
                    // TODO: Fix once it's obvious why this is broken?
//                    (mlTensor, otherIndices) = mlTensor.topK(min((mlTensor.scalarCount) / 2, Int(topK)))
//                }
                (mlTensor, otherIndices) = await mlTensor.topP(Float16(config.topP), indices: otherIndices)
                (mlTensor, otherIndices) = await mlTensor.penalizeRepetition(Float16(config.repetitionPenalty),
                                                                             atIndices: otherIndices)
                let otherPossibleTokenId: Int
                if mlTensor.rank == 0 {
                    otherPossibleTokenId = await Int(otherIndices.shapedArray(of: Int32.self).scalar!)
                } else {
                    otherPossibleTokenId = await Math.sample(
                        indexes: otherIndices,
                        probs: mlTensor.softmax()
                    )
                }
                outputBuffer.append(otherPossibleTokenId)
                if otherPossibleTokenId == config.eosTokenId {
                    logger.debug("Found end of service token.")
                    break
                }
            }
            jsonSchemaStateTracker.updateState(with: outputBuffer.last!, &outputBuffer)
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
            
            if totalDecoded.isEmpty && decoded.filter(\.isNewline).count > 0 {
                // do nothing
            } else {
                totalDecoded += decoded
                stream?.yield(decoded)
            }
//            print("Total decoded:", totalDecoded)
        }
        
        if outputBuffer.count >= self.contextSize {
            logger.warning("Context size exceeded")
        }
        
        stream?.finish()
        
        return totalDecoded
    }
    var hasShownInitialPrompt = false

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
    
    public func infer(prompt: String) async throws -> Grammar {
        try await JSONDecoder().decode(Grammar.self, from: infer(prompt: prompt)
            .reduce("", +)
            .data(using: .utf8)!)
    }
}
