import Foundation
import Generation
import JSONSchema
@preconcurrency import CoreML
import TensorUtils
import Tokenizers
import OSLog

// MARK: - Session

/// A unified session for LLM inference supporting both free-form text and JSON-constrained generation.
///
/// Session maintains conversation state (KV cache, token history) across multiple inference calls.
/// Each inference call can optionally specify a JSON schema to constrain output.
public actor Session {
    // MARK: - Core Properties
    let model: MLModel
    let tokenizer: Tokenizer
    let kvCache: MLState?
    let contextSize: Int
    let chatTemplate: any ChatTemplate
    let systemPrompt: String
    var config: GenerationConfig
    var outputBuffer: [Int] = []
    var hasShownInitialPrompt = false

    // MARK: - Optional Features
    let tools: (any Llama32Tools)?

    // MARK: - Internal
    let requiresCausal: Bool
    let requiresAttention: Bool
    var logger: Logger

    // MARK: - Initialization

    public init(
        model: MLModel,
        tokenizer: Tokenizer,
        systemPrompt: String,
        contextSize: Int,
        config: GenerationConfig,
        chatTemplate: any ChatTemplate,
        tools: (any Llama32Tools)? = nil,
        logging: Bool = false
    ) async {
        self.model = model
        self.tokenizer = tokenizer
        self.systemPrompt = systemPrompt
        self.contextSize = contextSize
        self.chatTemplate = chatTemplate
        self.tools = tools

        self.config = config
        self.config.eosTokenId = tokenizer.eosTokenId

        self.requiresCausal = model.modelDescription.inputDescriptionsByName["causalMask"] != nil
        self.requiresAttention = model.modelDescription.inputDescriptionsByName["attentionMask"] != nil

        if logging {
            self.logger = .llamaSession
        } else {
            self.logger = .init(OSLog.disabled)
        }

        if !model.modelDescription.stateDescriptionsByName.isEmpty {
            kvCache = model.makeState()
        } else {
            kvCache = nil
        }
    }

    // MARK: - Public API

    /// Generate free-form text response (streaming)
    public func infer(prompt: String) -> AsyncStream<String> {
        let formattedPrompt = formatPrompt(prompt)
        return AsyncStream { stream in
            Task {
                try await self.runInference(
                    prompt: formattedPrompt,
                    grammarType: nil,
                    stream: stream
                )
            }
        }
    }

    /// Generate JSON-constrained response (streaming)
    ///
    /// The schema description (including @SchemaGuide descriptions and constraints)
    /// is automatically injected into the prompt to help the LLM understand what to generate.
    /// Grammar-constrained decoding then ensures the output is valid JSON.
    public func infer<T: JSONSchemaConvertible>(
        prompt: String,
        as type: T.Type
    ) -> AsyncStream<String> {
        // Inject schema description into prompt
        let schemaInjectedPrompt = formatPromptWithSchema(prompt, schema: type)
        return AsyncStream { stream in
            Task {
                try await self.runInference(
                    prompt: schemaInjectedPrompt,
                    grammarType: type,
                    stream: stream
                )
            }
        }
    }

    /// Generate JSON-constrained response and parse it
    public func infer<T: JSONSchemaConvertible>(
        prompt: String,
        as type: T.Type
    ) async throws -> T {
        let jsonString = await infer(prompt: prompt, as: type).reduce("", +)
        guard let data = jsonString.data(using: .utf8) else {
            throw LlamaANEError.invalidLogitsOutput
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Generate response from JSON-encodable input (streaming)
    public func infer<Input: Encodable>(input: Input) -> AsyncStream<String> {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(input),
              let jsonString = String(data: data, encoding: .utf8) else {
            return AsyncStream { $0.finish() }
        }
        return infer(prompt: jsonString)
    }

    /// Generate JSON response from JSON-encodable input
    public func infer<Input: Encodable, Output: JSONSchemaConvertible>(
        input: Input,
        as type: Output.Type
    ) async throws -> Output {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(input),
              let jsonString = String(data: data, encoding: .utf8) else {
            throw LlamaANEError.invalidLogitsOutput
        }
        return try await infer(prompt: jsonString, as: type)
    }

    // MARK: - Private Implementation

    private func formatPrompt(_ prompt: String) -> String {
        var formattedPrompt = chatTemplate.formatUserMessage(prompt) + chatTemplate.formatAssistantPrefix()
        if !hasShownInitialPrompt {
            formattedPrompt = chatTemplate.beginOfText + chatTemplate.formatSystemPrompt(systemPrompt) + formattedPrompt
            hasShownInitialPrompt = true
        }
        return formattedPrompt
    }

    /// Format prompt with JSON schema injection for structured output
    ///
    /// This injects the schema description into the prompt so the LLM understands:
    /// - What fields to generate
    /// - What each field means (from @SchemaGuide descriptions)
    /// - Any constraints (max length, ranges, allowed values)
    private func formatPromptWithSchema(_ prompt: String, schema: any JSONSchemaConvertible.Type) -> String {
        let schemaDescription = schema.schemaDescription()

        // Build the schema-aware prompt
        let schemaPrompt = """
        \(prompt)

        Respond with a JSON object matching this schema:
        \(schemaDescription)

        Output only valid JSON, no additional text.
        """

        var formattedPrompt = chatTemplate.formatUserMessage(schemaPrompt) + chatTemplate.formatAssistantPrefix()
        if !hasShownInitialPrompt {
            formattedPrompt = chatTemplate.beginOfText + chatTemplate.formatSystemPrompt(systemPrompt) + formattedPrompt
            hasShownInitialPrompt = true
        }
        return formattedPrompt
    }

    @discardableResult
    private func runInference(
        prompt: String,
        grammarType: (any JSONSchemaConvertible.Type)?,
        stream: AsyncStream<String>.Continuation?
    ) async throws -> String {
        // Create grammar tracker if needed (scoped to this inference call)
        var grammarTracker: JSONSchemaStateTracker? = nil
        if let grammarType {
            grammarTracker = JSONSchemaStateTracker(schema: grammarType, tokenizer: tokenizer)
        }

        // Encode prompt and add to buffer
        let addSpecialTokens = outputBuffer.isEmpty
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: addSpecialTokens)
        outputBuffer.append(contentsOf: tokens)

        // Prepare initial input
        var input = try await self.input(from: outputBuffer, totalBuffer: outputBuffer)
        let currentLength = input["inputIds"]?.multiArrayValue?.count ?? 1
        if currentLength > contextSize {
            logger.warning("Context length exceeded: \(currentLength) > \(self.contextSize)")
            throw LlamaANEError.contextLengthExceeded(current: currentLength, maximum: contextSize)
        }

        var totalDecoded = ""
        var isExpectedToolCall = false

        // Generation loop
        while outputBuffer.count < contextSize {
            var time = Date.now
            let output = try await asynchronousPredict(input: input)
            var elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
            logger.debug("\(Thread.current) Prediction took \(elapsed)s")
            time = Date.now

            guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float16.self) else {
                throw LlamaANEError.invalidLogitsOutput
            }

            // Flatten tensor for processing
            var mlTensor = MLTensor(logitsValue).cast(to: Float.self).flattened()

            // Apply grammar constraints if active
            if let tracker = grammarTracker {
                mlTensor = await tracker.applyPenalty(mlTensor)
            }

            // Apply repetition penalty
            if config.repetitionPenalty != 1.0 {
                mlTensor = await mlTensor.applyRepetitionPenalty(
                    Float(config.repetitionPenalty),
                    generatedTokenIds: outputBuffer
                )
            }

            // Token selection
            let nextToken: Int
            if config.generationMode == .greedy {
                let shaped = await mlTensor.argmax().shapedArray(of: Int32.self)
                nextToken = Int(shaped.scalars[0])
            } else {
                // Sampling path - but grammar constraints make this tricky.
                // The grammar has already masked invalid tokens to -inf.
                // We need to sample from valid tokens weighted by model probability.

                var logits = await mlTensor.shapedArray(of: Float.self).scalars
                let minValidLogit = -Float.greatestFiniteMagnitude / 2

                // First, find all valid tokens (not masked by grammar)
                var validIndices: [Int] = []
                var validLogits: [Float] = []
                for (idx, logit) in logits.enumerated() {
                    if logit > minValidLogit {
                        validIndices.append(idx)
                        validLogits.append(logit)
                    }
                }

                // If only one valid token, just use it (no sampling needed)
                if validIndices.count <= 1 {
                    nextToken = validIndices.first ?? 0
                } else {
                    // Apply topK to the VALID tokens only
                    let topK = min(validLogits.count, Int(config.topK))
                    if topK > 0 && topK < validLogits.count {
                        (validIndices, validLogits) = TopKLogitsWarper(k: topK).warp(indices: validIndices, logits: validLogits)
                    }

                    let actualK = validLogits.count
                    mlTensor = MLTensor(shape: [1, actualK], scalars: validLogits)
                    var otherIndices = MLTensor(shape: [1, actualK], scalars: validIndices.map(Int32.init))

                    (mlTensor, otherIndices) = await mlTensor.topP(Float16(config.topP), indices: otherIndices)
                    (mlTensor, otherIndices) = await mlTensor.penalizeRepetition(
                        Float16(config.repetitionPenalty),
                        atIndices: otherIndices
                    )

                    if mlTensor.rank == 0 {
                        nextToken = await Int(otherIndices.shapedArray(of: Int32.self).scalar!)
                    } else {
                        nextToken = await Math.sample(
                            indexes: otherIndices,
                            probs: mlTensor.softmax()
                        )
                    }
                }
            }

            outputBuffer.append(nextToken)

            // Check for end of sequence
            if nextToken == config.eosTokenId {
                logger.debug("Found end of sequence token.")
                break
            }

            // Update grammar state if active
            var grammarComplete = false
            if grammarTracker != nil {
                grammarTracker!.updateState(with: nextToken, &outputBuffer)
                grammarComplete = grammarTracker!.isComplete
            }

            // Decode and stream token BEFORE checking grammar completion
            // This ensures the final token is yielded even when grammar completes
            let decoded = tokenizer.decode(tokens: [nextToken])

            // Handle tool calls (only for free-form text)
            if grammarTracker == nil && tools != nil && decoded.starts(with: "[") {
                totalDecoded += decoded
                isExpectedToolCall = true
            } else {
                if totalDecoded.isEmpty && decoded.filter(\.isNewline).count > 0 {
                    // Skip leading newlines
                } else {
                    totalDecoded += decoded
                    stream?.yield(decoded)
                }
            }

            // Stop generation if grammar is complete (after yielding the final token)
            if grammarComplete {
                logger.debug("Grammar complete, stopping generation.")
                break
            }

            // Prepare next input
            time = Date.now
            input = try await self.input(from: [nextToken], totalBuffer: outputBuffer)
            elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
            logger.debug("Next input gen took \(elapsed)")
        }

        if outputBuffer.count >= contextSize {
            logger.warning("Context size exceeded")
        }

        // Handle tool responses (only for free-form text without grammar)
        if grammarTracker == nil, let tools {
            let functionCalls = tools.parseFunctionCalls(totalDecoded)

            if !functionCalls.isEmpty {
                var responses: [FunctionResponse] = []
                for call in functionCalls {
                    do {
                        responses.append(
                            FunctionResponse(
                                result: try await tools.callTool(call),
                                functionCall: call
                            )
                        )
                    } catch {}
                }

                let toolResponsePrompt = """
                <|start_header_id|>user<|end_header_id|>

                \(responses.map({
                    """
                    The response for \($0.functionCall.name) is: \($0.result)
                    Please report the result for the user and do not call the function again.
                    """
                }).joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """

                try await runInference(prompt: toolResponsePrompt, grammarType: nil, stream: stream)
            }
        }

        stream?.finish()
        return totalDecoded
    }

    // MARK: - Model Prediction Helpers

    private func asynchronousPredict(input: MLDictionaryFeatureProvider) async throws -> any MLFeatureProvider {
        do {
            if let kvCache {
                return try await model.prediction(from: input, using: kvCache)
            } else {
                return try await model.prediction(from: input)
            }
        } catch {
            throw LlamaANEError.predictionFailed(underlying: error)
        }
    }

    // MARK: - Input Preparation

    private func input(from tokens: [Int], totalBuffer: [Int]) async throws -> MLDictionaryFeatureProvider {
        let inputDescription = model.modelDescription.inputDescriptionsByName["inputIds"]
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            throw LlamaANEError.missingShapeConstraint(inputName: "inputIds")
        }

        var inputDictionary: [String: MLFeatureValue] = [:]

        if shapeConstraint.type == .range {
            let queryLen = min(tokens.count, contextSize)
            let inputTokens = Array(tokens.prefix(queryLen))

            let inputIds = MLShapedArray<Int32>(
                scalars: inputTokens.map { Int32($0) },
                shape: [1, queryLen]
            )
            inputDictionary["inputIds"] = MLFeatureValue(shapedArray: inputIds)

            if requiresCausal {
                let kvLen = min(totalBuffer.count, contextSize)
                let causalMask: MLShapedArray<Float16>

                if queryLen == 1 {
                    // Incremental decoding: single token attends to all previous
                    causalMask = MLShapedArray<Float16>(
                        repeating: 1.0,
                        shape: [1, 1, 1, kvLen]
                    )
                } else {
                    // Prefill: full lower triangular mask
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
                throw LlamaANEError.contextLengthExceeded(current: tokens.count, maximum: contextSize)
            }

            let maxTokens = min(tokens.count, fixed_seq_len)
            let padLength = fixed_seq_len - maxTokens

            let paddedTokens = Array(tokens.prefix(maxTokens))
                + Array(repeating: config.padTokenId ?? 0, count: padLength)

            let inputIds = MLShapedArray<Int32>(
                scalars: paddedTokens.map { Int32($0) },
                shape: [1, fixed_seq_len]
            )

            let attentionMaskValues = Array(repeating: Int32(1), count: maxTokens)
                + Array(repeating: Int32(0), count: padLength)

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
