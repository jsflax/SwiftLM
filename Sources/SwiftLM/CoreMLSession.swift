import Foundation
import Generation
import JSONSchema
import MiniBPE
@preconcurrency import CoreML
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
    /// Our own tokenizer (MiniBPE) used for grammar-constrained decoding — replaces
    /// the swift-transformers fork's token↔id maps. nil = no constrained decoding.
    let grammarTokenizer: (any GrammarTokenizer)?
    var kvCache: MLState?
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
    let additionalStopTokens: Set<Int>

    // MARK: - Initialization

    public init(
        model: MLModel,
        tokenizer: Tokenizer,
        systemPrompt: String,
        contextSize: Int,
        config: GenerationConfig,
        chatTemplate: any ChatTemplate,
        tools: (any Llama32Tools)? = nil,
        grammarTokenizer: (any GrammarTokenizer)? = nil,
        logging: Bool = false
    ) async {
        self.model = model
        self.tokenizer = tokenizer
        self.grammarTokenizer = grammarTokenizer
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

        // Set up additional stop tokens based on model type
        // Qwen models use <|endoftext|> (151643) in addition to <|im_end|> (151645)
        if chatTemplate is QwenChatTemplate || chatTemplate is DeepSeekChatTemplate {
            self.additionalStopTokens = [151643] // <|endoftext|>
        } else {
            self.additionalStopTokens = []
        }
    }

    // MARK: - Context Info

    /// Total context size (maximum tokens the session can hold)
    public var maxContextTokens: Int { contextSize }

    /// Number of tokens currently used in the context
    public var usedTokens: Int { outputBuffer.count }

    /// Number of tokens remaining before context is full
    public var remainingTokens: Int { max(0, contextSize - outputBuffer.count) }

    /// Percentage of context used (0.0 to 1.0)
    public var contextUsage: Double { Double(outputBuffer.count) / Double(contextSize) }

    // MARK: - Session Management

    /// Serializable conversation state for save/restore
    public struct ConversationState: Codable, Sendable {
        /// The token buffer representing the full conversation
        public let tokens: [Int]
        /// The system prompt used for this conversation
        public let systemPrompt: String
        /// Timestamp when the state was saved
        public let savedAt: Date

        public init(tokens: [Int], systemPrompt: String, savedAt: Date = Date()) {
            self.tokens = tokens
            self.systemPrompt = systemPrompt
            self.savedAt = savedAt
        }
    }

    /// Reset the session, clearing the KV cache and conversation history.
    /// The system prompt is preserved and will be re-applied on the next inference.
    public func reset() {
        outputBuffer.removeAll()
        hasShownInitialPrompt = false
        // Create fresh KV cache state
        if !model.modelDescription.stateDescriptionsByName.isEmpty {
            kvCache = model.makeState()
        }
    }

    /// Save the current conversation state for later restoration.
    /// Returns a serializable state that can be persisted to disk.
    public func save() -> ConversationState {
        ConversationState(
            tokens: outputBuffer,
            systemPrompt: systemPrompt
        )
    }

    /// Restore a previously saved conversation state.
    /// This resets the session and prefills with the saved tokens.
    /// - Parameter state: The conversation state to restore
    /// - Parameter ignoreSystemPromptMismatch: If true, allows restoring even if system prompts differ
    /// - Throws: If the state's system prompt doesn't match and ignoreSystemPromptMismatch is false
    public func restore(from state: ConversationState, ignoreSystemPromptMismatch: Bool = false) async throws {
        if !ignoreSystemPromptMismatch && state.systemPrompt != systemPrompt {
            throw SwiftLMError.systemPromptMismatch(expected: systemPrompt, got: state.systemPrompt)
        }
        reset()
        try await prefill(tokens: state.tokens)
    }

    /// Prefill the session with text, processing it through the model
    /// without generating new tokens. Useful for injecting context or history.
    /// - Parameter text: The text to prefill (will be tokenized)
    public func prefill(text: String) async throws {
        let tokens = tokenizer.encode(text: text)
        try await prefill(tokens: tokens)
    }

    /// Prefill the session with tokens, processing them through the model
    /// without generating new tokens.
    /// - Parameter tokens: The token IDs to prefill
    public func prefill(tokens: [Int]) async throws {
        guard !tokens.isEmpty else { return }

        // Add tokens to the buffer FIRST so input() computes correct mask dimensions.
        // This matches the runInference pattern where tokens are appended before input() is called.
        outputBuffer.append(contentsOf: tokens)

        // Process all buffered tokens through the model to populate KV cache
        let input = try await self.input(from: outputBuffer, totalBuffer: outputBuffer)
        _ = try await asynchronousPredict(input: input)

        hasShownInitialPrompt = true
    }

    /// Prefill with a formatted conversation history.
    /// Each message should be a tuple of (role, content) where role is "user" or "assistant".
    /// - Parameters:
    ///   - messages: Array of (role, content) tuples
    ///   - includeSystemPrompt: Whether to include the system prompt (default: true)
    public func prefill(messages: [(role: String, content: String)], includeSystemPrompt: Bool = true) async throws {
        let formattedText = chatTemplate.formatFullPrompt(
            system: includeSystemPrompt ? systemPrompt : "",
            userMessages: messages
        )
        try await prefill(text: formattedText)
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
            throw SwiftLMError.invalidLogitsOutput
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            print("❌", "JSON Decoding Error for: \(String(data: data, encoding: .utf8) ?? "??")")
            throw error
        }
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
            throw SwiftLMError.invalidLogitsOutput
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
        // Create grammar tracker if needed (scoped to this inference call).
        // Uses our own MiniBPE token↔id maps, not a third-party tokenizer.
        var grammarTracker: JSONSchemaStateTracker? = nil
        if let grammarType, let grammarTokenizer {
            grammarTracker = JSONSchemaStateTracker(schema: grammarType, tokenizer: grammarTokenizer)
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
            throw SwiftLMError.contextLengthExceeded(current: currentLength, maximum: contextSize)
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

            #if arch(arm64)
            guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float16.self) else {
                throw SwiftLMError.invalidLogitsOutput
            }
            #else
            guard let logitsValue = output.featureValue(for: "logits")?.shapedArrayValue(of: Float.self) else {
                throw SwiftLMError.invalidLogitsOutput
            }
            #endif

            // Extract logits for the LAST token position only
            // Shape is [1, seq_len, vocab_size] - we want [vocab_size] from position [-1]
            let logitsTensor = MLTensor(logitsValue).cast(to: Float.self)
            let seqLen = logitsTensor.shape[1]
            var mlTensor = logitsTensor[0, seqLen - 1].flattened()

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

            // Token selection - GPU-native pipeline
            let nextToken: Int
            if config.generationMode == .greedy {
                // Greedy: just take argmax (GPU operation)
                let shaped = await mlTensor.argmax().shapedArray(of: Int32.self)
                // argmax returns a scalar tensor (shape []), access via .scalar
                nextToken = Int(shaped.scalar!)
            } else if grammarTracker != nil {
                // Grammar-constrained sampling: filter to valid tokens first, then sample
                // Must pull to CPU since grammar already applied -inf to invalid tokens
                let logits = await mlTensor.shapedArray(of: Float.self).scalars
                let minValidLogit = -Float.greatestFiniteMagnitude / 2

                var validIndices: [Int] = []
                var validLogits: [Float] = []
                for (idx, logit) in logits.enumerated() {
                    if logit > minValidLogit {
                        validIndices.append(idx)
                        validLogits.append(logit)
                    }
                }

                if validIndices.count <= 1 {
                    nextToken = validIndices.first ?? 0
                } else {
                    // Apply temperature
                    let temperature = config.temperature
                    if temperature != 1.0 && temperature > 0 {
                        validLogits = validLogits.map { $0 / Float(temperature) }
                    }

                    // Apply topK if needed (usually few valid tokens so often skipped)
                    let topK = config.topK > 0 ? min(Int(config.topK), validLogits.count) : validLogits.count
                    if topK < validLogits.count {
                        (validIndices, validLogits) = TopKLogitsWarper(k: topK).warp(indices: validIndices, logits: validLogits)
                    }

                    // Apply topP
                    let topP = config.topP
                    if topP < 1.0 && validLogits.count > 1 {
                        // Softmax
                        let maxLogit = validLogits.max() ?? 0
                        let exps = validLogits.map { Foundation.exp($0 - maxLogit) }
                        let expSum = exps.reduce(0, +)
                        let probs = exps.map { $0 / expSum }

                        // Sort by probability descending
                        var indexed = zip(validIndices, zip(validLogits, probs)).map { ($0, $1.0, $1.1) }
                        indexed.sort { $0.2 > $1.2 }

                        // Find cutoff
                        var cumsum: Float = 0
                        var cutoff = indexed.count
                        for (i, (_, _, prob)) in indexed.enumerated() {
                            cumsum += prob
                            if cumsum > Float(topP) {
                                cutoff = i + 1
                                break
                            }
                        }
                        cutoff = max(cutoff, 1)

                        validIndices = indexed.prefix(cutoff).map { $0.0 }
                        validLogits = indexed.prefix(cutoff).map { $0.1 }
                    }

                    // Sample from filtered distribution
                    let maxLogit = validLogits.max() ?? 0
                    let exps = validLogits.map { Foundation.exp($0 - maxLogit) }
                    let expSum = exps.reduce(0, +)
                    let probs = exps.map { $0 / expSum }
                    nextToken = Math.sample(indexes: validIndices, probs: probs)
                }
            } else {
                // Standard sampling pipeline using GPU-native operations
                let temperature = config.temperature
                let topK = config.topK > 0 ? Int(config.topK) : 40
                let topP = config.topP

                // Step 1: Temperature scaling (GPU)
                if temperature != 1.0 && temperature > 0 {
                    mlTensor = mlTensor / Float(temperature)
                }

                // Step 2: Top-K via argsort + gathering (GPU)
                let sortedIndices = mlTensor.argsort(descendingOrder: true)
                let topKIndices = sortedIndices[0..<topK]
                let topKLogits = mlTensor.gathering(atIndices: topKIndices, alongAxis: -1)

                // Step 3: Top-P (nucleus) filtering (GPU)
                var filteredLogits = topKLogits
                let filteredIndices = topKIndices
                if topP < 1.0 {
                    let probs = topKLogits.softmax(alongAxis: -1)
                    let cumsum = probs.cumulativeSum(alongAxis: -1)
                    // Shift cumsum left so we always keep at least the first token
                    // Original: exclude where cumsum > topP (but this excludes token 0 if prob[0] > topP)
                    // Fixed: exclude where shifted_cumsum > topP (token 0 is never excluded)
                    let shiftedCumsum = MLTensor(
                        concatenating: [MLTensor(repeating: Float(0), shape: [1]), cumsum[0..<(topK-1)]],
                        alongAxis: 0
                    )
                    let excludeMask = shiftedCumsum .> Float(topP)
                    filteredLogits = topKLogits.replacing(
                        with: MLTensor(repeating: -Float.greatestFiniteMagnitude, shape: topKLogits.shape),
                        where: excludeMask
                    )
                }

                // Step 4: Final softmax (GPU)
                let finalProbs = filteredLogits.softmax(alongAxis: -1)

                // Step 5: Sample (requires CPU for random selection)
                nextToken = await Math.sample(indexes: filteredIndices, probs: finalProbs)
            }

            outputBuffer.append(nextToken)

            // Check for end of sequence (main EOS token or additional stop tokens)
            if nextToken == config.eosTokenId || additionalStopTokens.contains(nextToken) {
                logger.debug("Found end of sequence token: \(nextToken)")
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

            // Check if token decodes to only replacement characters (U+FFFD)
            // This can happen with special tokens that aren't properly handled
            let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
            let isReplacementChar = !trimmed.isEmpty && trimmed.allSatisfy({ $0 == "\u{FFFD}" })

            if isReplacementChar {
                // Don't yield replacement characters but continue generation
                logger.debug("Skipping replacement character token from stream: \(nextToken)")
            } else if grammarTracker == nil && tools != nil && decoded.starts(with: "[") {
                // Handle tool calls (only for free-form text)
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
            throw SwiftLMError.predictionFailed(underlying: error)
        }
    }

    // MARK: - Input Preparation

    private func input(from tokens: [Int], totalBuffer: [Int]) async throws -> MLDictionaryFeatureProvider {
        let inputDescription = model.modelDescription.inputDescriptionsByName["inputIds"]
        guard let shapeConstraint = inputDescription?.multiArrayConstraint?.shapeConstraint else {
            throw SwiftLMError.missingShapeConstraint(inputName: "inputIds")
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
                #if arch(arm64)
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
                #else
                let causalMask: MLShapedArray<Float>

                if queryLen == 1 {
                    causalMask = MLShapedArray<Float>(
                        repeating: 1.0,
                        shape: [1, 1, 1, kvLen]
                    )
                } else {
                    let shape = [1, 1, kvLen, kvLen]
                    let onesTensor = MLTensor(ones: shape, scalarType: Float.self)
                    causalMask = await onesTensor.bandPart(lowerBandCount: -1, upperBandCount: 0)
                        .shapedArray(of: Float.self)
                }
                #endif
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
                throw SwiftLMError.contextLengthExceeded(current: tokens.count, maximum: contextSize)
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
