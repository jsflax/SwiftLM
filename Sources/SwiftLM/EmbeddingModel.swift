import Foundation
@preconcurrency import CoreML
import Tokenizers
import Hub
import Accelerate

// MARK: - Embedding Model Protocol

/// Protocol for embedding models that convert text to dense vector representations.
public protocol EmbeddingModel: Sendable {
    /// The dimensionality of the embedding vectors
    var embeddingDimension: Int { get }

    /// Embed a single text string
    func embed(text: String) async throws -> [Float]

    /// Embed multiple text strings (batch processing)
    func embed(texts: [String]) async throws -> [[Float]]

    /// Compute cosine similarity between two texts
    func similarity(_ text1: String, _ text2: String) async throws -> Float
}

// MARK: - Embedding Configuration

/// Configuration for an embedding model loaded from CoreML
public struct EmbeddingConfiguration: Sendable {
    public let modelId: String?
    public let embeddingDimension: Int
    public let maxSequenceLength: Int
    public let poolingStrategy: PoolingStrategy
    public let normalizeEmbeddings: Bool

    private static let metadataPrefix = "co.swiftlm."
    private static let tokenizerKey = "co.huggingface.exporters.name"

    public init(from model: MLModel) {
        let metadata = (model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]) ?? [:]

        self.modelId = metadata[Self.tokenizerKey]
        self.embeddingDimension = Int(metadata["\(Self.metadataPrefix)embedding_dim"] ?? metadata["\(Self.metadataPrefix)hidden_size"] ?? "") ?? 768
        self.maxSequenceLength = Int(metadata["\(Self.metadataPrefix)max_position_embeddings"] ?? "") ?? 512

        let poolingStr = metadata["\(Self.metadataPrefix)pooling_strategy"] ?? "mean"
        self.poolingStrategy = PoolingStrategy(rawValue: poolingStr) ?? .mean

        let normalizeStr = metadata["\(Self.metadataPrefix)normalize_embeddings"] ?? "true"
        self.normalizeEmbeddings = normalizeStr == "true"
    }
}

// MARK: - Pooling Strategy

/// Strategy for pooling token embeddings into a single vector
public enum PoolingStrategy: String, Sendable {
    /// Average of all token embeddings (weighted by attention mask)
    case mean
    /// Use the [CLS] token embedding (first token)
    case cls
    /// Use the last non-padding token embedding
    case last
    /// No pooling - return all token embeddings
    case none
}

// MARK: - CoreML Embedding Model

/// A CoreML-based embedding model for text-to-vector encoding.
///
/// Use this for semantic search, clustering, similarity comparison, and other
/// embedding-based NLP tasks.
///
/// ```swift
/// let model = try CoreMLEmbeddingModel.load(url: modelURL)
///
/// // Single embedding
/// let embedding = try await model.embed(text: "Hello world")
///
/// // Batch embeddings
/// let embeddings = try await model.embed(texts: ["Hello", "World", "Test"])
///
/// // Similarity
/// let sim = try await model.similarity("cat", "dog")
/// ```
public final class CoreMLEmbeddingModel: @unchecked Sendable, EmbeddingModel {

    public let model: MLModel
    public let config: EmbeddingConfiguration
    public let tokenizer: Tokenizer

    public var embeddingDimension: Int { config.embeddingDimension }

    static let inputIdsKey = "inputIds"
    static let attentionMaskKey = "attentionMask"
    static let embeddingsKey = "embeddings"

    // MARK: - Initialization

    /// Initialize with a CoreML model and tokenizer
    public init(model: MLModel, tokenizer: Tokenizer) {
        self.model = model
        self.config = EmbeddingConfiguration(from: model)
        self.tokenizer = tokenizer
    }

    /// Initialize with a CoreML model and optional tokenizer directory
    /// If no tokenizer directory is provided, downloads tokenizer from HuggingFace Hub using model ID from metadata
    public convenience init(model: MLModel, tokenizerDirectory: URL) throws {
        let tokenizer: Tokenizer
        tokenizer = try Self.loadTokenizer(from: tokenizerDirectory)
        self.init(model: model, tokenizer: tokenizer)
    }

    /// Initialize with a CoreML model and optional tokenizer directory
    /// If no tokenizer directory is provided, downloads tokenizer from HuggingFace Hub using model ID from metadata
    public convenience init(model: MLModel) async throws {
        let tokenizer: Tokenizer
        // Fall back to downloading from Hub using model ID from metadata
        let config = EmbeddingConfiguration(from: model)
        guard let modelId = config.modelId else {
            throw SwiftLMError.tokenizerLoadFailed(underlying: NSError(
                domain: "SwiftLM",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No model ID in metadata and no tokenizer directory provided"]
            ))
        }
        tokenizer = try await AutoTokenizer.from(pretrained: modelId)
        self.init(model: model, tokenizer: tokenizer)
    }
    // MARK: - Loading

    /// Load and compile a CoreML embedding model from a URL
    /// Handles both .mlpackage (uncompiled) and .mlmodelc (compiled) formats
    /// Automatically finds tokenizer directory based on model path convention, or downloads from Hub
    public static func load(url: URL, computeUnits: MLComputeUnits = .cpuAndGPU) async throws -> CoreMLEmbeddingModel {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        do {
            let modelURL: URL
            if url.pathExtension == "mlpackage" {
                // Compile the model first
                modelURL = try await MLModel.compileModel(at: url)
            } else {
                modelURL = url
            }

            let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

            // Try to find tokenizer directory alongside the model
            // Convention: model is "ModelName_Embedding.mlpackage", tokenizer is "ModelName_tokenizer/"
            if let tokenizerDir = Self.findTokenizerDirectory(for: url) {
                // Use async init which handles Hub fallback if no local tokenizer
                return try CoreMLEmbeddingModel(model: model, tokenizerDirectory: tokenizerDir)
            } else {
                return try await CoreMLEmbeddingModel(model: model)
            }
        } catch {
            throw SwiftLMError.modelLoadFailed(underlying: error)
        }
    }

    /// Load with explicit tokenizer directory
    public static func load(url: URL, tokenizerDirectory: URL?, computeUnits: MLComputeUnits = .cpuAndGPU) async throws -> CoreMLEmbeddingModel {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        do {
            let modelURL: URL
            if url.pathExtension == "mlpackage" {
                modelURL = try await MLModel.compileModel(at: url)
            } else {
                modelURL = url
            }

            let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            if let tokenizerDirectory {
                return try CoreMLEmbeddingModel(model: model, tokenizerDirectory: tokenizerDirectory)
            } else {
                return try await CoreMLEmbeddingModel(model: model)
            }
        } catch {
            throw SwiftLMError.modelLoadFailed(underlying: error)
        }
    }

    /// Find tokenizer directory based on model path convention
    private static func findTokenizerDirectory(for modelURL: URL) -> URL? {
        // Model: "ModelName_Embedding.mlpackage" -> Tokenizer: "ModelName_tokenizer"
        let modelDir = modelURL.deletingLastPathComponent()
        let modelName = modelURL.deletingPathExtension().lastPathComponent

        // Remove "_Embedding" suffix if present
        let baseName: String
        if modelName.hasSuffix("_Embedding") {
            baseName = String(modelName.dropLast("_Embedding".count))
        } else {
            baseName = modelName
        }

        let tokenizerDir = modelDir.appendingPathComponent("\(baseName)_tokenizer")

        // Check if tokenizer directory exists
        if FileManager.default.fileExists(atPath: tokenizerDir.path) {
            return tokenizerDir
        }

        return nil
    }

    /// Load from a pre-compiled model
    public static func loadCompiled(url: URL, tokenizerDirectory: URL? = nil, computeUnits: MLComputeUnits = .cpuAndGPU) async throws -> CoreMLEmbeddingModel {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        do {
            let model = try MLModel(contentsOf: url, configuration: mlConfig)
            if let tokenizerDirectory {
                return try CoreMLEmbeddingModel(model: model, tokenizerDirectory: tokenizerDirectory)
            } else {
                return try await CoreMLEmbeddingModel(model: model)
            }
        } catch {
            throw SwiftLMError.modelLoadFailed(underlying: error)
        }
    }

    /// Load a tokenizer from a local directory
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

    // MARK: - Embedding

    /// Embed a single text string
    public func embed(text: String) async throws -> [Float] {
        let embeddings = try await embed(texts: [text])
        return embeddings[0]
    }

    /// Embed multiple texts (batch processing)
    public func embed(texts: [String]) async throws -> [[Float]] {
        var results: [[Float]] = []

        // Process one at a time for now (batch processing can be added later)
        for text in texts {
            let embedding = try await embedSingle(text: text)
            results.append(embedding)
        }

        return results
    }

    /// Internal single-text embedding
    private func embedSingle(text: String) async throws -> [Float] {
        // Tokenize
        let tokens = tokenizer.encode(text: text)
        let truncatedTokens = Array(tokens.prefix(config.maxSequenceLength))

        // Prepare inputs
        let inputIds = MLShapedArray<Int32>(
            scalars: truncatedTokens.map { Int32($0) },
            shape: [1, truncatedTokens.count]
        )

        let attentionMask = MLShapedArray<Float>(
            scalars: Array(repeating: Float(1.0), count: truncatedTokens.count),
            shape: [1, truncatedTokens.count]
        )

        let inputDict: [String: MLFeatureValue] = [
            Self.inputIdsKey: MLFeatureValue(shapedArray: inputIds),
            Self.attentionMaskKey: MLFeatureValue(shapedArray: attentionMask),
        ]

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)

        // Run inference
        let output = try await model.prediction(from: input)

        guard let embeddingsValue = output.featureValue(for: Self.embeddingsKey)?.multiArrayValue else {
            throw SwiftLMError.invalidLogitsOutput
        }

        // Extract embedding (shape depends on pooling - either [1, dim] or [1, seq, dim])
        let embedding = embeddingsValue.toFloatArray()

        // If the model outputs [1, hidden_dim], we're done
        // If it outputs [1, seq_len, hidden_dim], we need to pool
        if embeddingsValue.shape.count == 2 {
            // Already pooled by the model
            return Array(embedding.suffix(config.embeddingDimension))
        } else if embeddingsValue.shape.count == 3 {
            // Need to pool here (this shouldn't happen if model does pooling)
            let seqLen = embeddingsValue.shape[1].intValue
            let hiddenDim = embeddingsValue.shape[2].intValue

            // Mean pooling
            var pooled = [Float](repeating: 0, count: hiddenDim)
            for i in 0..<seqLen {
                for j in 0..<hiddenDim {
                    pooled[j] += embedding[i * hiddenDim + j]
                }
            }
            for j in 0..<hiddenDim {
                pooled[j] /= Float(seqLen)
            }

            // Normalize if configured
            if config.normalizeEmbeddings {
                return l2Normalize(pooled)
            }
            return pooled
        }

        return embedding
    }

    // MARK: - Similarity

    /// Compute cosine similarity between two texts
    public func similarity(_ text1: String, _ text2: String) async throws -> Float {
        let embeddings = try await embed(texts: [text1, text2])
        return cosineSimilarity(embeddings[0], embeddings[1])
    }

    /// Compute similarity between a query and multiple documents
    public func similarity(query: String, documents: [String]) async throws -> [Float] {
        let queryEmbedding = try await embed(text: query)
        let docEmbeddings = try await embed(texts: documents)

        return docEmbeddings.map { cosineSimilarity(queryEmbedding, $0) }
    }

    /// Find the most similar documents to a query
    public func search(query: String, documents: [String], topK: Int = 5) async throws -> [(index: Int, score: Float)] {
        let similarities = try await similarity(query: query, documents: documents)

        let indexed = similarities.enumerated().map { (index: $0.offset, score: $0.element) }
        let sorted = indexed.sorted { $0.score > $1.score }

        return Array(sorted.prefix(topK))
    }
}

// MARK: - Math Utilities

/// L2 normalize a vector
public func l2Normalize(_ vector: [Float]) -> [Float] {
    var norm: Float = 0
    vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
    norm = sqrt(norm)

    if norm == 0 {
        return vector
    }

    var result = [Float](repeating: 0, count: vector.count)
    var divisor = norm
    vDSP_vsdiv(vector, 1, &divisor, &result, 1, vDSP_Length(vector.count))
    return result
}

/// Compute cosine similarity between two vectors
public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }

    var dotProduct: Float = 0
    var normA: Float = 0
    var normB: Float = 0

    vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
    vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
    vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))

    let denom = sqrt(normA) * sqrt(normB)
    if denom == 0 { return 0 }

    return dotProduct / denom
}

/// Compute euclidean distance between two vectors
public func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return Float.infinity }

    var diff = [Float](repeating: 0, count: a.count)
    vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

    var sumSquares: Float = 0
    vDSP_svesq(diff, 1, &sumSquares, vDSP_Length(diff.count))

    return sqrt(sumSquares)
}

/// Compute dot product between two vectors
public func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }

    var result: Float = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
    return result
}
