import Foundation
import Testing
@testable import SwiftLM
import CoreML

// Path to test models
private let embeddingModelsPath = "/Users/jason/Documents/SwiftLM/Plugins/LLMGenerator/models"

@Suite("Embedding Model Tests")
struct EmbeddingTests {

    // MARK: - Model Loading Tests

    @Test("Load BGE embedding model")
    func testLoadBGEModel() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        #expect(model.embeddingDimension == 384, "BGE-small should have 384 dimensions")
        #expect(model.config.maxSequenceLength == 512, "Max sequence length should be 512")
    }

    @Test("Load multilingual E5 embedding model")
    func testLoadE5Model() async throws {
        let modelPath = "\(embeddingModelsPath)/multilingual-e5-small_Embedding.mlpackage"

        // E5 uses XLMRobertaTokenizer which may not be supported by swift-transformers
        // Skip test if tokenizer is not supported
        do {
            let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))
            #expect(model.embeddingDimension == 384, "E5-small should have 384 dimensions")
        } catch SwiftLMError.modelLoadFailed(let underlying) {
            if case SwiftLMError.tokenizerLoadFailed = underlying {
                // XLMRobertaTokenizer not supported - skip test
                print("Skipping E5 test: XLMRobertaTokenizer not supported by swift-transformers")
                return
            }
            throw underlying
        }
    }

    // MARK: - Basic Embedding Tests

    @Test("Embed single text")
    func testEmbedSingleText() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let embedding = try await model.embed(text: "Hello world")

        #expect(embedding.count == 384, "Embedding should have 384 dimensions")
        #expect(!embedding.contains(where: { $0.isNaN }), "Embedding should not contain NaN")
        #expect(!embedding.contains(where: { $0.isInfinite }), "Embedding should not contain Inf")
    }

    @Test("Embed multiple texts")
    func testEmbedMultipleTexts() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let texts = ["Hello", "World", "Test"]
        let embeddings = try await model.embed(texts: texts)

        #expect(embeddings.count == 3, "Should return 3 embeddings")
        for (i, emb) in embeddings.enumerated() {
            #expect(emb.count == 384, "Embedding \(i) should have 384 dimensions")
            #expect(!emb.contains(where: { $0.isNaN }), "Embedding \(i) should not contain NaN")
        }
    }

    // MARK: - Similarity Tests

    @Test("Similar texts have high similarity")
    func testSimilarTextsHighSimilarity() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let similarity = try await model.similarity(
            "I love programming",
            "Coding is my passion"
        )

        print("Similar texts similarity: \(similarity)")
        #expect(similarity > 0.6, "Similar texts should have similarity > 0.6, got \(similarity)")
    }

    @Test("Different texts have lower similarity")
    func testDifferentTextsLowerSimilarity() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let similarSim = try await model.similarity(
            "I love programming",
            "Coding is my passion"
        )

        let differentSim = try await model.similarity(
            "I love programming",
            "The weather is nice today"
        )

        print("Similar topics: \(similarSim)")
        print("Different topics: \(differentSim)")

        #expect(similarSim > differentSim, "Similar topics should have higher similarity than different topics")
        #expect(differentSim < 0.5, "Unrelated texts should have similarity < 0.5, got \(differentSim)")
    }

    @Test("Identical texts have similarity close to 1")
    func testIdenticalTextsSimilarity() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let text = "This is a test sentence."
        let similarity = try await model.similarity(text, text)

        print("Identical text similarity: \(similarity)")
        #expect(similarity > 0.99, "Identical texts should have similarity > 0.99, got \(similarity)")
    }

    // MARK: - Search Tests

    @Test("Search returns relevant documents first")
    func testSearchRelevantFirst() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let documents = [
            "The weather forecast shows rain tomorrow",
            "Python is a popular programming language",
            "JavaScript frameworks like React are widely used",
            "Sunny skies expected this weekend",
            "Machine learning requires large datasets",
        ]

        let results = try await model.search(
            query: "software development and coding",
            documents: documents,
            topK: 3
        )

        print("Search results for 'software development and coding':")
        for result in results {
            print("  [\(result.index)] \(String(format: "%.4f", result.score)): \(documents[result.index])")
        }

        // Programming-related documents should be in top results
        let topIndices = results.map { $0.index }
        let programmingIndices = [1, 2, 4] // Python, JavaScript, ML

        let hasRelevantInTop = topIndices.contains(where: { programmingIndices.contains($0) })
        #expect(hasRelevantInTop, "At least one programming-related document should be in top 3")

        // Weather documents should not be first
        let weatherIndices = [0, 3]
        #expect(!weatherIndices.contains(results[0].index), "Weather document should not be the top result")
    }

    @Test("Search with topK limits results")
    func testSearchTopK() async throws {
        let modelPath = "\(embeddingModelsPath)/bge-small-en-v1.5_Embedding.mlpackage"
        let model = try await CoreMLEmbeddingModel.load(url: URL(fileURLWithPath: modelPath))

        let documents = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]

        let results = try await model.search(query: "test", documents: documents, topK: 2)

        #expect(results.count == 2, "Should return exactly 2 results")
    }

    // MARK: - Math Utility Tests

    @Test("Cosine similarity of identical vectors is 1")
    func testCosineSimilarityIdentical() {
        let v = [Float](repeating: 0.5, count: 10)
        let similarity = cosineSimilarity(v, v)
        #expect(abs(similarity - 1.0) < 0.0001, "Identical vectors should have similarity 1.0")
    }

    @Test("Cosine similarity of orthogonal vectors is 0")
    func testCosineSimilarityOrthogonal() {
        let v1: [Float] = [1, 0, 0]
        let v2: [Float] = [0, 1, 0]
        let similarity = cosineSimilarity(v1, v2)
        #expect(abs(similarity) < 0.0001, "Orthogonal vectors should have similarity 0")
    }

    @Test("Cosine similarity of opposite vectors is -1")
    func testCosineSimilarityOpposite() {
        let v1: [Float] = [1, 2, 3]
        let v2: [Float] = [-1, -2, -3]
        let similarity = cosineSimilarity(v1, v2)
        #expect(abs(similarity + 1.0) < 0.0001, "Opposite vectors should have similarity -1")
    }

    @Test("L2 normalize produces unit vector")
    func testL2Normalize() {
        let v: [Float] = [3, 4] // 3-4-5 triangle
        let normalized = l2Normalize(v)

        let norm = sqrt(normalized.reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 0.0001, "Normalized vector should have unit length")
        #expect(abs(normalized[0] - 0.6) < 0.0001, "First component should be 0.6")
        #expect(abs(normalized[1] - 0.8) < 0.0001, "Second component should be 0.8")
    }

    @Test("Euclidean distance")
    func testEuclideanDistance() {
        let v1: [Float] = [0, 0]
        let v2: [Float] = [3, 4]
        let distance = euclideanDistance(v1, v2)
        #expect(abs(distance - 5.0) < 0.0001, "Distance should be 5.0 (3-4-5 triangle)")
    }

    @Test("Dot product")
    func testDotProduct() {
        let v1: [Float] = [1, 2, 3]
        let v2: [Float] = [4, 5, 6]
        let result = dotProduct(v1, v2)
        #expect(abs(result - 32.0) < 0.0001, "Dot product should be 1*4 + 2*5 + 3*6 = 32")
    }
}
