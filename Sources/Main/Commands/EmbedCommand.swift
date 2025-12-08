import ArgumentParser
import Foundation
import SwiftLM

struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Test an embedding model or compute embeddings"
    )

    @Argument(help: "Path to CoreML embedding model (.mlpackage)")
    var modelPath: String

    @Option(name: [.short, .long], help: "Text to embed (can be specified multiple times)")
    var text: [String] = []

    @Option(name: .customLong("query"), help: "Query text for similarity search")
    var query: String?

    @Option(name: .customLong("documents"), help: "Documents to search (comma-separated)")
    var documents: String?

    @Option(name: .customLong("top-k"), help: "Number of top results to return")
    var topK: Int = 5

    @Flag(name: .customLong("similarity"), help: "Compute pairwise similarity between texts")
    var computeSimilarity: Bool = false

    mutating func run() async throws {
        print("Loading embedding model from: \(modelPath)")

        let modelURL = URL(fileURLWithPath: modelPath)
        let model = try await CoreMLEmbeddingModel.load(url: modelURL)

        print("Model loaded:")
        print("  Embedding dimension: \(model.embeddingDimension)")
        print("  Max sequence length: \(model.config.maxSequenceLength)")
        print("  Pooling: \(model.config.poolingStrategy.rawValue)")
        print("  Normalize: \(model.config.normalizeEmbeddings)")
        print()

        // If query and documents provided, do similarity search
        if let query = query {
            let docs: [String]
            if let documentsStr = documents {
                docs = documentsStr.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            } else if !text.isEmpty {
                docs = text
            } else {
                print("Error: --documents or --text required with --query")
                return
            }

            print("Searching for: \"\(query)\"")
            print("In \(docs.count) documents...")
            print()

            let results = try await model.search(query: query, documents: docs, topK: topK)

            print("Top \(min(topK, results.count)) results:")
            for (rank, result) in results.enumerated() {
                let doc = docs[result.index]
                let preview = doc.prefix(50) + (doc.count > 50 ? "..." : "")
                print("  \(rank + 1). [\(String(format: "%.4f", result.score))] \(preview)")
            }
            return
        }

        // If texts provided, embed them
        if !text.isEmpty {
            print("Embedding \(text.count) text(s)...")
            print()

            let embeddings = try await model.embed(texts: text)

            for (i, (txt, emb)) in zip(text, embeddings).enumerated() {
                let preview = txt.prefix(40) + (txt.count > 40 ? "..." : "")
                let embPreview = emb.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
                print("[\(i)] \"\(preview)\"")
                print("    Embedding: [\(embPreview), ...] (dim=\(emb.count))")
                print()
            }

            // Compute pairwise similarities if requested
            if computeSimilarity && text.count > 1 {
                print("Pairwise cosine similarities:")
                for i in 0..<text.count {
                    for j in (i + 1)..<text.count {
                        let sim = cosineSimilarity(embeddings[i], embeddings[j])
                        print("  [\(i)] vs [\(j)]: \(String(format: "%.4f", sim))")
                    }
                }
            }
            return
        }

        // Default: run a quick test
        print("Running quick test...")
        let testTexts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn canine leaps above a sleepy hound.",
            "Machine learning is transforming the world.",
        ]

        print()
        for (i, txt) in testTexts.enumerated() {
            print("[\(i)] \"\(txt)\"")
        }
        print()

        let embeddings = try await model.embed(texts: testTexts)

        print("Embeddings computed (dimension: \(embeddings[0].count))")
        print()
        print("Pairwise cosine similarities:")
        for i in 0..<testTexts.count {
            for j in (i + 1)..<testTexts.count {
                let sim = cosineSimilarity(embeddings[i], embeddings[j])
                print("  [\(i)] vs [\(j)]: \(String(format: "%.4f", sim))")
            }
        }
        print()
        print("Expected: [0] vs [1] should be high (similar meaning), [0]/[1] vs [2] should be lower")
    }
}
