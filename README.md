# SwiftLM

On-device LLM toolkit for Swift. Run language models and embedding models on Apple devices using CoreML.

## Features

- **HuggingFace Export** - Convert models to CoreML with quantization support
- **Grammar-Constrained Generation** - Generate valid JSON using GBNF grammars
- **Text Embeddings** - Semantic search, similarity, and clustering
- **FoundationModels Compatible** - Types work with Apple's on-device models (macOS 26+/iOS 26+)

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/jsflax/SwiftLM.git", branch: "main")
]
```

## Quick Start

### Text Generation

```swift
import SwiftLM

// Load a model
let model = try CoreMLLanguageModel.loadCompiled(url: modelURL)

// Create a session
let session = await model.makeSession(systemPrompt: "You are a helpful assistant.")

// Stream responses
for await token in await session.infer(prompt: "What is 2+2?") {
    print(token, terminator: "")
}
```

### Structured Output

```swift
@JSONSchema
struct UserProfile: Codable, JSONSchemaConvertible {
    @SchemaGuide("User's full name", .maxLength(100))
    var name: String

    @SchemaGuide("Age in years", .range(0...150))
    var age: Int

    @SchemaGuide("List of hobbies", .count(1...10))
    var hobbies: [String]
}

// Generate structured data
let profile: UserProfile = try await session.infer(prompt: "Generate a user profile", as: UserProfile.self)
```

### Text Embeddings

```swift
let model = try CoreMLEmbeddingModel.load(url: embeddingModelURL)

// Embed text
let embedding = try await model.embed(text: "Hello world")

// Semantic similarity
let similarity = try await model.similarity("cat", "dog")

// Semantic search
let results = try await model.search(
    query: "machine learning",
    documents: ["AI research", "cooking recipes", "neural networks"],
    topK: 2
)
```

## CLI Tool

### Export Models

```bash
# Causal LMs
swiftlm export Qwen/Qwen2.5-1.5B-Instruct --quantize int4

# Embedding models
swiftlm export nomic-ai/nomic-embed-text-v1.5 --embedding
```

### Test Models

```bash
swiftlm test ./models/Qwen2.5-1.5B-Instruct.mlpackage --prompt "Hello!"
```

## @SchemaGuide Constraints

| Constraint | Usage | Description |
|------------|-------|-------------|
| `.maxLength(Int)` | `@SchemaGuide("desc", .maxLength(100))` | Max string length |
| `.range(ClosedRange<Int>)` | `@SchemaGuide("desc", .range(1...100))` | Integer range |
| `.doubleRange(ClosedRange<Double>)` | `@SchemaGuide("desc", .doubleRange(0.0...1.0))` | Double range |
| `.count(ClosedRange<Int>)` | `@SchemaGuide("desc", .count(1...10))` | Array element count |
| `.anyOf([String])` | `@SchemaGuide("desc", .anyOf(["a", "b"]))` | Enum-like string options |
| `.skip` | `@SchemaGuide("desc", .skip)` | Skip generation, use default |

## Supported Architectures

### Causal LMs

| Architecture | Models | Notes |
|--------------|--------|-------|
| LlamaForCausalLM | Llama 3.x, Llama 2 | CPU+GPU |
| MistralForCausalLM | Mistral 7B | CPU+GPU |
| Qwen2ForCausalLM | Qwen 2.5 | GPU required |
| Qwen3ForCausalLM | Qwen 3 | GPU required |
| DeepseekV3ForCausalLM | DeepSeek | GPU required |

### Embedding Models

| Architecture | Models | Pooling |
|--------------|--------|---------|
| BertModel | BERT, RoBERTa, DistilBERT | CLS |
| XLMRobertaModel | E5, BGE, multilingual | Mean/CLS |
| NomicBertModel | Nomic Embed | Mean |

## Requirements

- macOS 14.0+ / iOS 17.0+
- Swift 5.9+
- Xcode 15.0+

## License

MIT
