// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "SwiftLM",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "SwiftLM",
            targets: ["SwiftLM"]),
        .library(
            name: "MLXBackend",
            targets: ["MLXBackend"]),
        .executable(
            name: "swiftlm",
            targets: ["Main"]),
        .executable(
            name: "agent",
            targets: ["agent"]),
        .library(
            name: "SelfImprove",
            targets: ["SelfImprove"]),
        .executable(
            name: "selfloop",
            targets: ["selfloop"]),
    ],
    dependencies: [
        // Pinned (was branch:main) so it unifies with MLX/transformers' swift-syntax (603.x).
        .package(url: "https://github.com/apple/swift-syntax.git", from: "600.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // MLX backend deps (Mac: serve + LoRA train + self-improve). Metal → xcodebuild.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.1.0"),
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk", from: "0.12.0"),
    ],
    targets: [
        .macro(
            name: "JSONSchemaMacros",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ]
//            path: "JSONSchemaMacros"
        ),
        .macro(
            name: "LlamaKitMacros",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ]
//            path: "LlamaKitMacros"
        ),
        .target(
            name: "JSONSchema",
            dependencies: ["JSONSchemaMacros"]
//            path: "JSONSchema"
        ),
        // Our own byte-level BPE tokenizer — pure Swift, ZERO deps. Shared core for
        // both backends (MLX on Mac, CoreML on iPhone). Replaces the swift-transformers
        // fork in the runtime. 38/38 exact parity vs HF on Qwen2.5-Coder.
        .target(
            name: "MiniBPE"
        ),
        .target(
            name: "SwiftLM",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                "JSONSchema",
                "LlamaKitMacros",
                "MiniBPE"
            ],
            resources: [.process("Resources")],
            cSettings: [.define("ACCELERATE_NEW_LAPACK")],
            linkerSettings: [.linkedFramework("Accelerate")]),
        // MLX backend (Mac): serve + LoRA train + self-improvement loop + MCP tools.
        // The third `LanguageModel` conformer alongside CoreML + FoundationModels.
        // MLX uses Metal → build this target with xcodebuild, not plain `swift build`.
        .target(
            name: "MLXBackend",
            dependencies: [
                "SwiftLM",
                "MiniBPE",
                "SelfImprove",
                "Orchestration",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .testTarget(
            name: "SwiftLMTests",
            dependencies: ["SwiftLM"],
            linkerSettings: [
                .linkedFramework("XCTest"),
                .linkedFramework("Testing")]
        ),
        .executableTarget(
            name: "Main",
            dependencies: [
                "SwiftLM",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
        ),
        // The runnable agent: MLX serve + live MCP tools (+ LoRA self-improvement).
        // Builds with xcodebuild (Metal). macOS.
        .executableTarget(
            name: "agent",
            dependencies: ["MLXBackend", "SelfImprove"]
        ),
        // Orchestration spine (the "ComputePool" — the orchestration twin of the LanguageModel
        // dual-backend): one API, two execution realizations (LocalPool continuous-batching on one
        // box; ClusterPool over the network later). Pure Swift, ZERO deps → builds + tests under
        // plain `swift build`/`swift test` (no Metal). The MLX-backed LocalPool lives in MLXBackend.
        .target(
            name: "Orchestration"
        ),
        .testTarget(
            name: "OrchestrationTests",
            dependencies: ["Orchestration"]
        ),
        // Self-improvement data layer: transcript harvester + redactor + registry.
        // Pure Foundation (swift build); reads ~/.claude transcripts at runtime.
        .target(
            name: "SelfImprove"
        ),
        // Pure-Foundation unit tests for the SelfImprove layer (v2a verifier/fence/manifest logic).
        // No Metal/MLX dependency → runs under plain `swift test --filter SelfImproveTests`.
        .testTarget(
            name: "SelfImproveTests",
            dependencies: ["SelfImprove"]
        ),
        // The self-improvement loop: harvest → LoRA train → eval gate → promote.
        // Depends on MLXBackend → build with xcodebuild (Metal).
        .executableTarget(
            name: "selfloop",
            dependencies: ["SelfImprove", "MLXBackend", "Orchestration"]
        ),
    ]
)
