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
        // Range covers 600–603 so swift-syntax unifies with Lattice (pins [603,604)) when SwiftLM is a
        // PATH DEP of Orbital. mlx-swift-lm + swift-transformers accept up to 603; SwiftPM picks 603.x.
        // URL is `swiftlang/` (not `apple/`) to match mlx-swift-lm — avoids the package-identity conflict.
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0"..<"604.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // MLX backend deps (Mac: serve + LoRA train + self-improve). Metal → xcodebuild.
        // Pinned to a commit (was branch:main) so the graph can't drift mid-build.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", revision: "e3cb1e1b4fb373391a414c33e9221516d02134ef"),
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
                "Serving",
                "NativeTools",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),   // vision models (Qwen3.5-MoE VLM) — auto-routed at load
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
            dependencies: ["MLXBackend", "SelfImprove", "Serving", "NativeTools", "MiniBPE", "JSONSchema"]
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
        // Serving layer (the "useful agent first" product): retrieval grounding, abstention, and the
        // verifiable-vs-freeform router that wrap the serve-time loop. Pure Swift with INJECTED
        // backends (recall closure, generate closure) → builds + tests under plain `swift build`/`swift
        // test`, no Metal. The agent executable does the thin MLX/MCP wiring.
        .target(
            name: "Serving",
            dependencies: ["MiniBPE"]   // pure-Swift tokenizer for context-compaction token counting (no MLX)
        ),
        .testTarget(
            name: "ServingTests",
            dependencies: ["Serving"]
        ),
        // Native (in-process) agent tools: Read/Write/Edit/Glob/Grep/Bash. Pure Foundation, ZERO deps →
        // builds + tests under plain `swift build`/`swift test`. MLXBackend merges these into the model's
        // MCP tool surface; the `agent` CLI registers the standard set.
        .target(
            name: "NativeTools"
        ),
        .testTarget(
            name: "NativeToolsTests",
            dependencies: ["NativeTools"]
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
