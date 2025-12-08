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
        .executable(
            name: "swift-lm",
            targets: ["Main"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-syntax.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/jsflax/swift-transformers.git", branch: "main")
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
//        .plugin(name: "LLMGenerator",
//                capability: .command(intent: .custom(verb: "generate", description: "Generate a CoreML Model"),
//                                     permissions: [.writeToPackageDirectory(reason: "To add generated files")]),
//                exclude: ["export.py", "modeling_llama.py", "venv"]),
        .target(
            name: "JSONSchema",
            dependencies: ["JSONSchemaMacros"]
//            path: "JSONSchema"
        ),
        .target(
            name: "SwiftLM",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                "JSONSchema",
                "LlamaKitMacros"
            ],
            resources: [.process("Resources")],
            cSettings: [.define("ACCELERATE_NEW_LAPACK")],
            linkerSettings: [.linkedFramework("Accelerate")]),
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
            resources: [.copy("Resources/swiftlm-export")]
        ),
    ]
)
