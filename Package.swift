// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "LlamaANE",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "LlamaANE",
            targets: ["LlamaANE"]),
        .executable(
            name: "LlamaANEMain",
            targets: ["Main"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-syntax.git", branch: "main"),
        .package(path: "../swift-transformers")
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
            name: "LlamaANE",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
                "JSONSchema",
                "LlamaKitMacros"
            ],
            resources: [.process("Resources")],
            cSettings: [.define("ACCELERATE_NEW_LAPACK")],
            linkerSettings: [.linkedFramework("Accelerate")]),
        .testTarget(
            name: "LlamaANETests",
            dependencies: ["LlamaANE"],
            linkerSettings: [
                .linkedFramework("XCTest"),
                .linkedFramework("Testing")]
        ),
        .executableTarget(
            name: "Main",
            dependencies: ["LlamaANE"]
        ),
    ]
)
