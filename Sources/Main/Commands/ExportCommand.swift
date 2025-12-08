import ArgumentParser
import Foundation

struct Export: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export",
        abstract: "Export HuggingFace models to CoreML format"
    )

    @Argument(help: "HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    var modelId: String

    @Option(name: [.short, .customLong("output-dir")], help: "Output directory for exported models")
    var outputDir: String = "models"

    @Option(name: [.short, .customLong("max-context")], help: "Maximum context length")
    var maxContext: Int = 8192

    @Option(name: [.short, .customLong("quantize")], help: "Quantization type (int4)")
    var quantize: String?

    @Flag(name: .customLong("skip-test"), help: "Skip generation test after export")
    var skipTest: Bool = false

    mutating func run() async throws {
        // Find the bundled swiftlm-export binary
        guard let binaryPath = findExportBinary() else {
            throw ExportError.binaryNotFound
        }

        // Build arguments
        var args = [modelId, "--output-dir", outputDir, "--max-context", String(maxContext)]

        if let quantize = quantize {
            args.append(contentsOf: ["--quantize", quantize])
        }

        if skipTest {
            args.append("--skip-test")
        }

        // Run the export binary
        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = args
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            throw ExportError.exportFailed(code: process.terminationStatus)
        }
    }

    private func findExportBinary() -> String? {
        // Check Bundle.module first (SwiftPM resource bundle)
        if let bundledURL = Bundle.module.url(forResource: "swiftlm-export", withExtension: nil) {
            let path = bundledURL.path
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }

        // Check if running from Xcode/SwiftPM build
        if let resourcePath = Bundle.main.resourcePath {
            let bundledPath = (resourcePath as NSString).appendingPathComponent("swiftlm-export")
            if FileManager.default.isExecutableFile(atPath: bundledPath) {
                return bundledPath
            }
        }

        // Check relative to executable
        let executableDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        let relativePath = (executableDir as NSString).appendingPathComponent("swiftlm-export")
        if FileManager.default.isExecutableFile(atPath: relativePath) {
            return relativePath
        }

        // Check in Resources directory relative to executable
        let resourcesPath = (executableDir as NSString).appendingPathComponent("Resources/swiftlm-export")
        if FileManager.default.isExecutableFile(atPath: resourcesPath) {
            return resourcesPath
        }

        // Check in current directory
        let currentPath = FileManager.default.currentDirectoryPath
        let currentDirPath = (currentPath as NSString).appendingPathComponent("swiftlm-export")
        if FileManager.default.isExecutableFile(atPath: currentDirPath) {
            return currentDirPath
        }

        return nil
    }
}

enum ExportError: Error, CustomStringConvertible {
    case binaryNotFound
    case exportFailed(code: Int32)

    var description: String {
        switch self {
        case .binaryNotFound:
            return "Could not find swiftlm-export binary. Ensure it is bundled with the CLI."
        case .exportFailed(let code):
            return "Export failed with exit code \(code)"
        }
    }
}
