import ArgumentParser
import Foundation

struct Finetune: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "finetune",
        abstract: "Fine-tune language models with LoRA",
        discussion: """
        Fine-tune HuggingFace models for CoreML export, or train adapters for
        Apple's foundation model (.fmadapter format).

        Requires Python with: transformers, peft, accelerate, datasets, trl
        Install with: pip install transformers peft accelerate datasets trl

        Examples:
          # Fine-tune Qwen for CoreML
          swiftlm finetune --data train.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --merge

          # Train adapter for Apple's foundation model
          swiftlm finetune --data train.jsonl --train-fmadapter --export-fmadapter
        """
    )

    @Option(name: [.short, .customLong("data")], help: "Path to training data (JSONL format)")
    var data: String

    @Option(name: [.short, .customLong("model")], help: "HuggingFace model ID or local path")
    var model: String = "Qwen/Qwen2.5-1.5B-Instruct"

    @Option(name: [.short, .customLong("output")], help: "Output directory for checkpoints")
    var output: String = "finetuned-model"

    @Option(name: [.short, .customLong("epochs")], help: "Number of training epochs")
    var epochs: Int = 3

    @Option(name: [.customLong("batch-size")], help: "Batch size per device")
    var batchSize: Int = 4

    @Option(name: [.customLong("lr")], help: "Learning rate")
    var learningRate: Double = 2e-4

    @Option(name: [.customLong("lora-rank")], help: "LoRA rank")
    var loraRank: Int = 16

    @Option(name: [.customLong("eval-data")], help: "Path to evaluation data (optional)")
    var evalData: String?

    @Flag(name: .customLong("merge"), help: "Merge LoRA weights after training (for CoreML export)")
    var merge: Bool = false

    @Flag(name: .customLong("train-fmadapter"), help: "Train for Apple's foundation model")
    var trainFmadapter: Bool = false

    @Flag(name: .customLong("export-fmadapter"), help: "Export to .fmadapter format")
    var exportFmadapter: Bool = false

    @Option(name: .customLong("adapter-name"), help: "Name for .fmadapter export")
    var adapterName: String?

    mutating func run() async throws {
        // Find the bundled finetune.py script
        guard let scriptPath = findFinetuneScript() else {
            throw FinetuneError.scriptNotFound
        }

        // Find Python
        guard let pythonPath = findPython() else {
            throw FinetuneError.pythonNotFound
        }

        // Build arguments
        var args = [scriptPath, "--data", data, "--model", model, "--output", output]
        args.append(contentsOf: ["--epochs", String(epochs)])
        args.append(contentsOf: ["--batch-size", String(batchSize)])
        args.append(contentsOf: ["--lr", String(learningRate)])
        args.append(contentsOf: ["--lora-rank", String(loraRank)])

        if let evalData = evalData {
            args.append(contentsOf: ["--eval-data", evalData])
        }

        if merge {
            args.append("--merge")
        }

        if trainFmadapter {
            args.append("--train-fmadapter")
        }

        if exportFmadapter {
            args.append("--export-fmadapter")
        }

        if let adapterName = adapterName {
            args.append(contentsOf: ["--adapter-name", adapterName])
        }

        // Run the Python script
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = args
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            throw FinetuneError.trainingFailed(code: process.terminationStatus)
        }
    }

    private func findFinetuneScript() -> String? {
        let scriptName = "finetune.py"
        let pluginPath = "Plugins/LLMGenerator/\(scriptName)"

        // Check current working directory first
        let cwd = FileManager.default.currentDirectoryPath
        let cwdPath = (cwd as NSString).appendingPathComponent(pluginPath)
        if FileManager.default.fileExists(atPath: cwdPath) {
            return cwdPath
        }

        // Check relative to executable (for installed binary)
        let executableDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent

        // Go up from .build/debug to package root
        let buildPath = (executableDir as NSString).deletingLastPathComponent
        let packageRoot = (buildPath as NSString).deletingLastPathComponent
        let fromBuildPath = (packageRoot as NSString).appendingPathComponent(pluginPath)
        if FileManager.default.fileExists(atPath: fromBuildPath) {
            return fromBuildPath
        }

        // Check SWIFTLM_HOME environment variable
        if let swiftlmHome = ProcessInfo.processInfo.environment["SWIFTLM_HOME"] {
            let envPath = (swiftlmHome as NSString).appendingPathComponent(pluginPath)
            if FileManager.default.fileExists(atPath: envPath) {
                return envPath
            }
        }

        // Check in same directory as executable
        let sameDirPath = (executableDir as NSString).appendingPathComponent(scriptName)
        if FileManager.default.fileExists(atPath: sameDirPath) {
            return sameDirPath
        }

        return nil
    }

    private func findPython() -> String? {
        // Try common Python paths
        let pythonPaths = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ]

        for path in pythonPaths {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }

        // Try to find python3 via which
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        task.arguments = ["python3"]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice

        do {
            try task.run()
            task.waitUntilExit()

            if task.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !output.isEmpty {
                    return output
                }
            }
        } catch {
            // Fall through
        }

        return nil
    }
}

enum FinetuneError: Error, CustomStringConvertible {
    case scriptNotFound
    case pythonNotFound
    case trainingFailed(code: Int32)

    var description: String {
        switch self {
        case .scriptNotFound:
            return "Could not find bundled finetune.py script."
        case .pythonNotFound:
            return """
            Could not find Python 3.
            Install Python and required packages:
              pip install transformers peft accelerate datasets trl
            """
        case .trainingFailed(let code):
            return "Training failed with exit code \(code)"
        }
    }
}
