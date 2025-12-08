import ArgumentParser
import Foundation
import SwiftLM

struct Test: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "test",
        abstract: "Test a CoreML model with a prompt"
    )

    @Argument(help: "Path to CoreML model (.mlpackage)")
    var modelPath: String

    @Option(name: [.short, .long], help: "Prompt to test with")
    var prompt: String = "What is 2+2?"

    @Option(name: [.customLong("max-tokens")], help: "Maximum tokens to generate")
    var maxTokens: Int = 100

    @Option(name: [.customLong("system")], help: "System prompt")
    var systemPrompt: String = "You are a helpful assistant."

    mutating func run() async throws {
        print("Loading model from: \(modelPath)")
        print("Prompt: \(prompt)")
        print()

        let modelURL = URL(fileURLWithPath: modelPath)
        let model = try CoreMLLanguageModel.loadCompiled(url: modelURL)

        print("Model loaded: \(model.modelConfig.modelFamily)")
        print("Generating response...")
        print("---")

        let session = await model.makeSession(systemPrompt: systemPrompt)
        var output = ""
        for await token in await session.infer(prompt: prompt) {
            print(token, terminator: "")
            fflush(stdout)
            output += token
        }

        print()
        print("---")
        print("Generated \(output.count) characters")
    }
}
