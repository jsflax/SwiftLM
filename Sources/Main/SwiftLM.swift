import ArgumentParser
import Foundation

@main
struct SwiftLMCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "swiftlm",
        abstract: "Swift on-device LLM toolkit",
        version: "0.1.0",
        subcommands: [
            Export.self,
            Test.self,
            Embed.self,
        ],
        defaultSubcommand: nil
    )
}
