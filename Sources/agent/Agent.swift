import Foundation
import MLXBackend
import SelfImprove

// The runnable SwiftLM agent: loads an MLX model, hot-swaps the current champion
// adapter (self-improvement redeploy), connects MCP servers, and runs the round-capped
// tool loop in a REPL. The self-improving agent's pieces, assembled.
//   MODEL=...        override the model id
//   MCP_SERVER=...   path to an MCP server binary to connect (default: claude-utils)
//   BASE=1           skip the champion adapter (run the base model)

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

@main
struct Agent {
    static func main() async throws {
        let modelId = ProcessInfo.processInfo.environment["MODEL"]
            ?? "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
        log("loading \(modelId) ...")
        let model = try await MLXLanguageModel.load(modelId: modelId)
        log("loaded.")

        // Hot-swap the current champion adapter (redeploy = pointer flip on frozen base).
        if ProcessInfo.processInfo.environment["BASE"] == nil,
           let registry = try? Registry(), let champ = registry.currentChampion() {
            do {
                try await model.loadAdapter(directory: URL(fileURLWithPath: champ.adapterPath))
                log("hot-swapped champion adapter \(champ.cycleId) (held-out loss \(champ.heldoutLoss)).")
            } catch {
                log("champion adapter load failed (\(error)) — running base model.")
            }
        }

        // Connect MCP servers (default: the user's own claude-utils).
        let host = MCPHost()
        let server = ProcessInfo.processInfo.environment["MCP_SERVER"]
            ?? "/Users/jason/localdev/ClaudeUtils/.build/release/ClaudeUtils"
        if FileManager.default.fileExists(atPath: server) {
            let names = try await host.connect(.init(command: server))
            log("connected MCP server — tools: \(names.joined(separator: ", "))")
        } else {
            log("no MCP server at \(server) — running tool-less")
        }

        let instructions = "You are a helpful local agent with tools. When a question needs "
            + "live system data, call the appropriate tool, then answer the user in plain words."

        print("SwiftLM agent ready. Enter a prompt (blank line or 'quit' to exit):")
        while let line = readLine(strippingNewline: true) {
            let prompt = line.trimmingCharacters(in: .whitespaces)
            if prompt.isEmpty || prompt == "quit" { break }
            do {
                let answer = try await model.runWithTools(prompt, host: host, instructions: instructions)
                print(answer)
            } catch {
                print("error: \(error)")
            }
        }
        await host.shutdown()
        log("bye.")
    }
}
