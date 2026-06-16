import Foundation
import MLXLMCommon
import MCP
import System

// MCP tool layer for the MLX backend — the validated S6 (toolspike) loop, productized.
// MCPHost spawns/connects MCP servers and aggregates their tools; `runWithTools`
// drives the round-capped agent loop (emit tool_call -> dispatch -> inject -> finish).

/// Convert an MCP `inputSchema` (Value) into a Sendable JSON tree for a ToolSpec.
func sendableJSON(_ v: MCP.Value) -> any Sendable {
    switch v {
    case .null: return "null"
    case .bool(let b): return b
    case .int(let i): return i
    case .double(let d): return d
    case .string(let s): return s
    case .data(_, let d): return d.base64EncodedString()
    case .array(let a): return a.map { sendableJSON($0) }
    case .object(let o): return o.mapValues { sendableJSON($0) }
    }
}

/// Hosts one or more MCP servers, exposes their tools to the model, and dispatches calls.
public actor MCPHost {
    public struct ServerConfig: Sendable {
        public let command: String
        public let args: [String]
        public init(command: String, args: [String] = []) {
            self.command = command
            self.args = args
        }
    }

    private var processes: [Process] = []
    private var dispatchMap: [String: Client] = [:]      // tool name -> owning client
    public private(set) var specs: [ToolSpec] = []       // OpenAI fn-schema for the model

    public init() {}

    /// Spawn an MCP server over stdio, connect, and register its tools.
    @discardableResult
    public func connect(_ config: ServerConfig) async throws -> [String] {
        let inPipe = Pipe(), outPipe = Pipe()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: config.command)
        proc.arguments = config.args
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        try proc.run()
        processes.append(proc)

        let transport = StdioTransport(
            input: FileDescriptor(rawValue: outPipe.fileHandleForReading.fileDescriptor),
            output: FileDescriptor(rawValue: inPipe.fileHandleForWriting.fileDescriptor))
        let client = Client(name: "swiftlm-agent", version: "0.1.0")
        _ = try await client.connect(transport: transport)

        let (tools, _) = try await client.listTools()
        var names: [String] = []
        for t in tools {
            names.append(t.name)
            dispatchMap[t.name] = client
            let params = (sendableJSON(t.inputSchema) as? [String: any Sendable])
                ?? ["type": "object", "properties": [String: any Sendable]()]
            specs.append([
                "type": "function",
                "function": [
                    "name": t.name,
                    "description": t.description ?? "",
                    "parameters": params,
                ] as [String: any Sendable],
            ])
        }
        return names
    }

    /// Dispatch a model-emitted tool call to the owning MCP server; return result text.
    public func dispatch(name: String, arguments: [String: JSONValue]) async throws -> String {
        guard let client = dispatchMap[name] else { return "ERROR: unknown tool \(name)" }
        let argsAny = arguments.mapValues { $0.anyValue }
        let mcpArgs: [String: MCP.Value]? = argsAny.isEmpty
            ? nil
            : try? JSONDecoder().decode([String: MCP.Value].self,
                                        from: JSONSerialization.data(withJSONObject: argsAny))
        let (content, isError) = try await client.callTool(name: name, arguments: mcpArgs)
        let text = content.compactMap { c -> String? in
            if case let .text(t, _, _) = c { return t }
            return nil
        }.joined(separator: "\n")
        return (isError == true ? "ERROR: " : "") + text
    }

    public func shutdown() {
        for p in processes where p.isRunning { p.terminate() }
        processes.removeAll()
    }
}

extension MLXLanguageModel {
    /// Round-capped agent loop (S6): the model emits tool calls, we dispatch them via
    /// the MCP host, inject results, and continue. Once a result is in hand we drop the
    /// tools so the model is forced to produce a final text answer (no infinite re-call).
    @discardableResult
    public func runWithTools(
        _ prompt: String,
        host: MCPHost,
        instructions: String? = nil,
        maxRounds: Int = 5,
        maxTokensPerTurn: Int = 256
    ) async throws -> String {
        try await runWithToolsTracked(
            prompt, host: host, instructions: instructions,
            maxRounds: maxRounds, maxTokensPerTurn: maxTokensPerTurn).answer
    }

    /// Like `runWithTools`, but also returns the names of every tool the model invoked —
    /// the signal the correctness (tool-pass@1) gate scores against.
    public func runWithToolsTracked(
        _ prompt: String,
        host: MCPHost,
        instructions: String? = nil,
        maxRounds: Int = 5,
        maxTokensPerTurn: Int = 256
    ) async throws -> (answer: String, toolsCalled: [String]) {
        let specs = await host.specs
        var params = GenerateParameters(maxTokens: maxTokensPerTurn, temperature: 0.0)
        params.repetitionPenalty = 1.15
        params.repetitionContextSize = 20
        let session = ChatSession(container, instructions: instructions, generateParameters: params, tools: specs)
        var cont: [Chat.Message]? = nil
        var answer = ""
        var toolsCalled: [String] = []
        var dispatched = false
        var round = 0
        while round < maxRounds {
            if dispatched { session.tools = nil }   // force a final text answer
            let stream = (cont == nil)
                ? session.streamDetails(to: prompt)
                : session.streamDetails(to: cont!)
            var text = ""
            var toolCalls: [ToolCall] = []
            for try await g in stream {
                if let tc = g.toolCall { toolCalls.append(tc) }
                else if let ch = g.chunk { text += ch }
            }
            round += 1
            if toolCalls.isEmpty { answer = text; break }
            var msgs: [Chat.Message] = []
            for tc in toolCalls {
                toolsCalled.append(tc.function.name)
                msgs.append(.tool(try await host.dispatch(
                    name: tc.function.name, arguments: tc.function.arguments)))
            }
            dispatched = true
            cont = msgs
        }
        return (answer.isEmpty ? "(hit round cap without final text)" : answer, toolsCalled)
    }
}
