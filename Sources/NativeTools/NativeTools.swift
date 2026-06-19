import Foundation

// ── Native (in-process) agent tools: the basics an agent needs WITHOUT an external MCP server —
// Read / Write / Edit / Glob / Grep / Bash. Pure Foundation (no MLX) so it builds + tests under plain
// `swift build`/`swift test`; MLXBackend adapts these into the model's tool surface (MCPHost) and the
// `agent` CLI registers them. Each tool declares an OpenAI-fn-schema `parameters` object whose
// `asSendableDictionary()` is directly usable as part of an MLX `ToolSpec` (= `[String: any Sendable]`),
// so there's no coupling back to MLX types here.

public enum NativeToolError: Error, CustomStringConvertible, Equatable {
    case missingArgument(String)
    case io(String)
    public var description: String {
        switch self {
        case .missingArgument(let a): return "missing required argument: \(a)"
        case .io(let m):              return m
        }
    }
}

/// Decoded JSON arguments for a native tool call — keeps NativeTools free of MLX's `JSONValue`.
public struct ToolArguments: @unchecked Sendable {
    private let raw: [String: Any]
    public init(_ raw: [String: Any]) { self.raw = raw }
    public init(json: String) {
        if let data = json.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data),
           let dict = obj as? [String: Any] {
            raw = dict
        } else {
            raw = [:]
        }
    }
    public func string(_ key: String) -> String? { raw[key] as? String }
    public func bool(_ key: String) -> Bool? {
        if let b = raw[key] as? Bool { return b }
        if let s = raw[key] as? String { return s == "true" }
        return nil
    }
    public func int(_ key: String) -> Int? {
        if let i = raw[key] as? Int { return i }
        if let d = raw[key] as? Double { return Int(d) }
        if let s = raw[key] as? String { return Int(s) }
        return nil
    }
    public func requireString(_ key: String) throws -> String {
        guard let v = string(key), !v.isEmpty else { throw NativeToolError.missingArgument(key) }
        return v
    }
}

/// A minimal OpenAI-fn-schema "parameters" object (flat string/number/boolean props).
public struct JSONSchemaObject: Sendable {
    public struct Property: Sendable {
        public let type: String          // "string" | "integer" | "number" | "boolean"
        public let description: String
        public init(_ type: String, _ description: String) { self.type = type; self.description = description }
    }
    public let properties: [(name: String, prop: Property)]   // ordered for stable schema output
    public let required: [String]
    public init(_ properties: [(name: String, prop: Property)], required: [String] = []) {
        self.properties = properties; self.required = required
    }
    /// The "parameters" object as a Sendable dict — directly droppable into an MLX `ToolSpec`.
    public func asSendableDictionary() -> [String: any Sendable] {
        var props: [String: any Sendable] = [:]
        for (name, p) in properties {
            props[name] = ["type": p.type, "description": p.description] as [String: any Sendable]
        }
        return ["type": "object", "properties": props, "required": required]
    }
}

/// A built-in, in-process tool the model can call alongside MCP tools.
public protocol NativeTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: JSONSchemaObject { get }
    func run(_ args: ToolArguments) async throws -> String
}

/// Holds the built-in tools, exposes their fn-schema `specs()`, and dispatches a call by name. The
/// dispatch never throws — errors come back as an `ERROR: …` string (the same convention MCPHost uses),
/// so a tool failure feeds back to the model as an observation instead of aborting the turn.
public struct NativeToolRegistry: Sendable {
    public let tools: [any NativeTool]
    public init(_ tools: [any NativeTool]) { self.tools = tools }

    /// The standard agent toolset (Claude-Code-shaped names so transcripts/habits transfer). When a
    /// `subagentRunner` is injected, the Claude-faithful `Task` tool is included (the sub-agent capability);
    /// it's omitted by default so the pure module needs no backend.
    public static func standard(subagentRunner: SubagentRunner? = nil) -> NativeToolRegistry {
        var tools: [any NativeTool] = [ReadFileTool(), WriteFileTool(), EditFileTool(), GlobTool(),
                                       GrepTool(), BashTool(), WebFetchTool(), ExitPlanModeTool()]
        if let subagentRunner { tools.append(TaskTool(runner: subagentRunner)) }
        return .init(tools)
    }

    public var names: [String] { tools.map(\.name) }
    public func tool(named name: String) -> (any NativeTool)? { tools.first { $0.name == name } }

    /// OpenAI fn-schema specs (the `ToolSpec` shape MCPHost feeds the model).
    public func specs() -> [[String: any Sendable]] {
        tools.map {
            ["type": "function",
             "function": ["name": $0.name, "description": $0.description,
                          "parameters": $0.parameters.asSendableDictionary()] as [String: any Sendable]]
        }
    }

    /// Dispatch by name with raw JSON args; `nil` if this registry doesn't own the tool (caller falls
    /// back to MCP). Errors are returned as `ERROR: …` text (not thrown).
    public func dispatch(name: String, argsJSON: String) async -> String? {
        guard let t = tool(named: name) else { return nil }
        do { return try await t.run(ToolArguments(json: argsJSON)) }
        catch { return "ERROR: \(error)" }
    }
}
