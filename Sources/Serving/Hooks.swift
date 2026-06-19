import Foundation

// ── Serve-time lifecycle hooks (the "session hooks like Claude Code" subsystem).
//
// The MCP serve loop (`runWithTools`) fires lifecycle EVENTS at four seams; HOOKS subscribe and return
// a HookResult that the loop applies. This lets Engram be invoked the way the MCP intends — an advise
// hook on UserPromptSubmit that calls `recall` through MCPHost like any tool — and the SAME mechanism
// later hosts the serve-time verifier (PostToolUse), memory-writes (Stop), and tool vetoes (PreToolUse).
//
// Two hook flavors share one `Hook` protocol: in-process Swift hooks (e.g. `EngramAdviseHook`) and
// Claude-Code-style `ExternalCommandHook`s (config-driven subprocesses). This file is the pure-Swift
// core — no MLX, no Process, no MCP — so it builds + tests under plain `swift build`/`swift test`.

/// Session context Claude Code hands hooks on stdin (so the SAME hook binaries — e.g. `memory-hooks
/// advise` — key memory/state the way they do under `claude -p`). Injected by the agent; `nil` in pure
/// unit tests (then the session fields are simply omitted from the payload).
public struct HookContext: Sendable, Equatable {
    public var sessionID: String?
    public var transcriptPath: String?
    public var cwd: String?
    public init(sessionID: String? = nil, transcriptPath: String? = nil, cwd: String? = nil) {
        self.sessionID = sessionID
        self.transcriptPath = transcriptPath
        self.cwd = cwd
    }
}

public enum HookEvent: Sendable, Equatable {
    case userPromptSubmit(prompt: String)
    case preToolUse(tool: String, argumentsJSON: String)
    case postToolUse(tool: String, result: String)
    case stop(answer: String, toolsCalled: [String])

    /// Stable event name (matches `ExternalCommandHook.Spec.event` and Claude Code's hook event names).
    public var name: String {
        switch self {
        case .userPromptSubmit: return "UserPromptSubmit"
        case .preToolUse:       return "PreToolUse"
        case .postToolUse:      return "PostToolUse"
        case .stop:             return "Stop"
        }
    }

    /// JSON handed to an external command hook on stdin. Claude-Code-FAITHFUL so the same hook binaries
    /// work unchanged: snake_case `hook_event_name`, `tool_name`/`tool_input` (`tool_input` embedded as a
    /// JSON OBJECT when the argument string parses, else raw), and the session `context` Claude provides.
    public func jsonPayload(context: HookContext? = nil) -> String {
        var obj: [String: Any] = ["hook_event_name": name]
        if let c = context {
            if let s = c.sessionID { obj["session_id"] = s }
            if let t = c.transcriptPath { obj["transcript_path"] = t }
            if let w = c.cwd { obj["cwd"] = w }
        }
        switch self {
        case .userPromptSubmit(let p):
            obj["prompt"] = p
        case .preToolUse(let t, let a):
            obj["tool_name"] = t
            obj["tool_input"] = Self.jsonObjectOrString(a)
        case .postToolUse(let t, let r):
            obj["tool_name"] = t
            obj["tool_response"] = r
        case .stop(let ans, let tc):
            obj["stop_hook_active"] = true
            obj["answer"] = ans
            obj["toolsCalled"] = tc
        }
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else { return "{}" }
        return s
    }

    /// Parse a JSON-string argument into an object/array for embedding; fall back to the raw string.
    private static func jsonObjectOrString(_ s: String) -> Any {
        guard let data = s.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: data) else { return s }
        return any
    }
}

/// What a hook tells the loop to do. Fields are event-appropriate: `additionalContext` for
/// UserPromptSubmit, `decision` for PreToolUse, `replacementResult` for PostToolUse; Stop hooks return
/// `.passthrough` (side-effect only).
public struct HookResult: Sendable, Equatable {
    public enum ToolDecision: Sendable, Equatable { case allow; case deny(reason: String) }

    public var additionalContext: String?
    public var decision: ToolDecision
    public var replacementResult: String?

    public init(additionalContext: String? = nil,
                decision: ToolDecision = .allow,
                replacementResult: String? = nil) {
        self.additionalContext = additionalContext
        self.decision = decision
        self.replacementResult = replacementResult
    }

    public static let passthrough = HookResult()

    /// Parse an external hook's stdout JSON into a result. Claude-Code-faithful + lenient:
    ///  • `additionalContext` is read from the TOP level OR from `hookSpecificOutput.additionalContext`
    ///    — the shape `memory-hooks advise` / SessionStart emit. Without the nested read, Engram grounding
    ///    is silently dropped (it IS nested), so this is the crux of "engram operates as it does now".
    ///  • a deny is recognized from `decision` ∈ {deny, block}, `hookSpecificOutput.permissionDecision`,
    ///    or `continue == false`; reason from `reason` / `permissionDecisionReason` / `stopReason`.
    ///  • `replacementResult` (our own PostToolUse transform) is read from the top level.
    /// Missing/unknown fields → defaults; non-JSON → `.passthrough`.
    public static func decode(_ json: String) -> HookResult {
        // A hook's stdout can carry trailing lines AFTER the JSON object — `memory-hooks advise` appends a
        // `Log file path: …` line — which strict JSON parsing rejects. Extract the first balanced `{…}`
        // object first; otherwise Engram grounding is silently dropped on every turn.
        guard let data = firstJSONObject(json),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .passthrough
        }
        let hso = obj["hookSpecificOutput"] as? [String: Any]
        let addCtx = (obj["additionalContext"] as? String) ?? (hso?["additionalContext"] as? String)

        var decision: ToolDecision = .allow
        if let d = (obj["decision"] as? String)?.lowercased(), d == "deny" || d == "block" {
            decision = .deny(reason: (obj["reason"] as? String) ?? "denied by hook")
        } else if let pd = (hso?["permissionDecision"] as? String)?.lowercased(), pd == "deny" || pd == "block" {
            decision = .deny(reason: (hso?["permissionDecisionReason"] as? String) ?? "denied by hook")
        } else if let cont = obj["continue"] as? Bool, cont == false {
            decision = .deny(reason: (obj["stopReason"] as? String) ?? "hook requested stop")
        }

        return HookResult(additionalContext: addCtx,
                          decision: decision,
                          replacementResult: obj["replacementResult"] as? String)
    }

    /// Extract the first balanced top-level `{…}` JSON object from a string (braces inside string literals
    /// are ignored), so trailing lines after the JSON (e.g. a `Log file path:` line) don't break parsing.
    /// Returns nil when no complete object is present.
    static func firstJSONObject(_ s: String) -> Data? {
        guard let start = s.firstIndex(of: "{") else { return nil }
        var depth = 0, inString = false, escaped = false
        var i = start
        while i < s.endIndex {
            let c = s[i]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else if c == "\"" {
                inString = true
            } else if c == "{" {
                depth += 1
            } else if c == "}" {
                depth -= 1
                if depth == 0 { return String(s[start...i]).data(using: .utf8) }
            }
            i = s.index(after: i)
        }
        return nil
    }
}

public protocol Hook: Sendable {
    func handle(_ event: HookEvent) async -> HookResult
}

/// Fires an event to every hook in order and FOLDS their results: `additionalContext` is concatenated
/// (every hook can contribute), the FIRST `.deny` wins (any veto blocks the tool), and the LAST
/// `replacementResult` wins (later hooks override earlier transforms).
public struct HookChain: Sendable {
    let hooks: [Hook]
    public init(_ hooks: [Hook]) { self.hooks = hooks }
    public var isEmpty: Bool { hooks.isEmpty }

    public func fire(_ event: HookEvent) async -> HookResult {
        var contextParts: [String] = []
        var decision: HookResult.ToolDecision = .allow
        var replacement: String? = nil
        for hook in hooks {
            let r = await hook.handle(event)
            if let c = r.additionalContext, !c.isEmpty { contextParts.append(c) }
            if case .deny = r.decision, case .allow = decision { decision = r.decision }  // first deny wins
            if let rep = r.replacementResult { replacement = rep }                         // last replace wins
        }
        return HookResult(additionalContext: contextParts.isEmpty ? nil : contextParts.joined(separator: "\n\n"),
                          decision: decision,
                          replacementResult: replacement)
    }
}
