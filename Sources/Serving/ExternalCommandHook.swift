import Foundation

// ── Claude-Code-style external-command hooks: config-driven shell commands that fire on lifecycle
// events, exchanging JSON over stdin/stdout. The "de-Engram" path: instead of a special-cased in-process
// EngramAdviseHook, an MLX model loads the user's OWN `~/.claude/settings.json` and runs the SAME hooks
// `claude -p` does — so the Engram `memory-hooks advise` UserPromptSubmit hook (and every other hook)
// fires identically for a local model. Each spec's `run` (the subprocess spawn) is INJECTED so this stays
// pure-Swift + testable — the agent provides a real Foundation.Process runner.

public struct ExternalCommandHook: Hook {
    public struct Spec: Sendable, Codable, Equatable {
        public let event: String        // matches HookEvent.name ("UserPromptSubmit", "PreToolUse", …)
        public let command: String      // shell command line (Claude format) or executable path
        public let args: [String]       // extra argv (empty for Claude shell-string commands)
        public let matcher: String?     // tool-name regex for Pre/PostToolUse; nil/"" = match all tools
        public let timeoutSeconds: Int? // hard per-hook timeout (Claude `timeout`, seconds)
        public let runAsync: Bool       // fire-and-forget (Claude `async`): output ignored, never blocks

        public init(event: String, command: String, args: [String] = [],
                    matcher: String? = nil, timeoutSeconds: Int? = nil, runAsync: Bool = false) {
            self.event = event; self.command = command; self.args = args
            self.matcher = matcher; self.timeoutSeconds = timeoutSeconds; self.runAsync = runAsync
        }

        // Custom Codable: the legacy `~/.swiftlm/hooks.json` shape ({event,command,args}) decodes with the
        // new fields defaulted (matcher nil, timeout nil, async false). Claude's keys are `timeout`/`async`.
        enum CodingKeys: String, CodingKey {
            case event, command, args, matcher
            case timeoutSeconds = "timeout"
            case runAsync = "async"
        }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            event = try c.decode(String.self, forKey: .event)
            command = try c.decode(String.self, forKey: .command)
            args = try c.decodeIfPresent([String].self, forKey: .args) ?? []
            matcher = try c.decodeIfPresent(String.self, forKey: .matcher)
            timeoutSeconds = try c.decodeIfPresent(Int.self, forKey: .timeoutSeconds)
            runAsync = try c.decodeIfPresent(Bool.self, forKey: .runAsync) ?? false
        }
        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(event, forKey: .event)
            try c.encode(command, forKey: .command)
            try c.encode(args, forKey: .args)
            try c.encodeIfPresent(matcher, forKey: .matcher)
            try c.encodeIfPresent(timeoutSeconds, forKey: .timeoutSeconds)
            if runAsync { try c.encode(runAsync, forKey: .runAsync) }
        }
    }

    /// Runs `command` (with `args`), writing `stdinJSON` to stdin, returning stdout. `timeoutSeconds`
    /// lets the runner kill a hung hook (Claude's per-hook `timeout`). Injected → pure-Swift + testable.
    public typealias Runner = @Sendable (_ command: String, _ args: [String],
                                         _ stdinJSON: String, _ timeoutSeconds: Int?) async -> String

    let spec: Spec
    let run: Runner
    let context: HookContext?

    public init(spec: Spec, run: @escaping Runner, context: HookContext? = nil) {
        self.spec = spec
        self.run = run
        self.context = context
    }

    public func handle(_ event: HookEvent) async -> HookResult {
        guard event.name == spec.event else { return .passthrough }              // wrong event
        guard Self.matches(matcher: spec.matcher, event: event) else { return .passthrough }  // matcher gate
        let payload = event.jsonPayload(context: context)
        if spec.runAsync {
            // Claude `async`: detached fire-and-forget (e.g. notification pings) — never blocks the turn,
            // output ignored, contributes nothing to the result fold.
            let command = spec.command, args = spec.args, timeout = spec.timeoutSeconds, run = self.run
            Task.detached { _ = await run(command, args, payload, timeout) }
            return .passthrough
        }
        let out = await run(spec.command, spec.args, payload, spec.timeoutSeconds)
        return HookResult.decode(out)
    }

    /// A Pre/PostToolUse hook fires only when its `matcher` (a regex over the tool name, Claude-style)
    /// matches; an empty/nil matcher matches every tool. Non-tool events (UserPromptSubmit/Stop) have no
    /// tool dimension → the matcher is N/A and they always fire.
    static func matches(matcher: String?, event: HookEvent) -> Bool {
        let tool: String
        switch event {
        case .preToolUse(let t, _), .postToolUse(let t, _): tool = t
        default: return true
        }
        guard let m = matcher, !m.isEmpty else { return true }
        if let re = try? NSRegularExpression(pattern: m) {
            return re.firstMatch(in: tool, range: NSRange(tool.startIndex..., in: tool)) != nil
        }
        return tool == m   // invalid regex → exact-match fallback
    }
}

/// Loads external-hook specs and builds hooks with an injected runner. Two sources:
///  • `~/.swiftlm/hooks.json` (SwiftLM's own flat shape) via `decode`/`load`.
///  • `~/.claude/settings.json` (Claude's nested shape) via `loadClaudeSettings` — the de-Engram path.
public struct HookConfig: Sendable, Codable, Equatable {
    public let hooks: [ExternalCommandHook.Spec]
    public init(hooks: [ExternalCommandHook.Spec]) { self.hooks = hooks }

    /// Decode SwiftLM's flat config from raw JSON (testable without touching the filesystem).
    /// Shape: `{ "hooks": [ { "event": "PostToolUse", "command": "/path/log.sh", "args": [] } ] }`.
    public static func decode(_ json: String) -> HookConfig {
        guard let data = json.data(using: .utf8),
              let cfg = try? JSONDecoder().decode(HookConfig.self, from: data) else {
            return HookConfig(hooks: [])
        }
        return cfg
    }

    /// Load SwiftLM's flat config from a path; missing/unreadable/invalid → no external hooks (not an error).
    public static func load(path: String) -> HookConfig {
        guard let data = FileManager.default.contents(atPath: path),
              let json = String(data: data, encoding: .utf8) else { return HookConfig(hooks: []) }
        return decode(json)
    }

    /// Load Claude Code's `~/.claude/settings.json` hook config (the de-Engram path: the SAME config that
    /// drives `claude -p`, so e.g. the Engram `memory-hooks advise` UserPromptSubmit hook fires identically
    /// for local MLX models). Parses the NESTED Claude shape:
    ///   { "hooks": { "<Event>": [ { "matcher"?: "...", "hooks": [ { "type":"command", "command":"...",
    ///                                                              "timeout"?: N, "async"?: bool } ] } ] } }
    /// Only the four events SwiftLM models (UserPromptSubmit/PreToolUse/PostToolUse/Stop) are mapped; every
    /// other event (SessionStart, PreCompact, Notification, SubagentStop, …) is ignored. A missing/invalid
    /// file → no external hooks (not an error). Spec order is deterministic (fixed event order, array order
    /// within an event) so the HookChain folds `additionalContext` reproducibly.
    public static func loadClaudeSettings(path: String) -> HookConfig {
        guard let data = FileManager.default.contents(atPath: path),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let hooksObj = root["hooks"] as? [String: Any] else { return HookConfig(hooks: []) }
        var specs: [ExternalCommandHook.Spec] = []
        for event in ["UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"] {
            guard let groups = hooksObj[event] as? [[String: Any]] else { continue }
            for group in groups {
                let matcher = group["matcher"] as? String   // nil or "" → match all
                guard let hookList = group["hooks"] as? [[String: Any]] else { continue }
                for h in hookList {
                    guard (h["type"] as? String) == "command",
                          let command = h["command"] as? String else { continue }
                    specs.append(.init(event: event, command: command, args: [],
                                       matcher: matcher,
                                       timeoutSeconds: h["timeout"] as? Int,
                                       runAsync: (h["async"] as? Bool) ?? false))
                }
            }
        }
        return HookConfig(hooks: specs)
    }

    /// Build the external hooks, injecting the subprocess runner + optional session context (session_id /
    /// transcript_path / cwd Claude hands hooks on stdin).
    public func externalHooks(context: HookContext? = nil,
                              run: @escaping ExternalCommandHook.Runner) -> [ExternalCommandHook] {
        hooks.map { ExternalCommandHook(spec: $0, run: run, context: context) }
    }
}
