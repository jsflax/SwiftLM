import Foundation

// ── Claude-Code agent definitions (`~/.claude/agents/*.md`).
//
// A sub-agent type (e.g. `session-learner`) is defined by a markdown file with a YAML-ish `---` frontmatter
// block (name / description / tools / model / maxTurns) followed by the system-prompt body. Loading the
// user's REAL definitions — instead of a SwiftLM-bespoke prompt — is what makes a sub-agent a faithful
// drop-in: `MLXSubagentRunner` runs the loaded `systemPrompt` with the declared `tools`. Pure Foundation
// (file read + a tiny frontmatter parser), so it builds + tests under plain `swift build`/`swift test`.

public struct AgentDefinition: Sendable, Equatable {
    public var name: String
    public var description: String
    /// Allowed tool names (Claude's `tools:` field). `nil` = inherit all tools. `*` / "All tools" → nil too.
    public var tools: [String]?
    /// Advisory model tier ("sonnet"/"haiku"/…). SwiftLM runs the one loaded local model regardless.
    public var model: String?
    public var maxTurns: Int?
    public var systemPrompt: String

    public init(name: String, description: String = "", tools: [String]? = nil,
                model: String? = nil, maxTurns: Int? = nil, systemPrompt: String) {
        self.name = name; self.description = description; self.tools = tools
        self.model = model; self.maxTurns = maxTurns; self.systemPrompt = systemPrompt
    }
}

public enum AgentDefinitionLoader {
    /// Load every `*.md` agent definition under `dir` (default `~/.claude/agents`), keyed by `name`.
    /// Files that fail to parse (no frontmatter) are skipped, not fatal.
    public static func load(dir: String = ("~/.claude/agents" as NSString).expandingTildeInPath)
        -> [String: AgentDefinition] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir) else { return [:] }
        var out: [String: AgentDefinition] = [:]
        for file in entries where file.hasSuffix(".md") {
            let path = (dir as NSString).appendingPathComponent(file)
            guard let text = try? String(contentsOfFile: path, encoding: .utf8),
                  let def = parse(text, fallbackName: (file as NSString).deletingPathExtension)
            else { continue }
            out[def.name] = def
        }
        return out
    }

    /// Parse one agent `.md` (`---` frontmatter + markdown body). `nil` when there's no frontmatter block.
    public static func parse(_ text: String, fallbackName: String) -> AgentDefinition? {
        let lines = text.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return nil }
        // Find the closing `---`.
        guard let close = lines.dropFirst().firstIndex(where: { $0.trimmingCharacters(in: .whitespaces) == "---" })
        else { return nil }

        var fields: [String: String] = [:]
        for line in lines[1..<close] {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            if !key.isEmpty { fields[key] = value }
        }
        let body = lines[(close + 1)...].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)

        return AgentDefinition(
            name: fields["name"] ?? fallbackName,
            description: fields["description"] ?? "",
            tools: parseTools(fields["tools"]),
            model: fields["model"],
            maxTurns: fields["maxTurns"].flatMap { Int($0) },
            systemPrompt: body)
    }

    /// `tools:` is a comma-separated list; absent / `*` / "All tools" means "inherit all" (→ nil).
    static func parseTools(_ raw: String?) -> [String]? {
        guard let raw, !raw.isEmpty else { return nil }
        if raw == "*" || raw.lowercased() == "all tools" || raw.lowercased() == "all" { return nil }
        let names = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        return names.isEmpty ? nil : names
    }
}
