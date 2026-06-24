import Foundation

// ── ROOM TRANSCRIPT HARVESTER (Track 2): mine Orbital room `.lattice` files for role-discipline DPO data.
// WON = a strong-model (cloud Claude/codex/gemini) role turn in the REAL Orbital agentic format (gold). LOST = a
// small local-model failure (narrate-without-act / errored / missed-tool-error). Pure Foundation + the `sqlite3`
// CLI (the lattice is reactive SQLite); no model, no Lattice dependency — runs offline as a data-prep step.
//
// The linchpin is IN-DISTRIBUTION format: every example must emit the tool calls in the EXACT native names the
// local model serves (read_file/write_file/edit_file/bash/…). Cloud-Claude turns may carry Claude-CLI names
// (Read/Bash/Edit) — those are NORMALIZED (M2) or the trace is flagged, so we never re-teach the wrong names (the
// documented 100%→0% tool-use regression). The M1 stale-trace gate (re-verify a LOST turn on the CURRENT 122B
// before training) is a downstream step that needs the model — this harvester just labels + records the signals.

/// One harvested agent turn, structured for downstream pairing + (serve-identical) re-render.
public struct HarvestedTurn: Sendable {
    public enum Label: String, Sendable { case won, lost, skip }
    public struct ToolUse: Sendable {
        public let rawName: String     // as recorded in the lattice (may be a Claude-CLI name)
        public let name: String        // normalized to the native Orbital name (M2)
        public let input: String       // JSON args
        public let result: String
        public let status: String      // ok / errored / …
        public let wasError: Bool
    }
    public let room: String
    public let agentName: String
    public let role: String
    public let modelId: String
    public let isLocal: Bool
    public let isReferee: Bool
    public let systemPrompt: String
    public let priorContext: [(role: String, text: String)]   // earlier messages, in order (capped)
    public let outputText: String                             // this turn's visible message text
    public let status: String                                 // ChatMessage.status (done/errored/…)
    public let toolUses: [ToolUse]
    public let label: Label
    public let reason: String
    /// A flat `TrainPair`-compatible view (stop-gap until the pre-render-to-tokens pass renders structured turns).
    public var trainPairUser: String {
        let head = systemPrompt.isEmpty ? "" : "[system] \(systemPrompt)\n\n"
        let ctx = priorContext.map { "[\($0.role)] \($0.text)" }.joined(separator: "\n\n")
        return head + ctx
    }
    public var trainPairAssistant: String {
        let calls = toolUses.map { "<tool_call>\($0.name) \($0.input)</tool_call>" }.joined(separator: "\n")
        return outputText + (calls.isEmpty ? "" : "\n" + calls)
    }
}

public enum RoomTranscriptHarvester {

    /// Default room data sources (from the lattice inventory). Gold = strong-model role turns; failure = local.
    public static let goldRooms = ["mlx-mount", "orbital", "claude-control", "a2demo", "planbuild", "team-fixture"]
    public static let failureRooms = ["scratch-b1gate", "lru2", "lru-80b", "chess122team-branch2", "chess122team", "lru122"]

    /// M2 — map a Claude-CLI / common tool name to the native Orbital name. Unknown names pass through unchanged
    /// (so MCP/domain tools and the native names are preserved); the caller flags any residual non-native name.
    public static func normalizeToolName(_ raw: String) -> String {
        switch raw {
        case "Read": return "read_file"
        case "Write": return "write_file"
        case "Edit", "MultiEdit": return "edit_file"
        case "Bash", "BashOutput": return "bash"
        case "Glob": return "glob"
        case "Grep": return "grep"
        case "WebFetch": return "web_fetch"
        default: return raw
        }
    }

    static let nativeFileExecTools: Set<String> = ["read_file", "write_file", "edit_file", "bash", "glob", "grep", "web_fetch"]
    static let mutatingTools: Set<String> = ["write_file", "edit_file", "bash"]
    static let routingTools: Set<String> = ["handoff", "consult", "done"]
    /// Claude-CLI tools with NO Orbital-local equivalent — a local agent can't emit these, so a gold turn
    /// DOMINATED by them isn't in-distribution (even though the surrounding role behavior is still gold).
    static let claudeOnlyTools: Set<String> = ["Agent", "ToolSearch", "TaskCreate", "TaskUpdate", "TaskList",
        "TaskGet", "Skill", "ExitPlanMode", "AskUserQuestion", "todo_list", "TodoWrite", "NotebookEdit", "SlashCommand"]

    enum ToolCategory { case native, routing, mcp, claudeOnly, other }
    static func category(_ normalized: String, raw: String) -> ToolCategory {
        if nativeFileExecTools.contains(normalized) { return .native }
        if routingTools.contains(normalized) || raw.hasPrefix("mcp__orbital__") { return .routing }
        if raw.hasPrefix("mcp__") { return .mcp }
        if claudeOnlyTools.contains(raw) { return .claudeOnly }
        return .other
    }
    /// A turn is "in-format" gold iff its tool uses are NOT dominated by Claude-only tools (the role reasoning +
    /// native/routing/mcp calls transfer; a turn that is mostly Agent/ToolSearch/Task does not).
    static func inFormat(_ t: HarvestedTurn) -> Bool {
        let cats = t.toolUses.map { category($0.name, raw: $0.rawName) }
        let claudeOnly = cats.filter { $0 == .claudeOnly }.count
        return t.toolUses.isEmpty || Double(claudeOnly) / Double(t.toolUses.count) < 0.34
    }

    // ── sqlite3 -json bridge ────────────────────────────────────────────────────────────────────────────────
    /// Run a read-only query against a lattice (COPIED to /tmp first so the original — possibly WAL-dirty — is
    /// never touched) and decode the JSON array of row dicts. Returns [] on any failure (offline best-effort).
    static func query(_ latticeCopyPath: String, _ sql: String) -> [[String: Any]] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["sqlite3", "-json", latticeCopyPath, sql]
        let out = Pipe(); p.standardOutput = out; p.standardError = Pipe()
        do { try p.run() } catch { return [] }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard !data.isEmpty, let obj = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return obj
    }

    static func str(_ row: [String: Any], _ key: String) -> String {
        if let s = row[key] as? String { return s }
        if let n = row[key] as? NSNumber { return n.stringValue }
        return ""
    }
    static func intVal(_ row: [String: Any], _ key: String) -> Int { (row[key] as? NSNumber)?.intValue ?? Int(str(row, key)) ?? 0 }

    /// Copy a room lattice (+ -wal/-shm) into /tmp and return the copy path (so queries never touch the original).
    static func copyLattice(_ room: String) -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let src = home.appending(path: "Library/Application Support/Orbital/rooms/\(room).lattice")
        guard FileManager.default.fileExists(atPath: src.path) else { return nil }
        let dst = "/tmp/harvest-\(room).lattice"
        let fm = FileManager.default
        for suffix in ["", "-wal", "-shm"] {
            let s = src.path + suffix, d = dst + suffix
            try? fm.removeItem(atPath: d)
            try? fm.copyItem(atPath: s, toPath: d)
        }
        return FileManager.default.fileExists(atPath: dst) ? dst : nil
    }

    // ── Harvest one room → labeled turns ────────────────────────────────────────────────────────────────────
    public static func harvest(room: String, maxPriorContext: Int = 12) -> [HarvestedTurn] {
        guard let path = copyLattice(room) else { return [] }

        // Agents: globalId → (name, role, modelId, isLocal, isReferee, systemPrompt)
        struct AgentInfo { let name: String; let role: String; let modelId: String; let isLocal: Bool; let isReferee: Bool; let system: String }
        var agents: [String: AgentInfo] = [:]
        for r in query(path, "SELECT globalId,name,role,modelId,isLocal,isReferee,systemPrompt FROM Agent;") {
            agents[str(r, "globalId")] = AgentInfo(
                name: str(r, "name"), role: str(r, "role"), modelId: str(r, "modelId"),
                isLocal: intVal(r, "isLocal") != 0, isReferee: intVal(r, "isReferee") != 0, system: str(r, "systemPrompt"))
        }

        // Messages with author, time-ordered.
        struct Msg { let globalId: String; let text: String; let status: String; let author: String; let createdAt: Double }
        let msgRows = query(path,
            "SELECT c.globalId AS gid, c.text AS text, c.status AS status, c.createdAt AS ts, " +
            "(SELECT a.rhs FROM _ChatMessage_Agent_author a WHERE a.lhs = c.globalId LIMIT 1) AS author " +
            "FROM ChatMessage c ORDER BY c.createdAt ASC;")
        let msgs = msgRows.map { Msg(globalId: str($0, "gid"), text: str($0, "text"), status: str($0, "status"),
                                     author: str($0, "author"), createdAt: (($0["ts"] as? NSNumber)?.doubleValue ?? 0)) }

        // Tool events grouped by their owning message.
        var toolsByMsg: [String: [HarvestedTurn.ToolUse]] = [:]
        for r in query(path,
            "SELECT t.name AS name, t.input AS input, t.result AS result, t.status AS status, " +
            "(SELECT m.lhs FROM _ChatMessage_ToolEvent_tools m WHERE m.rhs = t.globalId LIMIT 1) AS msg " +
            "FROM ToolEvent t;") {
            let msg = str(r, "msg"); guard !msg.isEmpty else { continue }
            let raw = str(r, "name"); let status = str(r, "status")
            let wasError = status.lowercased().contains("error") || str(r, "result").hasPrefix("ERROR")
            toolsByMsg[msg, default: []].append(.init(rawName: raw, name: normalizeToolName(raw),
                input: str(r, "input"), result: str(r, "result"), status: status, wasError: wasError))
        }

        var out: [HarvestedTurn] = []
        for (i, m) in msgs.enumerated() {
            guard let ag = agents[m.author], !ag.role.isEmpty, ag.role != "operator" else { continue }   // skip the operator/un-authored
            let tools = toolsByMsg[m.globalId] ?? []
            let prior = Array(msgs[0..<i].suffix(maxPriorContext)).map { pm -> (String, String) in
                (agents[pm.author]?.name ?? "user", pm.text)
            }
            let mutating = tools.contains { mutatingTools.contains($0.name) }
            let anyCall = !tools.isEmpty
            let anyError = tools.contains { $0.wasError }
            let done = m.status == "done" || m.status == "complete"
            let errored = m.status == "errored"
            let isBuilder = ag.role.contains("builder")

            // Labeling (transparent; downstream M1 gate re-verifies LOST on the current 122B before training).
            var label = HarvestedTurn.Label.skip; var reason = ""
            if !ag.isLocal {
                if done && (!m.text.isEmpty || anyCall) { label = .won; reason = "strong-model completed role turn" }
                else { reason = "strong-model but not cleanly completed (status=\(m.status))" }
            } else {
                if errored { label = .lost; reason = "local turn errored" }
                else if isBuilder && !m.text.isEmpty && !mutating { label = .lost; reason = "builder produced text but no mutating tool call (narrate-without-act)" }
                else if anyError { label = .lost; reason = "tool error in turn" }
                else { reason = "local turn, no clear failure signal" }
            }
            out.append(HarvestedTurn(
                room: room, agentName: ag.name, role: ag.role, modelId: ag.modelId, isLocal: ag.isLocal,
                isReferee: ag.isReferee, systemPrompt: ag.system, priorContext: prior, outputText: m.text,
                status: m.status, toolUses: tools, label: label, reason: reason))
        }
        return out
    }

    // ── Env-gated runner: harvest the configured rooms + print the distribution (ROOM_HARVEST=1). ─────────────
    public static func runReport() -> String {
        var out: [String] = ["=== ROOM_HARVEST — role-discipline data inventory ==="]
        var allTurns: [HarvestedTurn] = []
        var rawNameCounts: [String: Int] = [:]
        for room in goldRooms + failureRooms {
            let turns = harvest(room: room)
            guard !turns.isEmpty else { continue }
            allTurns += turns
            for t in turns { for u in t.toolUses { rawNameCounts[u.rawName, default: 0] += 1 } }
            let won = turns.filter { $0.label == .won }.count
            let lost = turns.filter { $0.label == .lost }.count
            let skip = turns.filter { $0.label == .skip }.count
            let local = turns.first?.isLocal ?? false
            out.append("  \(room.padding(toLength: 22, withPad: " ", startingAt: 0)) turns=\(turns.count)  WON=\(won) LOST=\(lost) skip=\(skip)  (\(local ? "local" : "strong") roster)")
        }
        let wonTurns = allTurns.filter { $0.label == .won }
        let won = wonTurns.count
        let lost = allTurns.filter { $0.label == .lost }.count
        let cleanWon = wonTurns.filter(inFormat).count   // gold whose tools transfer (not Claude-only-dominated)
        out.append("")
        out.append("TOTAL: \(allTurns.count) turns  →  WON=\(won) (in-format/usable=\(cleanWon), Claude-only-heavy=\(won - cleanWon))  LOST=\(lost)  skip=\(allTurns.count - won - lost)")
        // The empirical answer to "what tool names do Claude-in-Orbital turns emit?" (drives the M2 normalizer).
        let nonNative = rawNameCounts.filter { !nativeFileExecTools.contains($0.key) && !["handoff","consult","done"].contains($0.key) && !$0.key.hasPrefix("mcp__") }
        out.append("RAW tool-name distribution: \(rawNameCounts.sorted { $0.value > $1.value }.prefix(20).map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
        out.append("NON-NATIVE / non-routing / non-mcp names (need M2 normalization): \(nonNative.isEmpty ? "(none)" : nonNative.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " "))")
        // Show a sample WON + LOST so the labels are legible.
        if let w = allTurns.first(where: { $0.label == .won && !$0.toolUses.isEmpty }) {
            out.append("\n---- SAMPLE WON (\(w.room)/\(w.role), \(w.modelId)) ----\n\(w.reason)\noutput: \(w.outputText.prefix(200))\ntools: \(w.toolUses.map { "\($0.rawName)→\($0.name)[\($0.status)]" }.joined(separator: ", "))")
        }
        if let l = allTurns.first(where: { $0.label == .lost }) {
            out.append("\n---- SAMPLE LOST (\(l.room)/\(l.role), \(l.modelId)) ----\n\(l.reason)\noutput: \(l.outputText.prefix(200))\ntools: \(l.toolUses.map { "\($0.rawName)[\($0.status)]" }.joined(separator: ", "))")
        }
        return out.joined(separator: "\n")
    }
}
