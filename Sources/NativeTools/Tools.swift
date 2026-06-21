import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking   // URLSession lives here on Linux
#endif
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

// ── The standard native toolset. Claude-Code-shaped (read_file/write_file/edit_file/glob/grep/bash) so
// habits + harvested transcripts transfer. Each is a small, deterministic Foundation function — unit-
// tested against temp dirs / safe commands. Mutating/exec tools (write/edit/bash) are gated at serve time
// by the PreToolUse hook seam (plan-mode read-only, etc.); they don't self-restrict here.
//
// CWD (bug D fix): the file/search tools resolve a RELATIVE path — and the glob/grep root + the bash cwd —
// against `cwd`, the agent's working directory. It defaults to the process dir for back-compat, but the MLX
// driver injects the ROOM cwd via `NativeToolRegistry.standard(cwd:)`, so a room agent's `glob **/*.swift` /
// relative `write_file` lands in the ROOM, not wherever orbital-loop happened to be launched.

/// Resolve a possibly-relative tool path against the working dir; an absolute path passes through unchanged.
func resolveToolPath(_ path: String, cwd: String) -> String {
    path.hasPrefix("/") ? path : URL(fileURLWithPath: cwd).appendingPathComponent(path).standardizedFileURL.path
}

/// Read a UTF-8 text file.
public struct ReadFileTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "read_file"
    public let description = "Read a UTF-8 text file (absolute path, or relative to the working directory)."
    public var parameters: JSONSchemaObject {
        .init([("path", .init("string", "Path to the file (absolute, or relative to the working dir)"))], required: ["path"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let path = resolveToolPath(try args.requireString("path"), cwd: cwd)
        do { return try String(contentsOfFile: path, encoding: .utf8) }
        catch { throw NativeToolError.io("cannot read \(path): \(error.localizedDescription)") }
    }
}

/// Create or overwrite a UTF-8 text file (creating intermediate directories).
public struct WriteFileTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "write_file"
    public let description = "Write (create or overwrite) a UTF-8 text file at an absolute path."
    public var parameters: JSONSchemaObject {
        .init([("path", .init("string", "Absolute path to write")),
               ("content", .init("string", "Full file contents"))], required: ["path", "content"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let path = resolveToolPath(try args.requireString("path"), cwd: cwd)
        let content = try args.requireString("content")
        let url = URL(fileURLWithPath: path)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
            return "wrote \(content.utf8.count) bytes to \(path)"
        } catch { throw NativeToolError.io("cannot write \(path): \(error.localizedDescription)") }
    }
}

/// Exact-string replace in a file. Refuses an ambiguous edit (old_string non-unique) unless replace_all.
public struct EditFileTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "edit_file"
    public let description = "Replace an exact string in a file. Fails if old_string is absent, or "
        + "(unless replace_all) is not unique. Returns the number of replacements."
    public var parameters: JSONSchemaObject {
        .init([("path", .init("string", "Absolute path")),
               ("old_string", .init("string", "Exact text to replace")),
               ("new_string", .init("string", "Replacement text")),
               ("replace_all", .init("boolean", "Replace every occurrence (default false)"))],
              required: ["path", "old_string", "new_string"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let path = resolveToolPath(try args.requireString("path"), cwd: cwd)
        let oldS = try args.requireString("old_string")
        let newS = args.string("new_string") ?? ""   // empty replacement (deletion) is allowed
        let all = args.bool("replace_all") ?? false
        let content: String
        do { content = try String(contentsOfFile: path, encoding: .utf8) }
        catch { throw NativeToolError.io("cannot read \(path): \(error.localizedDescription)") }

        let count = content.components(separatedBy: oldS).count - 1
        guard count > 0 else { throw NativeToolError.io("old_string not found in \(path)") }
        if !all && count > 1 {
            throw NativeToolError.io("old_string is not unique in \(path) (\(count) matches); "
                + "add surrounding context or set replace_all")
        }
        let updated: String
        if all { updated = content.replacingOccurrences(of: oldS, with: newS) }
        else { let r = content.range(of: oldS)!; updated = content.replacingCharacters(in: r, with: newS) }
        do { try updated.write(toFile: path, atomically: true, encoding: .utf8) }
        catch { throw NativeToolError.io("cannot write \(path): \(error.localizedDescription)") }
        return "replaced \(all ? count : 1) occurrence(s) in \(path)"
    }
}

/// Find files by glob pattern (POSIX fnmatch) under a directory.
public struct GlobTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "glob"
    public let description = "List files under a directory whose path matches a glob pattern "
        + "(e.g. **/*.swift). Returns up to 200 matching absolute paths, one per line."
    public var parameters: JSONSchemaObject {
        .init([("pattern", .init("string", "Glob pattern, e.g. *.swift or **/*.md")),
               ("path", .init("string", "Directory to search (default: current directory)"))],
              required: ["pattern"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let pattern = try args.requireString("pattern")
        let base = args.string("path").map { resolveToolPath($0, cwd: cwd) } ?? cwd
        let baseURL = URL(fileURLWithPath: base)
        guard let en = FileManager.default.enumerator(at: baseURL, includingPropertiesForKeys: [.isRegularFileKey],
                                                      options: [.skipsHiddenFiles]) else {
            throw NativeToolError.io("cannot enumerate \(base)")
        }
        var matches: [String] = []
        while let obj = en.nextObject() {
            guard let url = obj as? URL else { continue }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
            let rel = String(url.path.dropFirst(baseURL.path.count).drop(while: { $0 == "/" }))
            if Self.fnmatch(pattern, rel) || Self.fnmatch(pattern, url.lastPathComponent) {
                matches.append(url.path)
                if matches.count >= 200 { break }
            }
        }
        return matches.isEmpty ? "(no matches)" : matches.joined(separator: "\n")
    }
    /// POSIX glob match (`**` collapses to `*` so it spans path separators via FNM_PATHNAME-off).
    static func fnmatch(_ pattern: String, _ string: String) -> Bool {
        let p = pattern.replacingOccurrences(of: "**/", with: "*").replacingOccurrences(of: "**", with: "*")
        return Darwin.fnmatch(p, string, 0) == 0
    }
}

/// Search file contents by regex under a directory; returns `path:line: text` matches.
public struct GrepTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "grep"
    public let description = "Search text files under a path for a regular expression. Returns up to 100 "
        + "matches as `file:line: text`."
    public var parameters: JSONSchemaObject {
        .init([("pattern", .init("string", "Regular expression (NSRegularExpression syntax)")),
               ("path", .init("string", "File or directory to search (default: current directory)"))],
              required: ["pattern"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let pattern = try args.requireString("pattern")
        let base = args.string("path").map { resolveToolPath($0, cwd: cwd) } ?? cwd
        guard let re = try? NSRegularExpression(pattern: pattern) else {
            throw NativeToolError.io("invalid regex: \(pattern)")
        }
        let fm = FileManager.default
        var files: [String] = []
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: base, isDirectory: &isDir) else { throw NativeToolError.io("no such path: \(base)") }
        if isDir.boolValue {
            if let en = fm.enumerator(at: URL(fileURLWithPath: base), includingPropertiesForKeys: [.isRegularFileKey],
                                      options: [.skipsHiddenFiles]) {
                while let obj = en.nextObject() {
                    guard let url = obj as? URL else { continue }
                    if (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
                        files.append(url.path)
                    }
                    if files.count >= 5000 { break }
                }
            }
        } else { files = [base] }

        var out: [String] = []
        for file in files {
            guard let content = try? String(contentsOfFile: file, encoding: .utf8) else { continue }  // skip binary
            var lineNo = 0
            for line in content.split(separator: "\n", omittingEmptySubsequences: false) {
                lineNo += 1
                let s = String(line)
                if re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil {
                    out.append("\(file):\(lineNo): \(s.trimmingCharacters(in: .whitespaces))")
                    if out.count >= 100 { return out.joined(separator: "\n") }
                }
            }
        }
        return out.isEmpty ? "(no matches)" : out.joined(separator: "\n")
    }
}

/// Run a shell command (`/bin/sh -c`), returning combined stdout+stderr (truncated), killed on timeout.
/// The agent's hands for build/test/inspect — gated at serve time by the PreToolUse hook seam.
public struct BashTool: NativeTool {
    public let cwd: String
    public init(cwd: String = FileManager.default.currentDirectoryPath) { self.cwd = cwd }
    public let name = "bash"
    public let description = "Run a shell command via /bin/sh -c and return its combined stdout+stderr."
    public var parameters: JSONSchemaObject {
        .init([("command", .init("string", "The shell command to run")),
               ("timeout", .init("integer", "Timeout in seconds (default 60)"))], required: ["command"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let command = try args.requireString("command")
        let timeout = args.int("timeout") ?? 60
        return await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            DispatchQueue.global().async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/bin/sh")
                proc.arguments = ["-c", command]
                proc.currentDirectoryURL = URL(fileURLWithPath: cwd)   // run in the agent's working dir (bug D)
                let outPipe = Pipe()
                proc.standardOutput = outPipe
                proc.standardError = outPipe
                do { try proc.run() } catch { cont.resume(returning: "ERROR: \(error.localizedDescription)"); return }
                let killed = Killed()
                if timeout > 0 {
                    // Dedicated watchdog THREAD, not a global-queue timer: under heavy load (many concurrent
                    // processes) the global pool can be fully blocked, starving a timer so a "1s" timeout
                    // misses a 5s command. A real thread is scheduled regardless. Polls so it exits early
                    // when the command finishes on its own.
                    Thread.detachNewThread {
                        let deadline = Date().addingTimeInterval(TimeInterval(timeout))
                        while Date() < deadline {
                            if !proc.isRunning { return }
                            Thread.sleep(forTimeInterval: 0.05)
                        }
                        if proc.isRunning { killed.flag = true; proc.terminate() }
                    }
                }
                let data = outPipe.fileHandleForReading.readDataToEndOfFile()
                proc.waitUntilExit()
                var out = String(data: data, encoding: .utf8) ?? ""
                if out.utf8.count > 64_000 { out = String(out.prefix(64_000)) + "\n…(truncated)" }
                if killed.flag { out += "\nERROR: timed out after \(timeout)s" }
                cont.resume(returning: out.isEmpty ? "(no output, exit \(proc.terminationStatus))" : out)
            }
        }
    }
    private final class Killed: @unchecked Sendable { var flag = false }
}

/// Fetch a URL over HTTP(S) and return the page as readable text (HTML stripped to text; JSON/text passed
/// through). The agent's first-class web access — a parsed alternative to the unrestricted `bash curl`.
/// NOTE: this is network EGRESS. `bash` already permits egress, so this adds no NEW exposure; the plan's
/// P0 #7 (an egress-capability flag + Redactor on outbound args, for autonomous runs) gates it later.
public struct WebFetchTool: NativeTool {
    public init() {}
    public let name = "web_fetch"
    public let description = "Fetch an http(s) URL and return the page as readable text (HTML is stripped to "
        + "text; JSON/plain text is returned as-is). Use to read web pages or public APIs."
    public var parameters: JSONSchemaObject {
        .init([("url", .init("string", "The http(s) URL to fetch")),
               ("max_chars", .init("integer", "Cap the returned text length (default 20000)"))],
              required: ["url"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let urlStr = try args.requireString("url")
        let maxChars = max(500, args.int("max_chars") ?? 20_000)
        guard let url = URL(string: urlStr), let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw NativeToolError.io("invalid http(s) url: \(urlStr)")
        }
        var req = URLRequest(url: url, timeoutInterval: 30)
        req.setValue("SwiftLM-agent/0.1", forHTTPHeaderField: "User-Agent")
        let data: Data, resp: URLResponse
        do { (data, resp) = try await URLSession.shared.data(for: req) }
        catch { throw NativeToolError.io("fetch failed for \(urlStr): \(error.localizedDescription)") }

        let http = resp as? HTTPURLResponse
        let status = http?.statusCode ?? 0
        let contentType = (http?.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        guard var body = String(data: data, encoding: .utf8) else {
            return "[\(status)] \(url.host ?? urlStr): \(data.count) bytes of non-text content (\(contentType))"
        }
        if contentType.contains("html") || body.range(of: "<html", options: .caseInsensitive) != nil {
            body = Self.htmlToText(body)
        }
        let header = "[\(status)] \(url.host ?? urlStr)\n"
        if body.count > maxChars {
            return header + String(body.prefix(maxChars)) + "\n[truncated: \(body.count - maxChars) chars omitted]"
        }
        return header + body
    }

    /// Crude HTML → text: drop `<script>`/`<style>` blocks + comments, strip tags, decode a few entities,
    /// collapse whitespace. Enough for an agent to read a page; not a full renderer.
    static func htmlToText(_ html: String) -> String {
        func drop(_ s: String, _ pattern: String) -> String {
            s.replacingOccurrences(of: pattern, with: " ", options: [.regularExpression, .caseInsensitive])
        }
        var s = html
        s = drop(s, "<script[\\s\\S]*?</script>")
        s = drop(s, "<style[\\s\\S]*?</style>")
        s = drop(s, "<!--[\\s\\S]*?-->")
        s = drop(s, "<[^>]+>")                                   // strip remaining tags
        for (e, c) in ["&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"", "&#39;": "'", "&nbsp;": " "] {
            s = s.replacingOccurrences(of: e, with: c)
        }
        s = s.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\n[ \\t]*\\n[ \\t]*\\n+", with: "\n\n", options: .regularExpression)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Claude-faithful `ExitPlanMode`: a planning agent calls this with its finished plan to request approval
/// before making any changes. It has NO world effect of its own — the agent loop INTERCEPTS the call in
/// plan mode (surfaces the plan, asks for approval, flips to execution on yes). This `run` only fires in
/// the off-nominal case where it's somehow dispatched outside plan mode, where it's a no-op. Registering it
/// keeps the name on the model's tool surface (and in the constrainer's runtime enum) so the model can emit
/// it, and keeps SwiftLM a wire-level drop-in for `claude -p`'s ExitPlanMode.
public struct ExitPlanModeTool: NativeTool {
    public init() {}
    public let name = "ExitPlanMode"
    public let description = "Call this when your plan is ready (plan mode only) to present it for approval "
        + "before making any changes. The plan is shown to the approver; on approval you switch to execution."
    public var parameters: JSONSchemaObject {
        .init([("plan", .init("string", "The finished implementation plan (markdown)"))], required: ["plan"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        "(ExitPlanMode is handled by the agent loop; nothing to execute here.)"
    }
}

/// Claude-faithful `Task`: launch a sub-agent to handle a focused, self-contained job in its own context,
/// returning its final result. The actual spawning is an injected `SubagentRunner` (dependency injection),
/// so this stays MLX-free — the MLX-backed runner re-enters the agent loop with the named agent's system
/// prompt + scoped tools. Dispatches through the normal native path (no loop interception).
public struct TaskTool: NativeTool {
    let runner: SubagentRunner
    public init(runner: SubagentRunner) { self.runner = runner }
    public let name = "Task"
    public let description = "Launch a new sub-agent to handle a complex, multi-step task in its own "
        + "isolated context. Provide the sub-agent type and a complete, self-contained prompt; the sub-agent "
        + "runs to completion and returns its final result."
    public var parameters: JSONSchemaObject {
        .init([("description", .init("string", "A short (3-5 word) description of the task")),
               ("prompt", .init("string", "The full, self-contained task for the sub-agent to perform")),
               ("subagent_type", .init("string", "The type of sub-agent to launch (e.g. session-learner)"))],
              required: ["description", "prompt", "subagent_type"])
    }
    public func run(_ args: ToolArguments) async throws -> String {
        let type = try args.requireString("subagent_type")
        let prompt = try args.requireString("prompt")
        let desc = args.string("description") ?? ""
        return try await runner.run(subagentType: type, description: desc, prompt: prompt)
    }
}
