import Testing
import Foundation
@testable import NativeTools

/// Each test uses a UNIQUE temp dir (never real user files), so they're isolated + safe.
struct NativeToolsTests {
    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appending(path: "nt-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func readReturnsContents() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "a.txt")
        try "hello world".write(to: f, atomically: true, encoding: .utf8)
        let out = try await ReadFileTool().run(.init(json: #"{"path":"\#(f.path)"}"#))
        #expect(out == "hello world")
    }

    @Test func readMissingFileThrowsIO() async {
        await #expect(throws: NativeToolError.self) {
            try await ReadFileTool().run(.init(json: #"{"path":"/no/such/file-xyz"}"#))
        }
    }

    @Test func readMissingArgThrows() async {
        await #expect(throws: NativeToolError.missingArgument("path")) {
            try await ReadFileTool().run(.init(json: "{}"))
        }
    }

    @Test func writeThenReadRoundTrips() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "sub/b.txt")   // intermediate dir created
        _ = try await WriteFileTool().run(.init(["path": f.path, "content": "abc"]))
        #expect(try String(contentsOf: f, encoding: .utf8) == "abc")
    }

    @Test func editReplacesUniqueString() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "c.txt")
        try "let x = 1\nlet y = 2".write(to: f, atomically: true, encoding: .utf8)
        let msg = try await EditFileTool().run(.init(["path": f.path, "old_string": "y = 2", "new_string": "y = 3"]))
        #expect(msg.contains("1 occurrence"))
        #expect(try String(contentsOf: f, encoding: .utf8) == "let x = 1\nlet y = 3")
    }

    @Test func editRefusesAmbiguousWithoutReplaceAll() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "d.txt")
        try "a a a".write(to: f, atomically: true, encoding: .utf8)
        await #expect(throws: NativeToolError.self) {   // "a" is not unique
            try await EditFileTool().run(.init(["path": f.path, "old_string": "a", "new_string": "b"]))
        }
        // replace_all succeeds
        _ = try await EditFileTool().run(.init(["path": f.path, "old_string": "a", "new_string": "b", "replace_all": true]))
        #expect(try String(contentsOf: f, encoding: .utf8) == "b b b")
    }

    @Test func editMissingOldStringThrows() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "e.txt")
        try "hello".write(to: f, atomically: true, encoding: .utf8)
        await #expect(throws: NativeToolError.self) {
            try await EditFileTool().run(.init(["path": f.path, "old_string": "absent", "new_string": "x"]))
        }
    }

    @Test func globMatchesByExtension() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        try "x".write(to: dir.appending(path: "a.swift"), atomically: true, encoding: .utf8)
        try "x".write(to: dir.appending(path: "b.swift"), atomically: true, encoding: .utf8)
        try "x".write(to: dir.appending(path: "c.txt"), atomically: true, encoding: .utf8)
        let out = try await GlobTool().run(.init(["pattern": "*.swift", "path": dir.path]))
        #expect(out.contains("a.swift") && out.contains("b.swift") && !out.contains("c.txt"))
    }

    @Test func grepFindsRegexMatches() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        try "alpha\nfunc beta()\ngamma".write(to: dir.appending(path: "src.swift"), atomically: true, encoding: .utf8)
        let out = try await GrepTool().run(.init(["pattern": #"func \w+"#, "path": dir.path]))
        #expect(out.contains("func beta") && out.contains(":2:"))
        #expect(!out.contains("alpha"))
    }

    @Test func bashRunsCommand() async throws {
        let out = try await BashTool().run(.init(["command": "echo hi-there"]))
        #expect(out.contains("hi-there"))
    }

    @Test func bashTimesOut() async throws {
        let out = try await BashTool().run(.init(["command": "sleep 5", "timeout": 1]))
        #expect(out.contains("timed out"))
    }
}

struct NativeToolRegistryTests {
    @Test func standardExposesAllToolSpecs() {
        let reg = NativeToolRegistry.standard()
        #expect(Set(reg.names) == ["read_file", "write_file", "edit_file", "glob", "grep", "bash",
                                   "web_fetch", "ExitPlanMode"])
        let specs = reg.specs()
        #expect(specs.count == 8)
        // Injecting a runner adds the Claude-faithful `Task` tool (the sub-agent capability).
        #expect(NativeToolRegistry.standard(subagentRunner: StubRunner()).names.contains("Task"))
        // each spec is the OpenAI fn shape the model consumes
        let fn = specs.first { (($0["function"] as? [String: any Sendable])?["name"] as? String) == "read_file" }
        let function = try! #require(fn?["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "read_file")
        let params = try! #require(function["parameters"] as? [String: any Sendable])
        #expect(params["type"] as? String == "object")
        #expect((params["required"] as? [String]) == ["path"])
    }

    @Test func dispatchRoutesToOwnedToolElseNil() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "reg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appending(path: "x.txt")
        try "payload".write(to: f, atomically: true, encoding: .utf8)
        let reg = NativeToolRegistry.standard()
        let owned = await reg.dispatch(name: "read_file", argsJSON: #"{"path":"\#(f.path)"}"#)
        #expect(owned == "payload")
        let unowned = await reg.dispatch(name: "mcp__etrade__get_quote", argsJSON: "{}")
        #expect(unowned == nil)   // not ours → caller falls back to MCP
    }

    @Test func dispatchReturnsErrorStringOnFailure() async {
        let r = await NativeToolRegistry.standard().dispatch(name: "read_file", argsJSON: #"{"path":"/nope"}"#)
        #expect(r?.hasPrefix("ERROR") == true)   // failures come back as observations, not thrown
    }
}

/// A SubagentRunner that records its call and echoes — proves TaskTool forwards args via DI (no MLX).
final class StubRunner: SubagentRunner, @unchecked Sendable {
    let lock = NSLock()
    var calls: [(type: String, prompt: String)] = []
    func run(subagentType: String, description: String, prompt: String) async throws -> String {
        lock.withLock { calls.append((subagentType, prompt)) }
        return "ran \(subagentType): \(prompt)"
    }
}

struct TaskToolTests {
    @Test func taskToolForwardsToInjectedRunner() async throws {
        let runner = StubRunner()
        let tool = TaskTool(runner: runner)
        #expect(tool.name == "Task")
        let out = try await tool.run(ToolArguments([
            "subagent_type": "session-learner", "prompt": "review the session", "description": "learn"]))
        #expect(out == "ran session-learner: review the session")
        #expect(runner.calls.first?.type == "session-learner")
        #expect(runner.calls.first?.prompt == "review the session")
    }

    @Test func taskToolRequiresSubagentTypeAndPrompt() async {
        let tool = TaskTool(runner: StubRunner())
        await #expect(throws: NativeToolError.self) {
            _ = try await tool.run(ToolArguments(["prompt": "x"]))           // missing subagent_type
        }
        await #expect(throws: NativeToolError.self) {
            _ = try await tool.run(ToolArguments(["subagent_type": "x"]))    // missing prompt
        }
    }
}

struct WebFetchToolTests {
    // Pure (no network): the HTML→text reducer drops script/style/comments, strips tags, decodes entities,
    // and collapses whitespace — the part an agent actually reads.
    @Test func htmlToTextStripsTagsAndDecodesEntities() {
        let html = """
        <html><head><title>T</title><style>.x{color:red}</style>
        <script>var a = 1 < 2 && 3 > 0;</script><!-- a comment --></head>
        <body><h1>Hello&nbsp;World</h1><p>A &amp; B &lt;tag&gt; &#39;q&#39;</p></body></html>
        """
        let text = WebFetchTool.htmlToText(html)
        #expect(text.contains("Hello World"))
        #expect(text.contains("A & B <tag> 'q'"))
        #expect(!text.contains("color:red"))      // <style> body gone
        #expect(!text.contains("var a"))          // <script> body gone
        #expect(!text.contains("a comment"))      // <!-- --> gone
        #expect(!text.contains("<h1>") && !text.contains("</p>"))  // tags stripped
    }

    @Test func rejectsNonHttpScheme() async {
        let tool = WebFetchTool()
        // file:// (would read local disk) and a bare path must be refused before any network call.
        for bad in ["file:///etc/hosts", "/etc/hosts", "ftp://example.com"] {
            await #expect(throws: NativeToolError.self) {
                _ = try await tool.run(ToolArguments(["url": bad]))
            }
        }
    }

    @Test func requiresUrl() async {
        let tool = WebFetchTool()
        await #expect(throws: NativeToolError.self) {
            _ = try await tool.run(ToolArguments([:]))   // missing required `url`
        }
    }
}
