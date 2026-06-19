import Testing
import Foundation
@testable import Serving

struct ToolNameMatchTests {
    // A realistic surface: native tools + a few MCP tools, including `session_stats` (a real trader tool
    // that shares the "session" prefix with the hook sub-agent `session-learner` — the false-positive trap).
    let tools = ["read_file", "write_file", "edit_file", "glob", "grep", "bash", "web_fetch", "ExitPlanMode",
                 "current_time", "system_info", "session_stats"]

    @Test func exactNamePassesThrough() {
        #expect(closestToolName("read_file", in: tools) == "read_file")
        #expect(closestToolName("ExitPlanMode", in: tools) == "ExitPlanMode")
    }

    @Test func caseAndSeparatorVariationsRepair() {
        #expect(closestToolName("Read_File", in: tools) == "read_file")
        #expect(closestToolName("read-file", in: tools) == "read_file")
        #expect(closestToolName("readfile", in: tools) == "read_file")
        #expect(closestToolName("read file", in: tools) == "read_file")
        #expect(closestToolName("exitplanmode", in: tools) == "ExitPlanMode")   // case only
        #expect(closestToolName("web.fetch", in: tools) == "web_fetch")
    }

    @Test func smallTyposRepairToUniqueNearest() {
        #expect(closestToolName("read_fil", in: tools) == "read_file")    // 1 deletion
        #expect(closestToolName("writ_file", in: tools) == "write_file")  // 1 deletion
        #expect(closestToolName("grp", in: tools) == "grep")              // 1 insertion
    }

    @Test func unrelatedNameReturnsNilNotAGuess() {
        // THE REGRESSION: a hook-injected sub-agent name must NOT be repaired into a real (destructive) tool.
        #expect(closestToolName("session-learner", in: tools) == nil)
        // and it must NOT latch onto `session_stats` despite the shared "session" prefix
        #expect(closestToolName("session-learner", in: ["session_stats"]) == nil)
        #expect(closestToolName("Task", in: tools) == nil)
        #expect(closestToolName("spawn_subagent", in: tools) == nil)
    }

    @Test func ambiguousOrEmptyReturnsNil() {
        #expect(closestToolName("", in: tools) == nil)
        #expect(closestToolName("read_file", in: []) == nil)             // no tools → nil
        // two tools normalize identically → ambiguous → refuse to guess
        #expect(closestToolName("readfile", in: ["read_file", "read-file"]) == nil)
    }

    @Test func shortGarbageDoesNotMatchShortTools() {
        #expect(closestToolName("xyz", in: tools) == nil)
        #expect(closestToolName("foo", in: tools) == nil)
    }
}
