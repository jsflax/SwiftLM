import Testing
import Foundation
@testable import SelfImprove

/// Each test uses a UNIQUE temp root (never the real ~/.swiftlm/traces), so they're isolated + safe.
struct TraceCacheTests {
    private func tempRoot() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "traces-\(UUID().uuidString)")
    }

    @Test func recordsPersistAcrossInstances() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        // first "process": record two tasks
        let c1 = try TraceCache(cycleId: "cyc", root: root)
        try c1.record(taskId: "A", rollouts: [.init(text: "ok", passed: true), .init(text: "bad", passed: false)])
        try c1.record(taskId: "B", rollouts: [.init(text: "x", passed: true)])
        // second "process" (simulated crash + resume): a fresh cache sees the persisted work
        let c2 = try TraceCache(cycleId: "cyc", root: root)
        #expect(c2.completedTaskIds() == ["A", "B"])
        let a = try #require(c2.load().first { $0.taskId == "A" })
        #expect(a.rollouts == [.init(text: "ok", passed: true), .init(text: "bad", passed: false)])
    }

    @Test func resumeSkipsCompletedTasks() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let cache = try TraceCache(cycleId: "c", root: root)
        try cache.record(taskId: "done1", rollouts: [.init(text: "t", passed: true)])
        try cache.record(taskId: "done2", rollouts: [])
        let allTasks = ["done1", "done2", "todo1", "todo2"]
        let done = cache.completedTaskIds()
        #expect(allTasks.filter { !done.contains($0) } == ["todo1", "todo2"])  // only the unfinished regenerate
    }

    @Test func separateCyclesAreIsolated() throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        try TraceCache(cycleId: "c1", root: root).record(taskId: "A", rollouts: [])
        #expect(try TraceCache(cycleId: "c2", root: root).completedTaskIds().isEmpty)  // new cycle starts fresh
    }

    @Test func toleratesMalformedTrailingLine() throws {
        // a crash mid-write leaves a partial last line — load() must skip it, not throw
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let cache = try TraceCache(cycleId: "c", root: root)
        try cache.record(taskId: "A", rollouts: [.init(text: "t", passed: true)])
        let h = try FileHandle(forWritingTo: root.appending(path: "c.jsonl"))
        try h.seekToEnd(); try h.write(contentsOf: Data(#"{"taskId":"B","rol"#.utf8)); try h.close()
        let recs = cache.load()
        #expect(recs.count == 1)
        #expect(recs[0].taskId == "A")  // only the complete record survives a torn write
    }
}
