import Foundation

// ── Resumable flywheel checkpointing (the fix for "11h of trace-gen lost on a crash").
//
// The flywheel's expensive work is generation + verification: best-of-N rollouts per train task, each
// verified by compile+test. Two 32B runs died mid-trace-gen and threw ALL of it away because nothing
// was persisted until training/verdict. TraceCache fixes that: each task's rollouts (+ pass/fail
// verdicts) are appended to ~/.swiftlm/traces/<cycleId>.jsonl AND fsync'd the moment they're verified —
// so a crash loses only the in-progress task, and a re-run with the same cycleId skips already-done
// tasks (resume instead of restart). Reassembling preference pairs from cached (text, passed) is cheap
// and deterministic, so caching the raw rollouts is the right granularity.
//
// Pure-Swift + injectable root → unit-testable with no MLX and no real ~/.swiftlm.
public struct TraceCache: Sendable {
    public struct Rollout: Codable, Sendable, Equatable {
        public let text: String
        public let passed: Bool
        public init(text: String, passed: Bool) { self.text = text; self.passed = passed }
    }

    public struct TaskRecord: Codable, Sendable, Equatable {
        public let taskId: String
        public let rollouts: [Rollout]
        public init(taskId: String, rollouts: [Rollout]) { self.taskId = taskId; self.rollouts = rollouts }
    }

    let fileURL: URL

    /// `root` defaults to ~/.swiftlm/traces (created 0700). One JSONL file per cycle.
    public init(cycleId: String, root: URL? = nil) throws {
        let base = root ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".swiftlm/traces")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        self.fileURL = base.appending(path: "\(cycleId).jsonl")
    }

    /// Append one task's rollouts and fsync — so a crash immediately after still finds the record.
    public func record(taskId: String, rollouts: [Rollout]) throws {
        let line = String(data: try JSONEncoder().encode(TaskRecord(taskId: taskId, rollouts: rollouts)),
                          encoding: .utf8)! + "\n"
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil,
                                           attributes: [.posixPermissions: 0o600])
        }
        let h = try FileHandle(forWritingTo: fileURL)
        defer { try? h.close() }
        try h.seekToEnd()
        try h.write(contentsOf: Data(line.utf8))
        try h.synchronize()   // fsync: durable across a crash
    }

    /// All cached task records. Malformed lines (a partial final line from a crash mid-write) are
    /// skipped, not fatal — so the cache is always loadable.
    public func load() -> [TaskRecord] {
        guard let data = try? Data(contentsOf: fileURL),
              let text = String(data: data, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap {
            try? JSONDecoder().decode(TaskRecord.self, from: Data($0.utf8))
        }
    }

    /// Task IDs already fully recorded — the flywheel skips these on resume.
    public func completedTaskIds() -> Set<String> { Set(load().map(\.taskId)) }
}
