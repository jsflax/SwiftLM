import Testing
import Foundation
@testable import Orchestration

/// Tests for the cluster observability surface (`ClusterStatus`). It is a process-wide singleton with
/// no reset, so every test uses a UNIQUE worker label (UUID-suffixed) — its `WorkerStat` bucket is then
/// touched by nobody else, making per-worker assertions deterministic even though FanOutPool tests in
/// the same module also write to `ClusterStatus.shared`. We assert through the real JSON status FILE
/// (the artifact a human `cat`s), which exercises the actual serialization path including `Int? ?? NSNull()`.
struct ClusterStatusTests {
    private func uniq(_ p: String) -> String { "\(p)-\(UUID().uuidString.prefix(8))" }

    /// Read + parse the status file, return the worker object for `label` (nil if absent).
    private func workerEntry(_ label: String) async throws -> [String: Any]? {
        let path = await ClusterStatus.shared.statusFilePath()
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let workers = (obj?["workers"] as? [[String: Any]]) ?? []
        return workers.first { ($0["label"] as? String) == label }
    }

    @Test func servedAccountingAndValidJSON() async throws {
        let label = uniq("served")
        await ClusterStatus.shared.registerWorker(label: label, host: "h")
        for _ in 0..<3 { await ClusterStatus.shared.served(label: label, host: "h", ok: true) }
        await ClusterStatus.shared.served(label: label, host: "h", ok: false)
        await ClusterStatus.shared.flush()

        let entry = try await workerEntry(label)
        let w = try #require(entry, "worker \(label) missing from status file")
        #expect(w["served"] as? Int == 3)
        #expect(w["emptyOrFailed"] as? Int == 1)
        #expect(w["host"] as? String == "h")
        #expect(w["lastServedAgoSec"] is Int)  // served worker → Int age (non-null branch)
    }

    @Test func unservedWorkerSerializesNull() async throws {
        let label = uniq("idle")
        await ClusterStatus.shared.registerWorker(label: label, host: "h")
        await ClusterStatus.shared.flush()

        let entry = try await workerEntry(label)
        let w = try #require(entry)
        #expect(w["served"] as? Int == 0)
        #expect(w["emptyOrFailed"] as? Int == 0)
        #expect(w["lastServedAgoSec"] is NSNull)  // never-served age must be JSON null, not a crash
    }

    @Test func fanOutPoolRecordsServedPerWorker() async throws {
        let id = uniq("localpool")
        let local = LocalPool(descriptor: WorkerDescriptor(
            id: id, models: [ModelID("m")], batchWidth: 4,
            effectiveParallelism: 1, estTokensPerSecPerStream: 40)) { j in
            (0..<j.n).map { "\(j.prompt):\($0)" }   // always non-empty ⇒ ok=true
        }
        let fan = FanOutPool(workers: [local])
        _ = await fan.generateMany((0..<5).map { GenJob(model: ModelID("m"), prompt: "p\($0)", n: 2) })
        await ClusterStatus.shared.flush()

        let entry = try await workerEntry(id)
        let w = try #require(entry, "local worker \(id) missing")
        #expect(w["served"] as? Int == 5)   // 5 jobs all produced non-empty completions
        #expect(w["emptyOrFailed"] as? Int == 0)
    }

    @Test func emptyCompletionsCountAsFailed() async throws {
        let id = uniq("emptypool")
        let local = LocalPool(descriptor: WorkerDescriptor(
            id: id, models: [ModelID("m")], batchWidth: 4,
            effectiveParallelism: 1, estTokensPerSecPerStream: 40)) { j in
            (0..<j.n).map { _ in "" }   // all empty ⇒ ok=false
        }
        let fan = FanOutPool(workers: [local])
        _ = await fan.generateMany((0..<3).map { GenJob(model: ModelID("m"), prompt: "p\($0)", n: 2) })
        await ClusterStatus.shared.flush()

        let entry = try await workerEntry(id)
        let w = try #require(entry)
        #expect(w["served"] as? Int == 0)
        #expect(w["emptyOrFailed"] as? Int == 3)
    }
}
