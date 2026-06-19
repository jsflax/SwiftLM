import Testing
@testable import Orchestration

/// Records the peak number of workers running at once — proves FanOutPool actually parallelizes.
actor ConcurrencyTracker {
    private(set) var current = 0
    private(set) var maxSeen = 0
    func enter() { current += 1; maxSeen = max(maxSeen, current) }
    func leave() { current -= 1 }
}

/// A fake box: returns deterministic completions tagged with its worker id, optionally sleeping so a
/// concurrency tracker can observe overlap.
final class MockWorker: TracePool, @unchecked Sendable {
    let wid: String
    let tracker: ConcurrencyTracker?
    let sleepNanos: UInt64
    init(_ wid: String, tracker: ConcurrencyTracker? = nil, sleepNanos: UInt64 = 0) {
        self.wid = wid; self.tracker = tracker; self.sleepNanos = sleepNanos
    }
    func generate(_ job: GenJob) async -> [String] {
        await tracker?.enter()
        if sleepNanos > 0 { try? await Task.sleep(nanoseconds: sleepNanos) }
        await tracker?.leave()
        return (0..<job.n).map { "\(wid)#\(job.prompt)#\($0)" }
    }
    func capabilities() async -> [WorkerDescriptor] {
        [WorkerDescriptor(id: wid, models: [ModelID("m")], batchWidth: 1,
                          effectiveParallelism: 1, estTokensPerSecPerStream: 10)]
    }
}

struct TracePoolTests {
    private func job(_ p: String, n: Int = 1) -> GenJob { GenJob(model: ModelID("m"), prompt: p, n: n) }

    @Test func localPoolBestOfN() async {
        let pool = LocalPool(descriptor: WorkerDescriptor(id: "local", models: [ModelID("m")],
                             batchWidth: 8, effectiveParallelism: 1, estTokensPerSecPerStream: 40)) { j in
            (0..<j.n).map { "gen-\($0)" }
        }
        let out = await pool.generate(job("x", n: 4))
        #expect(out == ["gen-0", "gen-1", "gen-2", "gen-3"])
        let caps = await pool.capabilities()
        #expect(caps.count == 1)
        #expect(caps.first?.id == "local")
    }

    @Test func localPoolGenerateManyIsSerialAndOrdered() async {
        let pool = LocalPool(descriptor: WorkerDescriptor(id: "local", models: [ModelID("m")],
                             batchWidth: 1, effectiveParallelism: 1, estTokensPerSecPerStream: 40)) { j in
            [j.prompt]
        }
        let res = await pool.generateMany([job("p0"), job("p1"), job("p2")])
        #expect(res.map { $0[0] } == ["p0", "p1", "p2"])
    }

    @Test func fanOutRoundRobinPreservesOrder() async {
        let workers = (0..<3).map { MockWorker("w\($0)") }
        let pool = FanOutPool(workers: workers)
        let jobs = (0..<6).map { job("p\($0)") }
        let res = await pool.generateMany(jobs)
        #expect(res.count == 6)
        // job i is round-robined to worker i % 3, and results come back in input order
        for i in 0..<6 {
            #expect(res[i][0].hasPrefix("w\(i % 3)#p\(i)#"), "job \(i) → \(res[i])")
        }
    }

    @Test func fanOutRunsWorkersConcurrently() async {
        let tracker = ConcurrencyTracker()
        let workers = (0..<3).map { MockWorker("w\($0)", tracker: tracker, sleepNanos: 40_000_000) }
        let pool = FanOutPool(workers: workers)
        _ = await pool.generateMany((0..<6).map { job("p\($0)") })
        let peak = await tracker.maxSeen
        #expect(peak == 3, "all 3 workers (boxes) should run their buckets concurrently")
    }

    @Test func fanOutCapabilitiesAggregate() async {
        let pool = FanOutPool(workers: (0..<3).map { MockWorker("w\($0)") })
        let caps = await pool.capabilities()
        #expect(caps.count == 3)
        #expect(Set(caps.map(\.id)) == ["w0", "w1", "w2"])
    }

    @Test func fanOutSingleJobRoundRobins() async {
        let pool = FanOutPool(workers: (0..<2).map { MockWorker("w\($0)") })
        let a = await pool.generate(job("x"))
        let b = await pool.generate(job("y"))
        #expect(a[0].hasPrefix("w0#"))
        #expect(b[0].hasPrefix("w1#"))
    }
}
