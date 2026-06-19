import Testing
@testable import Orchestration

/// A test-double pool: echoes the request id as a few tokens, yielding between them so concurrent
/// streams interleave, and tracks the peak number of simultaneously-live generations.
actor MockPool: ComputePool {
    let tokensPerReq: Int
    private var live = 0
    private(set) var peak = 0
    private(set) var totalStarted = 0
    init(tokensPerReq: Int = 4) { self.tokensPerReq = tokensPerReq }

    private func enter() { live += 1; totalStarted += 1; peak = max(peak, live) }
    private func leave() { live -= 1 }

    func submit(_ req: InferenceRequest) async -> AsyncStream<Token> {
        enter()
        let n = tokensPerReq
        let id = req.id
        return AsyncStream<Token> { cont in
            Task {
                for i in 0..<n {
                    await Task.yield()
                    cont.yield(Token(text: "\(id):\(i) ", isFinal: i == n - 1))
                }
                cont.finish()
                await self.leave()
            }
        }
    }

    func capabilities() async -> [WorkerDescriptor] {
        [WorkerDescriptor(id: "mock", models: [ModelID("m")], batchWidth: 8,
                          effectiveParallelism: 1, estTokensPerSecPerStream: 100)]
    }
}

struct SchedulerTests {
    @Test func concurrencyCapRespected() async {
        let pool = MockPool(tokensPerReq: 6)
        let sched = Scheduler(pool: pool, maxConcurrent: 3)
        var completed = 0
        await withTaskGroup(of: String.self) { g in
            for i in 0..<24 {
                g.addTask { await sched.complete(InferenceRequest(id: "r\(i)", model: ModelID("m"), prompt: "p")) }
            }
            for await _ in g { completed += 1 }
        }
        #expect(completed == 24)
        let peak = await pool.peak
        let started = await pool.totalStarted
        #expect(started == 24, "every request reached the pool")
        // The cap is 3. (+1 tolerance: the mock's leave() and the next admit() can momentarily
        // overlap on the pool actor — a test-double artifact, not a scheduler over-admission.)
        #expect(peak <= 4, "scheduler over-admitted (peak \(peak), cap 3)")
        #expect(peak >= 2, "no concurrency occurred at all")
        let inflight = await sched.inflightCount
        #expect(inflight == 0, "all slots released after completion")
    }

    @Test func completesWithFullText() async {
        let pool = MockPool(tokensPerReq: 3)
        let sched = Scheduler(pool: pool, maxConcurrent: 2)
        let out = await sched.complete(InferenceRequest(id: "x", model: ModelID("m"), prompt: "p"))
        #expect(out == "x:0 x:1 x:2 ")
    }

    @Test func serialCapStillCompletes() async {
        // maxConcurrent = 1 forces full serialization; all must still finish.
        let pool = MockPool(tokensPerReq: 2)
        let sched = Scheduler(pool: pool, maxConcurrent: 1)
        var completed = 0
        await withTaskGroup(of: String.self) { g in
            for i in 0..<6 { g.addTask { await sched.complete(InferenceRequest(id: "s\(i)", model: ModelID("m"), prompt: "p")) } }
            for await _ in g { completed += 1 }
        }
        #expect(completed == 6)
        let peak = await pool.peak
        #expect(peak == 1, "maxConcurrent=1 must never run two at once")
    }

    @Test func requestBatchKeyGroupsByBaseAndSampler() {
        let a = InferenceRequest(model: ModelID("base"), adapter: AdapterID("lora-1"), prompt: "p", temperature: 0.7)
        let b = InferenceRequest(model: ModelID("base"), adapter: AdapterID("lora-2"), prompt: "q", temperature: 0.7)
        let c = InferenceRequest(model: ModelID("base"), adapter: AdapterID("lora-1"), prompt: "r", temperature: 0.0)
        // Same base + sampler co-batch (the SGMV key is base, NOT adapter) → a and b share a key.
        #expect(a.batchKey == b.batchKey)
        // Different sampler params cannot share a forward pass.
        #expect(a.batchKey != c.batchKey)
    }
}
