import Testing
import Foundation
@testable import Orchestration

/// Records the size of each coalesced batch the pool fused into one forward pass.
actor BatchRecorder {
    private(set) var sizes: [Int] = []
    func record(_ n: Int) { sizes.append(n) }
}

struct LocalBatchPoolTests {
    private func desc(_ batchWidth: Int) -> WorkerDescriptor {
        WorkerDescriptor(id: "local", models: [ModelID("m")], batchWidth: batchWidth,
                         effectiveParallelism: 1, estTokensPerSecPerStream: 40)
    }
    private func req(_ prompt: String, model: String = "m") -> InferenceRequest {
        InferenceRequest(model: ModelID(model), prompt: prompt)
    }

    /// K concurrent same-key requests fuse into ONE batch, and each caller gets ITS OWN prompt's result.
    @Test func concurrentSameKeyCoalesceIntoOneBatch() async {
        let rec = BatchRecorder()
        let K = 5
        let pool = LocalBatchPool(descriptor: desc(K), coalesceWindowMillis: 50) { reqs in
            await rec.record(reqs.count)
            return reqs.map { "echo:\($0.prompt)" }
        }
        let reqs = (0..<K).map { req("p\($0)") }
        let results = await withTaskGroup(of: (String, String).self) { group -> [String: String] in
            for r in reqs { group.addTask { (r.prompt, await pool.complete(r)) } }
            var m: [String: String] = [:]
            for await (p, r) in group { m[p] = r }
            return m
        }
        for i in 0..<K { #expect(results["p\(i)"] == "echo:p\(i)", "row p\(i) → \(results["p\(i)"] ?? "nil")") }
        let sizes = await rec.sizes
        #expect(sizes == [K], "expected one coalesced batch of \(K), got \(sizes)")
    }

    /// Requests with DIFFERENT batch keys (different base model) never share a forward pass.
    @Test func differentKeysDoNotMerge() async {
        let rec = BatchRecorder()
        // NOTE: keep the runBatch closure free of `#expect` — a testing macro inside a stored
        // @Sendable closure (run outside the @Test body) misbehaves. Assert from the test body instead.
        let pool = LocalBatchPool(descriptor: desc(8), coalesceWindowMillis: 50) { reqs in
            await rec.record(reqs.count)
            return reqs.map { "\($0.model.raw):\($0.prompt)" }
        }
        let reqs = [req("a", model: "m1"), req("b", model: "m1"), req("c", model: "m2")]
        let out = await withTaskGroup(of: String.self) { group -> [String] in
            for r in reqs { group.addTask { await pool.complete(r) } }
            var acc: [String] = []; for await r in group { acc.append(r) }
            return acc
        }
        #expect(Set(out) == ["m1:a", "m1:b", "m2:c"])
        // Single-key coalescing: m1's two requests fuse into one pass; m2 runs alone — never merged.
        let sizes = await rec.sizes.sorted()
        #expect(sizes == [1, 2], "m1 fuses 2, m2 runs alone — got \(sizes)")
    }

    /// More than a full batch accumulates → split into batches no larger than the width, none lost.
    @Test func overflowBeyondBatchWidthSplits() async {
        let rec = BatchRecorder()
        let pool = LocalBatchPool(descriptor: desc(2), coalesceWindowMillis: 50) { reqs in
            await rec.record(reqs.count)
            return reqs.map { "echo:\($0.prompt)" }
        }
        let reqs = (0..<5).map { req("p\($0)") }
        let results = await withTaskGroup(of: String.self) { group -> Set<String> in
            for r in reqs { group.addTask { await pool.complete(r) } }
            var acc: Set<String> = []; for await r in group { acc.insert(r) }
            return acc
        }
        #expect(results == Set((0..<5).map { "echo:p\($0)" }))
        let sizes = await rec.sizes
        #expect(sizes.allSatisfy { $0 <= 2 }, "no batch exceeds the width — got \(sizes)")
        #expect(sizes.reduce(0, +) == 5, "every request decoded exactly once — got \(sizes)")
    }

    /// A lone request still completes (batch of one) and reports the worker's capabilities.
    @Test func singleRequestRunsAsBatchOfOne() async {
        let pool = LocalBatchPool(descriptor: desc(8), coalesceWindowMillis: 20) { reqs in
            reqs.map { "solo:\($0.prompt)" }
        }
        let r = await pool.complete(req("hello"))
        #expect(r == "solo:hello")
        let caps = await pool.capabilities()
        #expect(caps.first?.id == "local")
    }
}
