import Foundation

// ── The trace-volume interface (#27, the empirically-forced fix).
//
// The v2a negatives proved the binding constraint is VERIFIED-TRACE VOLUME: one slow 32B generating
// best-of-N over many tasks is ~9h on one box. The fix is to DISTRIBUTE whole best-of-N jobs across
// boxes — each box runs same-prompt best-of-N (the existing `batchGenerate`, the easy batching case),
// the coordinator fans the per-task jobs across machines and aggregates. No hard ragged cross-prompt
// batching needed for this; that's the simpler half the recon flagged as tractable.
//
// `TracePool` is that interface. `LocalPool` runs a job on one box (via an INJECTED generator closure,
// so this stays pure-Swift + unit-testable — the MLX `batchGenerate` is wired in where the model
// lives). `FanOutPool` is the coordinator that spreads jobs across N workers; swap its workers from
// local closures to `RemotePool` RPC clients (next) and the same code fans across the cluster.

/// One best-of-N generation job: produce `n` completions of `prompt` under one sampler. This is the
/// unit the cluster distributes — exactly one flywheel task's rollouts.
public struct GenJob: Sendable, Identifiable, Codable {
    public let id: String
    public let model: ModelID
    public let adapter: AdapterID?
    public let prompt: String
    public let n: Int
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let repetitionPenalty: Float
    public let priority: Priority

    public init(id: String = UUID().uuidString, model: ModelID, adapter: AdapterID? = nil,
                prompt: String, n: Int, maxTokens: Int = 4096, temperature: Float = 0.7,
                topP: Float = 0.95, repetitionPenalty: Float = 1.0, priority: Priority = .background) {
        self.id = id; self.model = model; self.adapter = adapter; self.prompt = prompt; self.n = n
        self.maxTokens = maxTokens; self.temperature = temperature; self.topP = topP
        self.repetitionPenalty = repetitionPenalty; self.priority = priority
    }
}

/// A pool that runs best-of-N generation jobs. LocalPool (one box) and ClusterPool/FanOutPool (N
/// boxes) both conform; the flywheel calls `generateMany` and is regime-agnostic.
public protocol TracePool: Sendable {
    /// Produce `job.n` completions for one job (same-prompt best-of-N).
    func generate(_ job: GenJob) async -> [String]
    /// Run many jobs, returning completions in INPUT ORDER. Default: serial (one box). FanOutPool
    /// overrides to distribute across workers — the trace-volume win (wall-clock ≈ total / #workers).
    func generateMany(_ jobs: [GenJob]) async -> [[String]]
    func capabilities() async -> [WorkerDescriptor]
}

public extension TracePool {
    func generateMany(_ jobs: [GenJob]) async -> [[String]] {
        var out: [[String]] = []
        for j in jobs { out.append(await generate(j)) }
        return out
    }
}

/// One box's executor. The actual generation is INJECTED (`run`) so this type carries no MLX
/// dependency and is unit-testable; in production `run` closes over `MLXLanguageModel.batchGenerate`.
/// A box's GPU is serial, so `generateMany` keeps the default (serial) behavior here.
public final class LocalPool: TracePool, @unchecked Sendable {
    private let descriptor: WorkerDescriptor
    private let run: @Sendable (GenJob) async -> [String]

    public init(descriptor: WorkerDescriptor, run: @escaping @Sendable (GenJob) async -> [String]) {
        self.descriptor = descriptor
        self.run = run
    }

    public func generate(_ job: GenJob) async -> [String] { await run(job) }
    public func capabilities() async -> [WorkerDescriptor] { [descriptor] }
}

/// The cluster coordinator: spreads best-of-N jobs across N worker pools (local box + remote RPC
/// peers). Each worker runs its share serially (one GPU per box); all workers run concurrently, so
/// wall-clock ≈ (total jobs / #workers) × per-job time. THIS is what makes the broad 32B run tractable.
public actor FanOutPool: TracePool {
    private let workers: [TracePool]
    private var rr = 0   // round-robin cursor for single-job placement

    public init(workers: [TracePool]) {
        precondition(!workers.isEmpty, "FanOutPool needs ≥1 worker")
        self.workers = workers
    }

    public func generate(_ job: GenJob) async -> [String] {
        let w = workers[rr % workers.count]; rr += 1
        return await w.generate(job)
    }

    public func capabilities() async -> [WorkerDescriptor] {
        var all: [WorkerDescriptor] = []
        for w in workers { all += await w.capabilities() }
        return all
    }

    /// Round-robin the jobs into per-worker buckets, run each bucket on its worker (serially, since a
    /// box is one GPU), all buckets concurrently. Results are reassembled in INPUT ORDER.
    public func generateMany(_ jobs: [GenJob]) async -> [[String]] {
        let nW = workers.count
        var buckets: [[(Int, GenJob)]] = Array(repeating: [], count: nW)
        for (i, job) in jobs.enumerated() { buckets[i % nW].append((i, job)) }

        var results = [[String]](repeating: [], count: jobs.count)
        await withTaskGroup(of: [(Int, [String])].self) { group in
            for wi in 0..<nW where !buckets[wi].isEmpty {
                let worker = workers[wi]
                let bucket = buckets[wi]
                group.addTask {
                    var out: [(Int, [String])] = []
                    for (idx, job) in bucket { out.append((idx, await worker.generate(job))) }
                    return out
                }
            }
            for await chunk in group { for (idx, r) in chunk { results[idx] = r } }
        }
        return results
    }
}
