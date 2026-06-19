import Foundation

// ── The ComputePool abstraction — the orchestration twin of the LanguageModel dual-backend.
//
// SwiftLM's binding constraint (proven empirically by the v2a flywheel) is VERIFIED-TRACE VOLUME:
// one slow strong on-device model generates rollouts serially, so best-of-N starves. `ComputePool`
// is the spine that fixes this — "one API, two execution realizations," lifted from *inference* to
// *placement*:
//   • LocalPool (one box): logical-parallel agents are distinct (context, KVCache) streams
//     CONTINUOUS/KV-batched into one forward pass — the trace-volume multiplier, testable today.
//   • ClusterPool (N studios): the same API over a from-scratch RPC control plane + MLX.Distributed,
//     for oversized models or to fan best-of-N across machines. Built when ≥2 boxes exist.
//
// Orchestrator code is identical across regimes — only the conformer swaps. The COST differs, not
// the results/permissions: `capabilities()` exposes a cost profile so a planner can decide whether
// fanning out actually helps (≈1 effective parallelism on LocalPool past the batch ceiling).

/// Identifies a base model (e.g. a HF repo id). The routing/batching key is the BASE, NOT
/// `(base, adapter)` — SwiftLM's edge is many specialized adapters co-batched on one frozen base.
public struct ModelID: Hashable, Sendable, Codable, CustomStringConvertible {
    public let raw: String
    public init(_ raw: String) { self.raw = raw }
    public var description: String { raw }
}

/// Identifies a LoRA adapter riding on a frozen base (`nil` = the bare base).
public struct AdapterID: Hashable, Sendable, Codable, CustomStringConvertible {
    public let raw: String
    public init(_ raw: String) { self.raw = raw }
    public var description: String { raw }
}

/// Scheduling priority. Interactive turns preempt background flywheel work (the 1-box serial GPU
/// can't run both at once, so background yields — per the effort/scheduling note in the plan).
public enum Priority: Int, Comparable, Sendable, Codable {
    case background = 0   // overnight self-train / best-of-N trace generation
    case normal = 1
    case interactive = 2  // a live user turn — must not wait behind a flywheel batch
    public static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
}

/// One unit of inference work. The Scheduler tags its queue by
/// `(model, adapter, priority, estTokens, affinity)` so a co-batching pool can fuse same-base work.
public struct InferenceRequest: Sendable, Identifiable {
    public let id: String
    public let model: ModelID
    public let adapter: AdapterID?
    public let prompt: String
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var priority: Priority
    public var estTokens: Int        // estimated output length, for batch packing / fairness
    public var affinity: String?     // hint: keep related work on one worker (KV/prefix reuse)
    /// Pre-rendered prompt tokens (already chat-templated by the caller). When set, the pool decodes THESE
    /// directly instead of tokenizing `prompt` — the sub-agent path needs it because a sub-agent round's
    /// input is a whole CONVERSATION, not a single user string. `nil` ⇒ tokenize `prompt` as before.
    public var inputTokens: [Int32]?

    public init(id: String = UUID().uuidString, model: ModelID, adapter: AdapterID? = nil,
                prompt: String, maxTokens: Int = 512, temperature: Float = 0.0, topP: Float = 1.0,
                repetitionPenalty: Float = 1.0, priority: Priority = .normal,
                estTokens: Int = 256, affinity: String? = nil, inputTokens: [Int32]? = nil) {
        self.id = id; self.model = model; self.adapter = adapter; self.prompt = prompt
        self.maxTokens = maxTokens; self.temperature = temperature; self.topP = topP
        self.repetitionPenalty = repetitionPenalty; self.priority = priority
        self.estTokens = estTokens; self.affinity = affinity; self.inputTokens = inputTokens
    }

    /// The co-batching key — BASE model + sampler params (NOT the adapter; SGMV co-batches adapters).
    public var batchKey: BatchKey {
        BatchKey(model: model, temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty)
    }
}

/// Requests sharing a `BatchKey` can be decoded together in one forward pass (same base + sampler).
public struct BatchKey: Hashable, Sendable {
    public let model: ModelID
    public let temperature: Float
    public let topP: Float
    public let repetitionPenalty: Float
}

/// A streamed unit of a generation (a decoded text chunk; `isFinal` marks the last).
public struct Token: Sendable {
    public let text: String
    public let isFinal: Bool
    public init(text: String, isFinal: Bool) { self.text = text; self.isFinal = isFinal }
}

/// What a worker can do + its COST profile. The cost-aware caveat: a pool guarantees identical
/// RESULTS and PERMISSIONS across regimes, NOT identical LATENCY — `effectiveParallelism` is ≈1 for
/// a LocalPool past its batch ceiling and ≈N for a ClusterPool, and the fan-out planner reads it to
/// avoid "parallelizing" work that won't actually go faster on one box.
public struct WorkerDescriptor: Sendable {
    public let id: String
    public let models: [ModelID]               // base models resident or loadable here
    public let batchWidth: Int                 // max concurrent streams fused into one forward pass
    public let effectiveParallelism: Double    // ≈1 (LocalPool past ceiling) … ≈N (ClusterPool)
    public let estTokensPerSecPerStream: Double // measured decode rate (drops as batch/KV grows)
    public let host: String?                   // where this worker lives (observability; nil = local/unset)

    public init(id: String, models: [ModelID], batchWidth: Int,
                effectiveParallelism: Double, estTokensPerSecPerStream: Double,
                host: String? = nil) {
        self.id = id; self.models = models; self.batchWidth = batchWidth
        self.effectiveParallelism = effectiveParallelism
        self.estTokensPerSecPerStream = estTokensPerSecPerStream
        self.host = host
    }
}

/// The execution backend. LocalPool (MLXBackend) and ClusterPool (later) both conform; orchestrator
/// code above the protocol is regime-agnostic.
public protocol ComputePool: Sendable {
    /// Run one request, streaming its tokens. The pool decides batching/placement internally.
    func submit(_ req: InferenceRequest) async -> AsyncStream<Token>
    /// Describe the workers (capacity + cost) so a planner can decide whether fan-out helps.
    func capabilities() async -> [WorkerDescriptor]
}

public extension ComputePool {
    /// Convenience: collect a request's full text (await completion). Most callers want this.
    func complete(_ req: InferenceRequest) async -> String {
        var out = ""
        for await tok in await submit(req) { out += tok.text }
        return out
    }
}
