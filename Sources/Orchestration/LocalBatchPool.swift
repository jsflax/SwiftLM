import Foundation

// ── SLICE 2 — the COALESCING ComputePool conformer (the sub-agent fan-out batching multiplier).
//
// `LocalPool`/`FanOutPool` (TracePool) cover best-of-N: N rollouts of the SAME prompt, started
// together — the easy batching case. Sub-agent fan-out is the HARDER case: N DIFFERENT agents, each a
// full `runWithTools` turn, that arrive at their per-round generation step at slightly different times
// and with DIFFERENT-length conversations. `LocalBatchPool` is a continuous micro-batcher for exactly
// that: concurrent `submit`s sharing a `batchKey` (same base + sampler) are buffered for a short
// coalescing window, then fused into ONE forward pass via the variable-length `batchDecodeDistinct`
// (SLICE 1b). A lone request just runs as a batch of one; a synchronized fan-out of K runs as one
// K-row pass — the GPU-batched win, degrading gracefully as agents desync.
//
// The actual decode is INJECTED (`runBatch`) so this type is pure-Swift + unit-testable; MLXBackend
// wires `batchGenerateDistinct` in where the model lives (see `MLXLanguageModel.makeBatchPool`). The
// `Scheduler` sits in front to cap admitted concurrency at the batch width.
public actor LocalBatchPool: ComputePool {
    private let descriptor: WorkerDescriptor
    private let batchWidth: Int
    private let coalesceWindow: Duration
    /// Decode a coalesced batch of requests (same `batchKey`) → one completion each, in input order.
    private let runBatch: @Sendable ([InferenceRequest]) async -> [String]

    private struct Pending { let req: InferenceRequest; let cont: CheckedContinuation<String, Never> }
    /// Requests waiting to be fused, bucketed by co-batch key (base + sampler).
    private var buffers: [BatchKey: [Pending]] = [:]
    /// Keys with a flush already scheduled (so the window timer is armed at most once per key).
    private var scheduled: Set<BatchKey> = []

    public init(descriptor: WorkerDescriptor, batchWidth: Int? = nil, coalesceWindowMillis: Int = 5,
                runBatch: @escaping @Sendable ([InferenceRequest]) async -> [String]) {
        self.descriptor = descriptor
        self.batchWidth = max(1, batchWidth ?? descriptor.batchWidth)
        self.coalesceWindow = .milliseconds(max(0, coalesceWindowMillis))
        self.runBatch = runBatch
    }

    public func capabilities() async -> [WorkerDescriptor] { [descriptor] }

    public func submit(_ req: InferenceRequest) async -> AsyncStream<Token> {
        // v1 is non-streaming: coalesced rows decode in lockstep, so a single final chunk is delivered
        // per request (sub-agent generation legs want the whole assistant turn, not token-streaming).
        let text = await coalescedGenerate(req)
        return AsyncStream<Token> { continuation in
            continuation.yield(Token(text: text, isFinal: true))
            continuation.finish()
        }
    }

    // MARK: - Coalescing core

    private func coalescedGenerate(_ req: InferenceRequest) async -> String {
        await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            buffers[req.batchKey, default: []].append(Pending(req: req, cont: cont))
            if buffers[req.batchKey]!.count >= batchWidth {
                Task { await self.flush(req.batchKey) }              // a full batch is ready — fire now
            } else {
                armFlush(req.batchKey)                               // else fire after the window
            }
        }
    }

    /// Arm the window timer for `key` if not already armed. When it fires, whatever has accumulated is
    /// fused (even a batch of one). `flush` drains the buffer atomically, so a redundant timer is a no-op.
    private func armFlush(_ key: BatchKey) {
        guard !scheduled.contains(key) else { return }
        scheduled.insert(key)
        Task {
            try? await Task.sleep(for: coalesceWindow)
            await self.flush(key)
        }
    }

    private func flush(_ key: BatchKey) async {
        scheduled.remove(key)
        guard let pending = buffers[key], !pending.isEmpty else { return }
        // Take up to one batch; if more than a full batch accumulated (e.g. width-capped fan-out), keep
        // the overflow for the next pass and re-arm.
        let batch = Array(pending.prefix(batchWidth))
        let overflow = Array(pending.dropFirst(batchWidth))
        buffers[key] = overflow.isEmpty ? nil : overflow
        if !overflow.isEmpty { armFlush(key) }

        let texts = await runBatch(batch.map { $0.req })            // the one fused forward pass
        for (i, p) in batch.enumerated() {
            p.cont.resume(returning: i < texts.count ? texts[i] : "")
        }
    }
}
