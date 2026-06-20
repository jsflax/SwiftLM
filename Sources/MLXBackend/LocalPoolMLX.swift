import Foundation
import Orchestration

// Wires the MLX model into the cluster's TracePool interface. A cluster WORKER is just this model
// behind a LocalPool behind a WorkerServer — generation only. Verification (the repo clonefile +
// build + test) stays on the COORDINATOR, so workers need only the model, not the user's repos.

extension MLXLanguageModel {
    /// Expose this model as a `LocalPool`: a one-box `TracePool` whose generation runs batched
    /// best-of-N. The cost descriptor reports this box's batch width + measured decode rate so a
    /// fan-out planner can reason about it.
    public func makeLocalPool(workerId: String? = nil, batchWidth: Int = 12,
                              estTokPerSec: Double = 12) -> LocalPool {
        let wid = workerId ?? (Host.current().localizedName ?? "local")
        let desc = WorkerDescriptor(id: wid, models: [ModelID(modelId)], batchWidth: batchWidth,
                                    effectiveParallelism: 1, estTokensPerSecPerStream: estTokPerSec)
        return LocalPool(descriptor: desc) { job in
            await self.batchGenerate(job.prompt, n: job.n, maxTokens: job.maxTokens,
                                     temperature: job.temperature, topP: job.topP)
        }
    }

    /// Expose this model as a `LocalBatchPool` (SLICE 2): a coalescing `ComputePool` that fuses
    /// concurrent DIFFERENT-prompt requests (sub-agent fan-out) into one variable-length forward pass
    /// via `batchDecodeDistinct`. All coalesced rows share a `BatchKey`, so they have one sampler; the
    /// batch's `maxTokens` is the max across its requests (rows evict on their own EOS). Put a
    /// `Scheduler` (concurrency cap = batchWidth) in front for priority admission.
    public func makeBatchPool(workerId: String? = nil, batchWidth: Int = 8,
                              coalesceWindowMillis: Int = 5, estTokPerSec: Double = 12) -> LocalBatchPool {
        let wid = workerId ?? (Host.current().localizedName ?? "local")
        let desc = WorkerDescriptor(id: wid, models: [ModelID(modelId)], batchWidth: batchWidth,
                                    effectiveParallelism: 1, estTokensPerSecPerStream: estTokPerSec)
        return LocalBatchPool(descriptor: desc, batchWidth: batchWidth,
                              coalesceWindowMillis: coalesceWindowMillis) { reqs in
            let maxTok = reqs.map { $0.maxTokens }.max() ?? 512
            let temp = reqs.first?.temperature ?? 0
            // Sub-agent path: requests carry PRE-RENDERED conversation tokens → decode them directly (no
            // chat template). Otherwise (flywheel / single-prompt) tokenize the prompt string as before.
            if reqs.allSatisfy({ $0.inputTokens != nil }) {
                if ProcessInfo.processInfo.environment["SWIFTLM_BATCH_DEBUG"] != nil {
                    FileHandle.standardError.write(Data(("[batch-pool] coalesced \(reqs.count) sub-agent row(s) "
                        + "into one forward pass (lens \(reqs.map { $0.inputTokens!.count }))\n").utf8))
                }
                return await self.batchGenerateRows(reqs.map { $0.inputTokens! }, maxTokens: maxTok, temperature: temp)
            }
            return await self.batchGenerateDistinct(reqs.map { $0.prompt }, maxTokens: maxTok, temperature: temp)
        }
    }

    /// B2 co-batch DI: a pool-backed `BatchGenerator` to pass into `makeAgentBackend(batchGenerator:)`. Hold
    /// ONE per model and share it across ALL that model's agent backends, so CONCURRENT turns (esp. a consult
    /// fan-out: N panelists answering one question simultaneously) coalesce into ONE batched forward pass via
    /// `LocalBatchPool` (the proven `MLXSubagentRunner` pattern). Hides the `Orchestration` types behind the
    /// `MLXBackend` API so the caller (Orbital's `SharedMLXScheduler`) needs only `import MLXBackend`.
    public func makeCoBatchGenerator(batchWidth: Int = 8, coalesceWindowMillis: Int = 200) -> BatchGenerator {
        let pool = makeBatchPool(batchWidth: batchWidth, coalesceWindowMillis: coalesceWindowMillis)
        let mid = modelId
        return { @Sendable (tokens: [Int32], maxTok: Int) in
            await pool.complete(InferenceRequest(model: ModelID(mid), prompt: "",
                                                 maxTokens: maxTok, inputTokens: tokens))
        }
    }
}
