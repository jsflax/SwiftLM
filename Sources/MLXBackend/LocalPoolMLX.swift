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
}
