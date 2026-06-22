import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import SelfImprove

// The self-improvement engine — the validated S5 ("the bet") pure-Swift LoRA cycle,
// productized onto the MLX backend: train a frozen-base LoRA adapter on a curriculum,
// gated by a before/after external eval (held-out loss down, retention within budget).

extension MLXLanguageModel {
    public struct LoRAConfig: Sendable {
        public var numLayers: Int = 16
        public var learningRate: Float = 1e-4
        public var iterations: Int = 300
        public var batchSize: Int = 2
        /// Hard cap on rendered (prefix+completion) token length. DPO retains BOTH chosen and
        /// rejected forward graphs at once, and rejected = FAILED rollouts that often ran to the
        /// full maxTok — so uncapped long sequences OOM a 32B. Truncates the completion tail.
        public var maxSeqLen: Int = 2048
        public init() {}
    }

    /// Before/after eval-gate numbers from one LoRA cycle.
    public struct LoRACycleResult: Sendable {
        public let beforeHeldout: Float
        public let afterHeldout: Float
        public let beforeRetention: Float
        public let afterRetention: Float
        public let adapterURL: URL
        /// Generalization: held-out cross-entropy went down.
        public var betPasses: Bool { afterHeldout < beforeHeldout }
        /// No catastrophic forgetting: non-transcript retention within 5%.
        public var retentionHolds: Bool { (afterRetention - beforeRetention) / beforeRetention <= 0.05 }
    }

    /// Diagnostic: render a few TrainPairs through the EXACT training path (applyChatTemplate
    /// + common-prefix mask boundary) and decode the trained region, to verify the masking
    /// trains on precisely the assistant turn. No training, no mutation.
    public func debugRenderTrainPairs(
        _ pairs: [TrainPair]
    ) async -> [(start: Int, total: Int, full: String, trained: String)] {
        await container.perform { ctx in
            MaskedTraining.debugRender(pairs, tokenizer: ctx.tokenizer)
        }
    }

    /// Train a frozen-base LoRA adapter on `train`, measuring held-out + retention loss
    /// before and after. Mutations to `ctx.model` persist in the resident container, so
    /// generation after this call uses the trained adapter (hot-swap, validated S5).
    public func trainLoRA(
        train: [TrainPair],
        heldout: [String],
        retention: [String],
        adapterDir: URL,
        config: LoRAConfig = LoRAConfig()
    ) async throws -> LoRACycleResult {
        MLX.GPU.clearCache()   // free batched-generation KV caches before training allocates (OOM fix)
        return try await container.perform { (ctx: ModelContext) -> LoRACycleResult in
            func eval(_ data: [String]) -> Float {
                LoRATrain.evaluate(model: ctx.model, dataset: data, tokenizer: ctx.tokenizer,
                                   batchSize: config.batchSize, batchCount: 0)
            }
            // BEFORE — frozen base, no adapter yet.
            let beforeHeld = eval(heldout)
            let beforeRet = eval(retention)

            // Apply LoRA (initially a no-op: B=0) and train on the curriculum.
            // Part C contract: the resident trait-bank and the LoRA trainer must NOT co-reside on one container —
            // LoRAContainer.from→replaceLayers casts `child as? Linear`, and a resident layer IS-A Linear, so it
            // would wrap a trainable LoRA AROUND the resident leaf (corrupting both). Fail fast on misconfig.
            precondition(!LoRABank.isInstalled(in: ctx.model),
                "LoRA training on a container with the resident trait-bank installed — mutually exclusive (resident "
                + "layers would be double-wrapped). Train without SWIFTLM_TRAIT_BANK set.")
            let loraConfig = LoRAConfiguration(numLayers: config.numLayers)
            _ = try LoRAContainer.from(model: ctx.model, configuration: loraConfig)
            let optimizer = AdamW(learningRate: config.learningRate)
            // Assistant-only loss masking (our own loop) — train on assistant tokens only,
            // not the user prompt + chat template. Removes the format-fitting confound.
            MaskedTraining.train(
                model: ctx.model, data: train, optimizer: optimizer, tokenizer: ctx.tokenizer,
                iterations: config.iterations, batchSize: config.batchSize)

            // Save as a standard adapter directory so it can be hot-swapped back via
            // LoRAContainer.from(directory:).load(into:).
            try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
            try JSONEncoder().encode(loraConfig)
                .write(to: adapterDir.appending(component: "adapter_config.json"), options: .atomic)
            try LoRATrain.saveLoRAWeights(
                model: ctx.model, url: adapterDir.appending(component: "adapters.safetensors"))

            // AFTER — same frozen base + trained adapter.
            let afterHeld = eval(heldout)
            let afterRet = eval(retention)
            return LoRACycleResult(
                beforeHeldout: beforeHeld, afterHeldout: afterHeld,
                beforeRetention: beforeRet, afterRetention: afterRet, adapterURL: adapterDir)
        }
    }

    /// DPO variant of the LoRA cycle: train on (chosen, rejected) PREFERENCE pairs instead of SFT on
    /// verified-only traces. Same frozen-base + adapter + before/after eval gate; the only change is
    /// the training objective (preference loss, using the failed rollouts as negatives).
    public func trainDPO(
        pairs: [DPOTraining.Pair], heldout: [String], retention: [String],
        adapterDir: URL, config: LoRAConfig = LoRAConfig(), beta: Float = 0.1
    ) async throws -> LoRACycleResult {
        MLX.GPU.clearCache()
        return try await container.perform { (ctx: ModelContext) -> LoRACycleResult in
            func eval(_ data: [String]) -> Float {
                LoRATrain.evaluate(model: ctx.model, dataset: data, tokenizer: ctx.tokenizer,
                                   batchSize: config.batchSize, batchCount: 0)
            }
            let beforeHeld = eval(heldout); let beforeRet = eval(retention)
            precondition(!LoRABank.isInstalled(in: ctx.model),   // Part C contract: trait-bank ⊥ LoRA trainer (see trainLoRA)
                "LoRA/DPO training on a container with the resident trait-bank installed — mutually exclusive. "
                + "Train without SWIFTLM_TRAIT_BANK set.")
            let loraConfig = LoRAConfiguration(numLayers: config.numLayers)
            _ = try LoRAContainer.from(model: ctx.model, configuration: loraConfig)
            let optimizer = AdamW(learningRate: config.learningRate)
            DPOTraining.train(model: ctx.model, pairs: pairs, optimizer: optimizer,
                              tokenizer: ctx.tokenizer, iterations: config.iterations,
                              batchSize: config.batchSize, beta: beta, maxLen: config.maxSeqLen)
            try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)
            try JSONEncoder().encode(loraConfig)
                .write(to: adapterDir.appending(component: "adapter_config.json"), options: .atomic)
            try LoRATrain.saveLoRAWeights(
                model: ctx.model, url: adapterDir.appending(component: "adapters.safetensors"))
            let afterHeld = eval(heldout); let afterRet = eval(retention)
            return LoRACycleResult(
                beforeHeldout: beforeHeld, afterHeldout: afterHeld,
                beforeRetention: beforeRet, afterRetention: afterRet, adapterURL: adapterDir)
        }
    }
}
