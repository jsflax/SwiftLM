import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import MLX
import Tokenizers
import Hub
import HuggingFace
import MiniBPE

// The MLX backend — SwiftLM's Mac runtime: serves, LoRA-trains, and runs the
// self-improvement loop. The third backend behind SwiftLM's `LanguageModel`
// protocol, alongside CoreMLLanguageModel (iPhone/ANE) and FoundationLanguageModel.
//
// MLX uses Metal, so this target must be built with xcodebuild (not `swift build`).
// Validated foundation (S1/S5/S6): 100 tok/s serve, pure-Swift LoRA, live MCP tools.

/// An MLX-served language model. Immutable (`Sendable`): the model container is
/// loaded once via `load()` and held by reference. `ModelContainer` is itself an
/// actor, so concurrent generation is serialized safely inside it.
public final class MLXLanguageModel: Sendable {
    public let modelId: String
    let container: ModelContainer

    private init(modelId: String, container: ModelContainer) {
        self.modelId = modelId
        self.container = container
    }

    /// Load an mlx-community model and return a ready backend. The base is overridable via the
    /// `SWIFTLM_MODEL` env var (e.g. `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit` for the
    /// reasoning-grade prod band) without a rebuild; 7B-Coder stays the fast default test harness.
    public static func load(
        modelId: String? = nil
    ) async throws -> MLXLanguageModel {
        let id = modelId
            ?? ProcessInfo.processInfo.environment["SWIFTLM_MODEL"]
            ?? "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(id: id))
        // Bound MLX's GPU buffer-cache pool. By default mlx-swift's cacheLimit == the memoryLimit
        // (~1.5× the device working-set size ≈ effectively unbounded on a high-RAM box), and on free() a
        // buffer is RECYCLED INTO the pool instead of returned to the OS — so over a long/heavy serving run
        // the in-process working set balloons well past the model weights (mlx's own docs warn of "several GB
        // of cached memory from accumulated buffers"). The TRAINING path already calls clearCache(); the
        // SERVING path never bounded it. Cap the pool (env-tunable; 0 disables the cache) so the working set
        // stays bounded — small caches typically match unconstrained throughput.
        let cacheGB = ProcessInfo.processInfo.environment["SWIFTLM_MLX_CACHE_LIMIT_GB"].flatMap(Int.init) ?? 4
        MLX.Memory.cacheLimit = cacheGB << 30
        let m = MLXLanguageModel(modelId: id, container: container)
        // Part C: install the Resident Trait-Bank once per container, gated by env (default OFF in C1 — there are
        // no trained traits yet, so live rooms stay byte-identical to base). When on, base Linear/QuantizedLinear
        // leaves are swapped to resident layers with EMPTY stores ⇒ still byte-identical until a trait is
        // registered AND made active via `LoRARuntime.activeTraits`. The capture+bind plumbing runs either way.
        if LoRARuntime.bankEnabled {
            await m.installResidentTraitBank()
        }
        return m
    }

    /// Text completion. A repetition penalty is on by default — LoRA adapters overfit to
    /// a narrow style and degenerate into loops under pure-greedy decoding without it. BUT for
    /// R1-distill reasoning bases, DeepSeek's recommended setup is temp 0.6 / top_p 0.95 / NO
    /// rep-penalty (a penalty fights long CoT and corrupts the <think> tags) — pass
    /// `repetitionPenalty: 1.0` to disable and set `topP: 0.95`.
    public func generate(
        _ prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.0,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.15
    ) async throws -> String {
        var params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: topP)
        if repetitionPenalty > 1.0 {            // disabled when ≤ 1.0 (reasoning models)
            params.repetitionPenalty = repetitionPenalty
            params.repetitionContextSize = 20
        }
        let session = ChatSession(container, generateParameters: params)
        return try await session.respond(to: prompt)
    }
}
