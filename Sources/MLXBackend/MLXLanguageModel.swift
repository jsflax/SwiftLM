import Foundation
import MLXLLM
import MLXVLM            // vision model factory (Qwen35MoE etc.) — registered via the trampoline below
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
    /// Set only on the LEADER (rank 0) of a pipeline-distributed load — carries the ring group + the
    /// side-channel to the follower. The turn loop (`CompactingSession.ownedRound`) reads it to feed each
    /// round's input row to the follower. nil ⇒ single-machine (the normal path).
    public let distributedContext: DistributedContext?

    private init(modelId: String, container: ModelContainer,
                 distributedContext: DistributedContext? = nil) {
        self.modelId = modelId
        self.container = container
        self.distributedContext = distributedContext
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
        // Register the VLM model factory so a vision model (e.g. Qwen3.5-122B's `qwen3_5_moe`) routes to MLXVLM's
        // `Qwen35MoE` instead of the text factory. `#huggingFaceLoadModelContainer` resolves the factory via a
        // DYNAMIC `NSClassFromString("MLXVLM.TrampolineModelFactory")` lookup in `ModelFactoryRegistry`, which only
        // succeeds when MLXVLM's object code is present in the binary — so reference the trampoline CLASS here to
        // stop the linker dead-stripping it (referencing `VLMTypeRegistry` is NOT enough). Text-only models throw
        // `unsupportedModelType` in the VLM factory and fall through to the LLM factory. Proven by the P0 spike.
        _ = MLXVLM.TrampolineModelFactory.self
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(id: id))
        // Guard the silent-text-only regression: if MLXVLM ever fails to register, a vision model loads through the
        // text factory with NO error and just never sees pixels. Log the resolved model class so that's visible.
        let loadedClass = await container.perform { ctx in String(describing: type(of: ctx.model)) }
        FileHandle.standardError.write(Data("[mlx-load] \(id) → \(loadedClass)\n".utf8))
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

    /// Load a model **pipeline-sharded across the ring** in `group` (M3). Same path as ``load``,
    /// but brackets the factory weight-load with `PipelineLoad` so each rank loads ONLY its shard
    /// (and warms it up) — the model is too big for one machine. Orbital stays agnostic: the
    /// returned `MLXLanguageModel` generates via the same `.generate(...)`. `boundary` (2-rank only)
    /// is the number of leading layers placed on the LAST rank (the memory-constrained peer); pass
    /// identically on every rank. Run the SAME binary+args on every Mac; the ring (env `MLX_RANK` /
    /// `MLX_HOSTFILE`) distinguishes ranks and the forward's collectives keep them lock-step.
    public static func loadDistributed(
        modelId: String,
        group: DistributedGroup,
        boundary: Int? = nil,
        distributedContext: DistributedContext? = nil
    ) async throws -> MLXLanguageModel {
        _ = MLXVLM.TrampolineModelFactory.self    // keep the VLM factory linked (see load())
        PipelineLoad.active = PipelineLoad.Config(group: group, boundary: boundary)
        defer { PipelineLoad.active = nil }
        // Load from the LOCAL HF snapshot dir when present, so a partially-cached rank loads ONLY the
        // shard files it already has — never a full-repo re-download (`resolve()` skips download for a
        // `.directory` config). Each rank holds exactly its own pipeline shard (Python sharded_load did
        // the same partial download). Falls back to the Hub `id` path if nothing is cached.
        let config: ModelConfiguration
        if let snapshot = Self.localHFSnapshot(modelId: modelId) {
            FileHandle.standardError.write(
                Data("[mlx-load-dist rank \(group.rank)] local snapshot \(snapshot.path) (no download)\n".utf8))
            config = ModelConfiguration(directory: snapshot)
        } else {
            config = ModelConfiguration(id: modelId)
        }
        let container = try await #huggingFaceLoadModelContainer(configuration: config)
        let loadedClass = await container.perform { ctx in String(describing: type(of: ctx.model)) }
        FileHandle.standardError.write(
            Data("[mlx-load-dist rank \(group.rank)/\(group.size)] \(modelId) → \(loadedClass)\n".utf8))
        let cacheGB = ProcessInfo.processInfo.environment["SWIFTLM_MLX_CACHE_LIMIT_GB"]
            .flatMap(Int.init) ?? 4
        MLX.Memory.cacheLimit = cacheGB << 30
        return MLXLanguageModel(modelId: modelId, container: container,
                                distributedContext: distributedContext)
    }

    /// Form the MLX ring from the environment (`MLX_HOSTFILE` / `MLX_RANK`, the proven Track-A
    /// contract — listener rank 0 first) and load the model pipeline-sharded across it. The group
    /// is retained by the loaded model. Returns the rank/size so a caller can print rank-aware
    /// (rank 0 is the output rank). This keeps Orbital free of any direct MLX/`DistributedGroup`
    /// import — the two-machine detail lives entirely in SwiftLM. (M4 will form the ring from the
    /// Lattice roster instead of env; this env path is the M3 bring-up vehicle.)
    public static func loadDistributedFromRingEnv(
        modelId: String,
        boundary: Int? = nil
    ) async throws -> (model: MLXLanguageModel, rank: Int, size: Int) {
        let group = try DistributedGroup(strict: .ring)
        let model = try await loadDistributed(modelId: modelId, group: group, boundary: boundary)
        return (model, group.rank, group.size)
    }

    /// Form the ring from an EXPLICIT host list (no environment) and load the model pipeline-sharded
    /// across it. This is the no-env path M4 uses: `hosts` come from the Lattice cluster roster, not
    /// `MLX_HOSTFILE`. `hosts` is one `"ip:port"` per rank in rank order (rank 0 = listener, first);
    /// `rank` is this machine's index. Otherwise identical to ``loadDistributedFromRingEnv``.
    public static func loadDistributedFromHosts(
        modelId: String,
        hosts: [String],
        rank: Int,
        boundary: Int? = nil
    ) async throws -> (model: MLXLanguageModel, rank: Int, size: Int) {
        let group = try DistributedGroup(ringHosts: hosts, rank: rank)
        let model = try await loadDistributed(modelId: modelId, group: group, boundary: boundary)
        return (model, group.rank, group.size)
    }

    /// Resolve a Hugging Face repo id to its local snapshot directory in the HF hub cache, or `nil`
    /// if not cached. Honors `HF_HUB_CACHE` / `HF_HOME`, else `~/.cache/huggingface/hub`. Picks the
    /// snapshot that actually has a `config.json` (skips incomplete/blob-only dirs).
    static func localHFSnapshot(modelId: String) -> URL? {
        let env = ProcessInfo.processInfo.environment
        let hubCache: URL
        if let c = env["HF_HUB_CACHE"], !c.isEmpty {
            hubCache = URL(fileURLWithPath: c)
        } else if let h = env["HF_HOME"], !h.isEmpty {
            hubCache = URL(fileURLWithPath: h).appendingPathComponent("hub")
        } else {
            hubCache = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub")
        }
        let munged = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
        let snapshots = hubCache.appendingPathComponent("\(munged)/snapshots")
        guard let dirs = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil) else { return nil }
        for d in dirs where FileManager.default.fileExists(
            atPath: d.appendingPathComponent("config.json").path) {
            return d
        }
        return nil
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
        // Keep the (100+ GB) weights GPU-resident during decode — mirrors mlx_lm's
        // `with wired_limit(model)` (mlx_lm/generate.py). WITHOUT this, on a memory-tight box the
        // routed-expert weight pages get evicted and EVERY gather_qmm re-faults them — measured ~40x
        // slower MoE (1.8 s/token vs ~0.04 s), i.e. ~100x the memory-bandwidth floor. The wired limit
        // is device-adaptive (≈ Metal's recommendedMaxWorkingSetSize, ~75% RAM) so the 64 GB peer is
        // safe too.
        // Keep the resident model wired in RAM during decode (mirrors mlx_lm's `with wired_limit`).
        // WITHOUT this, on a memory-tight box the routed-expert weight pages get evicted and EVERY
        // gather_qmm re-faults them — measured ~13x slower (0.56 vs 7.4 tok/s on GLM-4.6 across 2 Macs).
        // Metal's recommended working-set (115 GB on a 128 GB box) covers the 107 GB shard; mlx caps the
        // wired limit AT that value (setting higher throws), so this is the max safe wiring.
        let wired = MLX.GPU.maxRecommendedWorkingSetBytes() ?? (96 << 30)
        return try await MLX.GPU.withWiredLimit(wired) {
            try await session.respond(to: prompt)
        }
    }

    /// DIAGNOSTIC (perf): pure-model decode benchmark — detok-free per-token rate plus a per-32-token
    /// wall-clock curve, all in ONE thermally-consistent run. The synchronous token loop's `generateTime`
    /// EXCLUDES the streaming detokenizer (the output is decoded once at the end), so this isolates the
    /// model's true per-token speed from detok overhead and exposes any O(n) growth with sequence position
    /// — disambiguating "is the gap the model or the detokenizer?" without cross-run thermal noise. Greedy
    /// only (so the distributed ring stays lock-step when both ranks run this same loop). Logs via `log`.
    public func generateBenchmark(
        _ prompt: String, maxTokens: Int, temperature: Float = 0.0,
        log: @Sendable @escaping (String) -> Void
    ) async throws -> (output: String, tokPerSec: Double, generateTime: Double) {
        let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, topP: 1.0)
        let wired = MLX.GPU.maxRecommendedWorkingSetBytes() ?? (96 << 30)
        return try await MLX.GPU.withWiredLimit(wired) {
            try await container.perform { ctx in
                let input = try await ctx.processor.prepare(input: UserInput(prompt: prompt))
                let cache = ctx.model.newCache(parameters: params)
                let iterator = try TokenIterator(
                    input: input, model: ctx.model, cache: cache, parameters: params)
                var blockStart = Date()
                var lastCount = 0
                let result = MLXLMCommon.generate(
                    input: input, context: ctx, iterator: iterator
                ) { tokens in
                    let n = tokens.count
                    if n - lastCount >= 32 {
                        let dt = Date().timeIntervalSince(blockStart)
                        log(String(format: "  tok %4d..%4d: %.3fs  (%.1f tok/s)",
                                   lastCount, n, dt, Double(n - lastCount) / max(dt, 1e-9)))
                        blockStart = Date()
                        lastCount = n
                    }
                    return .more
                }
                return (result.output, result.tokensPerSecond, result.generateTime)
            }
        }
    }
}
