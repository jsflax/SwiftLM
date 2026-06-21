import Foundation
import MLXLMCommon
import MiniBPE
import Serving

// ── The MLX↔Serving boundary for model-family detection. Serving's `ModelProfile` is MLX-free; this turns
// the two on-device signals into a profile, no network:
//   1. tool-call STYLE — from mlx-swift-lm's already-inferred `ToolCallFormat` (no re-parse, no re-implement);
//   2. the reasoning tie-break — `model_type` from the cached `config.json` (since `qwen2` alone can't tell a
//      plain Coder from an R1-distill), read from the same HF snapshot dir the grammar tokenizer uses.
enum ModelFamilyDetector {
    static func profile(forModelId modelId: String, toolCallFormat: ToolCallFormat?) -> ModelProfile {
        let style: ToolCallStyle = (toolCallFormat == .glm4) ? .taggedReasoning : .rawJSON
        let config = readConfig(forModelId: modelId)
        return ModelProfile.forModelType(
            config?["model_type"] as? String, toolCallStyle: style, modelId: modelId,
            contextWindow: (config?["max_position_embeddings"] as? Int) ?? 8192)   // → ContextBudget
    }

    /// The model's cached `config.json` (nil if the model isn't cached) — holds `model_type` (family
    /// tie-break) + `max_position_embeddings` (the context window that sets the compaction budget). No network.
    private static func readConfig(forModelId modelId: String) -> [String: Any]? {
        guard let dir = MiniBPE.snapshotDir(forModelId: modelId),
              let data = try? Data(contentsOf: dir.appending(path: "config.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj
    }
}

extension MLXLanguageModel {
    /// The model's detected family profile (reasoning?, tool-call style, generation budget). Read once after
    /// load; reuses mlx-swift-lm's already-inferred tool-call format + the cached `config.json`. No network.
    public var profile: ModelProfile {
        get async {
            ModelFamilyDetector.profile(forModelId: modelId,
                                        toolCallFormat: await container.configuration.toolCallFormat)
        }
    }

    /// The config-driven LOCAL AGENT ADAPTER — the model's own files (`config.json` / `generation_config.json` /
    /// `tokenizer_config.json` / `chat_template.jinja`) resolved into the per-model harness (format / reasoning /
    /// stops / owned-render). Reuses mlx's inferred format only as a FALLBACK. Read once after load; no network.
    /// (Phase 3+ swap the consumers from `profile` to this; kept additive so the backbone lands without churn.)
    public var localAdapter: LocalAgentAdapter {
        get async {
            let snap = SnapshotConfig.read(forModelId: modelId,
                                           mlxToolCallFormat: await container.configuration.toolCallFormat)
            return LocalAgentAdapter.resolve(snap)
        }
    }
}

extension SnapshotConfig {
    /// Read the model's own config from its HF snapshot — the ONE impure read behind the (pure) `resolve`.
    /// Handles BOTH template forms (a standalone `chat_template.jinja`, OR the embedded
    /// `tokenizer_config.json["chat_template"]` that half the cached models ship — JSONSerialization un-escapes
    /// it for us). Degrades to a minimal config (window 8192, defer-to-mlx) when the model isn't cached, but
    /// keeps `modelType` so the resolver's known-strict safety net still fires for an unreadable template.
    public static func read(forModelId modelId: String, mlxToolCallFormat: ToolCallFormat?) -> SnapshotConfig {
        let dir = MiniBPE.snapshotDir(forModelId: modelId)
        let config = dir.flatMap { readJSON($0.appending(path: "config.json")) }
        let genConfig = dir.flatMap { readJSON($0.appending(path: "generation_config.json")) }
        let tokConfig = dir.flatMap { readJSON($0.appending(path: "tokenizer_config.json")) }

        let modelType = config?["model_type"] as? String
        let window = (config?["max_position_embeddings"] as? Int) ?? 8192

        // Template: prefer the standalone .jinja file; else the embedded tokenizer_config field.
        let jinja = dir.flatMap { try? String(contentsOf: $0.appending(path: "chat_template.jinja"), encoding: .utf8) }
        let template = jinja ?? (tokConfig?["chat_template"] as? String)

        // EOS ids: generation_config wins, else config (each may be a scalar or an array).
        let eosIds = intList(genConfig?["eos_token_id"]) ?? intList(config?["eos_token_id"]) ?? []
        // EOS string + any generation stop strings.
        let eosStr = tokenString(tokConfig?["eos_token"])
        let stopStrings = stringList(genConfig?["stop_strings"]) ?? []

        return SnapshotConfig(
            modelId: modelId, modelType: modelType, contextWindow: window, chatTemplate: template,
            eosTokenStrings: eosStr.map { [$0] } ?? [], eosTokenIds: eosIds,
            generationStopStrings: stopStrings, mlxInferredFormat: choice(from: mlxToolCallFormat))
    }

    /// Map mlx-swift-lm's inferred `ToolCallFormat` → our MLX-free choice; anything we don't model → defer to mlx.
    private static func choice(from f: ToolCallFormat?) -> ToolCallFormatChoice {
        switch f {
        case .some(.json): return .json
        case .some(.xmlFunction): return .xmlFunction
        case .some(.glm4): return .glm4
        default: return .deferToMLX
        }
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj
    }

    /// eos_token_id is a scalar Int OR an array of Ints (the 122B ships two).
    private static func intList(_ v: Any?) -> [Int]? {
        if let i = v as? Int { return [i] }
        if let a = v as? [Int] { return a }
        if let a = v as? [Any] { let ints = a.compactMap { $0 as? Int }; return ints.isEmpty ? nil : ints }
        return nil
    }

    private static func stringList(_ v: Any?) -> [String]? { (v as? [Any])?.compactMap { $0 as? String } }

    /// eos_token is a String OR a {"content": "<|…|>"} object (HF AddedToken serialization).
    private static func tokenString(_ v: Any?) -> String? {
        if let s = v as? String { return s }
        if let o = v as? [String: Any], let c = o["content"] as? String { return c }
        return nil
    }
}
