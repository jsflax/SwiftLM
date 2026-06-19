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
}
