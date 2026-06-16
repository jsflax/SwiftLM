import Foundation
import MLXLLM
import MLXLMCommon

extension MLXLanguageModel {
    /// Hot-swap a trained LoRA adapter onto the resident frozen base. `directory` holds
    /// the standard `adapter_config.json` + `adapters.safetensors` (as written by
    /// `trainLoRA`). After this call, `generate` / `runWithTools` use the adapter —
    /// redeploy = pointer flip, base weights untouched.
    public func loadAdapter(directory: URL) async throws {
        try await container.perform { (ctx: ModelContext) in
            let adapter = try LoRAContainer.from(directory: directory)
            try adapter.load(into: ctx.model)
        }
    }
}
