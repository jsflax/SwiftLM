import Foundation
import MLX
import MLXNN
import MLXLMCommon

// ── MoE training fix #2: make the GatedDeltaNet (linear-attention) layers backprop-safe ─────────────────────────
//
// The qwen3_5_moe / 122B-A10B is HYBRID: config `full_attention_interval: 4` ⇒ 3 of every 4 decoder layers are
// `GatedDeltaNet` (linear attention). mlx-swift-lm implements its recurrence with a hand-written Metal kernel
// (`gatedDeltaUpdate` → `MLXFast.metalKernel`, GatedDelta.swift:102) that has NO vjp; its pure-MLX differentiable
// fallback (`gatedDeltaOps`) exists but is unreachable (the kernel manager is `private`/immutable, always non-nil).
// So any gradient flowing THROUGH a linear-attention layer aborts the backward:
//     "[Primitive::vjp] Not implemented for CustomKernel."
// The loaded class is `Qwen35MoEModel` → `Qwen35Model`, decoder `Qwen35DecoderLayer` (MLXLLM/Models/Qwen35.swift
// ~437-494): `r = linearAttn!(inputLayerNorm(x), …)` then the residual `h = x + r`.
//
// FIX (non-vendoring; a 4-lens fan-out picked this over custom-vjp/kernel-disable/model-reregister, all blocked):
// DETACH the linear-attention branch's INPUT. We replace each LINEAR layer's `input_layernorm` (an `MLXNN.RMSNorm`,
// which IS `open`) with a stop-grad wrapper. The GatedDeltaNet then consumes only CONSTANTS, so in reverse-mode MLX
// no cotangent ever reaches the CustomKernel and its (missing) vjp is never requested. Gradient to the trainable
// self_attn LoRA still flows: (1) the residual SKIP `x` in `h = x + r` is NOT detached, so it carries gradient to
// every layer below; (2) the FULL-attention layers (1 of 4) keep their intact `input_layernorm`, so their
// `self_attn.{q,k,v,o}_proj` LoRA trains normally. `stopGradient` is a forward no-op and the fast kernel + frozen
// weights are untouched ⇒ inference stays byte-identical. (Reuses the `StopGradGate` idiom in MoEStopGradFix.swift.)
// COST: the linear-attention layers contribute no gradient (their `linear_attn.*` LoRA is dropped); the adapter
// trains on the full-attention layers' self_attn — a standard, effective attention-only LoRA footprint.

/// An `RMSNorm` that returns `stopGradient` of an inner real `RMSNorm`'s output. The inner norm (with its loaded
/// weight + eps) is held as a child so the forward value is identical; the dummy 1-dim base weight is never used
/// (`callAsFunction` is overridden).
public final class StopGradRMSNorm: RMSNorm {
    @ModuleInfo(key: "inner") var inner: RMSNorm

    public init(wrapping norm: RMSNorm) {
        self._inner.wrappedValue = norm
        super.init(dimensions: 1, eps: 1e-6)   // unused placeholder — forward is overridden to use `inner`
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        stopGradient(inner(x))
    }
}

public enum LinearAttnStopGrad {
    /// On every LINEAR (GatedDeltaNet) decoder layer — detected by a non-nil `linear_attn` child — wrap the
    /// `input_layernorm` in a `StopGradRMSNorm`, severing gradient into the no-vjp kernel while the residual skip
    /// keeps the lower self_attn LoRA in a correct graph. Returns the count wrapped (0 ⇒ dense / non-hybrid base —
    /// harmless). Idempotent. MUST run BEFORE `LoRAContainer.from`, alongside `MoEStopGrad.install`; pair with
    /// self_attn-only LoRA keys (the linear_attn projections sit in the detached branch).
    @discardableResult
    public static func install(into model: LanguageModel) -> Int {
        guard let lora = model as? LoRAModel else { return 0 }
        var total = 0
        for layer in lora.loraLayers {
            let mods = layer.namedModules()
            let isLinear = mods.contains { key, _ in key == "linear_attn" }
            guard isLinear else { continue }
            var updates: [(String, Module)] = []
            for (key, child) in mods where key == "input_layernorm" {
                if let rms = child as? RMSNorm, !(rms is StopGradRMSNorm) {
                    updates.append((key, StopGradRMSNorm(wrapping: rms)))
                }
            }
            if !updates.isEmpty {
                layer.update(modules: .unflattened(updates))
                total += updates.count
            }
        }
        return total
    }
}
