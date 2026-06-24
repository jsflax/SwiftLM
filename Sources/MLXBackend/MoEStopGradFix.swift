import Foundation
import MLX
import MLXNN
import MLXLMCommon

// ── MoE training fix: stop_gradient on the expert-routing path ─────────────────────────────────────────────────
//
// mlx-swift-lm's MoE blocks (e.g. `Qwen35Language.SparseMoeBlock`, the qwen3_5_moe / 122B-A10B base) compute
// `inds = argPartition(softmax(gate(x)))` and feed them to `switch_mlp(x, inds)` → `gather_qmm`. Unlike Python
// mlx-lm — which wraps the routing as `mx.stop_gradient(mx.argpartition(...))` — the Swift port DROPPED the
// stop_gradient. So whenever a trainable param sits below a MoE layer (e.g. attention LoRA), the integer routing
// `inds` stay a differentiable descendant of that param, and on the FIRST backward pass MLX asks `GatherQMM::vjp`
// for the gradient wrt the indices — undefined — aborting the run:
//     "[GatherQMM::vjp] cannot compute the gradient wrt the indices."  (primitives.cpp:3613)
// This is NOT an MLX limitation and needs NO mlx-swift-lm patch. `SwitchGLU`/`SparseMoeBlock` are non-open, but
// `Linear` IS `open`: we wrap the MoE router `gate` (the `Linear` whose output feeds argPartition) in a stop-grad
// `Linear` subclass. Detaching the gate output makes both `inds` AND the routing `scores` constants on the
// backward — the gate is frozen anyway, so the only gradient dropped is the (secondary) routing-weight term; the
// expert-output gradient (the real signal, via `switch_mlp(x, …)`, whose `x` is NOT detached) still flows.
// `stopGradient` is a no-op in the forward, so logits/inference are byte-identical.

/// A `Linear` that returns `stopGradient` of an inner router gate's output. The inner gate (a `Linear` or
/// `QuantizedLinear`) is held as a child so its loaded weights/quantization are preserved untouched; the dummy
/// 1×1 base weight is never used (`callAsFunction` is overridden).
public final class StopGradGate: Linear {
    @ModuleInfo(key: "real_gate") var realGate: Linear

    public init(wrapping gate: Linear) {
        self._realGate.wrappedValue = gate
        super.init(1, 1, bias: false)   // unused placeholder — forward is overridden to use realGate
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        stopGradient(realGate(x))
    }
}

public enum MoEStopGrad {
    /// Wrap every MoE router `mlp.gate` (a `Linear`) in the model's decoder layers with a `StopGradGate`, so LoRA/
    /// DPO training on a MoE base doesn't abort in `GatherQMM::vjp`. Returns the count wrapped (0 ⇒ a dense model /
    /// nothing to do — harmless). Idempotent. MUST run BEFORE `LoRAContainer.from`. (Leaves `shared_expert_gate`
    /// alone — its sigmoid weighting is a normal differentiable path with no gather/indices.)
    @discardableResult
    public static func install(into model: LanguageModel) -> Int {
        guard let lora = model as? LoRAModel else { return 0 }
        var total = 0
        for layer in lora.loraLayers {
            var updates: [(String, Module)] = []
            for (key, child) in layer.namedModules() where key.hasSuffix("mlp.gate") {
                if let gate = child as? Linear, !(gate is StopGradGate) {
                    updates.append((key, StopGradGate(wrapping: gate)))
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
