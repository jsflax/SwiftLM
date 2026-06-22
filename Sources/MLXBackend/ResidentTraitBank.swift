import Foundation
import MLX
import MLXNN
import MLXLMCommon

// ── Part C — Resident Trait-Bank: role-based STACKABLE LoRA adapters over ONE shared frozen base model, applied
// at decode time. Each agent's role implies a SET of traits; the layer forward SUMS the active set:
//
//     y + Σ_{t ∈ active, sorted} scaleₜ · (x · Aₜ · Bₜ)
//
// That summation IS the stack — no offline rank-concat, no sequential adapter loads (which stack LoRA-on-LoRA).
// An EMPTY active set returns `super.callAsFunction(x)` UNCHANGED ⇒ byte-identical to base.
//
// WHY THIS LIVES IN SWIFTLM (NOT a fork of mlx-swift-lm): that package is a PINNED REMOTE dependency
// (Package.swift `revision e3cb1e1`); its `.build/checkouts` copy is wiped by `swift package clean` and committed
// nowhere — editing it is the opposite of robust. We don't need to: its `Linear`/`QuantizedLinear` are `open`
// with `open callAsFunction`, and models declare `@ModuleInfo var …: Linear` whose hot-swap casts `value as?
// Linear` — so a `Linear` SUBCLASS is both feasible AND required (a non-Linear wrapper would fail the cast). The
// resident layers are therefore SUBCLASSES defined here; install mirrors mlx-swift-lm's own (private)
// `replaceLayers` walk over the PUBLIC `Module.namedModules()` / `Module.update(modules:)` + `LoRAModel.loraLayers`.

/// Canonical, role-derived trait ids. `Comparable` (by rawValue) ⇒ the active set sums in a DETERMINISTIC order,
/// so a given (set, weights) always produces identical logits. C1 wires only `.toolReliability` end-to-end; the
/// rest are the fixed C1 taxonomy (their weights arrive in Part D) so the resolver vocabulary never churns.
public enum TraitID: String, Sendable, Hashable, Comparable, CaseIterable {
    case toolReliability = "tool-reliability"   // ALWAYS (every local agent): well-formed tool call, close </think>, no spiral
    case actionPlan      = "action-plan"        // permissionMode == .plan: read-only / route decisively
    case actionAuto      = "action-auto"        // permissionMode == .auto: execute decisively
    case roleDiscipline  = "role-discipline"    // stay in role + honest under long/contradictory context
    case refereeDone     = "referee-done"       // isReferee: done() only when verified
    public static func < (l: TraitID, r: TraitID) -> Bool { l.rawValue < r.rawValue }
}

/// Call-scoped trait selection. Bound as a `@TaskLocal` around the decode loop — specifically INSIDE
/// `streamFromTokens`'s `container.perform` body, the ONE site verified to propagate to the layer forward with no
/// `Task` hop. The resident layer reads `activeTraits` at FORWARD time, so each turn's role selection lives only
/// for that decode (and concurrent decodes over the SHARED container each see their own bound set — the isolation
/// the interleave test proves).
public enum LoRARuntime {
    @TaskLocal public static var activeTraits: Set<TraitID> = []

    /// Whether the Resident Trait-Bank is enabled for this process (env `SWIFTLM_TRAIT_BANK`). Read ONCE — the
    /// single source of truth for both the once-per-container install AND (C1.5) routing every local agent
    /// through the owned-render decode. Owned-render (`streamFromTokens`) is the ONLY path the `activeTraits`
    /// @TaskLocal propagates (mlx-swift-lm's ChatSession spawns Task hops it can't cross), so when the bank is
    /// LIVE even a normally-ChatSession model (e.g. the 80B `qwen3_next`) must decode via owned-render for its
    /// traits to apply. Default OFF ⇒ production is unchanged (native decode path, no resident layers).
    public static let bankEnabled = ProcessInfo.processInfo.environment["SWIFTLM_TRAIT_BANK"] != nil

    /// Loud tripwire for a non-empty trait-set on a decode path that CANNOT carry the `@TaskLocal`: mlx-swift-lm's
    /// ChatSession (`session.streamDetails`) and the co-batch pool both spawn nested unstructured `Task{}` between
    /// the bind and the forward — the bound value does not cross them, so such a set would SILENTLY serve base
    /// while believing it's adapted. Fail loud instead. C1 already gates traits to owned-render in
    /// `makeAgentBackend`, so this is defense-in-depth (the "never silently serve base" guarantee), not the gate.
    public static func assertCarriable(_ traits: Set<TraitID>, path: @autoclosure () -> String) {
        guard !traits.isEmpty else { return }
        let msg = "[trait-bank] \(traits.sorted().map(\.rawValue)) routed to '\(path())' — a decode path that "
            + "cannot carry the activeTraits @TaskLocal (nested Task hop), so they would SILENTLY serve BASE. "
            + "Route this agent through owned-render or clear its trait-set."
        FileHandle.standardError.write(Data((msg + "\n").utf8))
        assertionFailure(msg)
    }
}

/// Per-layer adapter bank. A REFERENCE type held as a plain ivar on the resident layer so MLXNN's Mirror-based
/// parameter discovery routes it to `.other` (NOT `.parameters`) — keeping the adapter A/B arrays OUT of the
/// model's parameter tree (they must never be enumerated by `parameters()`/`quantize()`/`eval(model)` or trained
/// as base params). `adapters` is populated ONCE at install/registration, before any decode reads it.
public final class TraitAdapterStore {
    public var adapters: [TraitID: (loraA: MLXArray, loraB: MLXArray, scale: Float)] = [:]
    public init() {}
}

/// Common surface for the two resident twins so `LoRABank` can register a trait without caring dense vs quantized.
protocol ResidentTraitLayer: AnyObject {
    var traitStore: TraitAdapterStore { get }
    var loraDims: (input: Int, output: Int) { get }   // (in, out): A is [in, rank], B is [rank, out]
}

/// The active-set summation, shared by both twins. `y` is the base forward, which the caller computes via
/// `super.callAsFunction(x)` — deliberately UNCAST, unlike mlx's LoRALinear which casts `x.asType(weight.dtype)`:
/// passing x straight through makes `y` bit-identical to the un-swapped base layer, and THAT is what guarantees
/// the empty-set byte-identity invariant (proven Δ=0 on the live 122B). DO NOT add a cast to the base call. The
/// delta is cast back to `y.dtype` so an active trait never silently promotes the layer's output dtype.
@inline(__always)
private func applyActiveTraits(_ y: MLXArray, _ x: MLXArray, _ store: TraitAdapterStore) -> MLXArray {
    let active = LoRARuntime.activeTraits
    if active.isEmpty { return y }                     // the hot/default case: byte-identical to base
    var out = y
    for t in active.sorted() {                          // deterministic order ⇒ reproducible logits
        if let a = store.adapters[t] {
            let delta = a.scale * matmul(matmul(x.asType(a.loraA.dtype), a.loraA), a.loraB)
            out = out + delta.asType(y.dtype)
        }
    }
    return out
}

/// Resident trait layer over a dense `Linear` base. Forward = base(x) + Σ active adapter deltas; empty ⇒ base.
final class ResidentLoRALinear: Linear, ResidentTraitLayer {
    let traitStore: TraitAdapterStore
    let loraDims: (input: Int, output: Int)
    init(base: Linear, store: TraitAdapterStore) {
        self.traitStore = store
        let (out, inp) = base.shape
        self.loraDims = (inp, out)
        super.init(weight: base.weight, bias: base.bias)   // REUSE the frozen base arrays — no copy
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyActiveTraits(super.callAsFunction(x), x, traitStore)   // super(x) == base ⇒ byte-identical when inactive
    }
}

/// Resident trait layer over a quantized `QuantizedLinear` base — the path that matters for the 4-bit 122B/80B.
/// Mirrors `QLoRALinear`: the base call runs the QUANTIZED matmul via `super` (no dequantize); deltas are added
/// on top. `super.init` carries the base's packed weight + scales/biases + groupSize/bits forward (no re-quantize).
final class ResidentLoRAQuantizedLinear: QuantizedLinear, ResidentTraitLayer {
    let traitStore: TraitAdapterStore
    let loraDims: (input: Int, output: Int)
    init(base: QuantizedLinear, store: TraitAdapterStore) {
        self.traitStore = store
        let (out, inp) = base.shape
        self.loraDims = (inp, out)
        super.init(weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
                   groupSize: base.groupSize, bits: base.bits, mode: base.mode)   // preserve the base quant mode (not just .affine)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyActiveTraits(super.callAsFunction(x), x, traitStore)
    }
}

/// The resident trait-bank operations: install (swap base leaves → resident twins) and register (populate a
/// trait's per-layer (A,B,scale)). Pure module-tree work over the PUBLIC mlx primitives — no fork, no private API.
public enum LoRABank {
    /// Install EMPTY resident trait layers across the model's LoRA-eligible `Linear`/`QuantizedLinear` leaves (the
    /// same layer set + keys mlx-swift-lm's LoRA training adapts). Idempotent: already-resident layers are skipped.
    /// With empty stores every resident forward returns base ⇒ the model is byte-identical until a trait is
    /// registered. Mirrors mlx-swift-lm's private `replaceLayers` walk; checks `QuantizedLinear` BEFORE `Linear`
    /// (the quantized twin IS-A Linear, so the order matters).
    ///
    /// MUTUAL EXCLUSION (contract, enforce before C2): the resident bank and the LoRA TRAINER must not co-reside on
    /// ONE container — `LoRAContainer.from`→`replaceLayers` also casts `child as? Linear`, and a resident layer
    /// IS-A Linear, so training on an installed container would wrap a trainable LoRA AROUND the resident leaf. In
    /// C1 this can't happen (install is env-gated default-OFF; no serving container is ever trained), so it's
    /// documented, not guarded — add a fail-fast guard when C2 wires traits on by default.
    public static func installResidentLayers(into model: LanguageModel) {
        guard let lora = model as? LoRAModel else { return }   // base doesn't expose its layer set ⇒ nothing to do
        let keys = Set(lora.loraDefaultKeys)
        for layer in lora.loraLayers {
            var update: [(String, Module)] = []
            for (key, child) in layer.namedModules() where keys.contains(key) {
                if child is ResidentTraitLayer { continue }                       // idempotent
                if let q = child as? QuantizedLinear {                            // quantized FIRST (it IS-A Linear)
                    update.append((key, ResidentLoRAQuantizedLinear(base: q, store: TraitAdapterStore())))
                } else if let lin = child as? Linear {
                    update.append((key, ResidentLoRALinear(base: lin, store: TraitAdapterStore())))
                }
            }
            if !update.isEmpty { layer.update(modules: .unflattened(update)) }
        }
    }

    /// Register a trait's per-layer (A, B, scale) into the already-installed resident layers. `makeAB` receives
    /// each layer's (input, output) dims and returns (loraA[in,rank], loraB[rank,out]); both are `eval`'d to stable
    /// constants before being stored. This is how the test harness injects a synthetic trait and how (Part D) a
    /// trained adapter is loaded — keyed by layer path in the real loader; broadcast across layers here for C1.
    public static func register(trait: TraitID, into model: LanguageModel, scale: Float,
                                makeAB: (_ input: Int, _ output: Int) -> (MLXArray, MLXArray)) {
        for (_, m) in model.namedModules() {
            guard let r = m as? ResidentTraitLayer else { continue }
            let (a, b) = makeAB(r.loraDims.input, r.loraDims.output)
            eval(a, b)
            r.traitStore.adapters[trait] = (a, b, scale)
        }
    }

    /// True iff any resident layer is installed in the model (introspection / test guard).
    public static func isInstalled(in model: LanguageModel) -> Bool {
        for (_, m) in model.namedModules() where m is ResidentTraitLayer { return true }
        return false
    }
}

extension MLXLanguageModel {
    /// Install the resident trait-bank into the shared container's model (idempotent, once per container). Runs
    /// inside `container.perform` (the actor boundary) exactly as the retired `loadAdapter` did. After this the
    /// model is byte-identical to base until a trait is registered + made active via `LoRARuntime.activeTraits`.
    public func installResidentTraitBank() async {
        await container.perform { ctx in
            LoRABank.installResidentLayers(into: ctx.model)
        }
    }

    /// Register a (synthetic or trained) trait into the resident bank. `container.perform` keeps it actor-safe.
    public func registerTrait(_ trait: TraitID, scale: Float,
                              makeAB: @Sendable @escaping (_ input: Int, _ output: Int) -> (MLXArray, MLXArray)) async {
        await container.perform { ctx in
            LoRABank.register(trait: trait, into: ctx.model, scale: scale, makeAB: makeAB)
        }
    }

    /// Whether the resident trait-bank is installed (test/introspection).
    public func traitBankInstalled() async -> Bool {
        await container.perform { ctx in LoRABank.isInstalled(in: ctx.model) }
    }
}
