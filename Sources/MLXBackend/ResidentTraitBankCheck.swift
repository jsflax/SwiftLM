import Foundation
import MLX
import MLXRandom
import MLXLMCommon
import Serving

// ── C1 correctness gate (TRAITBANK_CHECK=1). Proves the Resident Trait-Bank on a LIVE model, without training:
//
//   T0  empty-set byte-identity   — install the bank (empty stores); first-gen logits MUST equal base exactly.
//   T1  registered-but-inactive   — register traits, leave activeTraits=[]; logits MUST still equal base.
//   T2  B=0 active ⇒ base          — a trait whose loraB is zero contributes 0 even when active (the gating proof).
//   T3  B≠0 active ⇒ measurable Δ  — a non-zero trait actually changes the logits (the adapter IS applied).
//   T4  INTERLEAVE isolation       — two CONCURRENT tasks bind distinct single-trait sets over the ONE shared
//                                     container; each forward must see ONLY its own set (the @TaskLocal isolation
//                                     proof — the load-bearing concurrency claim of the whole design).
//
// Reuses `prefillFirstLogits` (the B2 probe): it prefills `row` into a fresh cache and returns the first-gen
// logit vector — deterministic (no sampling), so the byte-identity cases compare EXACTLY to 0. Because
// `prefillFirstLogits` runs the forward inside `container.perform` (an INLINE continuation, no Task hop),
// wrapping the CALL in `LoRARuntime.$activeTraits.withValue(set){…}` makes the bound set reach the resident
// layer's forward — exactly as it does on the real owned-render decode. The 122B/80B are 4-bit, so this
// exercises `ResidentLoRAQuantizedLinear` (the quantized twin that matters). Run DETACHED + memory-watchdog.

extension MLXLanguageModel {
    public func residentTraitBankCheck() async throws -> String {
        let adapter = await self.localAdapter
        func maxAbs(_ a: [Float], _ b: [Float]) -> Float {
            var m: Float = 0; for i in 0..<min(a.count, b.count) { let d = abs(a[i] - b[i]); if d > m { m = d } }; return m
        }
        func argmax(_ v: [Float]) -> Int { var bi = 0; for i in 1..<v.count where v[i] > v[bi] { bi = i }; return bi }

        // A small deterministic prompt row.
        let row = try await renderTurnMessages(
            [TurnMessage(role: .user, content: "Name two primary colors.")],
            tools: nil, enableThinking: adapter.emitsReasoning)

        // T0a — BASE logits (no resident layers installed yet).
        let lBase = await prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        let scale = (lBase.max() ?? 0) - (lBase.min() ?? 0)

        // Param-tree invariant (T5): the resident swap + trait registration must NOT change the model's flattened
        // parameter count — the adapter A/B live in a `.other` store that MLXNN's Mirror walk never enumerates.
        // This proves the invisibility claim on the LIVE model instead of resting on a code-reading argument.
        let pBase = await paramCount()

        // Install the resident bank (empty stores) — must not perturb the forward.
        await installResidentTraitBank()
        let installed = await traitBankInstalled()
        let pInstalled = await paramCount()

        // T0b — empty-set after install ⇒ byte-identical to base.
        let lEmpty = await prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        let dEmpty = maxAbs(lBase, lEmpty)

        // Register two synthetic traits: A (B≠0, has effect) and Z (B=0, inert even when active). Small magnitude
        // so the per-layer deltas stay bounded as they stack across depth (LayerNorms keep activations ~O(1)).
        let rank = 4
        await registerTrait(.toolReliability, scale: 1.0) { inp, out in
            (MLXRandom.uniform(low: -0.01, high: 0.01, [inp, rank]),
             MLXRandom.uniform(low: -0.01, high: 0.01, [rank, out]))
        }
        await registerTrait(.actionPlan, scale: 1.0) { inp, out in
            (MLXRandom.uniform(low: -0.01, high: 0.01, [inp, rank]),
             MLXArray.zeros([rank, out]))                      // loraB == 0 ⇒ zero delta even when active
        }
        let pRegistered = await paramCount()   // adapters now exist in the stores — the count must STILL be unchanged

        // T1 — registered but inactive (activeTraits=[]) ⇒ still base.
        let lInactive = await prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        let dInactive = maxAbs(lBase, lInactive)

        // T2 — B=0 trait active ⇒ base (proves the per-trait lookup + sum is correct: 0 delta).
        let lZero = await LoRARuntime.$activeTraits.withValue([.actionPlan]) {
            await self.prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        }
        let dZero = maxAbs(lBase, lZero)

        // T3 — B≠0 trait active ⇒ measurable change (the adapter is actually applied).
        let lA = await LoRARuntime.$activeTraits.withValue([.toolReliability]) {
            await self.prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        }
        let dA = maxAbs(lBase, lA)

        // T4 — INTERLEAVE: two CONCURRENT tasks, distinct single-trait sets, over the ONE shared container. Each
        // `async let` is its own child Task binding its own @TaskLocal; the container serializes the forwards but
        // each must read ONLY its own set. Compare to the solo references: c1(toolReliability)==lA, c2(B=0)==base.
        async let c1f = LoRARuntime.$activeTraits.withValue([.toolReliability]) {
            await self.prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        }
        async let c2f = LoRARuntime.$activeTraits.withValue([.actionPlan]) {
            await self.prefillFirstLogits(row, kvBox: OwnedKVCacheBox())
        }
        let (lc1, lc2) = await (c1f, c2f)
        let interA = maxAbs(lc1, lA)        // concurrent tool-reliability == solo tool-reliability ⇒ no leak from c2
        let interZ = maxAbs(lc2, lBase)     // concurrent B=0 == base ⇒ c2 did NOT pick up c1's trait

        // Pass criteria. The byte-identity cases must be EXACTLY 0 (deterministic forward, same base math); the
        // active case must move the logits a non-trivial fraction of their range; interleave deltas ≈ 0.
        let eps: Float = 1e-4 * max(1, scale)   // float slack for the "active" comparisons (interleave determinism)
        let paramsStable = (pBase == pInstalled && pInstalled == pRegistered)
        let pass = installed
            && dEmpty == 0 && dInactive == 0 && dZero == 0
            && dA > scale * 0.01
            && interA <= eps && interZ <= eps
            && paramsStable

        var out = ["=== C1 RESIDENT TRAIT-BANK CHECK (model: \(modelId), owned-render=\(adapter.requiresOwnedRender)) ==="]
        out.append("installed=\(installed)  logit scale≈\(String(format: "%.1f", scale))  (row=\(row.count) tok)")
        out.append(String(format: "T0 empty-set byte-identical : Δlogits=%.6f   %@", dEmpty, dEmpty == 0 ? "✓" : "✗ MUST be 0"))
        out.append(String(format: "T1 registered-but-inactive  : Δlogits=%.6f   %@", dInactive, dInactive == 0 ? "✓" : "✗ MUST be 0"))
        out.append(String(format: "T2 B=0 trait active         : Δlogits=%.6f   %@", dZero, dZero == 0 ? "✓" : "✗ MUST be 0"))
        out.append(String(format: "T3 B≠0 trait active         : Δlogits=%.6f   %@ (argmax base=%d active=%d)",
                          dA, dA > scale * 0.01 ? "✓ measurable" : "✗ no effect", argmax(lBase), argmax(lA)))
        out.append(String(format: "T4 interleave isolation     : c1−solo=%.6f  c2−base=%.6f   %@",
                          interA, interZ, (interA <= eps && interZ <= eps) ? "✓ no cross-talk" : "✗ TASK-LOCAL LEAK"))
        out.append(String(format: "T5 param-tree invariant     : base=%d installed=%d registered=%d   %@",
                          pBase, pInstalled, pRegistered, paramsStable ? "✓ adapters invisible" : "✗ params CHANGED"))
        out.append(pass
            ? "RESULT: PASS — empty/inactive/zero are byte-identical to base, a live trait moves the logits, and two concurrent agents' trait-sets do NOT contaminate each other."
            : "RESULT: FAIL — see the ✗ line(s) above.")
        return out.joined(separator: "\n")
    }

    /// Flattened model-parameter count (for the install-invisibility invariant): the resident swap + trait
    /// registration must NOT change it — the adapter A/B live in a `.other` store, invisible to `parameters()`.
    func paramCount() async -> Int {
        await container.perform { ctx in ctx.model.parameters().flattened().count }
    }
}
