import Foundation

// ── The verifiable-vs-freeform router (build step 2).
//
// The hinge of the serving design: it decides which correctness mechanism a turn gets.
//   .verifiable → serve-time best-of-N + the execution verifier picks the candidate that passes
//   .freeform   → retrieval grounding + abstention
//
// Honest scope (per the red-team): deciding whether an ARBITRARY prompt has a synthesizable test is
// itself an open problem, not a config flag. So v1 is a coarse, high-precision gate — it routes obvious
// code/implementation asks (where we can compile+test) to the verifier path and everything else to
// freeform. The classifier is injected and replaceable; until serve-time best-of-N exists (build step
// 3, gated on the P0 sandbox-FS-scope fix), the agent treats `.verifiable` as "generate normally" — the
// router is the structural seam that lets the verifier path slot in without reworking the loop.

public enum TurnKind: Sendable, Equatable {
    case verifiable   // a programmatic check exists/admits → best-of-N + verifier
    case freeform     // no cheap check → grounding + abstention
}

public struct TaskRouter: Sendable {
    let classify: @Sendable (String) -> TurnKind

    public init(classify: @escaping @Sendable (String) -> TurnKind = TaskRouter.defaultClassify) {
        self.classify = classify
    }

    public func route(_ prompt: String) -> TurnKind { classify(prompt) }

    /// v1 heuristic: VERIFIABLE iff the prompt is an obvious code/implementation request we could
    /// compile and test. Coarse and precision-biased — a missed code turn just falls back to freeform
    /// (no harm), whereas a false-positive would waste an N× verify on something untestable.
    public static let defaultClassify: @Sendable (String) -> TurnKind = { prompt in
        let p = prompt.lowercased()
        let codeCues = [
            "write a function", "write a swift", "write a python", "write code", "write a test",
            "implement ", "refactor ", "fix the bug", "fix this", "make it compile", "make it pass",
            "a function that", "code that", "def ", "func ", "class ", "struct ",
        ]
        return codeCues.contains { p.contains($0) } ? .verifiable : .freeform
    }
}
