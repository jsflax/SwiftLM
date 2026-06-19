import Foundation

// ── The serve-time strategy router. After the hooks pivot, retrieval grounding + abstention happen
// INSIDE the model loop (the EngramAdviseHook fires on UserPromptSubmit within `runWithTools`), so this
// session no longer grounds inline. It is now just the verifiable-vs-freeform dispatcher:
//   .verifiable → best-of-N + verifier (build step 3, gated on the P0 sandbox fix) if wired, else generate
//   .freeform   → generate (the hook chain injects grounding/hedge inside generate)
// `generate` is injected (the agent closes it over `runWithTools(prompt, host:, hooks:)`), keeping this
// pure-Swift and testable. The seam stays so the verifier path slots in without reworking the loop.

public struct Answer: Sendable, Equatable {
    public let text: String
    public let kind: TurnKind
    public init(text: String, kind: TurnKind) {
        self.text = text; self.kind = kind
    }
}

public struct ServingSession: Sendable {
    public typealias Generate = @Sendable (_ prompt: String, _ systemContext: String) async throws -> String

    let router: TaskRouter
    let generate: Generate
    /// Serve-time best-of-N + verifier (build step 3). When nil, verifiable turns fall back to `generate`.
    let verifyBestOfN: Generate?

    public init(
        router: TaskRouter = TaskRouter(),
        verifyBestOfN: Generate? = nil,
        generate: @escaping Generate
    ) {
        self.router = router
        self.generate = generate
        self.verifyBestOfN = verifyBestOfN
    }

    public func answer(_ prompt: String) async throws -> Answer {
        let kind = router.route(prompt)
        // Grounding/abstention are injected by the hook chain inside `generate` (runWithTools), so the
        // systemContext here is empty — the session only chooses the generation strategy.
        let text: String
        switch kind {
        case .verifiable: text = try await (verifyBestOfN ?? generate)(prompt, "")
        case .freeform:   text = try await generate(prompt, "")
        }
        return Answer(text: text, kind: kind)
    }
}
