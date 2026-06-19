import Foundation

// ── The built-in Engram advise hook (the "called as the MCP intends" piece).
//
// Fires on UserPromptSubmit: retrieve relevant memory (via the injected `recall`, which the agent wires
// to Engram's MCP `recall` tool through MCPHost.dispatch), format it with `RetrievalGrounding`, and
// return it as `additionalContext` for the loop to inject into the model's system context. If the turn
// asks for specific/recalled facts but nothing relevant was found, `AbstentionPolicy` turns the result
// into a hedge instruction instead — so the agent admits the gap rather than confabulating.
//
// This re-homes serving build steps 1–2 (grounding + abstention) into a hook: grounding is no longer a
// closure ServingSession calls inline; it's what this hook does when the lifecycle event fires.

public struct EngramAdviseHook: Hook {
    let grounding: RetrievalGrounding
    let abstention: AbstentionPolicy

    public init(grounding: RetrievalGrounding, abstention: AbstentionPolicy = AbstentionPolicy()) {
        self.grounding = grounding
        self.abstention = abstention
    }

    public func handle(_ event: HookEvent) async -> HookResult {
        guard case .userPromptSubmit(let prompt) = event else { return .passthrough }
        let g = await grounding.ground(prompt)
        var ctx = g.contextBlock
        if case .hedge(let reason) = abstention.decide(prompt: prompt, grounding: g) {
            ctx += AbstentionPolicy.hedgeInstruction(reason)
        }
        return ctx.isEmpty ? .passthrough : HookResult(additionalContext: ctx)
    }
}
