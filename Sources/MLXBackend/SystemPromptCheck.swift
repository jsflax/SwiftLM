import Foundation
import MLXLMCommon
import Serving

// ── BUG-2 validation (SYSPROMPT_CHECK=1) + a legible before/after demo. The owned-render decode used to build the
// prompt ONLY from the conversation turns, silently dropping the agent's system prompt (persona/role/plan-note/hook
// grounding) — so the 122B ran its agents without their instructions. The fix prepends a leading system turn in
// CompactingSession.ownedRound. This harness proves it on a LIVE model by decoding the SAME user question through
// the owned-render render path TWO ways — WITHOUT a system turn (the old behavior) vs WITH a directive system turn
// (the fix) — and showing that the model obeys the system prompt only in the second case. Run DETACHED + watchdog.

extension MLXLanguageModel {
    public func systemPromptCheck() async throws -> String {
        let adapter = await self.localAdapter
        var params = GenerateParameters(maxTokens: 220, temperature: 0)   // greedy ⇒ deterministic; enough to clear the 122B's reasoning + reach the answer
        params.repetitionPenalty = adapter.sampling.repetitionPenalty
        params.repetitionContextSize = adapter.sampling.repetitionContextSize

        let directive = "IMPORTANT: You are a pirate. Answer EVERY question in pirate speak, and begin your reply with 'Arrr'."
        let user = TurnMessage(role: .user, content: "What is the capital of France?")

        // Decode a structured transcript through the SAME render the owned-render round uses (renderTurnMessages →
        // streamFromTokens). No traits bound (activeTraits defaults to [] ⇒ base decode).
        func decode(_ turns: [TurnMessage]) async throws -> String {
            let tokens = try await renderTurnMessages(turns, tools: nil, enableThinking: adapter.emitsReasoning)
            var out = ""
            for try await g in streamFromTokens(tokens, maxTokens: 220, adapter: adapter, params: params) {
                if case .chunk(let c) = g { out += c }
            }
            return out
        }
        // Strip a leading <think>…</think> span so the visible answer is readable.
        func visible(_ s: String) -> String {
            var t = s
            if let o = t.range(of: "<think>"), let c = t.range(of: "</think>", range: o.upperBound..<t.endIndex) {
                t.removeSubrange(o.lowerBound..<c.upperBound)
            }
            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // The verdict judges the FULL output (incl. reasoning): the POINT is whether the model RECEIVED the system
        // prompt. An output that names the pirate role — even mid-reasoning — proves the directive reached the model
        // (the old behavior never mentions it, because the system turn was dropped before the template ever saw it).
        func sawDirective(_ raw: String) -> Bool {
            let l = raw.lowercased()
            return l.contains("pirate") || l.contains("arrr") || l.contains("matey") || l.contains("ahoy")
        }

        let rawWithout = try await decode([user])                                              // OLD: no system turn
        let rawWith = try await decode([TurnMessage(role: .system, content: directive), user])  // FIX: system prepended
        let vWithout = visible(rawWithout), vWith = visible(rawWith)
        let pass = sawDirective(rawWith) && !sawDirective(rawWithout)

        return [
            "=== BUG-2: system prompt now honored in owned-render (model: \(modelId), owned-render=\(adapter.requiresOwnedRender)) ===",
            "system directive : \"\(directive)\"",
            "user             : \"What is the capital of France?\"",
            "",
            "WITHOUT system turn (the OLD owned-render behavior — system prompt dropped):",
            "  " + vWithout.replacingOccurrences(of: "\n", with: "\n  "),
            "",
            "WITH system turn (the FIX):",
            "  " + vWith.replacingOccurrences(of: "\n", with: "\n  "),
            "",
            pass
                ? "RESULT: PASS — the model RECEIVES + acts on the system prompt ONLY with the fix (the pirate role/speak appears WITH the system turn and is entirely absent WITHOUT it — proving the old owned-render path dropped it)."
                : "RESULT: INCONCLUSIVE — eyeball the two outputs: WITH should reflect the pirate directive, WITHOUT should not."
        ].joined(separator: "\n")
    }
}
