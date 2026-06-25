import Foundation
import MLXLMCommon
import Serving

// VLM owned-render smoke. Proves the vision path of the owned-render decode end to end on a LIVE model:
// renderTurnMessages(images:) emits the `<|image_pad|>` placeholders + processed pixels, and streamFromTokens
// runs the cold vision-merge prefill via `model.prepare`, so the 122B actually grounds on the picture. A second
// round (the image turn + the model's answer + a follow-up about a DIFFERENT detail) proves the image survives a
// re-rendered transcript (A3b option A — the SwiftLM owned-render fit). Mirrors SystemPromptCheck. Run DETACHED.
extension MLXLanguageModel {
    public func vlmSmoke(imagePath: String, maxTokens: Int = 128) async throws -> String {
        let adapter = await self.localAdapter
        var params = GenerateParameters(maxTokens: maxTokens, temperature: 0)   // greedy ⇒ deterministic
        params.repetitionPenalty = adapter.sampling.repetitionPenalty
        params.repetitionContextSize = adapter.sampling.repetitionContextSize
        let url = URL(fileURLWithPath: imagePath)

        // Decode a transcript through the SAME owned-render path the 122B serves with — the image overload of
        // renderTurnMessages + the image-carrying streamFromTokens. `kvBox` nil ⇒ each round prefills fresh (the
        // cross-round KV-reuse fast path is exercised by the live scheduler run, not this isolated unit smoke).
        func decode(_ turns: [TurnMessage]) async throws -> String {
            let (tokens, imgBox) = try await renderTurnMessages(turns, tools: nil, enableThinking: false,
                                                                imageURLs: [url])
            var out = ""
            for try await g in streamFromTokens(tokens, maxTokens: maxTokens, adapter: adapter,
                                                params: params, image: imgBox) {
                if case .chunk(let c) = g { out += c }
            }
            return out
        }
        func visible(_ s: String) -> String {
            var t = s
            if let o = t.range(of: "<think>"), let c = t.range(of: "</think>", range: o.upperBound..<t.endIndex) {
                t.removeSubrange(o.lowerBound..<c.upperBound)
            }
            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        func has(_ s: String, _ n: String) -> Bool { s.lowercased().contains(n.lowercased()) }

        let q1 = TurnMessage(role: .user,
                             content: "What number is shown in large black digits in this image? Answer with just the number.",
                             imageURLs: [url])
        let a1 = try await decode([q1])
        // Round 2: re-render the image turn + the model's answer + a follow-up about a DIFFERENT detail (the
        // top-left shape's colour) so the model must still SEE the picture, not echo round 1.
        let q2 = TurnMessage(role: .user,
                             content: "In that same image, what COLOUR is the small shape in the top-left corner? Answer with one word.")
        let a2 = try await decode([q1, TurnMessage(role: .assistant, content: a1), q2])

        let v1 = visible(a1), v2 = visible(a2)
        let p1 = has(a1, "42"), p2 = has(a2, "red")
        return [
            "=== VLM owned-render smoke (model: \(modelId), owned-render=\(adapter.requiresOwnedRender)) ===",
            "round-1 image grounding   : \"\(v1.replacingOccurrences(of: "\n", with: " ").prefix(160))\"",
            "  → \(p1 ? "PASS — grounded on the picture (42)" : "FAIL — expected 42")",
            "round-2 persist (re-render): \"\(v2.replacingOccurrences(of: "\n", with: " ").prefix(160))\"",
            "  → \(p2 ? "PASS — image survived the re-rendered transcript (red)" : "FAIL — expected red")",
            "",
            (p1 && p2)
                ? "RESULT: PASS — the owned-render VLM path sees images and persists them across rounds."
                : "RESULT: FAIL — eyeball the outputs above."
        ].joined(separator: "\n")
    }
}
