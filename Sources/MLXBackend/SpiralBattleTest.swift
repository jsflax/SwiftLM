import Foundation
import MLX
import MLXLMCommon
import Serving

// ── A0 end-to-end battle-test. The pure `DegenerateRunDetector` unit tests prove the tripwire LOGIC; this
// proves it is WIRED into the real owned-render decode (`streamFromTokens`) and actually bounds a spiral on a
// live model. We can't rely on a healthy model to spiral on demand, so we INJECT the pathology
// (`SWIFTLM_SPIRAL_INJECT` forces a dominant logit spike every step) and contrast guard-on vs guard-off. This
// is the "confirm by repro, don't declare victory first" gate the plan demands.

extension MLXLanguageModel {
    /// Run the three-scenario battle-test and return a human-readable report (the `agent` exec prints it).
    /// A guard ON  — injected spiral, GENEROUS token cap: the tripwire (not the cap) must stop it in ≤ limit.
    /// B guard OFF — same injection, tripwire disabled: the unbounded spiral runs to a small cap (the control).
    /// C control   — no injection, guard ON: a healthy prompt generates normally and stops at EOS (no clip).
    public func spiralBattleTest() async throws -> String {
        let adapter = await self.localAdapter
        let limit = 48
        func params(_ maxTokens: Int) -> GenerateParameters {
            // Mirror SwiftLMServe's owned-render params: temp + rep-pen straight from the adapter's sampling.
            var p = GenerateParameters(maxTokens: maxTokens, temperature: adapter.sampling.temperature)
            p.repetitionPenalty = adapter.sampling.repetitionPenalty
            p.repetitionContextSize = adapter.sampling.repetitionContextSize
            return p
        }
        func decode(_ row: [Int32], maxTokens: Int) async throws -> (chunks: Int, seconds: Double, head: String) {
            let t0 = Date()
            var chunks = 0
            var head = ""
            for try await g in self.streamFromTokens(row, maxTokens: maxTokens, adapter: adapter, params: params(maxTokens)) {
                if case .chunk(let c) = g { chunks += 1; if head.count < 60 { head += c } }
            }
            return (chunks, Date().timeIntervalSince(t0), head.replacingOccurrences(of: "\n", with: "⏎"))
        }

        let row = try await self.renderTurnMessages(
            [TurnMessage(role: .user, content: "Say hello in one short sentence.")],
            tools: nil, enableThinking: false)

        var out = ["=== A0 SPIRAL BATTLE-TEST (model: \(modelId)) ==="]

        // A — guard ON, injected spiral, GENEROUS cap (2000): the tripwire, not the cap, must stop it.
        setenv("SWIFTLM_SPIRAL_INJECT", "100", 1)
        setenv("SWIFTLM_TRIPWIRE_IDENTICAL", "\(limit)", 1)
        setenv("SWIFTLM_TRIPWIRE_WHITESPACE", "\(limit)", 1)
        let a = try await decode(row, maxTokens: 2000)
        let aBounded = a.chunks <= limit + 4
        out.append(String(format: "A guard ON  + inject (cap 2000): %4d tok  %.2fs  → %@",
                          a.chunks, a.seconds, aBounded ? "BOUNDED ✓ (tripwire fired ~\(limit))" : "NOT BOUNDED ✗"))

        // B — same injection, tripwire DISABLED → the unbounded spiral runs to the cap (300) = the control.
        setenv("SWIFTLM_TRIPWIRE_IDENTICAL", "0", 1)
        setenv("SWIFTLM_TRIPWIRE_WHITESPACE", "0", 1)
        let b = try await decode(row, maxTokens: 300)
        let bRanAway = b.chunks >= 250
        out.append(String(format: "B guard OFF + inject (cap 300) : %4d tok  %.2fs  → %@",
                          b.chunks, b.seconds, bRanAway ? "RAN AWAY (rep-pen alone can't stop a spike)" : "stopped early?"))

        // C — control: no injection, guard ON, healthy prompt → normal generation, natural EOS, no clip.
        unsetenv("SWIFTLM_SPIRAL_INJECT")
        setenv("SWIFTLM_TRIPWIRE_IDENTICAL", "\(limit)", 1)
        setenv("SWIFTLM_TRIPWIRE_WHITESPACE", "\(limit)", 1)
        let c = try await decode(row, maxTokens: 2000)
        out.append(String(format: "C control (no inject)         : %4d tok  %.2fs  → %@  \"%@\"",
                          c.chunks, c.seconds,
                          c.chunks < limit ? "healthy (EOS, not clipped) ✓" : "long output — inspect",
                          String(c.head.prefix(40))))
        unsetenv("SWIFTLM_TRIPWIRE_IDENTICAL"); unsetenv("SWIFTLM_TRIPWIRE_WHITESPACE")

        // D — A1: a cancel must STOP the decode (free the GPU), not merely abandon the consumer. streamFromTokens
        // holds the SerialAccessContainer for its ENTIRE loop, so a second decode can only start once the first
        // RELEASES it. Start an injected, tripwire-OFF, 1M-cap spiral; cancel it after it's running; then race a
        // normal decode. Prompt ⇒ the cancelled decode broke out (Task.isCancelled) and freed the container; a
        // hang ⇒ the cancel never reached the decode loop and we're stuck behind a 1M-token spiral. This is the
        // differential proof that cancellation actually propagates to the decode (otherwise un-observable).
        setenv("SWIFTLM_SPIRAL_INJECT", "100", 1)
        setenv("SWIFTLM_TRIPWIRE_IDENTICAL", "0", 1)
        setenv("SWIFTLM_TRIPWIRE_WHITESPACE", "0", 1)
        let spiral = Task { () -> Int in
            var n = 0
            for try await g in self.streamFromTokens(row, maxTokens: 1_000_000, adapter: adapter, params: params(1_000_000)) {
                if case .chunk = g { n += 1 }
            }
            return n
        }
        try? await Task.sleep(for: .milliseconds(1500))   // let the spiral run + take the container
        spiral.cancel()
        unsetenv("SWIFTLM_SPIRAL_INJECT")                  // the probe must be a NORMAL decode
        let tD = Date()
        let probe = try await decode(row, maxTokens: 8)    // blocks here iff the container is still held
        let probeSecs = Date().timeIntervalSince(tD)
        _ = await spiral.result                            // reap the cancelled spiral
        let freed = probeSecs < 20
        out.append(String(format: "D cancel frees GPU (probe after cancel): %5.2fs %d tok  → %@",
                          probeSecs, probe.chunks, freed ? "container released ✓ (decode observed cancel)" : "BLOCKED ✗ (cancel ignored)"))
        unsetenv("SWIFTLM_TRIPWIRE_IDENTICAL"); unsetenv("SWIFTLM_TRIPWIRE_WHITESPACE")

        let pass = aBounded && bRanAway && freed
        out.append(pass
            ? "RESULT: PASS — tripwire bounds the spiral (A≈\(a.chunks) ≪ B≈\(b.chunks)); cancel frees the GPU (D=\(String(format: "%.1f", probeSecs))s)."
            : "RESULT: FAIL — A=\(a.chunks) B=\(b.chunks) freed=\(freed) (want A≤\(limit + 4) ≪ B≥250, D prompt).")
        return out.joined(separator: "\n")
    }
}
