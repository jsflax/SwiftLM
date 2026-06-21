import Testing
import Foundation
@testable import Serving

// A0 — the decode tripwire is the UNCONDITIONAL bound on a whitespace/stuck-token spiral. These prove the
// safety guarantee with no model: it fires exactly at the limit on a degenerate run, and NEVER on legitimate
// output. Hermetic — every detector is built with explicit limits (env is ignored when args are passed).

struct DegenerateRunDetectorTests {
    /// Feed `n` steps of (token, suffix); return the FIRST trip (or nil if the run never trips).
    private func run(_ detector: inout DegenerateRunDetector, steps: [(Int, String)]) -> DegenerateRunDetector.Trip? {
        for (tok, sfx) in steps { if let t = detector.observe(token: tok, suffix: sfx) { return t } }
        return nil
    }

    @Test func identicalTokenRunTripsExactlyAtLimit() {
        var d = DegenerateRunDetector(identicalRunLimit: 4, whitespaceRunLimit: 0)
        // 3 of the same token: not yet. The 4th trips.
        #expect(d.observe(token: 7, suffix: "a") == nil)
        #expect(d.observe(token: 7, suffix: "a") == nil)
        #expect(d.observe(token: 7, suffix: "a") == nil)
        #expect(d.observe(token: 7, suffix: "a") == .identicalRun(token: 7, length: 4))
    }

    @Test func differingTokenResetsIdenticalRun() {
        var d = DegenerateRunDetector(identicalRunLimit: 3, whitespaceRunLimit: 0)
        // alternate two tokens forever — never 3 in a row, never trips.
        var trip: DegenerateRunDetector.Trip? = nil
        for i in 0..<1000 { if let t = d.observe(token: i % 2, suffix: "x") { trip = t; break } }
        #expect(trip == nil)
    }

    @Test func whitespaceCycleTripsEvenWhenTokensDiffer() {
        // A space/newline CYCLE (distinct token ids) escapes the identical-token check but is caught by the
        // whitespace-run signal — the real shape of the observed spiral.
        var d = DegenerateRunDetector(identicalRunLimit: 0, whitespaceRunLimit: 5)
        let steps: [(Int, String)] = (0..<5).map { ($0 % 2 == 0 ? 1 : 2, $0 % 2 == 0 ? " " : "\n") }
        #expect(run(&d, steps: steps) == .whitespaceRun(length: 5))
    }

    @Test func realTokenBreaksWhitespaceRun() {
        var d = DegenerateRunDetector(identicalRunLimit: 0, whitespaceRunLimit: 4)
        // 3 whitespace, then a content token resets, so a following 3 whitespace still doesn't reach 4-in-a-row.
        #expect(d.observe(token: 1, suffix: " ") == nil)
        #expect(d.observe(token: 2, suffix: "\n") == nil)
        #expect(d.observe(token: 3, suffix: "\t") == nil)
        #expect(d.observe(token: 4, suffix: "code") == nil)   // breaks the run
        #expect(d.observe(token: 5, suffix: " ") == nil)
        #expect(d.observe(token: 6, suffix: " ") == nil)
        #expect(d.observe(token: 7, suffix: " ") == nil)      // only 3 since the reset
    }

    @Test func emptySuffixIsNeutralForWhitespaceRun() {
        // A multi-token grapheme yields an empty suffix mid-build: it must NOT reset a whitespace run (else a
        // spiral interleaving empty decodes would never trip), and must NOT count as whitespace itself.
        var d = DegenerateRunDetector(identicalRunLimit: 0, whitespaceRunLimit: 3)
        #expect(d.observe(token: 1, suffix: " ") == nil)
        #expect(d.observe(token: 2, suffix: "") == nil)        // neutral — run stays at 1
        #expect(d.observe(token: 3, suffix: " ") == nil)       // 2
        #expect(d.observe(token: 4, suffix: " ") == .whitespaceRun(length: 3))
    }

    @Test func legitimateCodeOutputNeverTrips() {
        // Indentation + blank lines + repeated punctuation interleaved with content — never trips at default
        // (high) limits. This is the false-positive guard: normal generation must run clean.
        var d = DegenerateRunDetector()   // defaults (48/48)
        let sample: [(Int, String)] = [
            (10, "func"), (11, " "), (12, "foo"), (13, "() {"), (14, "\n"), (15, "    "),
            (16, "return"), (17, " "), (18, "0"), (19, "\n"), (20, "}"), (21, "\n"), (22, "\n"),
            (23, "// "), (24, "===="), (25, "===="), (26, "===="), (27, " done"),
        ]
        // repeat the realistic block many times (distinct content each loop via the suffix) → still clean
        var trip: DegenerateRunDetector.Trip? = nil
        for _ in 0..<50 { if let t = run(&d, steps: sample) { trip = t; break } }
        #expect(trip == nil)
    }

    @Test func zeroLimitDisablesASignal() {
        // identicalRunLimit 0 ⇒ a stuck token never trips that signal; only whitespace can.
        var d = DegenerateRunDetector(identicalRunLimit: 0, whitespaceRunLimit: 0)
        var trip: DegenerateRunDetector.Trip? = nil
        for _ in 0..<10_000 { if let t = d.observe(token: 99, suffix: " ") { trip = t; break } }
        #expect(trip == nil)   // both disabled ⇒ never trips
    }

    @Test func identicalRunBeatsWhitespaceWhenBothWouldTrip() {
        // A stuck WHITESPACE token satisfies both; the identical-token check runs first and reports it.
        var d = DegenerateRunDetector(identicalRunLimit: 3, whitespaceRunLimit: 3)
        #expect(d.observe(token: 5, suffix: " ") == nil)
        #expect(d.observe(token: 5, suffix: " ") == nil)
        #expect(d.observe(token: 5, suffix: " ") == .identicalRun(token: 5, length: 3))
    }
}
