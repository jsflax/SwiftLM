import Testing
@testable import Serving

struct ServingSessionTests {
    // echoes the injected system context so we can confirm what (if anything) was passed
    let echo: ServingSession.Generate = { prompt, sys in "G[\(sys)]::\(prompt)" }

    @Test func freeformRoutesToGenerate() async throws {
        let s = ServingSession(generate: echo)
        let a = try await s.answer("what's the weather like?")
        #expect(a.kind == .freeform)
        #expect(a.text == "G[]::what's the weather like?")  // session injects no context; hooks do, inside generate
    }

    @Test func verifiableFallsBackToGenerateWhenNoVerifier() async throws {
        let s = ServingSession(generate: echo)  // verifyBestOfN nil
        let a = try await s.answer("write a function to reverse a string")
        #expect(a.kind == .verifiable)
        #expect(a.text.hasPrefix("G["))
    }

    @Test func verifiableUsesBestOfNWhenWired() async throws {
        let verifier: ServingSession.Generate = { p, _ in "VERIFIED::\(p)" }
        let s = ServingSession(verifyBestOfN: verifier, generate: echo)
        let a = try await s.answer("implement binary search")
        #expect(a.kind == .verifiable)
        #expect(a.text == "VERIFIED::implement binary search")  // verifier path used, not plain generate
    }
}
