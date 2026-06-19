import Testing
@testable import Serving

/// The interleaving activation logic for constrained tool-call emission — verified without a model.
struct GrammarActivationGateTests {
    @Test func staysFreeTextDuringProse() {
        var g = GrammarActivationGate()
        for t in ["Let", " me", " call", " a", " tool"] { #expect(g.observe(t) == false) }
        #expect(g.mode == .freeText)
    }

    @Test func activatesWhenTriggerArrivesAsOneToken() {
        var g = GrammarActivationGate()
        _ = g.observe("Sure. ")
        #expect(g.observe("<tool_call>") == true)   // the flipping token
        #expect(g.mode == .constrained)
    }

    @Test func activatesWhenTriggerSplitAcrossTokens() {
        var g = GrammarActivationGate()
        #expect(g.observe("<tool") == false)
        #expect(g.observe("_call") == false)
        #expect(g.observe(">") == true)             // trigger completes here
        #expect(g.mode == .constrained)
    }

    @Test func noReactivationWhileConstrained() {
        var g = GrammarActivationGate()
        _ = g.observe("<tool_call>")
        #expect(g.observe("<tool_call>") == false)  // already constrained → ignored
        #expect(g.mode == .constrained)
    }

    @Test func deactivateReturnsToFreeText() {
        var g = GrammarActivationGate()
        _ = g.observe("<tool_call>")
        g.deactivate()
        #expect(g.mode == .freeText)
        #expect(g.observe("more prose") == false)   // and can detect a SECOND call later
        #expect(g.observe("<tool_call>") == true)
    }

    @Test func customTrigger() {
        var g = GrammarActivationGate(trigger: "```json")
        #expect(g.observe("here: ```json") == true)
        #expect(g.mode == .constrained)
    }
}
