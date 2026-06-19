import Testing
@testable import Serving

struct AbstentionPolicyTests {
    let policy = AbstentionPolicy()
    private func grounded() -> Grounding {
        Grounding(items: [RetrievedItem(id: "x", text: "fact", distance: 0.1)], contextBlock: "x")
    }
    private let empty = Grounding(items: [], contextBlock: "")

    @Test func factQuestionNoGroundingHedges() {
        guard case .hedge = policy.decide(prompt: "what is the etrade issue?", grounding: empty) else {
            Issue.record("expected hedge"); return
        }
    }

    @Test func factQuestionWithGroundingProceeds() {
        #expect(policy.decide(prompt: "what is the etrade issue?", grounding: grounded()) == .proceed)
    }

    @Test func personalReferenceNoGroundingHedges() {
        guard case .hedge = policy.decide(prompt: "remember what we decided about sizing", grounding: empty) else {
            Issue.record("expected hedge"); return
        }
    }

    @Test func codeRequestProceedsEvenWithoutGrounding() {
        // a code/command request doesn't need recalled facts → never hedge
        #expect(policy.decide(prompt: "write a function to reverse a string", grounding: empty) == .proceed)
    }

    @Test func chitchatProceeds() {
        #expect(policy.decide(prompt: "thanks, that's great", grounding: empty) == .proceed)
    }

    @Test func injectedPredicateOverrides() {
        let always = AbstentionPolicy(needsGrounding: { _ in true })
        guard case .hedge = always.decide(prompt: "hi", grounding: empty) else {
            Issue.record("injected predicate should force the needs-grounding path"); return
        }
    }
}

struct TaskRouterTests {
    let router = TaskRouter()

    @Test func codeRequestRoutesVerifiable() {
        #expect(router.route("write a function that sums a list") == .verifiable)
        #expect(router.route("implement binary search in Swift") == .verifiable)
        #expect(router.route("fix the bug in this parser") == .verifiable)
    }

    @Test func freeformRoutesFreeform() {
        #expect(router.route("what's the weather like today?") == .freeform)
        #expect(router.route("summarize this article") == .freeform)
        #expect(router.route("what did we decide last time") == .freeform)
    }

    @Test func injectedClassifierOverrides() {
        #expect(TaskRouter(classify: { _ in .verifiable }).route("anything") == .verifiable)
    }
}
