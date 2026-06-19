import Testing
@testable import Serving

struct RetrievalGroundingTests {
    /// A fixture recall backend that returns a fixed item set (grounding logic tested deterministically).
    private func fixed(_ items: [RetrievedItem]) -> @Sendable (String, Int) async -> [RetrievedItem] {
        { _, _ in items }
    }

    @Test func keepsOnlyRelevantRanksAndCaps() async {
        let g = RetrievalGrounding(config: .init(maxItems: 2, maxDistance: 0.4), recall: fixed([
            RetrievedItem(id: "far",  text: "FAR",  distance: 0.9),   // dropped (> 0.4)
            RetrievedItem(id: "mid",  text: "MID",  distance: 0.30),
            RetrievedItem(id: "near", text: "NEAR", distance: 0.10),
            RetrievedItem(id: "also", text: "ALSO", distance: 0.35),  // dropped by maxItems cap (3rd)
        ]))
        let out = await g.ground("anything")
        #expect(out.items.map(\.id) == ["near", "mid"])  // ranked best-first, capped to 2, far dropped
        #expect(out.hasRelevant)
        #expect(out.contextBlock.contains("NEAR"))
        #expect(out.contextBlock.contains("MID"))
        #expect(!out.contextBlock.contains("FAR"))
        #expect(!out.contextBlock.contains("ALSO"))
    }

    @Test func noRelevantWhenAllTooFar() async {
        let g = RetrievalGrounding(config: .init(maxDistance: 0.2), recall: fixed([
            RetrievedItem(id: "a", text: "A", distance: 0.5),
            RetrievedItem(id: "b", text: "B", distance: 0.9),
        ]))
        let out = await g.ground("q")
        #expect(!out.hasRelevant)        // nothing within threshold → no grounding
        #expect(out.contextBlock == "")  // empty block so the agent injects nothing
    }

    @Test func emptyRecallYieldsEmptyGrounding() async {
        let out = await RetrievalGrounding(recall: fixed([])).ground("q")
        #expect(out.items.isEmpty)
        #expect(!out.hasRelevant)
    }

    @Test func headerAndBulletFormatting() async {
        let g = RetrievalGrounding(config: .init(header: "CTX:"), recall: fixed([
            RetrievedItem(id: "x", text: "fact one", distance: 0.1),
        ]))
        let out = await g.ground("q")
        #expect(out.contextBlock == "CTX:\n- fact one\n")
    }
}
