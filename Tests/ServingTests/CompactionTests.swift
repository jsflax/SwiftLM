import Testing
import MiniBPE
@testable import Serving

// 1 token per character — predictable counts for truncation tests, no model load.
private struct CharTokenizer: GrammarTokenizer {
    var tokensToIds: [String: Int] { [:] }
    var idsToTokens: [Int: String] { [:] }
    var eosTokenId: Int? { nil }
    func tokenize(text: String) -> [String] { text.map(String.init) }
}

struct CompactionTests {
    @Test func contextBudgetDerivation() {
        let b = ContextBudget.forWindow(131072)
        #expect(b.maxContextTokens == Int(Double(131072) * 0.70))
        #expect(b.keepRecentTokens == Int(Double(131072) * 0.25))
        #expect(b.maxKVSize == Int(Double(131072) * 0.70))
    }

    @Test func truncateCapsLongTextWithMarker() {
        let out = truncateToTokens(String(repeating: "x", count: 100), maxTokens: 10, tokenizer: CharTokenizer())
        #expect(out.hasPrefix("xxxxxxxxxx"))                       // first 10 tokens kept
        #expect(out.contains("[truncated: 90 tokens omitted]"))
    }

    @Test func truncateNoOpWhenUnderCap() {
        #expect(truncateToTokens("short", maxTokens: 100, tokenizer: CharTokenizer()) == "short")
    }

    @Test func splitKeepsRecentSummarizesOld() {
        // 4 messages × 100 tokens, keep 250 → summarize first 2, keep last 2 (=200 ≤ 250)
        #expect(compactionSplitIndex(messageTokens: [100, 100, 100, 100], keepTokens: 250) == 2)
    }

    @Test func splitAlwaysMakesProgress() {
        #expect(compactionSplitIndex(messageTokens: [10, 10], keepTokens: 1000) == 1)   // summarize ≥1, keep ≥1
        #expect(compactionSplitIndex(messageTokens: [50, 50, 500], keepTokens: 100) == 2) // huge last → keep just it
        #expect(compactionSplitIndex(messageTokens: [500], keepTokens: 100) == 0)         // single message → no split
    }
}
