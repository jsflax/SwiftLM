import Testing
@testable import Serving

// Detection registry, physics-derived budget, and profile-driven recovery/strip — all pure, no model load.

struct ModelProfileTests {
    // ── forModelType detection registry
    @Test func glmIsReasoningAndTagged() {
        let p = ModelProfile.forModelType("glm4_moe", toolCallStyle: .taggedReasoning,
                                          modelId: "mlx-community/GLM-4.5-Air-4bit")
        #expect(p.family == .glm4)
        #expect(p.emitsReasoning)
        #expect(p.toolCallStyle == .taggedReasoning)
    }

    @Test func glmDetectedByStyleAloneWhenTypeMissing() {
        let p = ModelProfile.forModelType(nil, toolCallStyle: .taggedReasoning, modelId: "x/y")
        #expect(p.family == .glm4)
        #expect(p.emitsReasoning)
    }

    @Test func qwenCoderIsNonReasoning() {
        let p = ModelProfile.forModelType("qwen2", toolCallStyle: .rawJSON,
                                          modelId: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
        #expect(p.family == .qwen)
        #expect(!p.emitsReasoning)
        #expect(p.toolCallStyle == .rawJSON)
    }

    @Test func r1DistillTieBreakIsReasoning() {
        // qwen2 model_type but an R1-distill id → reasoning (model_type alone can't tell them apart).
        let p = ModelProfile.forModelType("qwen2", toolCallStyle: .rawJSON,
                                          modelId: "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit")
        #expect(p.family == .deepseekR1)
        #expect(p.emitsReasoning)
    }

    @Test func unknownFallsBackToGeneric() {
        let p = ModelProfile.forModelType("some_new_arch", toolCallStyle: .rawJSON, modelId: "a/b")
        #expect(p.family == .generic)
        #expect(!p.emitsReasoning)
    }

    // ── physics-derived budget (floor never clipped, ceil never exceeded)
    @Test func budgetClampsBetweenFloorAndCeil() {
        let low = GenerationBudget(latencyTargetSeconds: 1, assumedTokensPerSecond: 1,
                                   floorTokens: 512, ceilTokens: 2048)
        #expect(low.maxTokens == 512)     // 1×1 → clamped UP to floor
        let mid = GenerationBudget(latencyTargetSeconds: 10, assumedTokensPerSecond: 100,
                                   floorTokens: 512, ceilTokens: 2048)
        #expect(mid.maxTokens == 1000)    // 10×100, within [512, 2048]
        let high = GenerationBudget(latencyTargetSeconds: 1000, assumedTokensPerSecond: 100,
                                    floorTokens: 512, ceilTokens: 2048)
        #expect(high.maxTokens == 2048)   // clamped DOWN to ceil
    }

    @Test func reasoningFloorNeverClipsCoT() {
        let p = ModelProfile.forModelType("glm4_moe", toolCallStyle: .taggedReasoning, modelId: "x/GLM")
        #expect(p.budget.floorTokens >= 4096)
        #expect(p.budget.maxTokens >= 4096)
    }

    // ── recovery keyed on style, not family
    @Test func taggedProfileRecoversBareNameCall() {
        let p = ModelProfile.forModelType("glm4_moe", toolCallStyle: .taggedReasoning, modelId: "x/GLM")
        let r = p.recoverMissedToolCall("done</think><tool_call>current_time\n</tool_call>")
        #expect(r?.name == "current_time")
        #expect(r?.argsJSON == "{}")
    }

    @Test func rawJsonProfileRecoversNothing() {
        let p = ModelProfile.forModelType("qwen2", toolCallStyle: .rawJSON, modelId: "x/Qwen")
        #expect(p.recoverMissedToolCall("<tool_call>current_time</tool_call>") == nil)
    }

    @Test func stripForDisplayRemovesThink() {
        #expect(ModelProfile.generic.stripForDisplay("<think>reasoning</think>The answer.") == "The answer.")
    }

    @Test func displayAnswerFallsBackToReasoningConclusion() {
        // a reasoning model that finishes inside <think> still says something (its last conclusion line)
        #expect(displayAnswer("<think>let me check\nThe file now compiles and prints Area: 12.0</think>")
                == "The file now compiles and prints Area: 12.0")
        #expect(displayAnswer("<think>x</think>Done.") == "Done.")   // post-think answer wins
        #expect(displayAnswer("plain answer") == "plain answer")     // no reasoning → passthrough
    }

    // ── underlying pure helpers (ReasoningOutput)
    @Test func stripReasoningHandlesMultipleAndTruncated() {
        #expect(stripReasoning("a<think>x</think>b<think>y</think>c") == "abc")
        #expect(stripReasoning("kept<think>truncated forever") == "kept")
        #expect(stripReasoning("ans<tool_call>foo</tool_call>") == "ans")
        #expect(stripReasoning("plain text") == "plain text")
    }

    @Test func reasoningStreamFilterSuppressesThink() {
        var f = ReasoningStreamFilter()
        #expect(f.feed("<think>secret</think>Hello") + f.flush() == "Hello")
    }

    @Test func reasoningStreamFilterHandlesSplitTags() {
        var f = ReasoningStreamFilter()
        var out = ""
        for chunk in ["ans", "<thi", "nk>hidden</thi", "nk>more"] { out += f.feed(chunk) }
        out += f.flush()
        #expect(out == "ansmore")   // think content suppressed even with tags split across chunks
    }

    @Test func reasoningStreamFilterPassesPlainText() {
        var f = ReasoningStreamFilter()
        #expect(f.feed("just < text") + f.flush() == "just < text")   // a stray '<' is not a tag
    }

    @Test func reasoningStreamFilterSuppressesToolCallTagToo() {
        var f = ReasoningStreamFilter()
        #expect(f.feed("Answer.<tool_call>current_time</tool_call>") + f.flush() == "Answer.")
        // both span types, split across chunks
        var g = ReasoningStreamFilter()
        var out = ""
        for c in ["A", "<think>r</think>B<tool_", "call>x</tool_call>C"] { out += g.feed(c) }
        #expect(out + g.flush() == "ABC")
    }

    @Test func recoverToolCallTagForms() {
        #expect(recoverToolCallTag("<tool_call>current_time</tool_call>")?.name == "current_time")
        #expect(recoverToolCallTag("no tag here")?.name == nil)
        let json = recoverToolCallTag(#"<tool_call>{"name":"read_file","arguments":{"path":"/x"}}</tool_call>"#)
        #expect(json?.name == "read_file")
        #expect(json?.argsJSON.contains("/x") == true)
    }
}
