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

    // ── output backstop: a single window-derived cap, clamped by the global runaway ceiling
    @Test func budgetBackstopFromWindow() {
        #expect(GenerationBudget.forWindow(8192).maxTokens == 8192)                              // small window → the window
        #expect(GenerationBudget.forWindow(262144).maxTokens == GenerationBudget.runawayCeiling) // huge window → ceiling
        #expect(GenerationBudget.forWindow(1024).maxTokens == 2048)                              // tiny → 2048 floor
    }

    @Test func backstopLeavesRoomForAFullFileWrite() {
        // The coder family must be able to emit a full source file in one write_file call — the old ~1000-token
        // budget truncated it. A real-window model now gets the runaway ceiling (≫ a file).
        let p = ModelProfile.forModelType("qwen3_next", toolCallStyle: .rawJSON, modelId: "x/Q", contextWindow: 262144)
        #expect(p.budget.maxTokens == GenerationBudget.runawayCeiling)
        #expect(p.budget.maxTokens >= 8192)
    }

    // ── recovery attempts the <tool_call> tag parser for ANY family; nil only when there's no usable tag
    @Test func taggedProfileRecoversBareNameCall() {
        let p = ModelProfile.forModelType("glm4_moe", toolCallStyle: .taggedReasoning, modelId: "x/GLM")
        let r = p.recoverMissedToolCall("done</think><tool_call>current_time\n</tool_call>")
        #expect(r?.name == "current_time")
        #expect(r?.argsJSON == "{}")
    }

    @Test func rawJsonProfileRecoversQwenJSONInTags() {
        // Qwen3-Next (qwen3_next → .rawJSON profile) emits JSON inside <tool_call> tags per its chat template;
        // mlx hardcodes Qwen → .xmlFunction, whose parser can't read the JSON, so the call leaks as text and
        // the round loop's recoverMissedToolCall must catch it. PRE-FIX this returned nil → the infinite loop.
        let p = ModelProfile.forModelType("qwen3_next", toolCallStyle: .rawJSON, modelId: "x/Qwen3-Next-80B")
        let r = p.recoverMissedToolCall(
            "I'll write the file.\n<tool_call>\n{\"name\": \"Write\", "
            + "\"arguments\": {\"file_path\": \"LRU.swift\", \"content\": \"x\"}}\n</tool_call>")
        #expect(r?.name == "Write")
        #expect(r?.argsJSON.contains("LRU.swift") == true)
    }

    @Test func rawJsonProfileRecoversNothingWithoutTag() {
        // A genuine raw-JSON emission with NO <tool_call> tag (or plain prose) has nothing to recover → nil.
        let p = ModelProfile.forModelType("qwen3_next", toolCallStyle: .rawJSON, modelId: "x/Qwen")
        #expect(p.recoverMissedToolCall(#"{"name":"current_time","arguments":{}}"#) == nil)
        #expect(p.recoverMissedToolCall("just explaining how tools work, no call here") == nil)
    }

    @Test func recoversMarkdownFencedJSONCall() {
        // A weaker model (Qwen2.5-7B under the batched render) wraps the call in ```json …``` instead of
        // <tool_call> tags. Recovery catches it — but requires BOTH name AND arguments, so a stray named JSON
        // example in prose is not mistaken for a call.
        let p = ModelProfile.forModelType("qwen3_next", toolCallStyle: .rawJSON, modelId: "x/Q")
        let r = p.recoverMissedToolCall(
            "Sure:\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"/tmp/x.txt\"}}\n```")
        #expect(r?.name == "write_file")
        #expect(r?.argsJSON.contains("x.txt") == true)
        #expect(p.recoverMissedToolCall("```json\n{\"name\": \"Alice\"}\n```") == nil)   // no arguments → not a call
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
