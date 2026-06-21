import Testing
@testable import Serving

// Config-driven adapter resolution — all pure, no model load. Fixtures are minimal template substrings that
// trigger exactly one detector each. The load-bearing case is the 80B-json / 122B-xmlFunction SPLIT: same
// `qwen3_*` family, two wire formats, so only the rendered template shape is a correct signal.

struct LocalAgentAdapterTests {
    // Representative template fragments (just enough to trip each detector; not the full jinja).
    static let qwenNextTmpl =   // qwen3_next (80B): JSON object inside <tool_call> tags, no <function=, no <think>
        "{%- for m in messages %}<tool_call>\n{\"name\": fn, \"arguments\": args | tojson}\n</tool_call>{%- endfor %}"
        + "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
    static let qwen35Tmpl =     // qwen3_5_moe (122B): XML-function calls, generated <think>, strict scan, reasoning_content
        "{%- if not(content.startswith('<tool_response>')) %}{%- set ns.last_query_index = index %}{%- endif %}"
        + "{%- if ns.multi_step_tool %}{{- raise_exception('No user query found in messages.') }}{%- endif %}"
        + "{%- if message.reasoning_content %}{{ message.reasoning_content }}{%- endif %}"
        + "<tool_call><function=name><parameter=k>v</parameter></function></tool_call>"
        + "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n{%- endif %}"
    static let glmTmpl =        // GLM: arg_key/arg_value tag form
        "<tool_call>{{ name }}\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>"
    static let coderTmpl =      // Qwen2.5-Coder: JSON-in-tags, NO <think> (non-reasoning instruct)
        "{%- for m in messages %}<tool_call>\n{\"name\": fn, \"arguments\": args}\n</tool_call>{%- endfor %}"

    // ── FORMAT: the critical split — same family, two wires, decided by the template, not model_type.
    @Test func qwenNextResolvesToJSONInTags() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Qwen3-Next-80B", modelType: "qwen3_next",
                                                    contextWindow: 262144, chatTemplate: Self.qwenNextTmpl,
                                                    mlxInferredFormat: .xmlFunction))   // mlx is WRONG here
        #expect(a.toolCallFormat == .json)            // template evidence overrides mlx's misinference
        #expect(a.requiresOwnedRender == false)       // 80B template doesn't raise
    }

    @Test func qwen35ResolvesToXMLFunction() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Qwen3.5-122B", modelType: "qwen3_5_moe",
                                                    contextWindow: 262144, chatTemplate: Self.qwen35Tmpl,
                                                    mlxInferredFormat: .xmlFunction))
        #expect(a.toolCallFormat == .xmlFunction)     // 122B really IS xml-function — must NOT be flipped to .json
    }

    @Test func glmResolvesToGlm4() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/GLM", modelType: "glm4_moe",
                                                    chatTemplate: Self.glmTmpl, mlxInferredFormat: .glm4))
        #expect(a.toolCallFormat == .glm4)
    }

    @Test func emptyTemplateDefersToMLXInferred() {
        // No template evidence → never make a model worse than mlx's own inference.
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Y", modelType: "qwen3_next",
                                                    chatTemplate: nil, mlxInferredFormat: .xmlFunction))
        #expect(a.toolCallFormat == .xmlFunction)
        let b = ModelProfile.resolve(SnapshotConfig(modelId: "x/Y", modelType: "qwen3_next",
                                                    chatTemplate: "", mlxInferredFormat: .deferToMLX))
        #expect(b.toolCallFormat == .deferToMLX)
    }

    // ── REASONING: flip qwen3_5 to reasoning via the template's generated <think>, not a family literal.
    @Test func qwen35FlipsToReasoningViaTemplate() {
        // forModelType("qwen3_5_moe") alone would say emitsReasoning:false (hasPrefix("qwen") branch); the
        // template's generation-prompt <think> corrects it.
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Qwen3.5-122B", modelType: "qwen3_5_moe",
                                                    chatTemplate: Self.qwen35Tmpl))
        #expect(a.emitsReasoning == true)
        #expect(a.reasoningContentField == "reasoning_content")
    }

    @Test func coderStaysNonReasoning() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Qwen2.5-Coder-7B", modelType: "qwen2",
                                                    chatTemplate: Self.coderTmpl))
        #expect(a.emitsReasoning == false)
        #expect(a.reasoningContentField == nil)
        #expect(a.toolCallFormat == .json)            // coder still emits JSON-in-tags
    }

    // ── OWNED RENDER: the strict reverse-scan that RAISES, plus the known-strict safety net.
    @Test func strictTemplateRequiresOwnedRender() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/122B", modelType: "qwen3_5_moe",
                                                    chatTemplate: Self.qwen35Tmpl))
        #expect(a.requiresOwnedRender == true)        // template contains "No user query"
    }

    @Test func knownStrictTypeIsOwnedRenderEvenWithUnreadableTemplate() {
        // Safety net: a known-strict model_type whose template couldn't be read must NOT degrade to the throwing
        // default — it still gets owned render.
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/122B", modelType: "qwen3_5_moe", chatTemplate: nil))
        #expect(a.requiresOwnedRender == true)
    }

    @Test func plainInstructIsNotOwnedRender() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/Coder", modelType: "qwen2",
                                                    chatTemplate: Self.coderTmpl))
        #expect(a.requiresOwnedRender == false)
    }

    // ── STOPS + budget basics carried through from the snapshot / forModelType.
    @Test func stopsAndEosCarryThrough() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/122B", modelType: "qwen3_5_moe",
                                                    chatTemplate: Self.qwen35Tmpl,
                                                    eosTokenStrings: ["<|im_end|>"], eosTokenIds: [248046, 248044],
                                                    generationStopStrings: ["<|endoftext|>"]))
        #expect(a.stopStrings == ["<|endoftext|>", "<|im_end|>"])   // deduped + sorted
        #expect(a.eosTokenIds == [248046, 248044])
    }

    @Test func resolvePreservesFamilyAndBudget() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/GLM", modelType: "glm4_moe",
                                                    contextWindow: 131072, chatTemplate: Self.glmTmpl,
                                                    mlxInferredFormat: .glm4))
        #expect(a.family == .glm4)
        #expect(a.emitsReasoning == true)             // glm via forModelType + template
        #expect(a.budget.maxTokens == GenerationBudget.runawayCeiling)   // big window → ceiling
    }

    // ── continuationMessages (122B precondition policy), pure [TurnMessage] -> [TurnMessage].
    @Test func continuationIsIdentityForNonStrict() {
        let p = ModelProfile.generic   // requiresOwnedRender == false
        let t: [TurnMessage] = [.init(role: .tool, content: "result")]
        #expect(p.continuationMessages(t) == t)
    }

    @Test func continuationKeepsRealUserQuery() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/122B", modelType: "qwen3_5_moe",
                                                    chatTemplate: Self.qwen35Tmpl))
        let t: [TurnMessage] = [.init(role: .user, content: "build an LRU"),
                                .init(role: .assistant, content: "ok"), .init(role: .tool, content: "r")]
        #expect(a.continuationMessages(t) == t)        // real user present → identity
    }

    @Test func continuationReassertsWhenOnlyToolResponseUser() {
        let a = ModelProfile.resolve(SnapshotConfig(modelId: "x/122B", modelType: "qwen3_5_moe",
                                                    chatTemplate: Self.qwen35Tmpl))
        let bareUser = TurnMessage(role: .user, content: "<tool_response>x</tool_response>")
        let t: [TurnMessage] = [bareUser, .init(role: .tool, content: "r")]
        let out = a.continuationMessages(t)
        #expect(out.first?.role == .user)
        #expect(out.count == t.count + 1)              // first user re-asserted at the head
    }
}
