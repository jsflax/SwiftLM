import Foundation

// ── The per-model LOCAL AGENT ADAPTER: everything model-specific the generic agent loop must vary by model,
// resolved from the model's OWN config (chat_template / config.json / generation_config / tokenizer_config) —
// NOT a hardcoded family branch. This is the local mirror of how a cloud `ProviderSession` delegates a whole
// turn to a CLI: the loop stays model-agnostic and consults the adapter for format, reasoning, stops, sampling,
// and message construction. `ModelProfile` (this module) IS the adapter value; these are the additive concerns
// + the config-driven `resolve`. MLX-free + Sendable, so it threads into both round loops and is unit-tested
// without loading a model. (mlx-swift-lm types stay out of Serving — MLXBackend bridges the choice to a parser.)

/// The conceptual name for the enriched `ModelProfile` value. Kept as an alias so new code / tests read as
/// "the adapter" while existing call sites that hold a `profile` keep compiling unchanged.
public typealias LocalAgentAdapter = ModelProfile

/// MLX-free mirror of the WIRE FORMAT a model emits its tool calls in. The adapter chooses this from the
/// model's REAL chat template (positive evidence only); `.deferToMLX` means "no template evidence — keep
/// mlx-swift-lm's own inference", so the change is strictly an improvement (a model is never made worse than
/// today). MLXBackend maps a concrete choice → an mlx `ToolCallFormat` → a parser; `.deferToMLX` → use the
/// loaded model's already-inferred format. The KEY correction this enables: `qwen3_next` emits JSON-in-tags
/// (`.json`) while `qwen3_5_moe` emits XML-function (`.xmlFunction`) — same model_type family, two wires, so
/// only the rendered template shape is a correct signal.
public enum ToolCallFormatChoice: Sendable, Equatable {
    case json          // a JSON object inside <tool_call> tags (Hermes/Qwen3-Next): <tool_call>{"name":…}</tool_call>
    case xmlFunction   // <tool_call><function=name><parameter=k>v</parameter></function></tool_call> (Qwen3.5-122B)
    case glm4          // GLM bare-name / arg-tag form
    case pythonic      // a [func(arg=…)] python-call list
    case deferToMLX    // no template evidence → use mlx-swift-lm's inferred format (never worse than today)
}

/// One conversation turn as a Sendable VALUE the (MLX-free) loop policy operates on — richer than a bare
/// (role, content) pair so a strict reasoning+tool template (the 122B) can be fed structured `reasoningContent`
/// + `toolCalls` through mlx's raw-dict prompt hatch. The MLXBackend render bridge converts these to the
/// template input; `Chat.Message` (mlx) can't carry reasoning/tool_calls, which is exactly why this exists.
public struct TurnMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable { case system, user, assistant, tool }
    public var role: Role
    public var content: String
    /// The model's `<think>` span from a completed assistant turn, replayed so a reasoning model keeps the
    /// thread of an in-progress tool chain (strict templates render this only for turns after the last user query).
    public var reasoningContent: String?
    /// Structured calls for an assistant turn — filled ONLY after the routed call is validated (`constrain`),
    /// so a parser miss is stored as plain `content`, never a malformed structured turn that poisons later rounds.
    public var toolCalls: [ToolCall]?
    /// VLM input: file URLs of images attached to this turn (a user turn carrying pictures). Carried on the turn
    /// (not a side channel) so the OWNED transcript re-emits the `<|vision_start|>…<|vision_end|>` markers on
    /// EVERY round — keeping the token-row prefix stable so the image's KV (encoded once on the cold prefill)
    /// is reused on later rounds instead of re-encoded. Empty for the overwhelming text-only majority.
    public var imageURLs: [URL] = []
    public init(role: Role, content: String, reasoningContent: String? = nil, toolCalls: [ToolCall]? = nil,
                imageURLs: [URL] = []) {
        self.role = role; self.content = content
        self.reasoningContent = reasoningContent; self.toolCalls = toolCalls
        self.imageURLs = imageURLs
    }
}

/// A routed tool call (name + best-effort args JSON) — what `recoverToolCallTag` / the backend parser surface.
public struct ToolCall: Sendable, Equatable {
    public var name: String
    public var argsJSON: String
    public init(name: String, argsJSON: String) { self.name = name; self.argsJSON = argsJSON }
}

/// Sampling knobs the loop needs, MLX-free (MLXBackend builds an mlx `GenerateParameters` from these + the
/// generation budget). ONE source instead of the temp/repPenalty/repContextSize triple copied into both loops.
public struct SamplingParams: Sendable, Equatable {
    public var temperature: Float
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int
    public init(temperature: Float = 0.0, repetitionPenalty: Float = 1.15, repetitionContextSize: Int = 20) {
        self.temperature = temperature
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

/// The model's own config, read ONCE from its HF snapshot. Sendable + pure: the impure file read lives in
/// MLXBackend (`SnapshotConfig.read`), the DERIVATION lives here (`ModelProfile.resolve`) so it's unit-testable
/// with literal fixtures. This is the ONE config read that replaces the two split reads (ModelFamilyDetector +
/// Orbital `MLXModelScan`), so format / reasoning / stops / window can't diverge. `chatTemplate` is the raw
/// template string from EITHER `chat_template.jinja` OR the embedded `tokenizer_config.json["chat_template"]`
/// (un-escaped by the reader) — half the cached models ship it embedded, not as a `.jinja` file.
public struct SnapshotConfig: Sendable {
    public var modelId: String
    public var modelType: String?
    public var contextWindow: Int                      // config.json max_position_embeddings (→ budgets)
    public var chatTemplate: String?                   // raw template (jinja file OR embedded), un-escaped
    public var eosTokenStrings: [String]               // tokenizer_config eos_token (+ explicit stop strings)
    public var eosTokenIds: [Int]                       // config/generation_config eos_token_id (the 122B has TWO)
    public var generationStopStrings: [String]          // generation_config.json stop_strings (rare)
    public var mlxInferredFormat: ToolCallFormatChoice  // mlx-swift-lm's own inference — the fallback signal

    public init(modelId: String, modelType: String? = nil, contextWindow: Int = 8192,
                chatTemplate: String? = nil, eosTokenStrings: [String] = [], eosTokenIds: [Int] = [],
                generationStopStrings: [String] = [], mlxInferredFormat: ToolCallFormatChoice = .deferToMLX) {
        self.modelId = modelId; self.modelType = modelType; self.contextWindow = contextWindow
        self.chatTemplate = chatTemplate; self.eosTokenStrings = eosTokenStrings; self.eosTokenIds = eosTokenIds
        self.generationStopStrings = generationStopStrings; self.mlxInferredFormat = mlxInferredFormat
    }
}

extension ModelProfile {
    /// Model types whose chat template RAISES when no real user query survives a continuation round — a SAFETY
    /// NET so a known-strict model whose template can't be read still gets owned render, instead of silently
    /// degrading to the throwing default. (The primary signal is the template scan below; this is the backstop.)
    public static let knownStrictModelTypes: Set<String> = ["qwen3_5_moe"]

    /// Build the per-model adapter from the model's own config — the config-driven resolution that replaces
    /// hardcoded family branches with TEMPLATE EVIDENCE. `forModelType` still supplies the family/budget basics
    /// (the irreducible reasoning tie-break that `model_type` alone can't make); this layers the leaked concerns
    /// (format, reasoning, stops, owned-render) on top, derived from the snapshot.
    public static func resolve(_ snap: SnapshotConfig) -> ModelProfile {
        // mlx's inferred style is the fallback signal `forModelType` already keys GLM detection on.
        let style: ToolCallStyle = (snap.mlxInferredFormat == .glm4) ? .taggedReasoning : .rawJSON
        let base = forModelType(snap.modelType, toolCallStyle: style,
                                modelId: snap.modelId, contextWindow: snap.contextWindow)
        let tmpl = snap.chatTemplate ?? ""
        // FORMAT — positive template evidence wins; else defer to mlx (never worse than today).
        let format = Self.chooseToolCallFormat(fromTemplate: tmpl, fallback: snap.mlxInferredFormat)
        // REASONING — true when the GENERATION PROMPT opens a <think> block (the model GENERATES reasoning),
        // not merely when the template accepts a `reasoning_content` field on input. Keep the family tie-break
        // (`base.emitsReasoning`) as an additional positive (GLM / R1-distill).
        let reasons = Self.templateEmitsReasoning(tmpl) || base.emitsReasoning
        let rcField: String? = tmpl.contains("reasoning_content") ? "reasoning_content" : nil
        // OWNED RENDER — the strict reverse user-query scan that raises (`No user query`), or a known-strict
        // model_type (safety net for an unreadable template). The 80B has the same scan but does NOT raise, so
        // matching the raise message (not just the scan idiom) avoids over-flagging it.
        let strict = tmpl.contains("No user query") || Self.knownStrictModelTypes.contains(snap.modelType ?? "")
        // STOPS — the model's OWN eos strings + any generation_config stop strings (deduped).
        let stops = Array(Set(snap.eosTokenStrings + snap.generationStopStrings)).sorted()
        return ModelProfile(
            family: base.family, emitsReasoning: reasons, toolCallStyle: base.toolCallStyle,
            budget: base.budget, budgetForcing: base.budgetForcing, contextBudget: base.contextBudget,
            toolCallFormat: format, reasoningContentField: rcField,
            stopStrings: stops, eosTokenIds: snap.eosTokenIds,
            sampling: base.sampling, requiresOwnedRender: strict)
    }

    /// Choose the wire format from what the template actually RENDERS for a call. Order matters: XML-function and
    /// GLM are unambiguous tag shapes; JSON-in-tags is a `<tool_call>` block whose body is a JSON object.
    static func chooseToolCallFormat(fromTemplate t: String, fallback: ToolCallFormatChoice) -> ToolCallFormatChoice {
        guard !t.isEmpty else { return fallback }
        if t.contains("<function=") { return .xmlFunction }                 // Qwen3.5-122B
        if t.contains("arg_key") || t.contains("arg_value") { return .glm4 } // GLM arg-tag form
        if t.contains("<tool_call>") &&
            (t.contains("tojson") || t.contains("\"name\"") || t.contains("'name'") || t.contains("arguments")) {
            return .json                                                     // Qwen3-Next / Hermes JSON-in-tags
        }
        if t.contains("<|python_tag|>") || t.contains("func_name") { return .pythonic }
        return fallback
    }

    /// A reasoning model GENERATES `<think>` because its generation prompt opens one. Detect that, rather than
    /// the mere presence of a `reasoning_content` input field (which a non-generating template may also handle).
    static func templateEmitsReasoning(_ t: String) -> Bool {
        guard t.contains("<think>") else { return false }
        return t.contains("add_generation_prompt") || t.contains("enable_thinking")
    }

    /// The message array fed to the template for a round. For a STRICT template (`requiresOwnedRender`) guarantee
    /// the precondition the real template needs: a non-`<tool_response>` user turn survives every continuation
    /// round (else the 122B's reverse user-query scan raises). Re-assert the original query at the head when a
    /// continuation round would otherwise carry only tool/assistant turns. Identity for non-strict models — the
    /// existing behavior. PURE [TurnMessage] → [TurnMessage], so it's unit-tested with no model load.
    public func continuationMessages(_ transcript: [TurnMessage]) -> [TurnMessage] {
        guard requiresOwnedRender else { return transcript }
        let hasRealUser = transcript.contains { $0.role == .user && !Self.isBareToolResponse($0.content) }
        if hasRealUser { return transcript }
        // No surviving real user query (e.g. a tool-result-only continuation) — re-assert the first user turn.
        if let firstUser = transcript.first(where: { $0.role == .user }) { return [firstUser] + transcript }
        return transcript
    }

    static func isBareToolResponse(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.hasPrefix("<tool_response>") && t.hasSuffix("</tool_response>")
    }
}
