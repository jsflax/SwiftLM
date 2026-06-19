import Foundation

// ── Per-model-FAMILY profile: the small set of behaviors the GENERIC agent loop must vary by model, so no
// family literal (no "glm4"/"qwen") ever appears in the loop. A Sendable VALUE type (not a protocol) — it
// threads trivially into the loop (and, later, the streaming path) and is unit-testable without a model.
// The loop consults the profile for exactly three things: how to bound generation (a physics-derived
// runaway rail, NOT a content cap), how to recover a tool call the backend parser missed, and what to hide
// from the displayed answer. New families are added in ONE place (`forModelType`), not scattered through code.

/// The one tool-call distinction the loop acts on: tag-emitting families may emit a bare-name
/// `<tool_call>name</tool_call>` the backend parser drops (recover it); raw-JSON families have everything
/// surfaced already (nothing to recover). Derived from mlx-swift-lm's per-model `ToolCallFormat`.
public enum ToolCallStyle: Sendable { case taggedReasoning, rawJSON }

/// A coarse family tag — for logging/diagnostics only. The loop NEVER switches on this (it switches on the
/// capability fields below, which keeps it generic); this just lets a human see what was detected.
public enum ModelFamily: String, Sendable { case glm4, qwen, deepseekR1, generic }

/// A PHYSICS-DERIVED generation bound — the plan's CoT rule made concrete: NOT an arbitrary cap. EOS is the
/// real terminator (normal turns stop well under this); the bound only catches a non-terminating runaway
/// (repetition / missing EOS). `maxTokens` = a latency budget (target seconds × assumed tok/s), clamped so
/// it can never fall below a reasoning `floor` (a long `<think>` is never clipped) nor exceed a hard `ceil`.
/// `assumedTokensPerSecond` is the documented per-family seed AND the seam where measured throughput lands
/// later — an inert constant for now.
public struct GenerationBudget: Sendable {
    public var latencyTargetSeconds: Double
    public var assumedTokensPerSecond: Double
    public var floorTokens: Int
    public var ceilTokens: Int
    public init(latencyTargetSeconds: Double, assumedTokensPerSecond: Double,
                floorTokens: Int, ceilTokens: Int) {
        self.latencyTargetSeconds = latencyTargetSeconds
        self.assumedTokensPerSecond = assumedTokensPerSecond
        self.floorTokens = floorTokens
        self.ceilTokens = ceilTokens
    }
    public var maxTokens: Int {
        min(ceilTokens, max(floorTokens, Int((latencyTargetSeconds * assumedTokensPerSecond).rounded(.up))))
    }
}

/// The CONTEXT budget — when to compact the conversation (Claude-Code-style) and how much to keep. Derived
/// from the model's context window (`config.json["max_position_embeddings"]`). Distinct from GenerationBudget
/// (which bounds OUTPUT tokens per turn); this bounds the INPUT conversation that accumulates across rounds.
public struct ContextBudget: Sendable {
    public var maxContextTokens: Int      // compact once the conversation crosses this (= window × safety)
    public var keepRecentTokens: Int      // on compaction, keep this many recent tokens VERBATIM (older → summary)
    public var maxToolOutputTokens: Int   // cap a SINGLE tool result to this many tokens (+ a truncation marker)
    public var maxKVSize: Int?            // RotatingKVCache floor (mlx) — bounds KV memory; nil = simple cache
    public init(maxContextTokens: Int, keepRecentTokens: Int,
                maxToolOutputTokens: Int = 2048, maxKVSize: Int? = nil) {
        self.maxContextTokens = maxContextTokens
        self.keepRecentTokens = keepRecentTokens
        self.maxToolOutputTokens = maxToolOutputTokens
        self.maxKVSize = maxKVSize
    }
    /// Derive from a model's context window: compact at 70%, keep the most recent 25% verbatim, cap one tool
    /// output at 2K tokens, KV floor 70%.
    public static func forWindow(_ contextWindow: Int) -> ContextBudget {
        ContextBudget(maxContextTokens: Int(Double(contextWindow) * 0.70),
                      keepRecentTokens: Int(Double(contextWindow) * 0.25),
                      maxToolOutputTokens: 2048,
                      maxKVSize: Int(Double(contextWindow) * 0.70))
    }
}

/// Reserved seam for the s1-style budget-forcing rail (on overflow, inject `</think>` + an answer prompt).
/// Inert in v1 — the loop adds NO branch for `.none`; this exists so the rail is additive later.
public enum BudgetForcing: Sendable { case none }

public struct ModelProfile: Sendable {
    public let family: ModelFamily
    public let emitsReasoning: Bool            // emits `<think>…</think>` → strip from display + budget headroom
    public let toolCallStyle: ToolCallStyle
    public let budget: GenerationBudget
    public let budgetForcing: BudgetForcing
    public let contextBudget: ContextBudget

    public init(family: ModelFamily, emitsReasoning: Bool, toolCallStyle: ToolCallStyle,
                budget: GenerationBudget, budgetForcing: BudgetForcing = .none,
                contextBudget: ContextBudget = .forWindow(8192)) {
        self.family = family
        self.emitsReasoning = emitsReasoning
        self.toolCallStyle = toolCallStyle
        self.budget = budget
        self.budgetForcing = budgetForcing
        self.contextBudget = contextBudget
    }

    /// The user-facing answer: reasoning/tool tags removed, with a conclusion fallback when a reasoning model
    /// ends inside `<think>` (so the turn never displays empty). A no-op for non-reasoning output (no tags).
    public func stripForDisplay(_ text: String) -> String { displayAnswer(text) }

    /// Recover a tool call the backend parser left unsurfaced — only for tag-emitting families (e.g. GLM's
    /// bare-name no-arg `<tool_call>name</tool_call>`). Raw-JSON families return nil (nothing to recover).
    public func recoverMissedToolCall(_ text: String) -> (name: String, argsJSON: String)? {
        toolCallStyle == .taggedReasoning ? recoverToolCallTag(text) : nil
    }

    /// Safe default for an unknown model: non-reasoning, raw-JSON, middle-of-the-road budget.
    public static let generic = ModelProfile(
        family: .generic, emitsReasoning: false, toolCallStyle: .rawJSON,
        budget: GenerationBudget(latencyTargetSeconds: 25, assumedTokensPerSecond: 60,
                                 floorTokens: 1024, ceilTokens: 4096))

    /// The registry (the ONE place families are enumerated): build a profile from `model_type` (config.json),
    /// the tool-call style (already inferred by mlx-swift-lm), and id substrings — the reasoning tie-break,
    /// since `qwen2` covers both a plain Coder AND an R1-distill. Most-specific first; unknown → `.generic`.
    public static func forModelType(_ modelType: String?, toolCallStyle: ToolCallStyle,
                                    modelId: String, contextWindow: Int = 8192) -> ModelProfile {
        let id = modelId.lowercased()
        let type = (modelType ?? "").lowercased()
        let isR1 = id.contains("distill") || id.contains("-r1") || id.contains("deepseek-r1")
        let ctx = ContextBudget.forWindow(contextWindow)   // compaction budget from the model's window

        // GLM-4 family: reasoning + TAGGED tool calls. MoE decode is slower → larger latency budget.
        if type.hasPrefix("glm4") || toolCallStyle == .taggedReasoning {
            return ModelProfile(
                family: .glm4, emitsReasoning: true, toolCallStyle: .taggedReasoning,
                budget: GenerationBudget(latencyTargetSeconds: 120, assumedTokensPerSecond: 40,
                                         floorTokens: 4096, ceilTokens: 8192),
                contextBudget: ctx)
        }
        // R1-distill: reasoning, raw-JSON tools. Long CoT → high reasoning floor.
        if isR1 {
            return ModelProfile(
                family: .deepseekR1, emitsReasoning: true, toolCallStyle: .rawJSON,
                budget: GenerationBudget(latencyTargetSeconds: 120, assumedTokensPerSecond: 60,
                                         floorTokens: 4096, ceilTokens: 8192),
                contextBudget: ctx)
        }
        // Instruct / coder model (no reasoning): tight budget — an answer doesn't need a CoT-sized floor.
        if type.hasPrefix("qwen") || type.hasPrefix("llama") || type.hasPrefix("mistral")
            || type.hasPrefix("gemma") || type.hasPrefix("phi") {
            return ModelProfile(
                family: .qwen, emitsReasoning: false, toolCallStyle: toolCallStyle,
                budget: GenerationBudget(latencyTargetSeconds: 10, assumedTokensPerSecond: 100,
                                         floorTokens: 512, ceilTokens: 2048),
                contextBudget: ctx)
        }
        return ModelProfile(family: .generic, emitsReasoning: false, toolCallStyle: .rawJSON,
                            budget: ModelProfile.generic.budget, contextBudget: ctx)
    }
}
