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

/// The per-turn OUTPUT-token BACKSTOP — NOT a budget. EOS is the real terminator: every legitimate turn (a
/// full-file write, a long reasoning chain) stops well under this. The cap exists ONLY because WE run the
/// local decode loop and a local model can spin forever without ever emitting EOS — there is no provider to
/// stop it (unlike the claude/codex/gemini CLIs, which own their own loop and get NO cap from us). Local
/// generation therefore runs to EOS like a hosted model. Bounded ONLY by the model's REAL context window (the
/// physical limit) — NO arbitrary ceiling. A real no-EOS spiral is caught by the DegenerateRunDetector tripwire
/// (entropy/repetition), not by a hand-set token cap. No latency target, no tok/s guess, no floor/ceiling pair,
/// no per-family tuning.
public struct GenerationBudget: Sendable {
    public let maxTokens: Int
    public init(maxTokens: Int) { self.maxTokens = maxTokens }
    /// The ONLY bound on a turn's output is the model's PHYSICAL context window — NO arbitrary token ceiling.
    /// A hand-set cap (the former 16384 "runawayCeiling") masks model problems instead of fixing them and
    /// truncates legitimate long output (it cut a correct 122B build mid-file, which read as a model failure).
    /// Real degeneracy is bounded by the DegenerateRunDetector tripwire (entropy/repetition), natural completion
    /// by EOS, and the hard upper bound is the window itself — not a magic number.
    public static func forWindow(_ contextWindow: Int) -> GenerationBudget {
        GenerationBudget(maxTokens: max(2048, contextWindow))
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
    /// Derive from a model's context window: compact at 70%, keep the most recent 25% verbatim, KV floor 70%.
    /// A SINGLE tool output may use up to HALF the window (min 4K tokens) — a code file / long grep must FIT, not
    /// get chopped to a fixed 2K that ignores the model's actual capacity. Truncation now only guards a genuinely
    /// huge (>½ window) read, and `truncateToTokens` DECODES what it keeps (no byte-level garble).
    public static func forWindow(_ contextWindow: Int) -> ContextBudget {
        ContextBudget(maxContextTokens: Int(Double(contextWindow) * 0.70),
                      keepRecentTokens: Int(Double(contextWindow) * 0.25),
                      maxToolOutputTokens: max(4096, Int(Double(contextWindow) * 0.5)),
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

    // ── Adapter concerns (see LocalAgentAdapter.swift). All default to the current behavior so existing call
    // sites (forModelType/generic/tests) are unchanged; `resolve(_:)` fills them from the model's real config.
    public let toolCallFormat: ToolCallFormatChoice   // the wire the model emits (template-derived); .deferToMLX = use mlx's
    public let reasoningTags: (open: String, close: String)   // the `<think>` span to strip / detect
    public let reasoningContentField: String?         // template's reasoning field (e.g. "reasoning_content"), or nil
    public let stopStrings: [String]                  // the model's own stop strings (∪ universal floor in MLXBackend)
    public let eosTokenIds: [Int]                      // explicit EOS ids (the 122B has TWO) — added to the decode stop set
    public let sampling: SamplingParams               // one source for temp/repPenalty/repContextSize (was copied ×2)
    public let requiresOwnedRender: Bool              // strict template → re-render the full conversation each round

    public init(family: ModelFamily, emitsReasoning: Bool, toolCallStyle: ToolCallStyle,
                budget: GenerationBudget, budgetForcing: BudgetForcing = .none,
                contextBudget: ContextBudget = .forWindow(8192),
                toolCallFormat: ToolCallFormatChoice = .deferToMLX,
                reasoningTags: (open: String, close: String) = ("<think>", "</think>"),
                reasoningContentField: String? = nil,
                stopStrings: [String] = [], eosTokenIds: [Int] = [],
                sampling: SamplingParams = SamplingParams(), requiresOwnedRender: Bool = false) {
        self.family = family
        self.emitsReasoning = emitsReasoning
        self.toolCallStyle = toolCallStyle
        self.budget = budget
        self.budgetForcing = budgetForcing
        self.contextBudget = contextBudget
        self.toolCallFormat = toolCallFormat
        self.reasoningTags = reasoningTags
        self.reasoningContentField = reasoningContentField
        self.stopStrings = stopStrings
        self.eosTokenIds = eosTokenIds
        self.sampling = sampling
        self.requiresOwnedRender = requiresOwnedRender
    }

    /// The user-facing answer: reasoning/tool tags removed, with a conclusion fallback when a reasoning model
    /// ends inside `<think>` (so the turn never displays empty). A no-op for non-reasoning output (no tags).
    public func stripForDisplay(_ text: String) -> String { displayAnswer(text) }

    /// Recover a tool call the backend parser left unsurfaced — a `<tool_call>…</tool_call>` tag the model
    /// emitted but mlx's per-format parser didn't match. Two real cases: GLM's bare-name no-arg
    /// `<tool_call>name</tool_call>`, and Qwen/Hermes JSON-in-tags `<tool_call>{"name":…,"arguments":…}</tool_call>`
    /// (mlx hardcodes Qwen → `.xmlFunction`, whose parser can't read the JSON body, so the call leaks as text).
    /// `recoverToolCallTag` returns nil when there's no tag, so a true raw-JSON model (bare JSON, no tags) is
    /// unaffected — making this safe to attempt for ANY family, not just `.taggedReasoning`.
    public func recoverMissedToolCall(_ text: String) -> (name: String, argsJSON: String)? {
        recoverToolCallTag(text)
    }

    /// Safe default for an unknown model: non-reasoning, raw-JSON, middle-of-the-road budget.
    public static let generic = ModelProfile(
        family: .generic, emitsReasoning: false, toolCallStyle: .rawJSON,
        budget: GenerationBudget.forWindow(8192))

    /// The registry (the ONE place families are enumerated): build a profile from `model_type` (config.json),
    /// the tool-call style (already inferred by mlx-swift-lm), and id substrings — the reasoning tie-break,
    /// since `qwen2` covers both a plain Coder AND an R1-distill. Most-specific first; unknown → `.generic`.
    public static func forModelType(_ modelType: String?, toolCallStyle: ToolCallStyle,
                                    modelId: String, contextWindow: Int = 8192) -> ModelProfile {
        let id = modelId.lowercased()
        let type = (modelType ?? "").lowercased()
        let isR1 = id.contains("distill") || id.contains("-r1") || id.contains("deepseek-r1")
        let ctx = ContextBudget.forWindow(contextWindow)   // compaction budget from the model's window

        // The OUTPUT backstop is the SAME for every family — run to EOS, capped only by the runaway backstop
        // (GenerationBudget.forWindow). Families still differ in what's real: reasoning, tool-call style. (The
        // old per-family "latency budget" capped a coder model at ~1000 tokens, which truncated a write_file
        // call mid-file → invalid JSON → no dispatch → the agent narrated but never built.)
        let budget = GenerationBudget.forWindow(contextWindow)
        // GLM-4 family: reasoning + TAGGED tool calls.
        if type.hasPrefix("glm4") || toolCallStyle == .taggedReasoning {
            return ModelProfile(family: .glm4, emitsReasoning: true, toolCallStyle: .taggedReasoning,
                                 budget: budget, contextBudget: ctx)
        }
        // R1-distill: reasoning, raw-JSON tools.
        if isR1 {
            return ModelProfile(family: .deepseekR1, emitsReasoning: true, toolCallStyle: .rawJSON,
                                 budget: budget, contextBudget: ctx)
        }
        // Instruct / coder model (no reasoning).
        if type.hasPrefix("qwen") || type.hasPrefix("llama") || type.hasPrefix("mistral")
            || type.hasPrefix("gemma") || type.hasPrefix("phi") {
            return ModelProfile(family: .qwen, emitsReasoning: false, toolCallStyle: toolCallStyle,
                                 budget: budget, contextBudget: ctx)
        }
        return ModelProfile(family: .generic, emitsReasoning: false, toolCallStyle: .rawJSON,
                            budget: budget, contextBudget: ctx)
    }
}
