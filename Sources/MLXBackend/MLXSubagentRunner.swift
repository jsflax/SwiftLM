import Foundation
import Serving
import NativeTools
import MiniBPE
import Orchestration

// ── The MLX-backed `SubagentRunner` (the concrete dependency the pure `TaskTool` is injected with).
//
// Running a sub-agent = running another agent TURN, re-entering the SAME loaded model with a FRESH,
// isolated context: the loaded `~/.claude/agents/<type>.md` system prompt, the agent's scoped tools, its
// own round budget, and NO hooks (so it doesn't re-fire the advise hook and recursively spawn itself).
// Each `run()` re-enters `runWithTools`, which builds its own `CompactingSession` → independent context;
// the model serializes inside `ModelContainer`, so concurrent `run()`s are safe (they interleave, not
// corrupt). v1 calls `runWithTools` directly — routing the GENERATION leg through `ComputePool` for true
// GPU-batched fan-out is the (measurement-gated) batching build, which plugs in under this same contract.
public final class MLXSubagentRunner: SubagentRunner, @unchecked Sendable {
    let model: MLXLanguageModel
    let host: MCPHost
    let agentDefs: [String: AgentDefinition]
    let grammarTokenizer: (any GrammarTokenizer)?
    let profile: ModelProfile
    /// SHARED across all run() calls so a concurrent fan-out coalesces into batched forward passes.
    let pool: LocalBatchPool

    public init(model: MLXLanguageModel, host: MCPHost, agentDefs: [String: AgentDefinition],
                grammarTokenizer: (any GrammarTokenizer)? = nil, profile: ModelProfile = .generic) {
        self.model = model; self.host = host; self.agentDefs = agentDefs
        self.grammarTokenizer = grammarTokenizer; self.profile = profile
        self.pool = model.makeBatchPool()
    }

    public func run(subagentType: String, description: String, prompt: String) async throws -> String {
        guard let def = agentDefs[subagentType] else {
            return "ERROR: unknown subagent type '\(subagentType)'. Available: "
                + agentDefs.keys.sorted().joined(separator: ", ")
        }
        // Scope tools to the agent definition (or all tools if it declares none), and NEVER expose `Task`
        // to a sub-agent — one level deep, like Claude (no recursive fan-out).
        let allTools = await host.toolNames
        var allow = def.tools.map(Set.init) ?? Set(allTools)
        allow.remove("Task")
        let maxRounds = def.maxTurns ?? 8

        // SLICE 3 (opt-in via SWIFTLM_BATCH_SUBAGENTS): route this sub-agent's generation through the shared
        // coalescing pool so a CONCURRENT fan-out fuses into batched forward passes. Off by default because a
        // LONE sub-agent (the common session-learner case) is faster on ChatSession's incremental KV than on
        // the pool's re-prefill-per-round; the batching win is real only under actual concurrency.
        let modelId = model.modelId
        let batchGen: BatchGenerator? = ProcessInfo.processInfo.environment["SWIFTLM_BATCH_SUBAGENTS"] != nil
            ? { @Sendable [pool] (tokens: [Int32], maxTok: Int) in
                await pool.complete(InferenceRequest(model: ModelID(modelId), prompt: "",
                                                     maxTokens: maxTok, inputTokens: tokens))
              }
            : nil

        // No hooks: a sub-agent's turn must not run the UserPromptSubmit advise hook (it would inject another
        // learning nudge and spawn another session-learner). It works from the prompt it was handed.
        return try await model.runWithTools(
            prompt, host: host, instructions: def.systemPrompt,
            maxRounds: maxRounds, hooks: nil,
            grammarTokenizer: grammarTokenizer, profile: profile,
            toolAllowlist: allow, batchGenerator: batchGen)
    }
}
