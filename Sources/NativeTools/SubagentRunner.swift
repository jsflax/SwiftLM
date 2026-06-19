import Foundation

// ── The dependency the `Task` tool delegates to (dependency injection).
//
// Spawning a sub-agent means running another agent turn, which means re-entering the model — an MLX
// concern. But the `Task` tool itself belongs in this pure, MLX-free NativeTools module (it's a first-class
// native tool, exactly like Claude Code's `Task`). The two are reconciled with DI: `TaskTool` depends on
// this PROTOCOL, and MLXBackend provides the concrete `MLXSubagentRunner` (which loads the agent definition
// and re-enters `runWithTools`) and injects it when the agent builds its tool registry. So NativeTools stays
// backend-agnostic and unit-testable (tests inject a stub runner), and there is no loop-intercept hack.

/// Runs a named sub-agent on a prompt and returns its final text answer.
public protocol SubagentRunner: Sendable {
    func run(subagentType: String, description: String, prompt: String) async throws -> String
}
