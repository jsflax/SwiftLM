import Foundation

// ── The permission / autonomy ladder (the concrete form of "plan mode").
//
// An agent turn runs under a PermissionMode that governs whether its tool calls actually dispatch. This is
// the on-device analogue of Claude Code's plan/default/accept/auto modes, mapped onto the Orbital room
// model where there is NO human in the inner loop — agents are peers of claude/codex agents, and the
// referee is an agent, not a person. So:
//   • .auto     — dispatch every tool call (the default; today's behavior). The trusted autopilot peer.
//   • .approval — a human must approve each call. In an autopilot room there IS no human, so this CLAMPS
//                 to allow (the documented Orbital behavior); the human-approval UI is the deferred piece
//                 that needs the not-yet-built ask-the-user channel.
//   • .plan     — planning only: read-only tools may run (so the planner can inspect to inform the plan),
//                 but anything with side effects (write/edit/bash/MCP) is denied. The agent should produce
//                 a plan and hand off for execution. This is "plan mode" — approval is implicit in the
//                 handoff, exactly as the Orbital room model specifies.
//
// Pure value types, no MLX/Process/MCP — the decision is unit-testable, and BOTH agent loops (the live
// `runWithTools` REPL and the `streamAgentTurn` Orbital sequencer) route their dispatch seam through it.

public enum PermissionMode: String, Sendable, Codable, Equatable, CaseIterable {
    case auto       // dispatch all tool calls
    case approval   // require approval per call (clamped to allow under autopilot — no human in the loop)
    case plan       // planning only: read-only tools allowed, mutating/unknown denied
}

/// The Claude-faithful tool a planning agent calls to present its finished plan for approval. Named/shaped
/// exactly like Claude Code's so SwiftLM stays a wire-level drop-in: a consumer (the CLI harness here, or
/// Orbital later) intercepts this tool call, surfaces the plan, and on approval flips the turn to `.auto`.
public let exitPlanModeToolName = "ExitPlanMode"

/// A human/consumer's verdict on a presented plan. `.approve` → flip to auto and implement; `.reject` →
/// stay in plan mode and revise.
public enum PlanApproval: Sendable, Equatable {
    case approve
    case reject(reason: String)
}

/// Injected approver: given the plan text, decide. The CLI reads stdin; Orbital/headless supply their own
/// (or auto-approve). Async so it can block on a human. `@Sendable` so it crosses the turn's task boundary.
public typealias PlanApprover = @Sendable (_ plan: String) async -> PlanApproval

/// Decides, per tool call, whether it may dispatch under the active PermissionMode. Conservative by
/// design: in plan mode ONLY tools on the read-only allow-list run, so an unknown or side-effecting tool
/// (including any MCP tool — e.g. `place_order`) is denied rather than guessed safe.
public struct ToolPermissionPolicy: Sendable, Equatable {
    public enum Decision: Sendable, Equatable {
        case allow
        case deny(reason: String)
        case needsApproval   // .approval mode with a human in the loop — routed to an approver (deferred)
    }

    public var mode: PermissionMode
    /// A room with no human in the inner loop. When true, `.approval` clamps to allow (the autopilot peer).
    public var autopilot: Bool
    /// Tools with no side effects — the set permitted in plan mode. Defaults to the native read-only tools;
    /// callers may extend it with known-read-only MCP tools.
    public var readOnlyTools: Set<String>

    /// When `true` (the CLI default), a plan-mode agent may call `ExitPlanMode` to present its plan and, on
    /// approval, flip the turn to `.auto` and implement. Orbital's multi-agent rooms set this `false`: there is
    /// no exit-plan step — a planner hands off to the builder and approval is implicit in the handoff. With
    /// `false`, both agent loops SKIP the `ExitPlanMode` interception, so the call falls to the deny gate (it is
    /// not a read-only tool) and the `.plan → .auto` self-escalation is unreachable.
    public var allowsPlanExit: Bool

    /// The built-in native tools that only observe the world (safe to run while planning).
    public static let readOnlyNativeTools: Set<String> = ["read_file", "glob", "grep", "web_fetch"]

    public init(mode: PermissionMode = .auto,
                autopilot: Bool = true,
                readOnlyTools: Set<String> = ToolPermissionPolicy.readOnlyNativeTools,
                allowsPlanExit: Bool = true) {
        self.mode = mode
        self.autopilot = autopilot
        self.readOnlyTools = readOnlyTools
        self.allowsPlanExit = allowsPlanExit
    }

    public func decide(tool: String) -> Decision {
        switch mode {
        case .auto:
            return .allow
        case .approval:
            return autopilot ? .allow : .needsApproval
        case .plan:
            return readOnlyTools.contains(tool)
                ? .allow
                : .deny(reason: "plan mode: '\(tool)' is not a read-only tool. Describe this step in your "
                        + "plan and hand off for execution instead of calling it now.")
        }
    }

    /// True when the mode lets a side-effecting tool run at all (so callers can short-circuit).
    public var allowsMutation: Bool { mode == .auto || (mode == .approval && autopilot) }

    /// The same policy switched to `.auto` — the post-approval state a turn flips to when its plan is
    /// approved (keeps `autopilot`/`readOnlyTools`, only the mode changes).
    public var approved: ToolPermissionPolicy {
        var p = self; p.mode = .auto; return p
    }

    /// A system-context note injected in plan mode so the model PLANS rather than flailing against denied
    /// write/exec tools. `nil` outside plan mode (no behavior change for .auto/.approval).
    public var instructionPrefix: String? {
        guard mode == .plan else { return nil }
        let readers = readOnlyTools.sorted().joined(separator: ", ")
        // The closing action depends on whether this environment HAS an exit-plan step. The CLI does
        // (ExitPlanMode → approve → implement); Orbital rooms do not — a planner hands off to the builder.
        let close = allowsPlanExit
            ? "call the \(exitPlanModeToolName) tool with your plan to request approval before making any changes."
            : "call `handoff` to pass your plan to the next agent for execution — handing off IS the approval "
              + "(there is no separate exit-plan step)."
        return "[Plan mode] You are PLANNING ONLY. You may inspect with read-only tools (\(readers)) but "
            + "must NOT modify files or run commands — those tools are blocked. Produce a concise, numbered "
            + "plan of the steps you WOULD take, then " + close
    }
}
