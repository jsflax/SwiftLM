import Testing
import Foundation
@testable import Serving

struct PermissionsTests {
    @Test func autoAllowsEverything() {
        let p = ToolPermissionPolicy(mode: .auto)
        #expect(p.decide(tool: "write_file") == .allow)
        #expect(p.decide(tool: "bash") == .allow)
        #expect(p.decide(tool: "mcp__etrade__place_order") == .allow)
        #expect(p.allowsMutation)
        #expect(p.instructionPrefix == nil)            // no plan note outside plan mode
    }

    @Test func planAllowsReadOnlyDeniesEverythingElse() {
        let p = ToolPermissionPolicy(mode: .plan)
        for ro in ["read_file", "glob", "grep", "web_fetch"] {
            #expect(p.decide(tool: ro) == .allow, "expected \(ro) allowed in plan mode")
        }
        // mutating native tools + any MCP tool (unknown read-only-ness) are conservatively denied
        for mut in ["write_file", "edit_file", "bash", "mcp__trader__log_trade", "anything_else"] {
            guard case .deny = p.decide(tool: mut) else { Issue.record("\(mut) should be denied"); return }
        }
        #expect(!p.allowsMutation)
        #expect(p.instructionPrefix?.contains("Plan mode") == true)
        #expect(p.instructionPrefix?.contains("read_file") == true)   // the reader set is named to the model
    }

    @Test func approvalClampsToAllowUnderAutopilotElseNeedsApproval() {
        // a room has no human in the loop → approval clamps to allow (the autopilot peer)
        let autopilot = ToolPermissionPolicy(mode: .approval, autopilot: true)
        #expect(autopilot.decide(tool: "bash") == .allow)
        #expect(autopilot.allowsMutation)
        // a human IS in the loop → routed to an approver (the deferred ask-channel)
        let human = ToolPermissionPolicy(mode: .approval, autopilot: false)
        #expect(human.decide(tool: "bash") == .needsApproval)
        #expect(!human.allowsMutation)
    }

    @Test func customReadOnlySetExtendsPlanMode() {
        // a caller can mark a known-read-only MCP tool safe to run while planning
        let p = ToolPermissionPolicy(mode: .plan,
                                     readOnlyTools: ToolPermissionPolicy.readOnlyNativeTools.union(["mcp__fmp__quote"]))
        #expect(p.decide(tool: "mcp__fmp__quote") == .allow)
        guard case .deny = p.decide(tool: "mcp__fmp__search") else { Issue.record("non-listed MCP denied"); return }
    }

    @Test func modeRoundTripsThroughRawValue() {
        // PERMISSION env / Orbital Agent.permissionMode decode path
        #expect(PermissionMode(rawValue: "plan") == .plan)
        #expect(PermissionMode(rawValue: "approval") == .approval)
        #expect(PermissionMode(rawValue: "auto") == .auto)
        #expect(PermissionMode(rawValue: "garbage") == nil)
        #expect(PermissionMode.allCases.count == 3)
    }

    @Test func allowsPlanExitDefaultsTrueAndKeepsCLIExitPlanCeremony() {
        // CLI default: the plan note still drives the ExitPlanMode → approve → implement ceremony, and the
        // flag survives the post-approval `.auto` flip (so a half-approved policy stays consistent).
        let cli = ToolPermissionPolicy(mode: .plan)
        #expect(cli.allowsPlanExit == true)
        #expect(cli.instructionPrefix?.contains(exitPlanModeToolName) == true)
        #expect(cli.approved.allowsPlanExit == true)
    }

    @Test func orbitalPlanPolicyRoutesButCannotExitOrWrite() {
        // Orbital's multi-agent plan policy: routing tools run; files/exec AND ExitPlanMode are denied, so a
        // planner can consult + hand off but can NEVER self-escalate to writing code within its turn.
        let p = ToolPermissionPolicy(
            mode: .plan,
            readOnlyTools: ToolPermissionPolicy.readOnlyNativeTools.union(["handoff", "consult", "done"]),
            allowsPlanExit: false)
        for ok in ["read_file", "glob", "grep", "web_fetch", "handoff", "consult", "done"] {
            #expect(p.decide(tool: ok) == .allow, "expected \(ok) allowed for an Orbital planner")
        }
        for no in ["write_file", "edit_file", "bash", exitPlanModeToolName] {
            guard case .deny = p.decide(tool: no) else { Issue.record("\(no) should be denied"); return }
        }
        #expect(!p.allowsMutation)
        // The prompt steers to handoff, never ExitPlanMode, while keeping the plan-mode framing + reader list.
        let prefix = p.instructionPrefix
        #expect(prefix?.contains("Plan mode") == true)
        #expect(prefix?.contains("read_file") == true)
        #expect(prefix?.contains("handoff") == true)
        #expect(prefix?.contains(exitPlanModeToolName) == false)
    }
}
