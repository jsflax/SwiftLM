import Testing
import Foundation
@testable import Serving

/// End-to-end smoke test of the turn sequencer with a SCRIPTED stub backend (no model load). Proves a
/// full turn (text → tool call → result) emits the exact AgentStreamEvent sequence whose NDJSON lines
/// route to the right Orbital `StreamJsonEvent` cases, with consistent tool ids and all four hook seams
/// firing. This is the "falls neatly into Orbital" guarantee at the orchestration level — the per-event
/// wire shapes are covered by AgentStreamEventTests.
struct AgentTurnTests {
    @Test func fullTurnEmitsOrbitalEventSequence() async throws {
        let backend = StubBackend([
            [.chunk("Hel"), .chunk("lo"), .toolCall(name: "search", argsJSON: #"{"q":"x"}"#)],
            [.chunk("the answer")],
        ])
        let stream = streamAgentTurn(prompt: "hi", instructions: "persona", hooks: nil,
                                     config: .init(modelLabel: "m", sessionID: "S1", cwd: "/c",
                                                   toolNames: ["search"], maxRounds: 8),
                                     backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }

        // the wire-type sequence Orbital's StreamJsonEvent decoder would route each line to:
        #expect(events.map { routeType($0.ndjsonLine()) }
                == ["system", "stream_event", "stream_event", "assistant", "user", "stream_event", "result"])
        // toolUse id == tool_result id, so Orbital keys the tool card back correctly:
        let useId = events.firstToolUseId
        #expect(useId != nil && useId == events.firstToolResultId)
        // the tool actually dispatched with the model's args:
        #expect(backend.dispatched.map(\.name) == ["search"])
        #expect(backend.dispatched.first?.argsJSON == #"{"q":"x"}"#)
        // final answer = round-2 text (round-1 prose streamed as deltas, kept by Orbital's sawDelta):
        guard case .result(let final, _, _, _)? = events.last else { Issue.record("no result"); return }
        #expect(final == "the answer")
    }

    @Test func plainTextTurnHasNoToolEvents() async throws {
        let backend = StubBackend([[.chunk("hi "), .chunk("there")]])
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: .init(), backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(events.map { routeType($0.ndjsonLine()) } == ["system", "stream_event", "stream_event", "result"])
        guard case .result(let final, _, _, _)? = events.last else { Issue.record("no result"); return }
        #expect(final == "hi there")
        #expect(backend.dispatched.isEmpty)
    }

    @Test func userPromptSubmitHookAugmentsInstructions() async throws {
        let backend = StubBackend([[.chunk("ok")]])
        let chain = HookChain([CtxHook(ctx: "RECALLED: etrade token expired")])
        let stream = streamAgentTurn(prompt: "q", instructions: "base", hooks: chain, config: .init(), backend: backend)
        for try await _ in stream {}
        // the grounding (memory-hooks advise shape) is appended to the persona the model sees:
        #expect(backend.seenInstructions.first ?? nil == "base\n\nRECALLED: etrade token expired")
    }

    @Test func preToolUseDenyBlocksDispatch() async throws {
        let backend = StubBackend([[.toolCall(name: "Bash", argsJSON: "{}")], [.chunk("done")]])
        let stream = streamAgentTurn(prompt: "q", instructions: nil,
                                     hooks: HookChain([DenyToolHook()]), config: .init(), backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(backend.dispatched.isEmpty)                       // vetoed → never dispatched
        let block = try #require(events.firstToolResult)
        #expect(block.content.contains("BLOCKED"))                // synthetic blocked result fed back
        #expect(block.isError)
    }

    @Test func reasoningStrippedFromDeltasAndResult() async throws {
        // a reasoning model: <think> must NOT stream as a delta, and the result must be the clean answer.
        let backend = StubBackend([[.chunk("<think>secret plan</think>"), .chunk("Final answer.")]])
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: .init(), backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        let deltas = events.compactMap { if case .textDelta(let t) = $0 { return t } else { return nil } }
        #expect(!deltas.joined().contains("secret plan"))     // think suppressed from the live stream
        #expect(deltas.joined() == "Final answer.")
        guard case .result(let final, _, _, _)? = events.last else { Issue.record("no result"); return }
        #expect(final == "Final answer.")                     // result also clean
    }

    @Test func intentNudgeWhenModelTrailsOffInReasoning() async throws {
        // round 0 = reasoning-only (no answer, no tool) → nudge → round 1 answers.
        let backend = StubBackend([[.chunk("<think>let me check the file</think>")], [.chunk("All set.")]])
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: .init(), backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        guard backend.seenResume.count >= 2,
              case .user(let nudge)? = backend.seenResume[1].first else {   // round 1 got a user nudge
            Issue.record("round 1 was not a user nudge"); return
        }
        #expect(nudge.contains("Continue"))
        guard case .result(let final, _, _, _)? = events.last else { Issue.record("no result"); return }
        #expect(final == "All set.")
    }

    @Test func postToolUseReplacesResult() async throws {
        let backend = StubBackend([[.toolCall(name: "edit", argsJSON: "{}")], [.chunk("done")]])
        backend.dispatchResult = ("raw tool output", false)
        let stream = streamAgentTurn(prompt: "q", instructions: nil,
                                     hooks: HookChain([ReplaceHook(with: "compile-checked: OK")]),
                                     config: .init(), backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(events.firstToolResult?.content == "compile-checked: OK")   // PostToolUse transform applied
    }

    @Test func planModeBlocksSideEffectingToolAndInjectsPlanNote() async throws {
        // Plan mode: a write tool must NOT dispatch; the model gets a BLOCKED result + a plan-mode note.
        let backend = StubBackend([[.toolCall(name: "write_file", argsJSON: "{}")], [.chunk("here's the plan")]])
        let cfg = AgentTurnConfig(permission: .init(mode: .plan))
        let stream = streamAgentTurn(prompt: "build X", instructions: "be helpful", hooks: nil,
                                     config: cfg, backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(backend.dispatched.isEmpty)                                  // write blocked before dispatch
        let block = try #require(events.firstToolResult)
        #expect(block.content.contains("BLOCKED (plan)"))
        #expect(block.isError)
        #expect(backend.seenInstructions.first??.contains("[Plan mode]") == true)  // plan note injected at SEAM A
    }

    @Test func planModeAllowsReadOnlyTool() async throws {
        // Plan mode still lets the planner INSPECT with read-only tools (read_file) to inform the plan.
        let backend = StubBackend([[.toolCall(name: "read_file", argsJSON: #"{"path":"x"}"#)], [.chunk("ok")]])
        backend.dispatchResult = ("file contents", false)
        let cfg = AgentTurnConfig(permission: .init(mode: .plan))
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: cfg, backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(backend.dispatched.map(\.name) == ["read_file"])             // read-only ran
        #expect(events.firstToolResult?.content == "file contents")
    }

    @Test func autoModeDispatchesEverything() async throws {
        // Default .auto is unchanged: a write tool dispatches normally.
        let backend = StubBackend([[.toolCall(name: "write_file", argsJSON: "{}")], [.chunk("done")]])
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: .init(), backend: backend)
        for try await _ in stream {}
        #expect(backend.dispatched.map(\.name) == ["write_file"])
    }

    @Test func planApprovedFlipsToAutoAndImplements() async throws {
        // round0 (plan mode): model presents a plan via ExitPlanMode → approved → round1 the write DISPATCHES.
        let backend = StubBackend([
            [.toolCall(name: "ExitPlanMode", argsJSON: #"{"plan":"1. write the file"}"#)],
            [.toolCall(name: "write_file", argsJSON: #"{"path":"x","content":"y"}"#)],
            [.chunk("implemented")],
        ])
        let approvals = Approvals()
        let cfg = AgentTurnConfig(maxRounds: 5, permission: .init(mode: .plan),
                                  approvePlan: { plan in approvals.record(plan); return .approve })
        let stream = streamAgentTurn(prompt: "build X", instructions: nil, hooks: nil, config: cfg, backend: backend)
        for try await _ in stream {}
        #expect(approvals.plans.first == "1. write the file")        // the plan reached the approver
        #expect(backend.dispatched.map(\.name) == ["write_file"])    // post-approval the write RAN (flip worked)
    }

    @Test func planRejectedStaysBlocked() async throws {
        // ExitPlanMode rejected → still plan mode → a subsequent write is BLOCKED, never dispatched.
        let backend = StubBackend([
            [.toolCall(name: "ExitPlanMode", argsJSON: #"{"plan":"do risky thing"}"#)],
            [.toolCall(name: "write_file", argsJSON: "{}")],
            [.chunk("revised plan")],
        ])
        let cfg = AgentTurnConfig(maxRounds: 5, permission: .init(mode: .plan),
                                  approvePlan: { _ in .reject(reason: "no") })
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: cfg, backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        #expect(backend.dispatched.isEmpty)                          // never flipped → write stayed blocked
        let blocked = events.compactMap { if case .toolResult(_, let c, _) = $0 { return c } else { return nil } }
        #expect(blocked.contains { $0.contains("Plan rejected") })
        #expect(blocked.contains { $0.contains("BLOCKED (plan)") })
    }

    @Test func quietToolRoundsForceAnswerInsteadOfRamblingToMaxRounds() async throws {
        // An over-eager model that keeps calling DISTINCT tools (each success) must be cut off after
        // defaultMaxQuietToolRounds and forced to answer — NOT run all the way to maxRounds (12).
        let backend = RamblingBackend(duplicate: false)
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil,
                                     config: .init(maxRounds: 12), backend: backend)
        var final: String? = nil
        for try await e in stream { if case .result(let f, _, _, _) = e { final = f } }
        #expect(backend.dispatched.count == defaultMaxQuietToolRounds)  // 4 quiet rounds, then forced answer
        #expect(final == "final answer")
    }

    @Test func duplicateCallsAreNotRedispatched() async throws {
        // Identical (name+args) call repeated → dispatched once; the repeats short-circuit with a "stop" hint.
        let backend = RamblingBackend(duplicate: true)
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil,
                                     config: .init(maxRounds: 12), backend: backend)
        var final: String? = nil, dupHints = 0
        for try await e in stream {
            if case .result(let f, _, _, _) = e { final = f }
            if case .toolResult(_, let c, _) = e, c.contains("already called") { dupHints += 1 }
        }
        #expect(backend.dispatched.count == 1)     // only the first identical call actually ran
        #expect(dupHints >= 1)                      // the rest were short-circuited
        #expect(final == "final answer")
    }
}

/// Thread-safe capture box for the injected plan approver.
private final class Approvals: @unchecked Sendable {
    private let lock = NSLock()
    private var _plans: [String] = []
    func record(_ p: String) { lock.lock(); _plans.append(p); lock.unlock() }
    var plans: [String] { lock.lock(); defer { lock.unlock() }; return _plans }
}

/// Backend that models an over-eager model: it emits a tool call every round tools are on (distinct args, or
/// identical when `duplicate`), and only produces a final answer once tools are forced off.
private final class RamblingBackend: AgentTurnBackend, @unchecked Sendable {
    let duplicate: Bool
    private let lock = NSLock()
    private var _dispatched: [(name: String, argsJSON: String)] = []
    private var _round = 0
    init(duplicate: Bool) { self.duplicate = duplicate }
    var dispatched: [(name: String, argsJSON: String)] { lock.lock(); defer { lock.unlock() }; return _dispatched }

    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error> {
        let n: Int
        lock.lock(); n = _round; _round += 1; lock.unlock()
        return AsyncThrowingStream { c in
            if toolsEnabled {
                c.yield(.toolCall(name: "peek", argsJSON: duplicate ? #"{"x":1}"# : "{\"x\":\(n)}"))
            } else {
                c.yield(.chunk("final answer"))
            }
            c.finish()
        }
    }
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool) {
        lock.withLock { _dispatched.append((name, argsJSON)) }
        return ("ok", false)
    }
}

extension AgentTurnTests {
    @Test func hungDispatchTimesOutAndTurnStillFinishes() async throws {
        // A tool whose dispatch never returns must NOT hang the turn: it's abandoned after the bound, a
        // "did not respond" observation is injected, and the turn completes.
        let backend = HangingBackend()
        let cfg = AgentTurnConfig(maxRounds: 4, toolTimeoutSeconds: 0.1)
        let stream = streamAgentTurn(prompt: "q", instructions: nil, hooks: nil, config: cfg, backend: backend)
        var events: [AgentStreamEvent] = []
        for try await e in stream { events.append(e) }
        let results = events.compactMap { if case .toolResult(_, let c, _) = $0 { return c } else { return nil } }
        #expect(results.contains { $0.contains("did not respond") })   // timeout observation fed back
        #expect(events.contains { if case .result = $0 { return true } else { return false } })  // turn finished
    }
}

/// Backend whose tool dispatch never returns in time — models a wedged MCP tool (the `clipboard` hang).
private final class HangingBackend: AgentTurnBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var _round = 0
    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error> {
        let n: Int
        lock.lock(); n = _round; _round += 1; lock.unlock()
        return AsyncThrowingStream { c in
            if toolsEnabled && n == 0 { c.yield(.toolCall(name: "hang", argsJSON: "{}")) }
            else { c.yield(.chunk("done")) }
            c.finish()
        }
    }
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool) {
        try? await Task.sleep(nanoseconds: 60_000_000_000)   // 60s — would hang the turn without the bound
        return ("never", false)
    }
}

// ── helpers ────────────────────────────────────────────────────────────────────────────────────────

/// Mirrors Orbital's `StreamJsonEvent.init(from:)` routing — the discriminator a line decodes to.
private func routeType(_ line: String) -> String {
    guard let o = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
          let t = o["type"] as? String else { return "unknown" }
    switch t {
    case "system":  return (o["subtype"] as? String) == "init" ? "system" : "unknown"
    case "user", "assistant", "stream_event", "result": return t
    default:        return "unknown"
    }
}

private extension Array where Element == AgentStreamEvent {
    var firstToolUseId: String? { for e in self { if case .toolUse(let id, _, _) = e { return id } }; return nil }
    var firstToolResultId: String? { for e in self { if case .toolResult(let id, _, _) = e { return id } }; return nil }
    var firstToolResult: (content: String, isError: Bool)? {
        for e in self { if case .toolResult(_, let c, let err) = e { return (c, err) } }; return nil
    }
}

/// Scripted AgentTurnBackend: yields each round's steps in order, captures what it was asked.
private final class StubBackend: AgentTurnBackend, @unchecked Sendable {
    let scripted: [[GenStep]]
    var roundIdx = 0
    var seenInstructions: [String?] = []
    var seenResume: [[ResumeMessage]] = []
    var dispatched: [(name: String, argsJSON: String)] = []
    var dispatchResult: (result: String, isError: Bool) = ("RESULT", false)

    init(_ scripted: [[GenStep]]) { self.scripted = scripted }

    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error> {
        seenInstructions.append(instructions)
        seenResume.append(resume)
        let steps = roundIdx < scripted.count ? scripted[roundIdx] : []
        roundIdx += 1
        return AsyncThrowingStream { c in
            for s in steps { c.yield(s) }
            c.finish()
        }
    }
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool) {
        dispatched.append((name, argsJSON))
        return dispatchResult
    }
}

private struct CtxHook: Hook {
    let ctx: String
    func handle(_ event: HookEvent) async -> HookResult {
        if case .userPromptSubmit = event { return HookResult(additionalContext: ctx) }
        return .passthrough
    }
}
private struct DenyToolHook: Hook {
    func handle(_ event: HookEvent) async -> HookResult {
        if case .preToolUse = event { return HookResult(decision: .deny(reason: "read-only mode")) }
        return .passthrough
    }
}
private struct ReplaceHook: Hook {
    let with: String
    func handle(_ event: HookEvent) async -> HookResult {
        if case .postToolUse = event { return HookResult(replacementResult: with) }
        return .passthrough
    }
}
