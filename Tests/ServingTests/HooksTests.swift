import Testing
import Foundation
@testable import Serving

/// thread-safe mutable box so tests can observe @Sendable closure side-effects (single-threaded awaits)
private final class Box<T>: @unchecked Sendable {
    var value: T
    init(_ v: T) { value = v }
}

private struct StubHook: Hook {
    let result: HookResult
    func handle(_ event: HookEvent) async -> HookResult { result }
}

struct HookChainTests {
    @Test func concatenatesAdditionalContext() async {
        let chain = HookChain([
            StubHook(result: HookResult(additionalContext: "A")),
            StubHook(result: .passthrough),
            StubHook(result: HookResult(additionalContext: "B")),
        ])
        let r = await chain.fire(.userPromptSubmit(prompt: "x"))
        #expect(r.additionalContext == "A\n\nB")
    }

    @Test func firstDenyWins() async {
        let chain = HookChain([
            StubHook(result: HookResult(decision: .deny(reason: "first"))),
            StubHook(result: HookResult(decision: .deny(reason: "second"))),
        ])
        let r = await chain.fire(.preToolUse(tool: "Bash", argumentsJSON: "{}"))
        #expect(r.decision == .deny(reason: "first"))
    }

    @Test func lastReplacementWins() async {
        let chain = HookChain([
            StubHook(result: HookResult(replacementResult: "one")),
            StubHook(result: HookResult(replacementResult: "two")),
        ])
        let r = await chain.fire(.postToolUse(tool: "t", result: "orig"))
        #expect(r.replacementResult == "two")
    }

    @Test func emptyChainPassthrough() async {
        let r = await HookChain([]).fire(.stop(answer: "a", toolsCalled: []))
        #expect(r == .passthrough)
    }
}

struct HookResultDecodeTests {
    @Test func decodeAdditionalContext() {
        let r = HookResult.decode(#"{"additionalContext":"ctx"}"#)
        #expect(r.additionalContext == "ctx")
        #expect(r.decision == .allow)
    }
    @Test func decodeDeny() {
        #expect(HookResult.decode(#"{"decision":"deny","reason":"nope"}"#).decision == .deny(reason: "nope"))
    }
    @Test func decodeBlockAlias() {
        #expect(HookResult.decode(#"{"decision":"block"}"#).decision == .deny(reason: "denied by hook"))
    }
    @Test func decodeGarbageIsPassthrough() {
        #expect(HookResult.decode("not json") == .passthrough)
    }
    // The REAL `memory-hooks advise` shape: nested additionalContext + a trailing log line after the JSON.
    // Strict parsing of the whole stdout would drop the grounding; the first-object extractor must survive it.
    @Test func decodeNestedWithTrailingLogLine() {
        let out = "{\"hookSpecificOutput\":{\"additionalContext\":\"## Relevant memories\\n[id:1] {brace in string}\"}}\nLog file path: /Users/x/.claude/memory-logs/y.log\n"
        let r = HookResult.decode(out)
        #expect(r.additionalContext == "## Relevant memories\n[id:1] {brace in string}")
        #expect(r.decision == .allow)
    }
}

struct HookEventPayloadTests {
    @Test func names() {
        #expect(HookEvent.userPromptSubmit(prompt: "p").name == "UserPromptSubmit")
        #expect(HookEvent.preToolUse(tool: "t", argumentsJSON: "{}").name == "PreToolUse")
        #expect(HookEvent.postToolUse(tool: "t", result: "r").name == "PostToolUse")
        #expect(HookEvent.stop(answer: "a", toolsCalled: []).name == "Stop")
    }
    @Test func jsonPayloadCarriesFields() throws {
        let json = HookEvent.userPromptSubmit(prompt: "hello").jsonPayload()
        let obj = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        #expect(obj["hook_event_name"] as? String == "UserPromptSubmit")
        #expect(obj["prompt"] as? String == "hello")
    }
}

struct EngramAdviseHookTests {
    private func grounder(_ items: [RetrievedItem]) -> RetrievalGrounding {
        RetrievalGrounding(config: .init(maxDistance: 1.0)) { _, _ in items }
    }

    @Test func groundsWhenRelevant() async {
        let hook = EngramAdviseHook(grounding: grounder([
            RetrievedItem(id: "m", text: "etrade token expired", distance: 0.1)
        ]))
        let r = await hook.handle(.userPromptSubmit(prompt: "what is the etrade issue?"))
        #expect((r.additionalContext ?? "").contains("etrade token expired"))
        #expect(!(r.additionalContext ?? "").contains("Do NOT fabricate"))
    }

    @Test func hedgesWhenFactQuestionUngrounded() async {
        let hook = EngramAdviseHook(grounding: grounder([]))
        let r = await hook.handle(.userPromptSubmit(prompt: "what is the etrade issue?"))
        #expect((r.additionalContext ?? "").contains("Do NOT fabricate"))
    }

    @Test func passthroughOnNonPromptEvent() async {
        let r = await EngramAdviseHook(grounding: grounder([])).handle(.postToolUse(tool: "t", result: "x"))
        #expect(r == .passthrough)
    }

    @Test func passthroughWhenChitchatUngrounded() async {
        let r = await EngramAdviseHook(grounding: grounder([])).handle(.userPromptSubmit(prompt: "thanks!"))
        #expect(r == .passthrough)
    }
}

struct ExternalCommandHookTests {
    @Test func firesOnlyOnMatchingEvent() async {
        let calls = Box(0)
        let runner: ExternalCommandHook.Runner = { _, _, _, _ in calls.value += 1; return #"{"additionalContext":"x"}"# }
        let hook = ExternalCommandHook(spec: .init(event: "PostToolUse", command: "/bin/echo"), run: runner)
        _ = await hook.handle(.userPromptSubmit(prompt: "p"))   // non-matching → runner not called
        #expect(calls.value == 0)
        let r = await hook.handle(.postToolUse(tool: "t", result: "r"))
        #expect(calls.value == 1)
        #expect(r.additionalContext == "x")
    }

    @Test func receivesEventJSONOnStdin() async {
        let seen = Box("")
        let runner: ExternalCommandHook.Runner = { _, _, stdin, _ in seen.value = stdin; return "{}" }
        let hook = ExternalCommandHook(spec: .init(event: "UserPromptSubmit", command: "c"), run: runner)
        _ = await hook.handle(.userPromptSubmit(prompt: "hi"))
        #expect(seen.value.contains("\"hook_event_name\":\"UserPromptSubmit\""))
        #expect(seen.value.contains("\"prompt\":\"hi\""))
    }
}

struct HookConfigTests {
    @Test func decodeSpecs() {
        let cfg = HookConfig.decode(#"{"hooks":[{"event":"PostToolUse","command":"/x.sh","args":["-v"]}]}"#)
        #expect(cfg.hooks == [.init(event: "PostToolUse", command: "/x.sh", args: ["-v"])])
    }
    @Test func garbageConfigIsEmpty() {
        #expect(HookConfig.decode("nope").hooks.isEmpty)
    }
    @Test func buildsExternalHooksWithRunner() async {
        let cfg = HookConfig.decode(#"{"hooks":[{"event":"Stop","command":"c","args":[]}]}"#)
        let hooks = cfg.externalHooks { _, _, _, _ in #"{"additionalContext":"done"}"# }
        #expect(hooks.count == 1)
        let r = await hooks[0].handle(.stop(answer: "a", toolsCalled: []))
        #expect(r.additionalContext == "done")
    }
}

// ── Claude-Code parity (the de-Engram path): the SAME stdout shapes + stdin payload + matcher/async
// semantics `claude -p` uses, so `memory-hooks advise` and friends fire identically for a local model.

struct ClaudeHookParityTests {
    // THE crux: memory-hooks advise nests additionalContext under hookSpecificOutput — without reading it,
    // Engram grounding is silently dropped.
    @Test func decodeNestedHookSpecificAdditionalContext() {
        let r = HookResult.decode(#"{"hookSpecificOutput":{"additionalContext":"grounded","hookEventName":"UserPromptSubmit"}}"#)
        #expect(r.additionalContext == "grounded")
        #expect(r.decision == .allow)
    }
    @Test func topLevelAdditionalContextStillWins() {
        let r = HookResult.decode(#"{"additionalContext":"top","hookSpecificOutput":{"additionalContext":"nested"}}"#)
        #expect(r.additionalContext == "top")
    }
    @Test func decodePermissionDecisionDeny() {
        let r = HookResult.decode(#"{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"blocked"}}"#)
        #expect(r.decision == .deny(reason: "blocked"))
    }
    @Test func decodeContinueFalseIsDeny() {
        #expect(HookResult.decode(#"{"continue":false,"stopReason":"halt"}"#).decision == .deny(reason: "halt"))
    }
    @Test func decodeContinueTrueIsAllow() {
        #expect(HookResult.decode(#"{"continue":true}"#).decision == .allow)
    }

    @Test func toolInputEmbeddedAsObject() throws {
        let json = HookEvent.preToolUse(tool: "Bash", argumentsJSON: #"{"cmd":"ls"}"#).jsonPayload()
        let obj = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        #expect(obj["tool_name"] as? String == "Bash")
        let input = try #require(obj["tool_input"] as? [String: Any])
        #expect(input["cmd"] as? String == "ls")
    }
    @Test func payloadCarriesSessionContext() throws {
        let json = HookEvent.userPromptSubmit(prompt: "p").jsonPayload(context: .init(sessionID: "S1", cwd: "/tmp"))
        let obj = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        #expect(obj["session_id"] as? String == "S1")
        #expect(obj["cwd"] as? String == "/tmp")
    }

    // matcher: a Pre/PostToolUse hook fires only for matching tool names (Claude's "matcher": "Bash"/"Agent").
    @Test func matcherGatesByToolName() async {
        let calls = Box(0)
        let hook = ExternalCommandHook(spec: .init(event: "PreToolUse", command: "c", matcher: "Bash"),
                                       run: { _, _, _, _ in calls.value += 1; return "{}" })
        _ = await hook.handle(.preToolUse(tool: "Read", argumentsJSON: "{}"))   // no match → not run
        #expect(calls.value == 0)
        _ = await hook.handle(.preToolUse(tool: "Bash", argumentsJSON: "{}"))   // match → run
        #expect(calls.value == 1)
    }
    @Test func emptyMatcherMatchesAllTools() async {
        let calls = Box(0)
        let hook = ExternalCommandHook(spec: .init(event: "PreToolUse", command: "c", matcher: ""),
                                       run: { _, _, _, _ in calls.value += 1; return "{}" })
        _ = await hook.handle(.preToolUse(tool: "AnyTool", argumentsJSON: "{}"))
        #expect(calls.value == 1)
    }

    // async: fire-and-forget (peon-ping notifications) — output never reaches the turn.
    @Test func asyncHookIsFireAndForgetPassthrough() async {
        let hook = ExternalCommandHook(spec: .init(event: "UserPromptSubmit", command: "c", runAsync: true),
                                       run: { _, _, _, _ in #"{"additionalContext":"should-be-ignored"}"# })
        #expect(await hook.handle(.userPromptSubmit(prompt: "p")) == .passthrough)
    }

    // timeout threads through to the injected runner.
    @Test func timeoutPassedToRunner() async {
        let seen = Box<Int?>(nil)
        let hook = ExternalCommandHook(spec: .init(event: "Stop", command: "c", timeoutSeconds: 42),
                                       run: { _, _, _, t in seen.value = t; return "{}" })
        _ = await hook.handle(.stop(answer: "a", toolsCalled: []))
        #expect(seen.value == 42)
    }
}

struct ClaudeSettingsHookConfigTests {
    // mirrors the real ~/.claude/settings.json nested shape (advise + async peon + a matcher'd pre-tool + an
    // unmodeled SessionStart that must be dropped).
    private let settings = #"""
    {
      "hooks": {
        "UserPromptSubmit": [
          { "hooks": [ { "type": "command", "command": "memory-hooks advise", "timeout": 300 } ] },
          { "matcher": "", "hooks": [ { "type": "command", "command": "peon.sh", "timeout": 10, "async": true } ] }
        ],
        "PreToolUse": [
          { "matcher": "Agent", "hooks": [ { "type": "command", "command": "memory-hooks pre-tool", "timeout": 300 } ] }
        ],
        "SessionStart": [
          { "hooks": [ { "type": "command", "command": "ignored", "timeout": 300 } ] }
        ]
      }
    }
    """#

    @Test func parsesNestedClaudeFormat() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "settings-\(UUID().uuidString).json")
        try Data(settings.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let cfg = HookConfig.loadClaudeSettings(path: url.path)

        #expect(cfg.hooks.count == 3)   // 2 UserPromptSubmit + 1 PreToolUse; SessionStart dropped
        let advise = try #require(cfg.hooks.first { $0.command == "memory-hooks advise" })
        #expect(advise.event == "UserPromptSubmit")
        #expect(advise.timeoutSeconds == 300)
        #expect(advise.runAsync == false)
        let peon = try #require(cfg.hooks.first { $0.command == "peon.sh" })
        #expect(peon.runAsync == true)
        #expect(peon.matcher == "")
        let pre = try #require(cfg.hooks.first { $0.event == "PreToolUse" })
        #expect(pre.matcher == "Agent")
        #expect(!cfg.hooks.contains { $0.command == "ignored" })   // unmodeled event never produces a hook
    }

    @Test func missingFileIsEmpty() {
        #expect(HookConfig.loadClaudeSettings(path: "/no/such/settings.json").hooks.isEmpty)
    }

    // end-to-end: the advise hook, built from settings + an injected runner stubbing memory-hooks' nested
    // stdout, surfaces grounding as additionalContext (exactly the Engram serving-path behavior).
    @Test func adviseHookSurfacesGroundingViaRunner() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "settings-\(UUID().uuidString).json")
        try Data(settings.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let hooks = HookConfig.loadClaudeSettings(path: url.path).externalHooks { cmd, _, _, _ in
            cmd == "memory-hooks advise"
                ? #"{"hookSpecificOutput":{"additionalContext":"recalled: etrade token expired","hookEventName":"UserPromptSubmit"}}"#
                : "{}"
        }
        let r = await HookChain(hooks).fire(.userPromptSubmit(prompt: "what is the etrade issue?"))
        #expect((r.additionalContext ?? "").contains("etrade token expired"))
    }
}
