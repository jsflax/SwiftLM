import Testing
import Foundation
@testable import Serving

/// Proves AgentStreamEvent.ndjsonLine() emits the EXACT claude-stream-json shapes Orbital decodes —
/// the "falls neatly into Orbital" contract. Each assertion mirrors a field that Orbital's
/// `StreamJsonEvent.init(from:)` (routes on top-level `type`/`subtype`) or `ClaudeTurnRunner.apply`
/// actually reads. If SwiftLM ever drifts off the wire format, these fail instead of a local model
/// silently degrading to `.unknown` events in a room.
struct AgentStreamEventTests {
    private func decode(_ line: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
    }

    @Test func systemInitRoutesToSystemInit() throws {
        let o = try decode(AgentStreamEvent.systemInit(model: "qwen", sessionID: "S1",
                                                       tools: ["handoff"], cwd: "/tmp").ndjsonLine())
        #expect(o["type"] as? String == "system")        // → .systemInit
        #expect(o["subtype"] as? String == "init")
        #expect(o["model"] as? String == "qwen")
        #expect(o["session_id"] as? String == "S1")
        #expect(o["tools"] as? [String] == ["handoff"])
    }

    @Test func textDeltaRoutesToStreamEventTextDelta() throws {
        let o = try decode(AgentStreamEvent.textDelta("hello").ndjsonLine())
        #expect(o["type"] as? String == "stream_event")  // → .streamEvent
        let ev = try #require(o["event"] as? [String: Any])
        #expect(ev["type"] as? String == "content_block_delta")  // apply(): live-render gate
        let delta = try #require(ev["delta"] as? [String: Any])
        #expect(delta["type"] as? String == "text_delta")
        #expect(delta["text"] as? String == "hello")     // apply(): the appended chunk
    }

    @Test func toolUseRoutesToAssistantToolUse() throws {
        let o = try decode(AgentStreamEvent.toolUse(id: "t1", name: "handoff",
                                                    inputJSON: #"{"note":"go"}"#).ndjsonLine())
        #expect(o["type"] as? String == "assistant")     // → .assistant
        let msg = try #require(o["message"] as? [String: Any])
        let content = try #require(msg["content"] as? [[String: Any]])
        let block = try #require(content.first)
        #expect(block["type"] as? String == "tool_use")  // apply(): opens tool card
        #expect(block["id"] as? String == "t1")
        #expect(block["name"] as? String == "handoff")
        let input = try #require(block["input"] as? [String: Any])  // embedded as object, not string
        #expect(input["note"] as? String == "go")
    }

    @Test func toolResultRoutesToUserToolResult() throws {
        let o = try decode(AgentStreamEvent.toolResult(id: "t1", content: "ok", isError: false).ndjsonLine())
        #expect(o["type"] as? String == "user")          // → .user
        let msg = try #require(o["message"] as? [String: Any])
        let content = try #require(msg["content"] as? [[String: Any]])  // applyToolResults: must be an array
        let block = try #require(content.first)
        #expect(block["type"] as? String == "tool_result")            // applyToolResults: type gate
        #expect(block["tool_use_id"] as? String == "t1")              // applyToolResults: keys back to the card
        #expect(block["content"] as? String == "ok")
        #expect(block["is_error"] as? Bool == false)
    }

    @Test func resultRoutesToResultWithUsage() throws {
        let o = try decode(AgentStreamEvent.result(finalText: "done", inputTokens: 12,
                                                   outputTokens: 7, isError: false).ndjsonLine())
        #expect(o["type"] as? String == "result")        // → .result
        #expect(o["result"] as? String == "done")        // apply(): finalText when no deltas streamed
        #expect(o["is_error"] as? Bool == false)
        let usage = try #require(o["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 12)     // apply(): agent.contextTokens / cost
        #expect(usage["output_tokens"] as? Int == 7)
    }

    @Test func toolUseToleratesNonJSONInput() throws {
        // a model that emits non-JSON args still produces a valid line (input falls back to a string).
        let o = try decode(AgentStreamEvent.toolUse(id: "t", name: "n", inputJSON: "not json").ndjsonLine())
        let block = try #require((o["message"] as? [String: Any])?["content"] as? [[String: Any]])
        #expect(block.first?["input"] as? String == "not json")
    }
}
