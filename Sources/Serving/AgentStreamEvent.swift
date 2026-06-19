import Foundation

// ── The Orbital mount point: a neutral agent-turn event that serializes to the EXACT claude
// `-p --output-format stream-json` NDJSON wire format Orbital already decodes.
//
// SwiftLM's serve loop (MLXBackend.streamAgent) emits these; the Orbital side decodes each `ndjsonLine()`
// with its EXISTING `StreamJsonEvent` decoder and feeds `ClaudeTurnRunner.apply` unchanged. That keeps
// SwiftLM free of any Orbital dependency (the repos meet at the documented claude wire format, not a
// shared Swift type) and makes a local MLX model a drop-in for `claude -p` in an Orbital room.
//
// The wire shapes below are pinned to what Orbital's `StreamJsonEvent.init(from:)` routes on (top-level
// `type`) and what `ClaudeTurnRunner.apply` reads — verified against the real source. AgentStreamEventTests
// is the regression guard that this still "falls neatly into Orbital".
public enum AgentStreamEvent: Sendable, Equatable {
    /// Turn start — model id + session + available tool names (Orbital ignores it but claude emits it).
    case systemInit(model: String, sessionID: String, tools: [String], cwd: String)
    /// One streamed text chunk (live-rendered as it arrives).
    case textDelta(String)
    /// The model invoked a tool (`inputJSON` embedded as a JSON object when it parses, else a string).
    case toolUse(id: String, name: String, inputJSON: String)
    /// A dispatched tool's result, keyed back to its `toolUse` by `id`.
    case toolResult(id: String, content: String, isError: Bool)
    /// Turn end — final text (used by Orbital only if no deltas streamed) + token usage.
    case result(finalText: String, inputTokens: Int, outputTokens: Int, isError: Bool)

    /// The single claude-stream-json NDJSON line for this event (no trailing newline; the caller adds it).
    public func ndjsonLine() -> String {
        let obj: [String: Any]
        switch self {
        case .systemInit(let model, let sessionID, let tools, let cwd):
            obj = ["type": "system", "subtype": "init",
                   "model": model, "session_id": sessionID, "tools": tools, "cwd": cwd]
        case .textDelta(let text):
            obj = ["type": "stream_event",
                   "event": ["type": "content_block_delta", "index": 0,
                             "delta": ["type": "text_delta", "text": text]]]
        case .toolUse(let id, let name, let inputJSON):
            obj = ["type": "assistant",
                   "message": ["role": "assistant",
                               "content": [["type": "tool_use", "id": id, "name": name,
                                            "input": Self.jsonObjectOrString(inputJSON)]]]]
        case .toolResult(let id, let content, let isError):
            obj = ["type": "user",
                   "message": ["role": "user",
                               "content": [["type": "tool_result", "tool_use_id": id,
                                            "content": content, "is_error": isError]]]]
        case .result(let finalText, let inputTokens, let outputTokens, let isError):
            obj = ["type": "result", "subtype": isError ? "error" : "success",
                   "is_error": isError, "result": finalText,
                   "usage": ["input_tokens": inputTokens, "output_tokens": outputTokens]]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else { return "{}" }
        return s
    }

    /// Parse a JSON-string argument into an object/array for embedding; fall back to the raw string.
    private static func jsonObjectOrString(_ s: String) -> Any {
        guard let data = s.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: data) else { return s }
        return any
    }
}
