import Foundation
import MLXLMCommon
import Serving
import NativeTools

// ── WRITE_DIAG: diagnose the chess-stress-test builder failure (the 122B "Forge" narrated "I'll create engine.py"
// but emitted NO usable Write tool call → working dir stayed empty). This reproduces a SINGLE builder turn in the
// REAL owned-render path with the REAL native tool schema (Read/Write/Edit/Bash/…), then DUMPS the raw decoded
// output so we can see what the model actually emitted: a well-formed write_file call, a TRUNCATED/MALFORMED one
// (the large-body tool-reliability failure), or pure narration with no call at all. A clean single-turn ask that
// SUCCEEDS would mean the chess failure is context-dependent (heavy multi-agent context), not fundamental.

extension MLXLanguageModel {
    public func writeReliabilityDiag() async throws -> String {
        let adapter = await self.localAdapter
        let host = MCPHost()
        await host.registerNative(NativeToolRegistry.standard(cwd: "/tmp/writediag"))
        let specs = await host.specs                       // the REAL native tool schema the builder sees
        var params = GenerateParameters(maxTokens: 1200, temperature: 0)   // greedy; generous cap for reasoning + a file body
        params.repetitionPenalty = adapter.sampling.repetitionPenalty
        params.repetitionContextSize = adapter.sampling.repetitionContextSize

        let sys = "You are a software builder agent with file tools (write_file, edit_file, bash, read_file, etc.). "
            + "When asked to create a file you MUST call the write_file tool with the full file content as the `content` "
            + "argument. Do NOT merely describe the file — emit the actual tool call."
        let user = "Create a file named board.py containing a minimal Python chess Board class: an __init__(self) that "
            + "sets up the standard starting position as an 8x8 list of piece characters, and a fen(self) method that "
            + "returns the starting FEN 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'. "
            + "Write the complete file now using the write_file tool."
        let turns = [TurnMessage(role: .system, content: sys), TurnMessage(role: .user, content: user)]
        let tokens = try await renderTurnMessages(turns, tools: specs, enableThinking: adapter.emitsReasoning)

        var text = ""
        var nChunks = 0
        for try await g in streamFromTokens(tokens, maxTokens: 1200, adapter: adapter, params: params) {
            if case .chunk(let c) = g { text += c; nChunks += 1 }
        }

        // RAW string-level analysis (independent of any parser — the parser would HIDE a malformed call).
        func count(_ needle: String) -> Int { text.components(separatedBy: needle).count - 1 }
        let tcOpen = count("<tool_call>"), tcClose = count("</tool_call>")
        let thinkOpen = text.contains("<think>"), thinkClose = text.contains("</think>")
        let mentionsWrite = text.contains("write_file")
        let funcStyle = text.contains("<function=")   // XML/Hermes style vs JSON {"name":...}

        // DECISIVE: does the REAL owned-render parser actually PARSE what the model emitted? CompactingSession.
        // ownedRound uses exactly `toolCallParser?.parse(stripThink(text), tools: specs)` (+ recoverMissedToolCall).
        // If the model emits a well-formed call but the parser returns nil, THAT is the bug (the call is dropped →
        // the agent looks like it "narrated without acting").
        let parser = await makeToolCallParser(adapter)
        let stripped = CompactingSession.stripThinkSpans(text)
        let parsed = parser?.parse(content: stripped, tools: specs)
        let parsedArgs = parsed.map { MLXLanguageModel.argsJSON($0.function.arguments) } ?? ""
        let recovered = adapter.recoverMissedToolCall(stripped)

        let parsedWrite = (parsed?.function.name == "write_file") || (recovered?.name == "write_file")
        let verdict: String
        if parsedWrite {
            verdict = "PARSER OK — the emitted call PARSES to write_file. The model emits AND the parser accepts it in a clean single turn ⇒ the chess builder failure is CONTEXT-DEPENDENT (heavy multi-agent context made Forge narrate without acting), NOT a format/parse bug."
        } else if (tcOpen >= 1 || funcStyle), parsed == nil, recovered == nil {
            verdict = "🐞 PARSER BUG — the model emitted a well-formed call (<function=write_file>/<tool_call>) but the owned-render parser + recover BOTH returned NIL ⇒ the call is silently DROPPED. This alone would make EVERY 122B file-write 'vanish' (the chess builder symptom). Likely a FORMAT MISMATCH (parser expects a different style than the emitted <function=...> XML)."
        } else if tcOpen >= 1, tcClose == 0 {
            verdict = "TRUNCATED — opened <tool_call> but never closed it within the budget. LARGE-BODY tool-reliability failure."
        } else if !thinkClose, thinkOpen {
            verdict = "STUCK IN <think> — never closed the reasoning block (spiral / over-reasoning); no call emitted."
        } else {
            verdict = "NO call emitted — narrated intent without acting."
        }

        var out = ["=== WRITE_DIAG (model: \(modelId), owned-render=\(adapter.requiresOwnedRender), promptTok=\(tokens.count), genCap=1200) ==="]
        out.append("decoded \(text.count) chars | <tool_call> open=\(tcOpen) close=\(tcClose) | <function=>=\(funcStyle) | <think> open=\(thinkOpen) close=\(thinkClose) | mentions write_file=\(mentionsWrite)")
        out.append("adapter.toolCallFormat=\(adapter.toolCallFormat) | parser=\(parser == nil ? "nil" : "present")")
        out.append("PARSER RESULT: \(parsed != nil ? "name=\(parsed!.function.name) argsLen=\(parsedArgs.count) hasContentArg=\(parsedArgs.contains("content"))" : "NIL (did not parse)")  |  recoverMissedToolCall=\(recovered?.name ?? "nil")")
        out.append("VERDICT: \(verdict)")
        out.append("---- RAW DECODE (first 1600 chars) ----")
        out.append(String(text.prefix(1600)))
        out.append("---- RAW DECODE (last 800 chars) ----")
        out.append(String(text.suffix(800)))
        return out.joined(separator: "\n")
    }
}
