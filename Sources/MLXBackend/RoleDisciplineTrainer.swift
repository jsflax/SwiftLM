import Foundation
import MLXLMCommon
import Serving
import NativeTools
import SelfImprove

// ── TRAIT_TRAIN / TRAIT_TRAIN_SELFTEST: Stage-2 — DPO-train the BUILDER role-discipline trait on the synthesized
// data (`~/Library/Application Support/SwiftLM/trait-synth/role-discipline.jsonl`), rendered SERVE-IDENTICALLY.
//
// The linchpin the prior 0-for-N runs missed: DPOTraining.renderOne trained on a BARE `applyChatTemplate([user])`
// (no system prompt, no tool schema, no reasoning) → train ≠ serve. This builds the structured turns into the
// SAME owned-render the 122B serves with — `renderTurnMessages(promptTurns, tools: specs, enableThinking: true)`
// for the prefix, and the EXACT xmlFunction wire form the model emits for the completion — caches the tokens on
// `TrainPair`, and the (already-modified) renderOne scores them directly. So the preference signal is
// acted-in-role vs narrated UNDER the real heavy context + real tool schema, not a flattened approximation.
//
// Completion wire form is derived from the 122B's own chat_template.jinja: the generation prompt ends with
// `<|im_start|>assistant\n<think>\n` (enable_thinking primes the open tag, add_generation_prompt=true), and a
// completed assistant turn renders `{reasoning|trim}\n</think>\n\n{content}` then, per tool call,
// `[\n\n|]<tool_call>\n<function=NAME>\n<parameter=KEY>\n{value}\n</parameter>\n…</function>\n</tool_call>`,
// closed by `<|im_end|>`. We reproduce exactly that AFTER the primed `<think>\n` (no leading think tag), so
// prefix ++ completion reconstructs the serve token stream and `completionStart = prefix.count` masks the prompt.
extension MLXLanguageModel {

    // ── role-discipline.jsonl line schema (mirrors the synthesized deliverable; `args` is a JSON STRING) ────────
    fileprivate struct SynthToolCall: Codable { let name: String; let args: String }
    fileprivate struct SynthTurn: Codable {
        let role: String; let content: String
        let reasoning: String?; let toolCalls: [SynthToolCall]?
    }
    fileprivate struct SynthCompletion: Codable {
        let reasoning: String?; let content: String; let toolCalls: [SynthToolCall]?
    }
    fileprivate struct SynthExample: Codable {
        let role: String
        let domain: String?; let heaviness: String?; let failureMode: String?
        let systemPrompt: String
        let context: [SynthTurn]
        let won: SynthCompletion
        let lost: SynthCompletion
    }

    fileprivate static func traitEnv(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }

    /// Read the JSONL (one object per line; skip malformed — mirrors `loadRecords`). Filter to the requested
    /// role(s) (default builder; `TRAIT_TRAIN_ROLE` csv to widen). Path override via `TRAIT_TRAIN_DATA`.
    fileprivate static func loadExamples() -> [SynthExample] {
        let path = traitEnv("TRAIT_TRAIN_DATA")
            ?? (NSHomeDirectory() + "/Library/Application Support/SwiftLM/trait-synth/role-discipline.jsonl")
        let roles = Set((traitEnv("TRAIT_TRAIN_ROLE") ?? "builder")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        let dec = JSONDecoder()
        var out: [SynthExample] = []
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            guard let d = line.data(using: .utf8),
                  let e = try? dec.decode(SynthExample.self, from: d) else { continue }
            if roles.contains(e.role) { out.append(e) }
        }
        return out
    }

    // ── completion wire form (template-exact; see header) ───────────────────────────────────────────────────────
    // Stable parameter order so the rendered call is deterministic (the template iterates an UNORDERED arg dict;
    // any consistent order is serve-valid, and a fixed one keeps the trained target reproducible).
    fileprivate static let argOrder = ["path", "content", "old_string", "new_string", "command", "timeout",
                                       "pattern", "target", "note", "summary", "question", "url", "prompt"]
    fileprivate static func orderedArgs(_ argsJSON: String) -> [(String, String)] {
        guard let d = argsJSON.data(using: .utf8),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return [] }
        func render(_ v: Any) -> String {
            switch v {
            case let s as String: return s
            case let b as Bool: return b ? "true" : "false"
            case let n as NSNumber: return n.stringValue
            default:
                if let data = try? JSONSerialization.data(withJSONObject: v),
                   let s = String(data: data, encoding: .utf8) { return s }
                return String(describing: v)
            }
        }
        var out: [(String, String)] = []; var seen = Set<String>()
        for k in argOrder where o[k] != nil { out.append((k, render(o[k]!))); seen.insert(k) }
        for (k, v) in o where !seen.contains(k) { out.append((k, render(v))) }
        return out
    }

    /// The assistant completion the model emits AFTER the primed `<think>\n` prefix — reasoning, `</think>`, the
    /// visible content, then each tool call in the 122B's xmlFunction wire form. NO leading `<think>` (primed) and
    /// NO trailing `<|im_end|>` (appended as a token by the pre-render).
    fileprivate static func wireCompletion(reasoning: String?, content: String,
                                           toolCalls: [SynthToolCall], adapter: ModelProfile) -> String {
        let r = (reasoning ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        // template line 101: '<think>\n' + reasoning + '\n</think>\n\n' + content  (prefix supplied '<think>\n')
        var s = r + "\n" + adapter.reasoningTags.close + "\n\n" + content
        let contentNonEmpty = !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        for (i, tc) in toolCalls.enumerated() {
            if i == 0 {
                s += contentNonEmpty ? "\n\n<tool_call>\n<function=\(tc.name)>\n"
                                     : "<tool_call>\n<function=\(tc.name)>\n"
            } else {
                s += "\n<tool_call>\n<function=\(tc.name)>\n"
            }
            for (k, v) in orderedArgs(tc.args) { s += "<parameter=\(k)>\n\(v)\n</parameter>\n" }
            s += "</function>\n</tool_call>"
        }
        return s
    }

    /// system + continuationMessages(context) — the exact ordering `calibrationRollout` uses, so the prompt is
    /// serve-identical (continuation scan first, then prepend the system turn).
    fileprivate func buildPromptTurns(_ e: SynthExample, adapter: ModelProfile) -> [TurnMessage] {
        let ctxTurns: [TurnMessage] = e.context.map { t in
            TurnMessage(role: TurnMessage.Role(rawValue: t.role) ?? .user, content: t.content,
                        reasoningContent: t.reasoning,
                        toolCalls: t.toolCalls?.map { ToolCall(name: $0.name, argsJSON: $0.args) })
        }
        let body = adapter.continuationMessages(ctxTurns)
        return e.systemPrompt.isEmpty ? body : [TurnMessage(role: .system, content: e.systemPrompt)] + body
    }

    /// Pre-render one example → a DPO pair carrying cached (prefix + wire-completion + im_end) tokens for both
    /// sides, with `completionStart = prefix.count`. renderTurnMessages and the encode each run their own
    /// `container.perform` (sequential, never nested).
    fileprivate func renderTraitPair(_ e: SynthExample, specs: [ToolSpec],
                                     adapter: ModelProfile) async throws -> DPOTraining.Pair? {
        let promptTurns = buildPromptTurns(e, adapter: adapter)
        let prefix = try await renderTurnMessages(promptTurns, tools: specs, enableThinking: true)
        guard prefix.count >= 2 else { return nil }
        let chosenWire = Self.wireCompletion(reasoning: e.won.reasoning, content: e.won.content,
                                             toolCalls: e.won.toolCalls ?? [], adapter: adapter)
        let rejectedWire = Self.wireCompletion(reasoning: e.lost.reasoning, content: e.lost.content,
                                               toolCalls: e.lost.toolCalls ?? [], adapter: adapter)
        let (cToks, rToks) = await container.perform { ctx -> ([Int32], [Int32]) in
            let tok = ctx.tokenizer
            let imEnd = Int32(tok.convertTokenToId("<|im_end|>") ?? tok.eosTokenId ?? 0)
            func enc(_ str: String) -> [Int32] { tok.encode(text: str, addSpecialTokens: false).map(Int32.init) }
            return (prefix + enc(chosenWire) + [imEnd], prefix + enc(rejectedWire) + [imEnd])
        }
        let start = prefix.count
        return DPOTraining.Pair(
            chosen: TrainPair(user: "", assistant: chosenWire, renderedTokens: cToks, completionStart: start),
            rejected: TrainPair(user: "", assistant: rejectedWire, renderedTokens: rToks, completionStart: start))
    }

    // ── TRAIT_TRAIN_SELFTEST: the train==serve eyeball gate (render only; NO training) ───────────────────────────
    public func roleDisciplineSelftest() async throws -> String {
        let adapter = await self.localAdapter
        let host = MCPHost()
        await host.registerNative(NativeToolRegistry.standard(cwd: "/tmp/roletrain"))
        let specs = await host.specs
        let examples = Self.loadExamples()
        guard let e = examples.first else {
            return "=== TRAIT_TRAIN_SELFTEST ABORT ===\nNo examples loaded (TRAIT_TRAIN_DATA / role filter?)."
        }
        let promptTurns = buildPromptTurns(e, adapter: adapter)
        let prefix = try await renderTurnMessages(promptTurns, tools: specs, enableThinking: true)
        let chosenWire = Self.wireCompletion(reasoning: e.won.reasoning, content: e.won.content,
                                             toolCalls: e.won.toolCalls ?? [], adapter: adapter)
        let rejectedWire = Self.wireCompletion(reasoning: e.lost.reasoning, content: e.lost.content,
                                               toolCalls: e.lost.toolCalls ?? [], adapter: adapter)
        let (cToks, rToks, prefixTail) = await container.perform { ctx -> ([Int32], [Int32], String) in
            let tok = ctx.tokenizer
            let imEnd = Int32(tok.convertTokenToId("<|im_end|>") ?? tok.eosTokenId ?? 0)
            func enc(_ str: String) -> [Int32] { tok.encode(text: str, addSpecialTokens: false).map(Int32.init) }
            let tail = tok.decode(tokenIds: prefix.suffix(40).map(Int.init))
            return (prefix + enc(chosenWire) + [imEnd], prefix + enc(rejectedWire) + [imEnd], tail)
        }
        var out: [String] = []
        out.append("=== TRAIT_TRAIN_SELFTEST (model=\(modelId), owned-render=\(adapter.requiresOwnedRender)) ===")
        out.append("examples loaded=\(examples.count); first=\(e.domain ?? "?")/\(e.heaviness ?? "?") fm=\(e.failureMode ?? "?")")
        out.append("promptTurns: " + promptTurns.map {
            "\($0.role.rawValue)(\($0.content.count)c\($0.toolCalls != nil ? "+tc" : "")\($0.reasoningContent != nil ? "+rc" : ""))"
        }.joined(separator: " "))
        out.append("prefix.count=\(prefix.count)  completionStart=\(prefix.count)")
        out.append("prefix TAIL (last ~40 tok, decoded): \(prefixTail.debugDescription)")
        out.append("chosenToks.count=\(cToks.count)  rejectedToks.count=\(rToks.count)")
        out.append("ASSERT prefix-is-prefix(chosen)=\(Array(cToks.prefix(prefix.count)) == prefix)")
        out.append("ASSERT prefix-is-prefix(rejected)=\(Array(rToks.prefix(prefix.count)) == prefix)")
        out.append("ASSERT completionStart<chosen.count=\(prefix.count < cToks.count)  <rejected.count=\(prefix.count < rToks.count)")
        out.append("---- CHOSEN WIRE COMPLETION ----\n" + chosenWire)
        out.append("---- REJECTED WIRE COMPLETION ----\n" + rejectedWire)
        return out.joined(separator: "\n")
    }

    // ── TRAIT_TRAIN: load → pre-render all pairs → DPO-train on a BARE container → write adapter + report gates ───
    public func roleDisciplineTrain() async throws -> String {
        func progress(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        let adapter = await self.localAdapter
        let host = MCPHost()
        await host.registerNative(NativeToolRegistry.standard(cwd: "/tmp/roletrain"))
        let specs = await host.specs
        let examples = Self.loadExamples()
        guard !examples.isEmpty else { return "=== TRAIT_TRAIN ABORT ===\nNo examples loaded." }
        progress("=== TRAIT_TRAIN: \(examples.count) examples; pre-rendering serve-identical pairs ===")

        var pairs: [DPOTraining.Pair] = []
        for (i, e) in examples.enumerated() {
            do {
                if let p = try await renderTraitPair(e, specs: specs, adapter: adapter) {
                    pairs.append(p)
                    if let s = p.chosen.completionStart, let c = p.chosen.renderedTokens?.count {
                        progress("  pair \(i) [\(e.domain ?? "?")/\(e.heaviness ?? "?")]: prefix=\(s) chosen=\(c) rejected=\(p.rejected.renderedTokens?.count ?? 0)")
                    }
                }
            } catch { progress("  pair \(i) render errored: \(error)") }
        }
        guard !pairs.isEmpty else { return "=== TRAIT_TRAIN ABORT ===\n0 pairs rendered." }

        // Coarse loss probe only — the REAL eval is the adapter-OFF/ON before/after demo (the user's "show me"
        // gate), not this held-out chat-loss. Kept minimal per the plan.
        let heldout = Array(examples.prefix(4).map { $0.systemPrompt + "\n\n" + ($0.context.first?.content ?? "") })
        let retention = heldout

        var config = MLXLanguageModel.LoRAConfig()
        config.batchSize = Self.traitEnv("TRAIT_BATCH").flatMap(Int.init) ?? 1
        config.iterations = Self.traitEnv("TRAIT_ITERS").flatMap(Int.init) ?? 200
        config.maxSeqLen = Self.traitEnv("TRAIT_MAXLEN").flatMap(Int.init) ?? 4096
        // Hybrid base: only ~1/4 layers carry self_attn (full_attention_interval=4) and the linear-attn layers are
        // detached for backprop, so span ALL layers (suffix caps at the real count) to maximize full-attention LoRA.
        config.numLayers = Self.traitEnv("TRAIT_NUMLAYERS").flatMap(Int.init) ?? 64
        let adapterDir = URL(fileURLWithPath:
            NSHomeDirectory() + "/Library/Application Support/SwiftLM/trait-adapters/role-discipline-builder")
        progress("=== TRAIT_TRAIN: DPO on \(pairs.count) pairs (iters=\(config.iterations) batch=\(config.batchSize) maxLen=\(config.maxSeqLen)) ===")
        let res = try await trainDPO(pairs: pairs, heldout: heldout, retention: retention,
                                     adapterDir: adapterDir, config: config, beta: 0.1)
        return [
            "=== TRAIT_TRAIN done (model=\(modelId)) ===",
            "pairs=\(pairs.count)  iters=\(config.iterations)  batch=\(config.batchSize)  maxLen=\(config.maxSeqLen)",
            "heldout loss: \(res.beforeHeldout) -> \(res.afterHeldout)  (betPasses=\(res.afterHeldout < res.beforeHeldout))",
            "retention loss: \(res.beforeRetention) -> \(res.afterRetention)",
            "adapter: \(res.adapterURL.path)",
            "NEXT: installTrait into the resident bank + the adapter-OFF/ON before/after demo (the real eval).",
        ].joined(separator: "\n")
    }
}
