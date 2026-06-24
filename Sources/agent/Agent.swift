import Foundation
import MLXBackend
import SelfImprove
import Serving
import NativeTools
import MiniBPE
import JSONSchema
import Orchestration

// Schema for the grammar-constrained-decoding smoke test (CONSTRAIN_DEMO=1). `name` is a plain String, but
// the demo constrains it at RUNTIME to a fixed action set (via `runtimeEnums`) — no compile-time enum /
// @Generable needed. This is the exact mechanism for "tool name ∈ the connected tools".
@JSONSchema struct ToolCall { let name: String; let target: String }
@JSONSchema struct ToolName { let name: String }   // phase-1 of the constrained tool call (name only)

// The runnable SwiftLM agent: loads an MLX model on the FROZEN BASE, connects MCP servers,
// and runs the round-capped tool loop in a REPL. Per the "useful agent first" pivot (Jun 17
// 2026), the frozen base is the foundation — value comes from retrieval grounding + serve-time
// verifier-gated selection, NOT from fine-tuning (0 verifier-gated FT promotions to date). The
// champion registry / LoRA hot-swap stays as OFFLINE R&D and is OPT-IN here, not the default.
//   MODEL=...        override the model id
//   MCP_SERVER=...   path to an MCP server binary to connect (default: claude-utils)
//   CHAMPION=1       opt in to hot-swap the registry champion adapter (offline-R&D experiments)

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

@main
struct Agent {
    static func main() async throws {
        // ROOM_HARVEST=1: Track 2 — mine Orbital room .lattice files for role-discipline DPO data (WON strong-model
        // role turns / LOST local failures). Model-FREE, so run it BEFORE loading any model.
        if ProcessInfo.processInfo.environment["ROOM_HARVEST"] != nil {
            print(RoomTranscriptHarvester.runReport())
            return
        }

        // The default model is a CONFIG choice only — the family (reasoning?, tool-call style, generation
        // budget) is AUTO-DETECTED from the loaded model via ModelProfile, so any MODEL=… works unchanged.
        let modelId = ProcessInfo.processInfo.environment["MODEL"]
            ?? "mlx-community/GLM-4.5-Air-4bit"
        log("loading \(modelId) ...")
        let model = try await MLXLanguageModel.load(modelId: modelId)
        log("loaded.")

        // SPIRAL_BATTLETEST=1: A0 end-to-end proof — drive the real owned-render decode into a forced spiral
        // and confirm the DegenerateRunDetector tripwire bounds it (guard-on vs guard-off vs healthy control).
        if ProcessInfo.processInfo.environment["SPIRAL_BATTLETEST"] != nil {
            print(try await model.spiralBattleTest())
            return
        }

        // OWNED_RENDER_BENCH=1: Part B investigation — cache trimmability + the re-prefill cost curve + the
        // incremental-advance potential, on THIS model. Decides whether incremental KV is feasible/worth it.
        if ProcessInfo.processInfo.environment["OWNED_RENDER_BENCH"] != nil {
            print(await model.ownedRenderBench())
            return
        }

        // INCR_KV_CHECK=1: B2 correctness gate — incremental KV (reuse carried cache + prefill the tail) must
        // produce token-identical output to a full re-prefill of the same row.
        if ProcessInfo.processInfo.environment["INCR_KV_CHECK"] != nil {
            print(try await model.incrementalKVEquivalenceCheck())
            return
        }

        // TRAITBANK_CHECK=1: C1 correctness gate — the Resident Trait-Bank must be byte-identical to base when no
        // trait is active (empty/inactive/B=0), apply a measurable delta when a live trait IS active, and keep two
        // CONCURRENT agents' trait-sets isolated (the @TaskLocal interleave proof). Do NOT also set SWIFTLM_TRAIT_BANK
        // (the check installs the bank itself so it can measure the pre-install base first).
        if ProcessInfo.processInfo.environment["TRAITBANK_CHECK"] != nil {
            print(try await model.residentTraitBankCheck())
            return
        }

        // SYSPROMPT_CHECK=1: BUG-2 gate + before/after demo — the owned-render decode must now honor the agent's
        // system prompt (it used to drop it). Decodes the SAME question with vs without a directive system turn.
        if ProcessInfo.processInfo.environment["SYSPROMPT_CHECK"] != nil {
            print(try await model.systemPromptCheck())
            return
        }

        // WRITE_DIAG=1: diagnose the chess-stress-test builder failure — a single builder turn (real native tool
        // schema, owned-render) asked to write a file, dumping the RAW decode so we can see whether it emits a
        // well-formed write_file call, a truncated/malformed one (large-body failure), or pure narration.
        if ProcessInfo.processInfo.environment["WRITE_DIAG"] != nil {
            print(try await model.writeReliabilityDiag())
            return
        }

        // TRAIT_CALIB=1: Part D Stage 1 — measure the ROLE-DISCIPLINE in-band rate. Reproduces a HEAVY
        // multi-agent builder turn (owned-render, real native tool schema) across task domains × a context
        // heaviness ladder, scores each best-of-N rollout via the REAL tool-call parser (acted-in-role vs
        // narrated), and reports the per-cell/overall in-band rate + trainability verdict. TRAIT_DATAGEN=1
        // additionally persists the won×lost rollouts as DPO pairs for the later DPO-train stage.
        if ProcessInfo.processInfo.environment["TRAIT_CALIB"] != nil
            || ProcessInfo.processInfo.environment["TRAIT_DATAGEN"] != nil
            || ProcessInfo.processInfo.environment["TRAIT_REAL_CONTEXT"] != nil {
            print(try await model.traitCalibration())
            return
        }

        // TRAIT_TRAIN_SELFTEST=1: Stage-2 train==serve eyeball gate — render the FIRST builder pair via the
        // owned-render path (system + tool schema + xmlFunction completion) and dump prefix tail + wire
        // completions + prefix-is-prefix/boundary asserts. NO training, NO decode.
        if ProcessInfo.processInfo.environment["TRAIT_TRAIN_SELFTEST"] != nil {
            print(try await model.roleDisciplineSelftest())
            return
        }

        // TRAIT_TRAIN=1: Stage-2 — DPO-train the BUILDER role-discipline trait on the synthesized data, rendered
        // SERVE-IDENTICALLY (renderTurnMessages prefix + the 122B xmlFunction completion, cached on TrainPair so
        // renderOne scores them directly). Trains on a BARE container, writes the adapter + reports the gates.
        if ProcessInfo.processInfo.environment["TRAIT_TRAIN"] != nil {
            print(try await model.roleDisciplineTrain())
            return
        }

        // TRAIT_DEMO=1: the "show me it working" gate — decode a heavy builder context adapter-OFF then adapter-ON
        // (Hotswap.loadAdapter), reporting the acted-in-role RATE delta + greedy completions side by side.
        if ProcessInfo.processInfo.environment["TRAIT_DEMO"] != nil {
            print(try await model.roleDisciplineDemo())
            return
        }

        // Default: run the FROZEN BASE (per the "useful agent first" pivot). The champion adapter is
        // OPT-IN (CHAMPION=1) — it belongs to the offline-R&D loop, not the interactive default path.
        if ProcessInfo.processInfo.environment["CHAMPION"] != nil {
            if let registry = try? Registry(), let champ = registry.currentChampion() {
                do {
                    try await model.loadAdapter(directory: URL(fileURLWithPath: champ.adapterPath))
                    log("hot-swapped champion adapter \(champ.cycleId) (held-out loss \(champ.heldoutLoss)).")
                } catch {
                    log("CHAMPION=1 set but adapter load failed (\(error)) — running frozen base.")
                }
            } else {
                log("CHAMPION=1 set but no registry champion found — running frozen base.")
            }
        }

        // CONSTRAIN_DEMO=1: grammar-constrained structured-generation smoke test — the output is
        // valid-by-construction JSON matching DemoToolCall, masked token-by-token by the schema grammar.
        if ProcessInfo.processInfo.environment["CONSTRAIN_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            log("grammar tokenizer: \(tok.tokensToIds.count) tokens")
            let actions = ["read_file", "run_bash", "search"]
            // Tempt the model toward a non-listed action ("delete"); runtimeEnums forces `name` ∈ actions —
            // proving constrained tool/action SELECTION against a RUNTIME set, by construction.
            let json = try await model.generateConstrained(
                "The user said: 'delete the file at /tmp/secret'. Choose an action and target.",
                schema: ToolCall.self, grammarTokenizer: tok,
                instructions: "Respond with ONLY a JSON object with fields name and target.",
                runtimeEnums: ["name": actions], maxTokens: 128)
            print("CONSTRAINED OUTPUT: \(json)")
            if let data = json.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let name = obj["name"] as? String {
                print(actions.contains(name)
                      ? "VALID SELECTION ✓  name=\(name) (forced from \(actions), not 'delete')"
                      : "INVALID SELECTION ✗  name=\(name)")
            } else {
                print("INVALID JSON ✗")
            }
            return
        }

        // Register the built-in native tools (Read/Write/Edit/Glob/Grep/Bash) — the agent's hands,
        // available with no external MCP server — then connect MCP servers on top.
        let host = MCPHost()
        await host.registerNative(.standard())
        log("registered native tools: \(NativeToolRegistry.standard().names.joined(separator: ", "))")
        // Connect the user's configured MCP servers (~/.claude.json `mcpServers`) — the same set `claude -p`
        // sees, namespaced `mcp__<server>__<tool>`. So `memory`, `claude-utils`, etc. are all real tools
        // (no hardcoded path, no single-server special case). SWIFTLM_MCP_ALLOWLIST=a,b scopes which to load.
        let allow = ProcessInfo.processInfo.environment["SWIFTLM_MCP_ALLOWLIST"]
            .map { Set($0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }) }
        let mcpResults = await host.connectClaudeServers(allowlist: allow)
        for (name, outcome) in mcpResults {
            switch outcome {
            case .success(let tools): log("connected MCP '\(name)' — \(tools.count) tools")
            case .failure(let e):     log("MCP '\(name)' skipped: \(e)")
            }
        }
        if mcpResults.isEmpty { log("no ~/.claude.json mcpServers found — native tools only") }

        // BATCH_BENCH=1: the crossover measurement that GATES the continuous-batching build. Times a batched
        // forward (B streams in ONE pass via the existing batchGenerate) vs B serialized B=1 passes. If B=4
        // isn't ~2x+ faster than 4 serial passes on THIS model/GPU, token-level batching isn't worth the
        // weeks (serial concurrency wins for the small-B + grammar agent workload).
        if ProcessInfo.processInfo.environment["BATCH_BENCH"] != nil {
            let p = "Write a detailed technical explanation of how a B-tree index works in a database."
            _ = await model.batchGenerate(p, n: 1, maxTokens: 8, temperature: 0)   // warm the GPU
            var t: [Int: Double] = [:]
            for b in [1, 2, 4, 8] {
                let t0 = Date()
                _ = await model.batchGenerate(p, n: b, maxTokens: 128, temperature: 0, stopOnEOS: false)
                t[b] = Date().timeIntervalSince(t0)
            }
            let t1 = t[1]!
            print("=== BATCH CROSSOVER (128 tok/stream, greedy) ===")
            for b in [1, 2, 4, 8] {
                let tb = t[b]!, serial = Double(b) * t1
                print(String(format: "B=%d: batched %.2fs | %d×serial %.2fs | speedup %.2fx | per-stream %.3fs",
                             b, tb, b, serial, serial / tb, tb / Double(b)))
            }
            print("verdict: batching worth building iff B=4 speedup >= ~2x")
            await host.shutdown(); return
        }

        // BATCH_DISTINCT_DEMO=1: SLICE 1b proof — N DIFFERENT, DIFFERENT-LENGTH prompts decoded in lockstep
        // via SEPARATE-PREFILL + MERGE-CACHES (no truncation, no pad-prefill). Asserts each row's merged-batch
        // output ≈ that row run ALONE, and that divergent rows stay ON-TOPIC (independence, not contamination).
        if ProcessInfo.processInfo.environment["BATCH_DISTINCT_DEMO"] != nil {
            // Deliberately different lengths: p2 is much longer than p1 (forces real left-padding on row 0).
            let p1 = "List exactly three primary colors, one per line, nothing else."
            let p2 = "You are a careful astronomy tutor. Considering only the eight classical planets that orbit "
                + "the Sun, name exactly three of them, one per line, with no extra commentary whatsoever."
            let rows = await model.batchEquivalenceCheck([p1, p2], maxTokens: 48)
            print("=== BATCH DISTINCT 1b (B=2 different-length prompts, separate-prefill + merged decode) ===")
            for (i, r) in rows.enumerated() {
                let eq = r.firstDiff == -1
                print("--- row \(i): batched==solo \(eq) (firstDiff token idx \(r.firstDiff), "
                    + "len batched/solo \(r.batchedIds.count)/\(r.soloIds.count)) ---")
                print("  batched: \(r.batchedText.replacingOccurrences(of: "\n", with: " ⏎ ").prefix(160))")
                print("  solo   : \(r.soloText.replacingOccurrences(of: "\n", with: " ⏎ ").prefix(160))")
            }
            print("DIAGNOSIS: rows have DIFFERENT prompt lengths, so a correct merge requires per-sequence RoPE "
                + "offsets + a left-pad mask. If each row's BATCHED text matches its SOLO text (or stays on its "
                + "OWN topic when float non-determinism flips a near-tie argMax), the merge is correct. "
                + "Contamination = a row's batched text drifts to the OTHER prompt's topic.")
            await host.shutdown(); return
        }

        // GRAMMAR_BATCH_BENCH=1: SLICE 4 measurement — is PER-ROW grammar in a batched pass worth it? Times
        // batched-unconstrained vs batched-constrained vs solo-constrained on B distinct prompts + a fixed
        // JSON schema, so the verdict (does batched grammar keep the batching win, or does the per-step mask
        // build erode it?) is a number, not an assertion.
        if ProcessInfo.processInfo.environment["GRAMMAR_BATCH_BENCH"] != nil {
            let gtok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let fields = [SchemaProperty("name", String.self), SchemaProperty("city", String.self)]
            let prompts = [
                "Give me a person record for a teacher in Paris.",
                "Give me a person record for a doctor in Tokyo.",
                "Give me a person record for an artist in Cairo.",
                "Give me a person record for a pilot in Lima.",
            ]
            let maxTok = Int(ProcessInfo.processInfo.environment["BENCH_TOK"] ?? "") ?? 64
            let r = await model.grammarBatchBench(prompts: prompts, fields: fields,
                                                  tokenizer: gtok, maxTokens: maxTok)
            print("=== GRAMMAR BATCH BENCH (B=\(r.batch), \(maxTok) tok/row, greedy, fixed-length) ===")
            print(String(format: "(a) batched UNconstrained : %.2fs", r.batchedUnconstrained))
            print(String(format: "(b) batched   constrained : %.2fs", r.batchedConstrained))
            print(String(format: "(c) solo      constrained : %.2fs  (B sequential 1-row runs)", r.soloConstrained))
            print(String(format: "mask overhead (b/a)       : %.2fx", r.batchedConstrained / r.batchedUnconstrained))
            print(String(format: "batching speedup (c/b)    : %.2fx", r.soloConstrained / r.batchedConstrained))
            print("verdict: per-row batched grammar is worth it iff (c/b) ≫ 1 while (b/a) stays modest.")
            // Correctness: run to NATURAL completion (stopOnComplete, generous cap) — NOT the bench's fixed 64
            // (that cap is only for the fair fixed-length timing above; capping correctness at 64 would
            // mislabel a row that simply hasn't closed its JSON yet as "invalid"). The BATCHED constrained
            // decode must match the SOLO one per row (proves the per-row masking is faithful, not a batched
            // bug); any remaining length is shared model verbosity within a VALID string, not a grammar fault.
            let cap = 256
            let batched = await model.batchGenerateConstrained(prompts, fields: fields, tokenizer: gtok, maxTokens: cap)
            var solo: [String] = []
            for p in prompts { solo += await model.batchGenerateConstrained([p], fields: fields, tokenizer: gtok, maxTokens: cap) }
            print("--- batched-vs-solo constrained, per row (faithful iff they match) ---")
            for i in 0..<prompts.count {
                let ok = (batched[i].data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) }) != nil
                print("  row \(i) match=\(batched[i] == solo[i]) validJSON=\(ok)")
                print("    batched: \(batched[i].replacingOccurrences(of: "\n", with: "⏎").prefix(90))")
                print("    solo   : \(solo[i].replacingOccurrences(of: "\n", with: "⏎").prefix(90))")
            }
            await host.shutdown(); return
        }

        // BATCH_SUBAGENT_DEMO=1: SLICE 3 proof — two CONCURRENT agent turns route their generation through the
        // shared coalescing pool and FUSE into one batched forward pass (set SWIFTLM_BATCH_DEBUG=1 to see the
        // "coalesced 2 rows" line), each still producing its OWN correct answer. This is the sub-agent fan-out.
        if ProcessInfo.processInfo.environment["BATCH_SUBAGENT_DEMO"] != nil {
            let pool = model.makeBatchPool()
            let gen: BatchGenerator = { (tokens: [Int32], maxTok: Int) in
                await pool.complete(InferenceRequest(model: ModelID(modelId), prompt: "",
                                                     maxTokens: maxTok, inputTokens: tokens))
            }
            let pA = "What is the capital of France? Reply with just the city name."
            let pB = "What is the chemical symbol for gold? Reply with just the symbol."
            let t0 = Date()
            async let a = model.runWithTools(pA, host: host, maxRounds: 1, toolAllowlist: [], batchGenerator: gen)
            async let b = model.runWithTools(pB, host: host, maxRounds: 1, toolAllowlist: [], batchGenerator: gen)
            let ra = try await a, rb = try await b
            print("=== BATCH SUBAGENT (2 concurrent turns fused via the coalescing pool) ===")
            print(String(format: "elapsed %.2fs", Date().timeIntervalSince(t0)))
            print("A (capital of France): \(ra.replacingOccurrences(of: "\n", with: " ").prefix(100))")
            print("B (symbol for gold)  : \(rb.replacingOccurrences(of: "\n", with: " ").prefix(100))")
            await host.shutdown(); return
        }

        // BATCH_SUBAGENT_TOOL_DEMO=1: SLICE 3 tool-calling proof — two CONCURRENT turns, each requiring a
        // TOOL call, route generation through the coalescing pool. Exercises the batched path's tool-call
        // parsing (mlx ToolCallFormat parser on the coalesced text) → dispatch → result → final answer.
        if ProcessInfo.processInfo.environment["BATCH_SUBAGENT_TOOL_DEMO"] != nil {
            let pool = model.makeBatchPool()
            let gen: BatchGenerator = { (tokens: [Int32], maxTok: Int) in
                await pool.complete(InferenceRequest(model: ModelID(modelId), prompt: "",
                                                     maxTokens: maxTok, inputTokens: tokens))
            }
            let names = await host.toolNames
            let timeTool = names.first { $0.hasSuffix("current_time") }
            let uuidTool = names.first { $0.hasSuffix("uuid") }
            let allow = Set([timeTool, uuidTool].compactMap { $0 })
            log("tool-fanout allowlist: \(allow.sorted())")
            let prof = await model.profile   // the DETECTED profile (GLM → taggedReasoning), NOT .generic —
            // a no-arg call's recovery (profile.recoverMissedToolCall) only fires for a taggedReasoning profile.
            let gtok = try? MiniBPE.grammarTokenizer(forModelId: modelId)
            let pA = "What is the current time? Call a tool to find out, then tell me."
            let pB = "Generate a random UUID. Call a tool to do it, then tell me."
            let t0 = Date()
            async let a = model.runWithToolsTracked(pA, host: host, maxRounds: 4, grammarTokenizer: gtok,
                                                    profile: prof, toolAllowlist: allow, batchGenerator: gen)
            async let b = model.runWithToolsTracked(pB, host: host, maxRounds: 4, grammarTokenizer: gtok,
                                                    profile: prof, toolAllowlist: allow, batchGenerator: gen)
            let ra = try await a, rb = try await b
            print("=== BATCH SUBAGENT TOOL (2 concurrent tool-calling turns via the coalescing pool) ===")
            print(String(format: "elapsed %.2fs", Date().timeIntervalSince(t0)))
            print("A toolsCalled=\(ra.toolsCalled): \(ra.answer.replacingOccurrences(of: "\n", with: " ").prefix(110))")
            print("B toolsCalled=\(rb.toolsCalled): \(rb.answer.replacingOccurrences(of: "\n", with: " ").prefix(110))")
            await host.shutdown(); return
        }

        // TOOLCALL_DEMO=1: interleaved constrained tool-call emission — the model free-texts then emits a
        // tool call whose `name` is FORCED ∈ the connected tools, by construction (the end-to-end of #31).
        if ProcessInfo.processInfo.environment["TOOLCALL_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let names = await host.toolNames
            let specs = await host.specs
            log("tool surface (\(names.count)): \(names.joined(separator: ", "))")
            let text = try await model.generateToolCallConstrained(
                "Read the file at /etc/hostname for me.",
                tools: specs, toolNames: names,
                schema: ToolCall.self, grammarTokenizer: tok,
                instructions: "Call a tool when the user needs file or system data.", maxTokens: 200)
            print("=== RAW MODEL OUTPUT ===\n\(text)\n========================")
            if let r = text.range(of: #""name"\s*:\s*""#, options: .regularExpression) {
                let name = String(text[r.upperBound...].prefix(while: { $0 != "\"" }))
                print(names.contains(name)
                      ? "VALID TOOL ✓  name=\(name) (∈ the connected tools, by construction)"
                      : "INVALID  name=\(name) (NOT a connected tool)")
            } else {
                print("(no <tool_call> emitted — model answered in prose; interleaving stayed passthrough)")
            }
            await host.shutdown()
            return
        }

        // ARGS_DEMO=1: constrain a tool call's ARGS to the chosen tool's RUNTIME schema — fixing the wrong-
        // args failure (the model emitted {"action":"read"} instead of {"path":...}).
        if ProcessInfo.processInfo.environment["ARGS_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let specs = await host.specs
            guard let rf = specs.first(where: {
                      (($0["function"] as? [String: any Sendable])?["name"] as? String) == "read_file" }),
                  let fn = rf["function"] as? [String: any Sendable],
                  let params = fn["parameters"] as? [String: any Sendable] else {
                print("read_file spec not found"); await host.shutdown(); return
            }
            let fields = requiredToolArgFields(parameters: params)
            log("read_file required args (constrained): \(fields.map(\.name))")
            let json = try await model.generateConstrained(
                "The user wants to read the file /etc/hostname. Produce the arguments for the read_file tool.",
                fields: fields, grammarTokenizer: tok,
                instructions: "Respond with ONLY a JSON object of the tool's arguments.", maxTokens: 128)
            print("CONSTRAINED ARGS: \(json)")
            if let data = json.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                print(obj["path"] != nil
                      ? "VALID ARGS ✓  keys=\(obj.keys.sorted()) (forced to read_file's schema — has `path`)"
                      : "keys=\(obj.keys.sorted()) (missing `path`)")
            } else {
                print("INVALID JSON ✗")
            }
            await host.shutdown()
            return
        }

        // TOOLDISPATCH_DEMO=1: the COMPLETE constrained tool call → DISPATCH (end-to-end of #31).
        if ProcessInfo.processInfo.environment["TOOLDISPATCH_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let call = try await model.callToolConstrained(
                "What is the current time?",
                host: host, grammarTokenizer: tok, nameSchema: ToolName.self)
            print("CONSTRAINED CALL → name=\(call.name)  args=\(call.argsJSON)")
            print("DISPATCH RESULT → \(call.result.prefix(400))")
            print((await host.toolNames).contains(call.name)
                  ? "✓ constrained tool call DISPATCHED (name ∈ tools + args ∈ schema, by construction)"
                  : "✗ name not a real tool")
            await host.shutdown()
            return
        }

        // LIVELOOP_DEMO=1: prove the LIVE-loop path (`constrainRoutedToolCall`) that runWithTools now runs
        // each round — the unconstrained pass ROUTES a tool (name maybe hallucinated, args maybe wrong); we
        // repair the name (∈ tools) and force the args (∈ the tool's schema) before dispatch.
        if ProcessInfo.processInfo.environment["LIVELOOP_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let specs = await host.specs
            let names = await host.toolNames
            func show(_ label: String, _ routedName: String, _ routedArgs: String, _ p: String) async throws {
                let c = try await model.constrainRoutedToolCall(
                    routedName: routedName, routedArgsJSON: routedArgs, prompt: p,
                    specs: specs, toolNames: names, grammarTokenizer: tok)
                print("\(label): routed(\(routedName), \(routedArgs)) → name=\(c.name) args=\(c.argsJSON)")
            }
            // PRESERVE distinct correct values — two same-tool calls no longer collapse to one (CRITICAL fix):
            try await show("preserve-A", "read_file", "{\"path\":\"/etc/hosts\"}", "Read /etc/hosts.")
            try await show("preserve-B", "read_file", "{\"path\":\"/etc/passwd\"}", "Read /etc/passwd.")
            // PRESERVE exact-match free-text verbatim — old_string keeps its trailing spaces + newline
            // (the trim no longer corrupts edit_file / write_file / grep args):
            try await show("preserve-edit", "edit_file",
                           "{\"path\":\"/tmp/x\",\"old_string\":\"foo  \\n\",\"new_string\":\"bar\"}",
                           "Edit the file.")
            // REPAIR bogus structure — drop the invented {"action":…}, fill the omitted required `path`:
            try await show("repair-args", "read_file", "{\"action\":\"read\"}", "Read the file at /etc/hosts.")
            // REPAIR a hallucinated tool name → a real tool:
            try await show("repair-name", "totally_made_up_tool", "{}", "What is the current time?")
            await host.shutdown()
            return
        }

        let instructions = "You are a helpful local agent with tools. When a question needs "
            + "live system data, call the appropriate tool, then answer the user in plain words."

        // The de-Engram path: run the user's OWN ~/.claude/settings.json hooks (Engram's `memory-hooks
        // advise` on UserPromptSubmit, etc.) verbatim — so a local model is grounded exactly like claude.
        let hooks = ClaudeHookRunner.loadChain()
        log(hooks == nil ? "no ~/.claude hooks found — running hook-less"
                         : "loaded ~/.claude hooks (Engram advise, etc.)")

        // STREAM_DEMO=1: run the ORBITAL-FACING streamAgent path and print the claude stream-json NDJSON
        // lines Orbital's StreamJsonEvent decoder consumes. Verifies on the real model that a room agent gets
        // constrained/dispatched tool calls, <think> stripped from deltas, and a clean result.
        if ProcessInfo.processInfo.environment["STREAM_DEMO"] != nil {
            let p = ProcessInfo.processInfo.environment["STREAM_PROMPT"] ?? "What is the current time?"
            let stream = await model.streamAgent(p, host: host, instructions: instructions, hooks: hooks)
            for try await ev in stream { print(ev.ndjsonLine()) }
            await host.shutdown()
            return
        }

        // COMPACT_DEMO=1: force Claude-Code-style compaction with a TINY context budget on a multi-round task,
        // then test CONTINUITY — the agent must still answer from info read BEFORE the compaction (run with
        // SWIFTLM_CTX_DEBUG=1 to see the `[compact]` lines fire).
        if ProcessInfo.processInfo.environment["COMPACT_DEMO"] != nil {
            let tok = try MiniBPE.grammarTokenizer(forModelId: modelId)
            let p = await model.profile
            // Small budget: fires compaction after a couple of rounds without thrashing (real budgets are
            // window×0.7 = ~91K for GLM-Air). keepRecent excludes the original prompt, so the secret code
            // survives ONLY if the summary preserved it — a clean continuity test.
            let tiny = ModelProfile(family: p.family, emitsReasoning: p.emitsReasoning,
                                    toolCallStyle: p.toolCallStyle, budget: p.budget,
                                    contextBudget: ContextBudget(maxContextTokens: 2500, keepRecentTokens: 1100,
                                                                 maxToolOutputTokens: 800))
            let answer = try await model.runWithTools(
                "Remember this secret code exactly: ZEBRA-4417-QX. Now use the bash tool to read /etc/hosts, "
                + "then /etc/passwd (one at a time). AFTER reading both, tell me the exact secret code I gave "
                + "you at the very start.",
                host: host, instructions: instructions, maxRounds: 10, hooks: hooks,
                grammarTokenizer: tok, profile: tiny)
            print("FINAL ANSWER: \(answer)")
            await host.shutdown()
            return
        }

        // Build the grammar tokenizer once so the REPL's tool calls are CONSTRAINED — every dispatched call
        // gets a valid name (∈ the connected tools) and valid args (∈ the tool's schema), regenerated under
        // grammar constraint instead of MLXLMCommon's unconstrained parse. nil → falls back to unconstrained.
        let grammarTok = try? MiniBPE.grammarTokenizer(forModelId: modelId)
        log(grammarTok == nil ? "no grammar tokenizer — tool calls UNCONSTRAINED"
                              : "grammar tokenizer ready — tool calls constrained to schema")

        // Auto-detect the model family → its behavior profile (reasoning handling, tool-call recovery, the
        // physics-derived generation budget). No model is special-cased; the profile drives the generic loop.
        let profile = await model.profile

        // Sub-agent capability (the Claude-faithful `Task` tool): load `~/.claude/agents/*.md` and build the
        // MLX runner, then RE-register the native tools WITH `Task` injected (DI). A sub-agent re-enters
        // `runWithTools` with the loaded system prompt + scoped tools, no advise hook, no `Task` (one level
        // deep). v1 runs serial/concurrent; the ComputePool-batched conformer plugs in under this contract.
        let agentDefs = AgentDefinitionLoader.load()
        let subagentRunner = MLXSubagentRunner(model: model, host: host, agentDefs: agentDefs,
                                               grammarTokenizer: grammarTok, profile: profile)
        await host.registerNative(.standard(subagentRunner: subagentRunner))
        log("sub-agents: \(agentDefs.count) defs (\(agentDefs.keys.sorted().joined(separator: ", "))) — Task tool enabled")
        // Agentic round budget: enough for read-several-files → implement → test → fix → re-test → verify.
        // Simple turns terminate early (the model answers), so a generous cap only helps hard tasks.
        let maxRounds = Int(ProcessInfo.processInfo.environment["MAXROUNDS"] ?? "") ?? 12
        // PERMISSION=plan|approval|auto (default auto). Plan mode runs read-only tools only — the model
        // produces a plan instead of acting (the autonomy ladder Orbital room agents run under).
        let permission = ToolPermissionPolicy(
            mode: PermissionMode(rawValue: ProcessInfo.processInfo.environment["PERMISSION"] ?? "auto") ?? .auto)
        // In plan mode the model calls ExitPlanMode when ready; the harness shows the plan and the human
        // approves on stdin. Approve → the same turn flips to .auto and implements (Claude's plan-mode UX).
        let approvePlan: PlanApprover = { plan in
            print("\n=== PROPOSED PLAN (approve to execute) ===\n\(plan)\n=== Approve and implement? [y/N] ===")
            let line = (readLine(strippingNewline: true) ?? "").trimmingCharacters(in: .whitespaces).lowercased()
            return (line == "y" || line == "yes") ? .approve : .reject(reason: "user did not approve the plan")
        }
        log("model family: \(profile.family.rawValue) (reasoning=\(profile.emitsReasoning), "
            + "toolCalls=\(profile.toolCallStyle), maxTokens=\(profile.budget.maxTokens), "
            + "permission=\(permission.mode.rawValue))")

        print("SwiftLM agent ready. Enter a prompt (blank line or 'quit' to exit):")
        while let line = readLine(strippingNewline: true) {
            let prompt = line.trimmingCharacters(in: .whitespaces)
            if prompt.isEmpty || prompt == "quit" { break }
            do {
                let answer = try await model.runWithTools(prompt, host: host,
                                                          instructions: instructions,
                                                          maxRounds: maxRounds,
                                                          hooks: hooks,
                                                          grammarTokenizer: grammarTok, profile: profile,
                                                          permission: permission,
                                                          approvePlan: permission.mode == .plan ? approvePlan : nil,
                                                          toolTimeoutSeconds:
                                                              Double(ProcessInfo.processInfo.environment["TOOL_TIMEOUT_SEC"] ?? "")
                                                              ?? defaultToolDispatchTimeoutSeconds)
                print(answer)
            } catch {
                print("error: \(error)")
            }
        }
        await host.shutdown()
        log("bye.")
    }
}
