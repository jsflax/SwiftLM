import Foundation
import MLXLMCommon
import Serving
import NativeTools

// ── TRAIT_CALIB / TRAIT_DATAGEN: Part D Stage 1 — measure the ROLE-DISCIPLINE in-band rate and (when
// TRAIT_DATAGEN is set) harvest the DPO won×lost pairs that train the role-discipline LoRA.
//
// THE TARGET BEHAVIOR (role-discipline, NOT narrow tool-formatting): under heavy multi-agent context the 122B
// builder "Forge" NARRATED its intent ("I'll build engine.py") and emitted ZERO tool calls — the working dir
// stayed empty while the spine spun. In CLEAN single-turn context (WriteReliabilityDiag) the SAME model emits a
// well-formed `<tool_call><function=write_file>` call ~100% of the time and the parser accepts it. So the gap is
// NOT a parse/format bug — it is the model failing to ACT IN ROLE once the context is long/noisy. The mechanical
// VERIFIER for "acted in role" is exactly the serve-path tool-call parser: a parser-accepted call = the builder
// acted; pure narration / truncation / a think-spiral = it did not. Tool-call emission is the SIGNAL, role-
// discipline is the trait.
//
// WHY A HARNESS, NOT bestOfNDomainPairs: the domain flywheel renders bare prompts through `pool.generateMany`
// and verifies via a repo clonefile→build→test oracle (DomainVerifier) — neither the owned-render path nor the
// tool-call verifier. Reusing it would (a) discard the heavy transcript ⇒ measure a false ~100%, and (b) verify
// the wrong thing. This harness instead reproduces the SERVE render exactly (CompactingSession.ownedRound) so
// train==serve, and scores with the REAL parser so the verifier==serve.
//
// THE WRONG-MEASUREMENT TRAP this guards: if the harness used clean context it would measure WRITE_DIAG's ~100%
// and wrongly conclude "no gap". So it builds a heaviness LADDER (light/medium/heavy) and REQUIRES the spread
// (light≈100%, heavy lower) as proof the harness is actually loaded — if heavy also ≈100%, the failure is
// orchestration-specific (room timing/handoffs), not standalone role-discipline, and an adapter won't fix it.
//
// Env knobs (all optional): TRAIT_N (rollouts/cell, def 8) · TRAIT_TEMP (def 0.8, >0 for diversity) ·
// TRAIT_MAXTOK (def 1600) · TRAIT_DOMAINS (csv of task ids, def all) · TRAIT_LEVELS (csv of light,medium,heavy,
// def all) · TRAIT_DATAGEN (persist DPO pairs) · TRAIT_DATAGEN_OUT (output path override).

extension MLXLanguageModel {

    // One builder task: a single-file creation ask with an unambiguous role-appropriate action (write the file).
    fileprivate struct BuilderTask: Sendable { let id: String; let file: String; let ask: String }

    fileprivate enum Heaviness: String, CaseIterable, Sendable { case light, medium, heavy }

    // Per-rollout score. `positive` = acted-in-role (a parser-accepted call to ANY available tool — the inverse
    // of the narrate-without-acting failure). `mutating` = the call was a file-producing/exec action
    // (write_file/edit_file/bash) — the stricter "did the role-appropriate thing" signal. `negKind` classifies a
    // failure so a budget-truncation artifact isn't miscounted as narration.
    fileprivate struct Rollout: Sendable {
        let text: String; let positive: Bool; let isMutating: Bool
        let calledName: String?; let negKind: String?
    }

    // The 7 non-chess builder domains (so the trait learned is ROLE-DISCIPLINE, not chess planning). Each has a
    // single-file success criterion; the role-appropriate action is to call write_file with the file body.
    fileprivate static let calibTasks: [BuilderTask] = [
        .init(id: "cli",      file: "cli.py",       ask: "Create cli.py: a parse_args(argv) using argparse with --input (required) and --verbose (a flag), returning the parsed namespace."),
        .init(id: "config",   file: "config.py",    ask: "Create config.py: a load_config(path) that reads a JSON file and returns a dict, defaulting to {} when the file is missing."),
        .init(id: "handler",  file: "handler.py",   ask: "Create handler.py: a BaseHTTPRequestHandler subclass whose do_GET returns HTTP 200 with the JSON body {\"ok\": true}."),
        .init(id: "parser",   file: "exprparser.py",ask: "Create exprparser.py: a tokenize(s) returning a list of int / '+' / '*' tokens, and an evaluate(tokens) honoring '*' over '+' precedence."),
        .init(id: "validate", file: "validate.py",  ask: "Create validate.py: a validate(record) that raises ValueError when record['email'] lacks '@' or record['age'] < 0, else returns True."),
        .init(id: "migration",file: "migration.sql",ask: "Create migration.sql: a CREATE TABLE users statement with id (primary key), email (unique, not null), and created_at (timestamp)."),
        .init(id: "logsetup", file: "logsetup.py",  ask: "Create logsetup.py: a get_logger(name) returning a logger with a StreamHandler at INFO level and a standard '%(asctime)s %(levelname)s %(name)s %(message)s' format."),
    ]

    // The builder persona = the SERVE-time system prompt (the `instructions` ownedRound prepends after BUG-2).
    // Names the REAL native tools the model sees in the injected schema (write_file/edit_file/…). Crucially it
    // states the room contract that makes narration a FAILURE: other agents see only your tool calls, not intent.
    fileprivate static let builderPersona = """
    You are Forge, the BUILDER agent in a multi-agent engineering room. You implement files using your native \
    tools: write_file, edit_file, read_file, glob, grep, bash. The Planner assigns implementation tasks; the \
    Critic reviews your output; the Referee ends the room only when the work is verified. When you are assigned \
    a task to create a file you MUST call the write_file tool with the COMPLETE file content as the `content` \
    argument — do not merely describe what you will do. The other agents cannot see your reasoning or your \
    intentions; they see ONLY the tool calls you emit and their results. An empty working directory means you \
    have not done your job, no matter what you said you would do.
    """

    public func traitCalibration() async throws -> String {
        func env(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }
        let datagen = env("TRAIT_DATAGEN") != nil
        let N = env("TRAIT_N").flatMap(Int.init) ?? 8
        let T = env("TRAIT_TEMP").flatMap(Float.init) ?? 0.8
        let maxTok = env("TRAIT_MAXTOK").flatMap(Int.init) ?? 1600
        let levels: [Heaviness] = {
            guard let csv = env("TRAIT_LEVELS") else { return Heaviness.allCases }
            let want = Set(csv.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
            return Heaviness.allCases.filter { want.contains($0.rawValue) }
        }()
        let tasks: [BuilderTask] = {
            guard let csv = env("TRAIT_DOMAINS") else { return Self.calibTasks }
            let want = Set(csv.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
            return Self.calibTasks.filter { want.contains($0.id) }
        }()

        let adapter = await self.localAdapter
        // The tool schema is identical across every rollout ⇒ build the native specs ONCE.
        let host = MCPHost()
        await host.registerNative(NativeToolRegistry.standard(cwd: "/tmp/traitcalib"))
        let specs = await host.specs
        // The mechanical verifier == the serve-path parser. If the model defers to MLX (no xmlFunction/json
        // template evidence) there is no standalone parser to score with — abort loudly rather than guess.
        guard let parser = await makeToolCallParser(adapter) else {
            return "=== TRAIT_CALIB ABORT ===\nNo standalone tool-call parser (adapter.toolCallFormat=\(adapter.toolCallFormat), likely .deferToMLX). "
                + "Stage-1 scoring needs the serve parser. Run on a model whose template evidences its tool-call format (the 122B = xmlFunction)."
        }
        let toolNames = Set(specs.compactMap { spec -> String? in
            // ToolSpec is [String: any Sendable] = {"type":"function","function":{"name":…}}
            ((spec["function"] as? [String: any Sendable])?["name"] as? String)
        })
        let mutatingTools: Set<String> = ["write_file", "edit_file", "bash"]

        func progress(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        progress("=== TRAIT_CALIB start: model=\(modelId) owned-render=\(adapter.requiresOwnedRender) "
            + "N=\(N) T=\(T) maxTok=\(maxTok) domains=\(tasks.map(\.id)) levels=\(levels.map(\.rawValue)) "
            + "tools=\(toolNames.sorted()) datagen=\(datagen) ===")

        // ── One rollout: delegates to the shared serve-identical render+decode+score (calibrationRollout). ───
        func runRollout(turns: [TurnMessage]) async throws -> Rollout {
            try await calibrationRollout(system: Self.builderPersona, turns: turns, specs: specs, parser: parser,
                                         toolNames: toolNames, mutatingTools: mutatingTools, adapter: adapter,
                                         maxTok: maxTok, temperature: T)
        }

        // ── REAL-CONTEXT REPLAY (the decisive test): if TRAIT_REAL_CONTEXT points at a harvested-failure JSON,
        // replay that EXACT context (Forge's real system prompt + the real plan+panel-debate turns) through the
        // ISOLATED decode — no orchestration — N times. Narration here ⇒ the failure is CONTEXT-INDUCED (trainable
        // role-discipline); acting here ⇒ the room failure was orchestration-specific and an adapter won't help.
        if let realPath = env("TRAIT_REAL_CONTEXT") {
            return try await replayRealContext(path: realPath, specs: specs, parser: parser, toolNames: toolNames,
                                               mutatingTools: mutatingTools, adapter: adapter, N: N, T: T, maxTok: maxTok)
        }

        // ── SELF-CHECK (fail fast): render the HEAVIEST cell, assert it is genuinely heavy + the parser is live +
        // one decode produces output — so a harness bug surfaces in ~1 rollout, not after 150. ────────────────
        if let probe = tasks.first {
            let heavyTurns = Self.buildContext(task: probe, level: .heavy)
            let probeTokens = (try? await renderTurnMessages(
                ([TurnMessage(role: .system, content: Self.builderPersona)] + adapter.continuationMessages(heavyTurns)),
                tools: specs, enableThinking: true))?.count ?? 0
            progress("[self-check] heaviest cell (\(probe.id)/heavy) renders to \(probeTokens) prompt tokens "
                + "(want ≫ light; a true heavy load is ≥1500).")
            if levels.contains(.heavy), probeTokens < 1200 {
                return "=== TRAIT_CALIB ABORT (self-check) ===\nHeaviest cell rendered to only \(probeTokens) tokens — "
                    + "the heavy ladder is not actually heavy, so calibration would measure a false ~100%. Fix buildContext(.heavy)."
            }
        }

        // ── SWEEP: domains × heaviness, N rollouts each. Print each cell as it completes (a killed run still
        // yields the light-vs-heavy signal). Decode is serial — the 122B is ~65GB resident, one at a time. ────
        struct Cell { let task: BuilderTask; let level: Heaviness; let rollouts: [Rollout] }
        var cells: [Cell] = []
        for task in tasks {
            for level in levels {
                let turns = Self.buildContext(task: task, level: level)
                var rs: [Rollout] = []
                for i in 0..<N {
                    do { rs.append(try await runRollout(turns: turns)) }
                    catch { progress("  [\(task.id)/\(level.rawValue)] rollout \(i) errored: \(error)"); continue }
                }
                cells.append(Cell(task: task, level: level, rollouts: rs))
                let won = rs.filter(\.positive).count
                let mut = rs.filter(\.isMutating).count
                let negs = rs.filter { !$0.positive }
                let nb = Dictionary(grouping: negs, by: { $0.negKind ?? "?" }).mapValues(\.count)
                progress("  [\(task.id)/\(level.rawValue)] acted \(won)/\(rs.count) "
                    + "(mutating \(mut)) — neg: \(nb.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " "))")
            }
        }

        // ── AGGREGATE + REPORT ────────────────────────────────────────────────────────────────────────────
        func rate(_ rs: [Rollout]) -> String {
            let w = rs.filter(\.positive).count
            return rs.isEmpty ? "—" : "\(w)/\(rs.count) (\(Int((Double(w) / Double(rs.count) * 100).rounded()))%)"
        }
        var out: [String] = []
        out.append("=== TRAIT_CALIB (model: \(modelId), owned-render=\(adapter.requiresOwnedRender), "
            + "trait=role-discipline, N=\(N), T=\(T), maxTok=\(maxTok)) ===")
        out.append("PRIMARY signal = acted-in-role (a parser-accepted call to an available tool); "
            + "negatives = narration / truncated-call / think-spiral.")
        out.append("")
        out.append("PER-CELL (domain × heaviness) acted-in-role rate:")
        for task in tasks {
            let parts = levels.map { lvl -> String in
                let rs = cells.first { $0.task.id == task.id && $0.level == lvl }?.rollouts ?? []
                return "\(lvl.rawValue): \(rate(rs))"
            }
            out.append("  \(task.id.padding(toLength: 10, withPad: " ", startingAt: 0)) \(parts.joined(separator: "   "))")
        }
        out.append("")
        out.append("BY HEAVINESS (across domains):")
        for lvl in levels {
            let rs = cells.filter { $0.level == lvl }.flatMap(\.rollouts)
            let mut = rs.filter(\.isMutating).count
            out.append("  \(lvl.rawValue.padding(toLength: 8, withPad: " ", startingAt: 0)) acted \(rate(rs))   mutating \(mut)/\(rs.count)")
        }
        out.append("")
        out.append("BY DOMAIN (across levels):")
        for task in tasks {
            let rs = cells.filter { $0.task.id == task.id }.flatMap(\.rollouts)
            out.append("  \(task.id.padding(toLength: 10, withPad: " ", startingAt: 0)) acted \(rate(rs))")
        }
        let all = cells.flatMap(\.rollouts)
        let overallWon = all.filter(\.positive).count
        let overallMut = all.filter(\.isMutating).count
        let negAll = all.filter { !$0.positive }
        let negBreak = Dictionary(grouping: negAll, by: { $0.negKind ?? "?" }).mapValues(\.count)
        // A cell is TRAINABLE when 0 < won < N (it can sometimes act, sometimes not — the capability-gap band).
        let trainableCells = cells.filter { let w = $0.rollouts.filter(\.positive).count; return w > 0 && w < $0.rollouts.count }.count
        let dryCells = cells.filter { $0.rollouts.allSatisfy { !$0.positive } && !$0.rollouts.isEmpty }.count
        let satCells = cells.filter { !$0.rollouts.isEmpty && $0.rollouts.allSatisfy(\.positive) }.count
        out.append("")
        out.append("OVERALL acted-in-role: \(overallWon)/\(all.count) "
            + "(\(all.isEmpty ? 0 : Int((Double(overallWon) / Double(all.count) * 100).rounded()))%)   "
            + "mutating \(overallMut)/\(all.count)")
        out.append("NEGATIVE breakdown: \(negBreak.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: "  "))")
        out.append("CELLS: trainable(0<won<N)=\(trainableCells)  dry(won==0)=\(dryCells)  saturated(won==N)=\(satCells)  of \(cells.count)")

        // VERDICT — the spread is the proof. light≈100% + heavy meaningfully lower + some trainable cells = a
        // real, context-induced, trainable gap. heavy≈100% = the failure is orchestration-specific, not the model.
        func levelRate(_ level: Heaviness) -> Double? {
            let rs = cells.filter { $0.level == level }.flatMap(\.rollouts)
            guard !rs.isEmpty else { return nil }
            return Double(rs.filter(\.positive).count) / Double(rs.count)
        }
        func pct(_ d: Double) -> String { "\(Int((d * 100).rounded()))%" }
        let lightRate = levelRate(.light)
        let heavyRate = levelRate(.heavy)
        let verdict: String
        if let lr = lightRate, let hr = heavyRate {
            if hr >= 0.93 {
                verdict = "NO-GAP IN HARNESS — heavy context still acts \(pct(hr)) (light \(pct(lr))). The narrate-without-acting "
                    + "failure did NOT reproduce from context alone ⇒ it is likely ORCHESTRATION-specific (room timing/handoffs), "
                    + "NOT standalone role-discipline. Harvest the REAL chess122team-branch2 transcript to confirm before training."
            } else if lr - hr >= 0.15, trainableCells >= 1 {
                verdict = "TRAINABLE GAP CONFIRMED — light \(pct(lr)) vs heavy \(pct(hr)) (spread \(pct(lr - hr))), \(trainableCells) trainable cells. "
                    + "The gap is context-induced and in-band ⇒ proceed to Stage 2 (DPO train) on the harvested pairs."
            } else if overallWon == 0 {
                verdict = "NO-SEED — the builder never acts, even light. Check parser presence + the NEGATIVE breakdown "
                    + "(if truncated-call dominates, raise TRAIT_MAXTOK; if narration, the persona/render may differ from serve)."
            } else {
                verdict = "WEAK/AMBIGUOUS spread (light \(pct(lr)) vs heavy \(pct(hr))) — increase heaviness or N, or harvest the real transcript."
            }
        } else {
            verdict = "PARTIAL — need both light and heavy levels for the spread verdict (set TRAIT_LEVELS=light,heavy)."
        }
        out.append("VERDICT: \(verdict)")

        // SAMPLES — the user wants to SEE the model working, not just metrics (memory 1BA7350C). Dump a real
        // acted-in-role rollout vs a real narration failure from the HEAVY level, side by side.
        if let hp = cells.first(where: { $0.level == .heavy && $0.rollouts.contains(where: \.positive) })?.rollouts.first(where: \.positive) {
            out.append("\n---- SAMPLE ACTED-IN-ROLE (heavy) [called \(hp.calledName ?? "?")] ----\n" + String(hp.text.prefix(1200)))
        }
        if let hn = cells.first(where: { $0.level == .heavy && $0.rollouts.contains(where: { !$0.positive }) })?.rollouts.first(where: { !$0.positive }) {
            out.append("\n---- SAMPLE NARRATION-FAILURE (heavy) [\(hn.negKind ?? "?")] ----\n" + String(hn.text.prefix(1200)))
        }

        // ── DPO DATAGEN — persist won×lost pairs (structured turns for the durable Stage-2 render fix + a
        // flattened user string as the stopgap for the current bare DPOTraining.renderOne). ─────────────────
        if datagen {
            var persisted: [PersistPair] = []
            for cell in cells {
                let won = cell.rollouts.filter(\.positive).map(\.text)
                let lost = cell.rollouts.filter { !$0.positive }.map(\.text)
                guard !won.isEmpty, !lost.isEmpty else { continue }   // only trainable cells yield pairs
                let turns = Self.buildContext(task: cell.task, level: cell.level)
                let promptTurns = ([TurnMessage(role: .system, content: Self.builderPersona)] + turns).map {
                    PersistTurn(role: $0.role.rawValue, content: $0.content, reasoningContent: $0.reasoningContent,
                                toolCallsJSON: $0.toolCalls.map { tcs in tcs.map { "\($0.name): \($0.argsJSON)" }.joined(separator: "\n") })
                }
                let userFlat = ([("system", Self.builderPersona)] + turns.map { ($0.role.rawValue, $0.content) })
                    .map { "\($0.0): \($0.1)" }.joined(separator: "\n\n")
                var made = 0
                outer: for w in won {
                    for l in lost {
                        persisted.append(PersistPair(domain: cell.task.id, heaviness: cell.level.rawValue,
                                                     promptTurns: promptTurns, userFlat: userFlat,
                                                     chosen: w, rejected: l, modelId: modelId))
                        made += 1; if made >= 24 { break outer }
                    }
                }
            }
            let outURL = Self.datagenOutputURL(modelId: modelId)
            do {
                try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted]
                try enc.encode(persisted).write(to: outURL)
                out.append("\nDATAGEN: wrote \(persisted.count) DPO pairs → \(outURL.path)")
            } catch {
                out.append("\nDATAGEN ERROR: \(error)")
            }
        }
        out.append("\n(Stage-2 NOTE: DPOTraining.renderOne renders only `applyChatTemplate([user])` — to keep train==serve "
            + "the DPO-train stage MUST render `promptTurns` via renderTurnMessages, not the flattened userFlat. The pairs "
            + "carry BOTH; the structured form is canonical.)")

        let report = out.joined(separator: "\n")
        progress("=== TRAIT_CALIB done ===")
        return report
    }

    // ── ONE ROLLOUT, SERVE-IDENTICAL (mirrors CompactingSession.ownedRound): prepend the system turn, run
    // continuationMessages, render, decode (re-inserting the primed <think> open tag on the first chunk), then
    // SCORE with the real parser + recover fallback + the garbage-name guard — so the verifier == serve. ────────
    fileprivate func calibrationRollout(
        system: String, turns: [TurnMessage], specs: [ToolSpec], parser: any ToolCallParser,
        toolNames: Set<String>, mutatingTools: Set<String>, adapter: ModelProfile,
        maxTok: Int, temperature: Float
    ) async throws -> Rollout {
        let body = adapter.continuationMessages(turns)
        let rendered = system.isEmpty ? body : [TurnMessage(role: .system, content: system)] + body
        let tokens = try await renderTurnMessages(rendered, tools: specs, enableThinking: true)
        var params = GenerateParameters(maxTokens: maxTok, temperature: temperature)
        params.repetitionPenalty = adapter.sampling.repetitionPenalty
        params.repetitionContextSize = adapter.sampling.repetitionContextSize

        let primeThink = adapter.emitsReasoning
        let openTag = adapter.reasoningTags.open
        var text = ""
        var firstChunk = true
        for try await g in streamFromTokens(tokens, maxTokens: maxTok, adapter: adapter,
                                            params: params, kvBox: nil, activeTraits: []) {
            if case .chunk(let c) = g {
                text += (firstChunk && primeThink) ? openTag + c : c
                firstChunk = false
            }
        }

        let stripped = CompactingSession.stripThinkSpans(text)
        let parsed = parser.parse(content: stripped, tools: specs)
        let recovered = (parsed == nil) ? adapter.recoverMissedToolCall(stripped) : nil
        let rawName = parsed?.function.name ?? recovered?.name
        func validName(_ n: String) -> Bool {
            !(n.contains("<") || n.contains("\n") || n.count > 64) && toolNames.contains(n)
        }
        if let n = rawName, validName(n) {
            return Rollout(text: text, positive: true, isMutating: mutatingTools.contains(n), calledName: n, negKind: nil)
        }
        func count(_ needle: String) -> Int { text.components(separatedBy: needle).count - 1 }
        let tcOpen = count("<tool_call>") + count("<function="), tcClose = count("</tool_call>") + count("</function>")
        let thinkOpen = text.contains(openTag), thinkClose = text.contains(adapter.reasoningTags.close)
        let kind: String
        if tcOpen >= 1, tcClose == 0 { kind = "truncated-call" }
        else if primeThink, thinkOpen, !thinkClose { kind = "think-spiral" }
        else { kind = "narration" }
        return Rollout(text: text, positive: false, isMutating: false, calledName: rawName, negKind: kind)
    }

    // ── REAL-CONTEXT REPLAY: load a harvested-failure JSON ({system, turns:[{role,content}]}) and decode it in
    // isolation. Rollout 0 is GREEDY (temperature 0 = the faithful serve setting that produced the failure); the
    // rest sample at T for diversity + training data. Decisive read: does the model narrate (gap context-induced)
    // or act (gap orchestration-specific)? Also serves as the user's real before/after demo surface. ───────────
    fileprivate func replayRealContext(
        path: String, specs: [ToolSpec], parser: any ToolCallParser, toolNames: Set<String>,
        mutatingTools: Set<String>, adapter: ModelProfile, N: Int, T: Float, maxTok: Int
    ) async throws -> String {
        func progress(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let ctx = try JSONDecoder().decode(RealContext.self, from: data)
        let turns = ctx.turns.map { TurnMessage(role: TurnMessage.Role(rawValue: $0.role) ?? .user, content: $0.content) }
        let renderedProbe = (ctx.system.isEmpty ? [] : [TurnMessage(role: .system, content: ctx.system)]) + adapter.continuationMessages(turns)
        let promptTok = (try? await renderTurnMessages(renderedProbe, tools: specs, enableThinking: true))?.count ?? 0
        progress("=== TRAIT_REPLAY: \(ctx.source ?? path) — system=\(ctx.system.count)c turns=\(turns.count) promptTok=\(promptTok) N=\(N) (rollout0=greedy, rest T=\(T)) ===")

        var rs: [Rollout] = []
        for i in 0..<max(1, N) {
            let temp: Float = (i == 0) ? 0.0 : T
            do {
                let r = try await calibrationRollout(system: ctx.system, turns: turns, specs: specs, parser: parser,
                                                     toolNames: toolNames, mutatingTools: mutatingTools, adapter: adapter,
                                                     maxTok: maxTok, temperature: temp)
                rs.append(r)
                progress("  rollout \(i)\(i == 0 ? " [greedy]" : ""): "
                    + (r.positive ? "ACTED [\(r.calledName ?? "?")]" : "FAILED [\(r.negKind ?? "?")]"))
            } catch { progress("  rollout \(i) errored: \(error)") }
        }
        let won = rs.filter(\.positive).count
        let greedyActed = rs.first?.positive ?? false
        let negBreak = Dictionary(grouping: rs.filter { !$0.positive }, by: { $0.negKind ?? "?" }).mapValues(\.count)

        var out: [String] = []
        out.append("=== TRAIT_REPLAY (model: \(modelId), trait=role-discipline) ===")
        out.append("SOURCE: \(ctx.source ?? path)")
        if let note = ctx.taskNote { out.append("ROLE-APPROPRIATE ACTION: \(note)") }
        out.append("CONTEXT: system=\(ctx.system.count) chars, \(turns.count) prior turns, \(promptTok) prompt tokens.")
        out.append("GREEDY (faithful serve setting): \(greedyActed ? "ACTED — emitted \(rs.first?.calledName ?? "?")" : "NARRATED/FAILED — \(rs.first?.negKind ?? "?")")")
        out.append("ALL \(rs.count) rollouts: acted-in-role \(won)/\(rs.count); negatives: \(negBreak.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " "))")
        let verdict: String
        if !greedyActed || won < rs.count * 7 / 10 {
            verdict = "CONTEXT-INDUCED FAILURE CONFIRMED — the model narrates/fails on the REAL failure context in isolation "
                + "(no orchestration). This is a trainable role-discipline gap; this exact context is a gold DPO negative seed."
        } else {
            verdict = "GAP DID NOT REPRODUCE in isolation (acted \(won)/\(rs.count), greedy acted) — the room failure was likely "
                + "ORCHESTRATION-specific (timing/handoffs/turn-trigger), NOT standalone role-discipline. An adapter won't fix it; "
                + "investigate the orchestration path instead."
        }
        out.append("VERDICT: \(verdict)")
        // SAMPLES (the user wants to SEE the model working): the greedy rollout, plus a contrasting outcome.
        if let g = rs.first {
            out.append("\n---- GREEDY ROLLOUT (\(g.positive ? "ACTED \(g.calledName ?? "")" : "FAILED \(g.negKind ?? "")")) ----\n" + String(g.text.prefix(1400)))
        }
        if let other = rs.dropFirst().first(where: { $0.positive != (rs.first?.positive ?? false) }) {
            out.append("\n---- CONTRASTING ROLLOUT (\(other.positive ? "ACTED \(other.calledName ?? "")" : "FAILED \(other.negKind ?? "")")) ----\n" + String(other.text.prefix(1400)))
        }
        progress("=== TRAIT_REPLAY done ===")
        return out.joined(separator: "\n")
    }

    // ── TRAIT_DEMO: the user's "show me it working" gate — adapter OFF vs ON before/after on a real heavy context.
    // Decodes the SAME context N times on the FROZEN BASE, then loads the trained role-discipline LoRA into the
    // resident model (Hotswap.loadAdapter) and decodes N more, reporting the ACTED-IN-ROLE RATE for each (greedy
    // rollout 0 + N−1 at T) + the greedy completions side by side. The acted-RATE (not a single greedy flip) is the
    // robust signal: the disposition gap is probabilistic, so the adapter shows as OFF acts k/N → ON acts more.
    // Inference needs NO StopGradGate/StopGradRMSNorm (those are backprop-only). Env: TRAIT_DEMO_CONTEXT (RealContext
    // JSON, default /tmp/forge_real_context.json), TRAIT_DEMO_ADAPTER (trait dir), TRAIT_N, TRAIT_TEMP, TRAIT_MAXTOK.
    public func roleDisciplineDemo() async throws -> String {
        func env(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }
        func progress(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        let N = env("TRAIT_N").flatMap(Int.init) ?? 6
        let T = env("TRAIT_TEMP").flatMap(Float.init) ?? 0.8
        let maxTok = env("TRAIT_MAXTOK").flatMap(Int.init) ?? 1600
        let contextPath = env("TRAIT_DEMO_CONTEXT") ?? "/tmp/forge_real_context.json"
        let adapterPath = env("TRAIT_DEMO_ADAPTER")
            ?? (NSHomeDirectory() + "/Library/Application Support/SwiftLM/trait-adapters/role-discipline-builder")

        let data = try Data(contentsOf: URL(fileURLWithPath: contextPath))
        let ctx = try JSONDecoder().decode(RealContext.self, from: data)
        let turns = ctx.turns.map { TurnMessage(role: TurnMessage.Role(rawValue: $0.role) ?? .user, content: $0.content) }

        let adapter = await self.localAdapter
        let host = MCPHost()
        await host.registerNative(NativeToolRegistry.standard(cwd: "/tmp/traitdemo"))
        let specs = await host.specs
        guard let parser = await makeToolCallParser(adapter) else {
            return "=== TRAIT_DEMO ABORT ===\nNo standalone tool-call parser (toolCallFormat=\(adapter.toolCallFormat))."
        }
        let toolNames = Set(specs.compactMap { ($0["function"] as? [String: any Sendable])?["name"] as? String })
        let mutatingTools: Set<String> = ["write_file", "edit_file", "bash"]

        // N rollouts (rollout 0 greedy T=0 = faithful serve, rest at T for the rate) — returns acted count + greedy.
        func runPhase(_ label: String) async -> (acted: Int, total: Int, greedy: Rollout?) {
            var rs: [Rollout] = []
            for i in 0..<max(1, N) {
                let temp: Float = (i == 0) ? 0.0 : T
                do {
                    let r = try await self.calibrationRollout(
                        system: ctx.system, turns: turns, specs: specs, parser: parser, toolNames: toolNames,
                        mutatingTools: mutatingTools, adapter: adapter, maxTok: maxTok, temperature: temp)
                    rs.append(r)
                    progress("  [\(label)] rollout \(i)\(i == 0 ? " [greedy]" : ""): "
                        + (r.positive ? "ACTED [\(r.calledName ?? "?")]" : "FAILED [\(r.negKind ?? "?")]"))
                } catch { progress("  [\(label)] rollout \(i) errored: \(error)") }
            }
            return (rs.filter(\.positive).count, rs.count, rs.first)
        }

        progress("=== TRAIT_DEMO: model=\(modelId) context=\(ctx.source ?? contextPath) adapter=\(adapterPath) N=\(N) T=\(T) ===")
        progress("--- PHASE OFF (frozen base, no adapter) ---")
        let off = await runPhase("OFF")
        progress("--- loading trained role-discipline adapter ---")
        try await loadAdapter(directory: URL(fileURLWithPath: adapterPath))
        progress("--- PHASE ON (adapter active) ---")
        let on = await runPhase("ON")

        func verdict(_ r: Rollout?) -> String {
            r.map { $0.positive ? "ACTED [\($0.calledName ?? "?")]" : "FAILED [\($0.negKind ?? "?")]" } ?? "?"
        }
        var out: [String] = []
        out.append("=== TRAIT_DEMO (model: \(modelId), trait=role-discipline) ===")
        out.append("SOURCE: \(ctx.source ?? contextPath)")
        if let note = ctx.taskNote { out.append("ROLE-APPROPRIATE ACTION: \(note)") }
        out.append("ADAPTER: \(adapterPath)")
        out.append("")
        out.append("ACTED-IN-ROLE RATE (greedy + \(max(0, N - 1)) sampled @T=\(T)):")
        out.append("  ADAPTER OFF: \(off.acted)/\(off.total)   greedy=\(verdict(off.greedy))")
        out.append("  ADAPTER ON : \(on.acted)/\(on.total)   greedy=\(verdict(on.greedy))")
        out.append("")
        if let g = off.greedy { out.append("---- OFF greedy completion ----\n" + String(g.text.prefix(1200))) }
        if let g = on.greedy { out.append("\n---- ON greedy completion ----\n" + String(g.text.prefix(1200))) }
        out.append("")
        if on.acted > off.acted {
            out.append("DELTA: +\(on.acted - off.acted)/\(on.total) acted — the adapter INCREASED acting-in-role (role-discipline learned).")
        } else if on.acted < off.acted {
            out.append("REGRESSION: the adapter DECREASED acting-in-role (overfit / instability).")
        } else {
            out.append("NO DELTA: equal acted-rate (\(off.acted)/\(off.total)). Try a heavier / held-out context that induces the gap.")
        }
        progress("=== TRAIT_DEMO done ===")
        return out.joined(separator: "\n")
    }

    fileprivate struct RealContext: Codable {
        let source: String?; let system: String; let taskNote: String?
        struct Turn: Codable { let role: String; let content: String }
        let turns: [Turn]
    }

    // ── The HEAVY-CONTEXT LADDER: a builder turn's context, reproduced at three heaviness levels. The induction
    // mechanism is prior-turn ACCUMULATION (ownedRound re-renders the whole transcript every round). Light ≈
    // WRITE_DIAG's clean baseline; heavy mirrors the 13-agent chess room (a long objective + several completed
    // build cycles with reasoning + multi-tool sequences + repeated critic "dir still empty" feedback). The
    // narrations are REALISTIC ("I'll create the next module"), never artificial-failure, so the heaviness — not
    // a planted "I'm stuck" — is what (if anything) pushes the model to narrate the current task without acting.
    fileprivate static func buildContext(task: BuilderTask, level: Heaviness) -> [TurnMessage] {
        let taskTurn = TurnMessage(role: .user,
            content: "<planner> Forge — next task: \(task.ask) Create \(task.file) now using the write_file tool.")
        switch level {
        case .light:
            return [TurnMessage(role: .user, content: "<planner> Team, we're assembling a small Python toolkit. Forge implements files; Critic reviews."),
                    taskTurn]
        case .medium:
            return planner(intro: true) + cycle(file: "util.py", reason: "It's the shared helper the other modules import, so it goes first.",
                                                 body: "def clamp(x, lo, hi):\\n    return max(lo, min(hi, x))\\n") + [
                TurnMessage(role: .user, content: "<critic> util.py looks correct. Planner: the next task is below."),
                taskTurn]
        case .heavy:
            // Several completed build cycles (varied files + reasoning + a read→write→bash sequence + recurring
            // critic feedback) → a genuinely long, noisy transcript like the room that produced the failure.
            var turns = planner(intro: true)
            let priors: [(String, String, String)] = [
                ("util.py",      "The shared helper the other modules import — it goes first.",                 "def clamp(x, lo, hi):\\n    return max(lo, min(hi, x))\\n"),
                ("errors.py",    "Central exception types so every module raises consistently.",                "class ToolkitError(Exception):\\n    pass\\n"),
                ("io_utils.py",  "Reading and writing files with a single encoding policy.",                    "def read_text(p):\\n    with open(p, encoding='utf-8') as f:\\n        return f.read()\\n"),
                ("registry.py",  "A name→factory registry the CLI will dispatch through.",                      "_R = {}\\ndef register(name):\\n    def deco(fn):\\n        _R[name] = fn; return fn\\n    return deco\\n"),
                ("timing.py",    "A context manager to time sections during debugging.",                        "import time\\nclass timer:\\n    def __enter__(self):\\n        self.t = time.time(); return self\\n    def __exit__(self, *a):\\n        self.dt = time.time() - self.t\\n"),
                ("cache.py",     "A tiny LRU so repeated lookups don't recompute.",                             "from functools import lru_cache\\ncached = lru_cache(maxsize=128)\\n"),
            ]
            for (i, p) in priors.enumerated() {
                turns += cycle(file: p.0, reason: p.1, body: p.2)
                if i == 2 {
                    // A read→inspect→note interlude (multi-tool noise like a real builder mid-room).
                    turns.append(TurnMessage(role: .assistant,
                        content: "Let me re-check util.py before continuing.",
                        reasoningContent: "I want to confirm clamp's signature so errors.py can rely on it.",
                        toolCalls: [Serving.ToolCall(name: "read_file", argsJSON: "{\"path\":\"util.py\"}")]))
                    turns.append(TurnMessage(role: .tool, content: "<tool_response>def clamp(x, lo, hi):\\n    return max(lo, min(hi, x))</tool_response>"))
                }
                turns.append(TurnMessage(role: .user, content: i % 2 == 0
                    ? "<critic> \(p.0) looks fine — keep going, the directory still needs the remaining modules."
                    : "<critic> Reviewed \(p.0). Acceptable. Planner, queue the next one."))
            }
            turns.append(TurnMessage(role: .user, content: "<planner> Good progress, Forge. Several modules are in. We still have the rest of the toolkit to build — stay on it."))
            turns.append(taskTurn)
            return turns
        }
    }

    private static func planner(intro: Bool) -> [TurnMessage] {
        intro ? [TurnMessage(role: .user, content:
            "<planner> Objective: build a small, well-factored Python toolkit (helpers, error types, I/O, a CLI, "
            + "an HTTP handler, validation, logging). Forge implements one file per task using the write_file tool; "
            + "the Critic reviews each file; the Referee ends the room only when every module exists and is verified. "
            + "Work task by task. Do not summarize — produce the files.")] : []
    }

    // One completed build cycle: the builder's assistant turn (reasoning + a write_file call) + the tool result.
    private static func cycle(file: String, reason: String, body: String) -> [TurnMessage] {
        [TurnMessage(role: .assistant,
            content: "Creating \(file).",
            reasoningContent: "\(reason) I'll write the file now with write_file.",
            toolCalls: [Serving.ToolCall(name: "write_file", argsJSON: "{\"path\":\"\(file)\",\"content\":\"\(body)\"}")]),
         TurnMessage(role: .tool, content: "<tool_response>wrote \(file)</tool_response>")]
    }

    // ── DPO pair persistence (the real DPOTraining.Pair/TrainPair are not Codable; this Codable mirror is decoded
    // back by the Stage-2 trainer). `promptTurns` is the canonical structured prompt (render serve-identically via
    // renderTurnMessages); `userFlat` is the stopgap for the current bare renderOne. ───────────────────────────
    fileprivate struct PersistTurn: Codable { let role: String; let content: String; let reasoningContent: String?; let toolCallsJSON: String? }
    fileprivate struct PersistPair: Codable {
        let domain: String; let heaviness: String
        let promptTurns: [PersistTurn]; let userFlat: String
        let chosen: String; let rejected: String; let modelId: String
    }
    fileprivate static func datagenOutputURL(modelId: String) -> URL {
        if let override = ProcessInfo.processInfo.environment["TRAIT_DATAGEN_OUT"] { return URL(fileURLWithPath: override) }
        let slug = modelId.replacingOccurrences(of: "/", with: "_")
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("SwiftLM/trait-calib/\(slug)/role-discipline-pairs.json")
    }
}
