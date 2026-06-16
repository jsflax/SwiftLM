import Foundation
import SelfImprove
import MLXBackend
import Orchestration

// The self-improvement loop, one cycle: harvest → masked LoRA train → eval gate →
// promote/reject. Gate = (held-out loss ↓) AND (retention within 5%) AND (tool-pass@1
// not regressed) — the correctness check that catches an adapter that got so chatty it
// stops calling tools. Plus a behavioral before/after A/B on held-out prompts.
//   EVAL_ONLY=1   skip training; just eval base vs current champion (cheap).
//   MCP_SERVER=…  MCP server binary for the tool-pass gate (default: claude-utils).

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func pct(_ a: Float, _ b: Float) -> String { String(format: "%+.1f%%", (b - a) / a * 100) }

func userTurn(_ formatted: String) -> String {
    guard let a = formatted.range(of: "<|im_start|>user\n"),
          let b = formatted.range(of: "<|im_end|>", range: a.upperBound..<formatted.endIndex)
    else { return formatted }
    return String(formatted[a.upperBound..<b.lowerBound])
}

/// Tool-pass@1: fraction of cases whose tool loop calls the expected tool.
func toolPass(_ model: MLXLanguageModel, _ host: MCPHost, _ cases: [ToolCase]) async -> Double {
    var pass = 0
    for c in cases {
        let r = try? await model.runWithToolsTracked(
            c.prompt, host: host,
            instructions: "You are a helpful agent with tools. Call the appropriate tool, then answer.")
        if r?.toolsCalled.contains(c.expectedTool) == true { pass += 1 }
    }
    return cases.isEmpty ? 0 : Double(pass) / Double(cases.count)
}

@main
struct SelfLoop {
    static func main() async throws {
        let env = ProcessInfo.processInfo.environment

        // WORKER=1 — run as a CLUSTER WORKER: load the model, serve batched best-of-N over the network,
        // and park. A second box runs this; the coordinator's FanOutPool RPCs generation jobs here.
        // No curriculum/registry/repo needed — workers only generate (verification is the coordinator's).
        //   WORKER_PORT (default 8787) · WORKER_ID · WORKER_BATCH (12) · SWIFTLM_MODEL (must match coord)
        if env["WORKER"] != nil {
            let port = UInt16(env["WORKER_PORT"] ?? "") ?? 8787
            log("cluster WORKER: loading model \(env["SWIFTLM_MODEL"] ?? "(default)") ...")
            let model = try await MLXLanguageModel.load()
            let pool = model.makeLocalPool(workerId: env["WORKER_ID"],
                                           batchWidth: Int(env["WORKER_BATCH"] ?? "") ?? 12)
            let server = WorkerServer(pool: pool, serviceName: env["WORKER_BONJOUR"])
            let bound = try await server.start(port: port)
            log("✅ SwiftLM cluster worker READY — model=\(model.modelId) port=\(bound). Waiting for jobs (Ctrl-C to stop) ...")
            while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }   // park
        }

        let evalOnly = env["EVAL_ONLY"] != nil
        let registry = try Registry()

        log("harvesting curriculum from transcripts ...")
        let curriculum = TranscriptHarvester.harvest()
        log("curriculum: train=\(curriculum.trainPairs.count) "
            + "(\(curriculum.trainPairs.count - curriculum.trainToolCount) chat + "
            + "\(curriculum.trainToolCount) tool) heldout=\(curriculum.heldout.count) "
            + "(\(curriculum.toolTraceExamples) tool traces harvested; productive "
            + "\(curriculum.productiveFiles)/\(curriculum.filesScanned) files); "
            + "redactions: \(curriculum.redactionSummary)")
        if !evalOnly {
            guard curriculum.trainPairs.count > 20, curriculum.heldout.count > 5 else {
                log("✗ curriculum too small."); exit(1)
            }
        }

        let cycleId = "cycle-\(Int(Date().timeIntervalSince1970))"
        let adapterDir = registry.adapterURL(cycleId: cycleId)
        if let champ = registry.currentChampion() {
            log("current champion: \(champ.cycleId) (held-out loss \(champ.heldoutLoss))")
        } else { log("no champion yet.") }

        log("loading model ...")
        let model = try await MLXLanguageModel.load()

        // BATCH_TEST=1 — validate batched best-of-N: time batched-of-N vs serial-of-N + show outputs.
        if ProcessInfo.processInfo.environment["BATCH_TEST"] != nil {
            let n = Int(ProcessInfo.processInfo.environment["BATCH_N"] ?? "") ?? 8
            let maxTok = Int(ProcessInfo.processInfo.environment["BATCH_MAXTOK"] ?? "") ?? 256
            let prompt = "Write a Swift function `binarySearch(_ a: [Int], _ target: Int) -> Int?` and explain step by step how it works and why it is O(log n)."
            // REAL path (stopOnEOS=true, eviction): variable-length best-of-N batched vs serial. This
            // is the case that was 0.59× before eviction (lockstep tail). With eviction it should be
            // >1× AND outputs identical-quality. The step-count print shows streams finishing early.
            log("BATCH_TEST: eviction path — batched best-of-\(n) vs \(n)×serial (maxTok \(maxTok), T=0.7) ...")
            let t0 = Date()
            let batched = await model.batchGenerate(prompt, n: n, maxTokens: maxTok, temperature: 0.7)
            let tBatched = Date().timeIntervalSince(t0)
            let t1 = Date()
            var serial: [String] = []
            for _ in 0..<n { serial.append((try? await model.generate(prompt, maxTokens: maxTok, temperature: 0.7, repetitionPenalty: 1.0)) ?? "") }
            let tSerial = Date().timeIntervalSince(t1)
            print("""

            ============ BATCH_TEST (eviction) — best-of-\(n) ============
            batched : \(String(format: "%.1f", tBatched))s   (\(batched.filter { !$0.isEmpty }.count)/\(n) non-empty)
            serial  : \(String(format: "%.1f", tSerial))s
            speedup : \(String(format: "%.2f", tSerial / max(0.001, tBatched)))×   (was 0.59× before eviction)
            sample[0]: \(batched.first.map { String($0.prefix(140)) } ?? "(none)")
            ====================================================
            """)
            return
        }

        // DUMP_BATCH=1 — verify the masking trains on precisely the assistant turn, then exit.
        if ProcessInfo.processInfo.environment["DUMP_BATCH"] != nil {
            let chat = curriculum.trainPairs.filter { !$0.assistant.contains("<tool_call>") }.prefix(2)
            let tools = curriculum.trainPairs.filter { $0.assistant.contains("<tool_call>") }.prefix(2)
            let sample = Array(chat) + Array(tools)
            let rendered = await model.debugRenderTrainPairs(sample)
            for (i, r) in rendered.enumerated() {
                func clip(_ s: String, _ n: Int) -> String {
                    String(s.prefix(n)).replacingOccurrences(of: "\n", with: "⏎")
                }
                print("\n=== PAIR \(i + 1)  boundary=\(r.start)/\(r.total) ===")
                print("FULL    : \(clip(r.full, 320))")
                print("TRAINED : \(clip(r.trained, 320))")
            }
            return
        }

        // CODE_FLYWHEEL=1 — Phase 4½ v2: the code-execution verifier axis. Generate Swift
        // solutions, compile+run them against hidden tests, distill what passes. No MCP needed;
        // the 7B has real headroom here (not saturated), so this is where LIFT can show.
        if ProcessInfo.processInfo.environment["CODE_FLYWHEEL"] != nil {
            try await runCodeFlywheel(model: model, registry: registry,
                                      cycleId: cycleId, adapterDir: adapterDir, curriculum: curriculum)
            return
        }

        // DOMAIN_FLYWHEEL=1 — v2a: reimplement a real function from the user's repo, verified by
        // that repo's OWN gradient-check test (DomainVerifier). The capability-gap surface. This
        // mode CALIBRATES (frozen-base pass@1/pass@k + verified-trace yield → is the task in the
        // 5–60% band?). THINK=1 for reasoning rollouts.
        if ProcessInfo.processInfo.environment["DOMAIN_FLYWHEEL"] != nil {
            try await runDomainFlywheel(model: model, registry: registry,
                                        cycleId: cycleId, adapterDir: adapterDir, curriculum: curriculum)
            return
        }

        // Connect an MCP server for the tool-pass@1 correctness gate.
        let host = MCPHost()
        let mcpServer = ProcessInfo.processInfo.environment["MCP_SERVER"]
            ?? "/Users/jason/localdev/ClaudeUtils/.build/release/ClaudeUtils"
        var toolBefore: Double? = nil
        if FileManager.default.fileExists(atPath: mcpServer) {
            _ = try? await host.connect(.init(command: mcpServer))
            log("evaluating tool-pass@1 (base) ...")
            toolBefore = await toolPass(model, host, EvalSuite.toolCases)
        } else { log("no MCP server at \(mcpServer) — tool-pass gate skipped.") }

        // EVAL_ONLY: load the current champion adapter and compare tool-pass to base.
        if evalOnly {
            guard let champ = registry.currentChampion() else { log("no champion to eval."); exit(1) }
            try await model.loadAdapter(directory: URL(fileURLWithPath: champ.adapterPath))
            let toolAfter = toolBefore == nil ? nil : await toolPass(model, host, EvalSuite.toolCases)
            print("""

            ============ EVAL_ONLY — base vs champion \(champ.cycleId) ============
            tool-pass@1   : base \(fmt(toolBefore)) → champion \(fmt(toolAfter))   (\(EvalSuite.toolCases.count) cases)
            ====================================================================
            """)
            await host.shutdown(); return
        }

        // FLYWHEEL=1 — Phase 4½ verifier-gated flywheel. Distill on VERIFIED self-play traces
        // (best-of-N, tool-verified) instead of raw transcripts, and measure the compounding
        // signal (pass@k before/after) on a held-out, disjoint eval set.
        if ProcessInfo.processInfo.environment["FLYWHEEL"] != nil {
            try await runFlywheel(model: model, host: host, registry: registry,
                                  cycleId: cycleId, adapterDir: adapterDir,
                                  curriculum: curriculum, toolBefore: toolBefore)
            return
        }

        // Behavioral A/B probes (base answers).
        let probes = curriculum.heldout.prefix(2).map { userTurn($0) }
        var before: [String] = []
        for p in probes { before.append(try await model.generate(p, maxTokens: 200)) }

        // Flywheel (Phase 4½): best-of-N rollouts → verify (tool-pass) → keep winners as
        // CLEAN self-play data, mixed into the curriculum. Verified > raw transcripts.
        var trainSet = curriculum.trainPairs
        if toolBefore != nil {   // MCP connected
            let verified = (try? await model.bestOfNToolTraces(
                tasks: EvalSuite.toolCases, host: host, n: 4)) ?? []
            log("flywheel: \(verified.count) verified tool-call examples added to curriculum")
            trainSet += verified
        }

        log("training LoRA cycle \(cycleId) (assistant-only masked, +flywheel) ...")
        let r = try await model.trainLoRA(
            train: trainSet, heldout: curriculum.heldout,
            retention: RetentionSet.examples, adapterDir: adapterDir)

        // Adapter is now live in the container — measure tool-pass + behavioral A/B.
        let toolAfter: Double? = toolBefore == nil ? nil : await toolPass(model, host, EvalSuite.toolCases)
        var after: [String] = []
        for p in probes { after.append(try await model.generate(p, maxTokens: 200)) }
        await host.shutdown()

        let toolHeld = (toolBefore == nil || toolAfter == nil) || (toolAfter! >= toolBefore! - 0.01)
        let promoted = r.betPasses && r.retentionHolds && toolHeld
        let timestamp = ISO8601DateFormatter().string(from: Date())
        try registry.appendHistory(.init(
            cycleId: cycleId, promoted: promoted,
            beforeHeldout: r.beforeHeldout, afterHeldout: r.afterHeldout,
            beforeRetention: r.beforeRetention, afterRetention: r.afterRetention,
            trainCount: curriculum.trainPairs.count, heldoutCount: curriculum.heldout.count,
            timestamp: timestamp))
        if promoted {
            try registry.promote(.init(
                cycleId: cycleId, adapterPath: adapterDir.path,
                heldoutLoss: r.afterHeldout, promotedAt: timestamp))
        }

        print("""

        ============ SELF-IMPROVEMENT CYCLE — \(cycleId) ============
        train/heldout   : \(curriculum.trainPairs.count) / \(curriculum.heldout.count)
        held-out loss   : \(String(format: "%.4f", r.beforeHeldout)) → \(String(format: "%.4f", r.afterHeldout))   (\(pct(r.beforeHeldout, r.afterHeldout)))   want DOWN
        retention loss  : \(String(format: "%.4f", r.beforeRetention)) → \(String(format: "%.4f", r.afterRetention))   (\(pct(r.beforeRetention, r.afterRetention)))   want ≤ +5%
        tool-pass@1     : \(fmt(toolBefore)) → \(fmt(toolAfter))   want NOT regressed
        gate            : generalization \(r.betPasses ? "✅" : "❌") · retention \(r.retentionHolds ? "✅" : "⚠️") · tools \(toolHeld ? "✅" : "❌")
        decision        : \(promoted ? "✅ PROMOTED — champion is now \(cycleId)" : "❌ REJECTED — champion unchanged")
        adapter         : \(adapterDir.path)
        ==========================================================
        """)
        print("\n— behavioral A/B (held-out prompts the model never trained on) —")
        for (i, p) in probes.enumerated() {
            func one(_ s: String) -> String { String(s.prefix(220)).replacingOccurrences(of: "\n", with: " ") }
            print("\nPROMPT \(i + 1): \(one(p))")
            print("  BEFORE: \(one(before[i]))")
            print("  AFTER : \(one(after[i]))")
        }
    }
}

func fmt(_ d: Double?) -> String { d == nil ? "n/a" : String(format: "%.0f%%", d! * 100) }

/// Phase 4½ flywheel cycle: distill on VERIFIED self-play traces (best-of-N over disjoint
/// training tasks, kept only if they call the right tool) + a clean chat anchor — NOT raw
/// transcripts. Measures the compounding signal (pass@k before/after) on the held-out eval
/// set, and gates promotion on tool-pass held + retention (the tool axis is what we trained).
func runFlywheel(
    model: MLXLanguageModel, host: MCPHost, registry: Registry,
    cycleId: String, adapterDir: URL, curriculum: Curriculum, toolBefore: Double?
) async throws {
    guard let g0 = toolBefore else { log("✗ FLYWHEEL needs an MCP server connected."); exit(1) }
    let eval = EvalSuite.toolCases
    let ks = [1, 2, 4, 8]

    func toolName(_ a: String) -> String? {
        guard let r = a.range(of: "\"name\": \"") else { return nil }
        let rest = a[r.upperBound...]
        guard let e = rest.firstIndex(of: "\"") else { return nil }
        return String(rest[..<e])
    }
    func curve(_ m: [Int: Double]) -> String {
        ks.map { "@\($0)=" + String(format: "%.0f%%", (m[$0] ?? 0) * 100) }.joined(separator: " ")
    }

    // BEFORE: greedy pass@1 (= g0) + sampled pass@k curve on the held-out cases.
    log("flywheel: measuring BEFORE pass@k on \(eval.count) held-out cases ...")
    var passBefore: [Int: Double] = [:]
    for k in ks { passBefore[k] = await model.toolPassAtK(cases: eval, host: host, k: k) }

    // Generate verified self-play traces over the DISJOINT training tasks.
    log("flywheel: generating verified traces (best-of-8 over \(EvalSuite.flywheelTrainTasks.count) tasks) ...")
    let verified = (try? await model.bestOfNToolTraces(
        tasks: EvalSuite.flywheelTrainTasks, host: host, n: 8)) ?? []
    let toolsCovered = Set(verified.compactMap { toolName($0.assistant) })
    log("flywheel: \(verified.count) unique verified traces covering \(toolsCovered.count) tools")
    guard verified.count >= 3 else {
        log("✗ too few verified traces (\(verified.count))."); await host.shutdown(); exit(1)
    }

    // Curriculum = TOOL-DOMINANT verified traces + a modest clean chat anchor. Empirically
    // (selfloop5 vs 7) tool-calling survives a LoRA cycle only when tool examples DOMINATE;
    // a balanced/chat-heavy mix collapses tools to ~0% even when the traces are clean. So
    // oversample the (few, low-arg-variety) verified traces to dominate, with a small chat
    // anchor for voice. Verified > raw, masking is fixed → expect tools held AND coherent chat.
    let toolTarget = 150
    var tools: [TrainPair] = []
    while tools.count < toolTarget && !verified.isEmpty { tools += verified }
    tools = Array(tools.prefix(toolTarget))
    let chatAnchor = curriculum.trainPairs.filter { !$0.assistant.contains("<tool_call>") }
    let anchor = Array(chatAnchor.prefix(50))
    let trainSet = tools + anchor
    log("flywheel: distilling on \(tools.count) verified (oversampled from \(verified.count) unique) "
        + "+ \(anchor.count) chat-anchor = \(trainSet.count) ...")

    // Behavioral A/B base answers (does freeform collapse after distill?).
    let probes = curriculum.heldout.prefix(2).map { userTurn($0) }
    var before: [String] = []
    for p in probes { before.append(try await model.generate(p, maxTokens: 200)) }

    // Distill.
    let r = try await model.trainLoRA(
        train: trainSet, heldout: curriculum.heldout,
        retention: RetentionSet.examples, adapterDir: adapterDir)

    // AFTER: greedy pass@1 + sampled pass@k + A/B.
    let gAfter = await toolPass(model, host, eval)
    var passAfter: [Int: Double] = [:]
    for k in ks { passAfter[k] = await model.toolPassAtK(cases: eval, host: host, k: k) }
    var after: [String] = []
    for p in probes { after.append(try await model.generate(p, maxTokens: 200)) }
    await host.shutdown()

    let toolHeld = gAfter >= g0 - 0.01
    let promoted = toolHeld && r.retentionHolds
    let timestamp = ISO8601DateFormatter().string(from: Date())
    try registry.appendHistory(.init(
        cycleId: cycleId, promoted: promoted,
        beforeHeldout: r.beforeHeldout, afterHeldout: r.afterHeldout,
        beforeRetention: r.beforeRetention, afterRetention: r.afterRetention,
        trainCount: trainSet.count, heldoutCount: curriculum.heldout.count,
        timestamp: timestamp))
    if promoted {
        try registry.promote(.init(
            cycleId: cycleId, adapterPath: adapterDir.path,
            heldoutLoss: r.afterHeldout, promotedAt: timestamp))
    }

    print("""

    ============ FLYWHEEL CYCLE (Phase 4½) — \(cycleId) ============
    distilled on    : \(tools.count) verified (\(verified.count) unique, oversampled) + \(anchor.count) chat-anchor
    tool coverage   : \(toolsCovered.count)/5 claude-utils tools
    greedy pass@1   : \(fmt(g0)) → \(fmt(gAfter))   want NOT regressed
    sampled pass@k  : BEFORE  \(curve(passBefore))
                      AFTER   \(curve(passAfter))   want ↑ or k-for-fixed-pass ↓
    retention loss  : \(String(format: "%.4f", r.beforeRetention)) → \(String(format: "%.4f", r.afterRetention))   (\(pct(r.beforeRetention, r.afterRetention)))   want ≤ +5%
    held-out (chat) : \(String(format: "%.4f", r.beforeHeldout)) → \(String(format: "%.4f", r.afterHeldout))   (\(pct(r.beforeHeldout, r.afterHeldout)))   (info only)
    gate            : tools \(toolHeld ? "✅" : "❌") · retention \(r.retentionHolds ? "✅" : "⚠️")
    decision        : \(promoted ? "✅ PROMOTED — champion is now \(cycleId)" : "❌ REJECTED — champion unchanged")
    adapter         : \(adapterDir.path)
    ============================================================
    """)
    print("\n— behavioral A/B (held-out prompts; checks freeform didn't collapse) —")
    for (i, p) in probes.enumerated() {
        func one(_ s: String) -> String { String(s.prefix(220)).replacingOccurrences(of: "\n", with: " ") }
        print("\nPROMPT \(i + 1): \(one(p))")
        print("  BEFORE: \(one(before[i]))")
        print("  AFTER : \(one(after[i]))")
    }
}

/// Phase 4½ v2 — the CODE-EXECUTION flywheel: generate Swift solutions, COMPILE + RUN them
/// against each task's hidden test (the most reliable verifier tier), distill what passes.
///   CODE_FLYWHEEL=1          answer-only (direct) rollouts.
///   CODE_FLYWHEEL=1 THINK=1  STaR/CoT: rollouts reason in <think>…</think> then answer; the
///                            verifier grades ONLY the post-</think> answer (outcome reward);
///                            the WHOLE winning trace (reasoning + answer) is distilled.
/// Gate (collapse-proof, per the CoT design): exec-pass not regressed AND retention held AND a
/// DIVERSITY FLOOR (after-pass@8 must not fall below the FROZEN base pass@8 — catches the
/// pass@1→pass@8 "convergence by collapse" failure). In THINK mode, also a CAUSAL LOAD-BEARING
/// ablation: think-pass@1 must beat no-think-pass@1 on the SAME adapter, else the reasoning is
/// decorative filler. No MCP host needed.
func runCodeFlywheel(
    model: MLXLanguageModel, registry: Registry,
    cycleId: String, adapterDir: URL, curriculum: Curriculum
) async throws {
    let env = ProcessInfo.processInfo.environment
    let think = env["THINK"] != nil
    let hard = env["HARD"] != nil   // HARD=1 → LeetCode-medium tasks that REQUIRE reasoning
    let mode = (think ? "CODE-FLYWHEEL+CoT (STaR)" : "CODE-FLYWHEEL") + (hard ? " [HARD]" : "")
    let eval = hard ? CodeEvalSuite.evalHardTasks : CodeEvalSuite.evalTasks
    let trainTasks = hard ? CodeEvalSuite.trainHardTasks : CodeEvalSuite.trainTasks
    // Env knobs — right-size for slow reasoning bases (R1-32B reasons long at ~12 tok/s):
    //   FLYWHEEL_N=4 best-of-N · FLYWHEEL_MAXTOK=2048 (give reasoning room) · FLYWHEEL_KS=1,4
    let n = env["FLYWHEEL_N"].flatMap { Int($0) } ?? 8
    let maxTok = env["FLYWHEEL_MAXTOK"].flatMap { Int($0) } ?? 700
    let ks = env["FLYWHEEL_KS"]?.split(separator: ",").compactMap { Int($0) }.sorted()
        ?? [1, 2, 4, 8]
    let kMax = ks.max() ?? 8   // diversity-floor reference (frozen-base pass@kMax)
    // Sampling — R1-distill REQUIRES temp≈0.6 / topP 0.95 / repPen OFF (greedy degenerates into
    // repetition; DeepSeek model card). Set FLYWHEEL_TEMP=0.6 FLYWHEEL_TOPP=0.95 FLYWHEEL_REPPEN=1.0
    // for reasoning bases. Default (unset) = greedy headline + 0.7 sampled (coder bases).
    let passTemp = env["FLYWHEEL_TEMP"].flatMap { Float($0) } ?? 0.0    // headline pass@1 + ablation
    let sampleTemp = env["FLYWHEEL_TEMP"].flatMap { Float($0) } ?? 0.7  // sampled pass@k + best-of-N
    let topP = env["FLYWHEEL_TOPP"].flatMap { Float($0) } ?? 1.0
    let repPen = env["FLYWHEEL_REPPEN"].flatMap { Float($0) } ?? 1.15
    func curve(_ m: [Int: Double]) -> String {
        ks.map { "@\($0)=" + String(format: "%.0f%%", (m[$0] ?? 0) * 100) }.joined(separator: " ")
    }

    log("\(mode): measuring BEFORE exec-pass on \(eval.count) held-out tasks (think=\(think), N=\(n), maxTok=\(maxTok), ks=\(ks), passT=\(passTemp), sampT=\(sampleTemp), topP=\(topP), repPen=\(repPen)) ...")
    let g0 = await model.codePass(tasks: eval, think: think, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    var passBefore: [Int: Double] = [:]   // passBefore[kMax] = FROZEN base pass@kMax (diversity reference)
    for k in ks { passBefore[k] = await model.codePassAtK(tasks: eval, k: k, think: think, temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) }
    log("\(mode): BEFORE pass@1(T=\(passTemp))=\(fmt(g0)) · pass@k \(curve(passBefore))")

    log("\(mode): generating verified \(think ? "reasoning traces" : "solutions") (best-of-\(n) over \(trainTasks.count) tasks) ...")
    let gen = try await model.bestOfNCodeTraces(tasks: trainTasks, think: think, n: n, temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    log("\(mode): \(gen.passed)/\(gen.attempted) rollouts passed compile+test (\(gen.truncated) truncated mid-think) → \(gen.traces.count) unique verified traces")
    guard gen.traces.count >= 3 else { log("✗ too few verified traces (\(gen.traces.count))."); exit(1) }

    // GENTLE curriculum: the base is ALREADY competent at code, so heavy oversampling OVERFITS and
    // forgets held-out breadth (observed 83%→17%). Nudge — light oversampling, balanced with chat,
    // fewer/softer steps. (Reasoning traces are longer/more stylized → even more collapse-prone.)
    let target = 60
    var code: [TrainPair] = []
    while code.count < target && !gen.traces.isEmpty { code += gen.traces }
    code = Array(code.prefix(target))
    let chatAnchor = curriculum.trainPairs.filter { !$0.assistant.contains("<tool_call>") }
    let anchor = Array(chatAnchor.prefix(60))
    let trainSet = code + anchor
    log("\(mode): distilling (gentle) on \(code.count) verified (\(gen.traces.count) unique) + \(anchor.count) chat-anchor = \(trainSet.count) ...")

    let probes = curriculum.heldout.prefix(2).map { userTurn($0) }
    var before: [String] = []
    for p in probes { before.append(try await model.generate(p, maxTokens: 512, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) }

    var cfg = MLXLanguageModel.LoRAConfig()
    cfg.iterations = 120
    cfg.learningRate = 5e-5
    let r = try await model.trainLoRA(
        train: trainSet, heldout: curriculum.heldout,
        retention: RetentionSet.examples, adapterDir: adapterDir, config: cfg)

    log("\(mode): measuring AFTER exec-pass ...")
    let gAfter = await model.codePass(tasks: eval, think: think, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    var passAfter: [Int: Double] = [:]
    for k in ks { passAfter[k] = await model.codePassAtK(tasks: eval, k: k, think: think, temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) }
    // CAUSAL load-bearing ablation (THINK only): same trained adapter + same sampler, reasoning OFF.
    let gAblate: Double? = think ? await model.codePass(tasks: eval, think: false, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) : nil
    var after: [String] = []
    for p in probes { after.append(try await model.generate(p, maxTokens: 512, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) }

    let codeHeld = gAfter >= g0 - 0.01
    let diversityFloor = (passAfter[kMax] ?? 0) >= (passBefore[kMax] ?? 0) - 0.001   // pass@kMax not collapsed
    // Reasoning must be load-bearing: think beats no-think on the same adapter (else decorative).
    let loadBearing = !think || (gAblate.map { gAfter > $0 + 0.001 } ?? true)
    let promoted = codeHeld && r.retentionHolds && diversityFloor && loadBearing
    let timestamp = ISO8601DateFormatter().string(from: Date())
    try registry.appendHistory(.init(
        cycleId: cycleId, promoted: promoted,
        beforeHeldout: r.beforeHeldout, afterHeldout: r.afterHeldout,
        beforeRetention: r.beforeRetention, afterRetention: r.afterRetention,
        trainCount: trainSet.count, heldoutCount: curriculum.heldout.count, timestamp: timestamp))
    if promoted {
        try registry.promote(.init(cycleId: cycleId, adapterPath: adapterDir.path,
            heldoutLoss: r.afterHeldout, promotedAt: timestamp))
    }

    print("""

    ============ \(mode) — \(cycleId) ============
    distilled on    : \(code.count) verified (\(gen.traces.count) unique) + \(anchor.count) chat-anchor\(think ? "  [reasoning traces]" : "")
    rollouts        : \(gen.passed)/\(gen.attempted) passed compile+test (\(gen.truncated) truncated mid-think)
    exec-pass@1     : \(fmt(g0)) → \(fmt(gAfter))   (T=\(passTemp))   want ↑ (headroom: not saturated)
    exec-pass@k     : BEFORE  \(curve(passBefore))
                      AFTER   \(curve(passAfter))   want ↑ or k-for-fixed-pass ↓
    diversity floor : after-pass@\(kMax) \(fmt(passAfter[kMax])) vs frozen-base pass@\(kMax) \(fmt(passBefore[kMax]))   \(diversityFloor ? "✅" : "❌ COLLAPSE")
    \(think ? "load-bearing  : think-pass@1 \(fmt(gAfter)) vs no-think \(fmt(gAblate))   \(loadBearing ? "✅ reasoning helps" : "❌ decorative filler")" : "")
    retention loss  : \(String(format: "%.4f", r.beforeRetention)) → \(String(format: "%.4f", r.afterRetention))   (\(pct(r.beforeRetention, r.afterRetention)))   want ≤ +5%
    gate            : exec-pass \(codeHeld ? "✅" : "❌") · retention \(r.retentionHolds ? "✅" : "⚠️") · diversity \(diversityFloor ? "✅" : "❌")\(think ? " · load-bearing \(loadBearing ? "✅" : "❌")" : "")
    decision        : \(promoted ? "✅ PROMOTED — champion is now \(cycleId)" : "❌ REJECTED — champion unchanged")
    adapter         : \(adapterDir.path)
    ============================================================
    """)
    print("\n— behavioral A/B (held-out chat prompts; checks freeform didn't collapse) —")
    for (i, p) in probes.enumerated() {
        func one(_ s: String) -> String { String(s.prefix(220)).replacingOccurrences(of: "\n", with: " ") }
        print("\nPROMPT \(i + 1): \(one(p))")
        print("  BEFORE: \(one(before[i]))")
        print("  AFTER : \(one(after[i]))")
    }
}

/// v2a — the DOMAIN flywheel. CALIBRATION mode (step 4): measure the FROZEN base's pass@1/pass@k
/// and verified-trace yield on real backward-pass reimplementation, verified by the repo's own
/// gradient checks. The decision-gate for the whole v2a thesis: a task earns a distill iteration
/// only if it sits in the 5–60% capability-gap band (pass@1<100% so a gap remains; pass@N>0 so ≥1
/// verified trace exists to bootstrap). THINK=1 → reasoning rollouts (the verifier grades only the
/// post-</think> answer). No MCP host; no training in this mode (that's step 5).
func runDomainFlywheel(
    model: MLXLanguageModel, registry: Registry,
    cycleId: String, adapterDir: URL, curriculum: Curriculum
) async throws {
    let env = ProcessInfo.processInfo.environment
    let think = env["THINK"] != nil
    let mode = think ? "DOMAIN-FLYWHEEL+CoT (STaR)" : "DOMAIN-FLYWHEEL"
    let tasks = DomainEvalSuite.active
    // Train/held-out split: DOMAIN_HOLDOUT=Id1,Id2 marks those tasks EVAL-ONLY; the rest are TRAIN.
    // Empty ⇒ in-distribution (train == eval). A held-out split makes memorization IMPOSSIBLE by
    // construction — any lift on a task the distill never saw is genuine skill transfer.
    let holdoutIds = Set((env["DOMAIN_HOLDOUT"] ?? "").split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty })
    let evalTasks = holdoutIds.isEmpty ? tasks : tasks.filter { holdoutIds.contains($0.id) }
    let trainTasks = holdoutIds.isEmpty ? tasks : tasks.filter { !holdoutIds.contains($0.id) }
    let heldOut = !holdoutIds.isEmpty
    let evalSamples = Int(env["DOMAIN_EVAL_SAMPLES"] ?? "") ?? 8   // multi-sample pass@1 (kills the noise)
    let baseName = env["SWIFTLM_MODEL"] ?? "Qwen2.5-Coder-7B-Instruct-4bit (default)"

    // Cluster TracePool: this box's LocalPool + any remote workers (CLUSTER_WORKERS="host:port,host:port").
    // Generation fans across all workers; verification stays on this coordinator. Local-only if unset.
    let localPool = model.makeLocalPool()
    let remoteWorkers: [TracePool] = (env["CLUSTER_WORKERS"] ?? "").split(separator: ",").compactMap { spec in
        let p = spec.split(separator: ":")
        guard p.count == 2, let port = UInt16(p[1]) else { return nil }
        return RemotePool(host: String(p[0]), port: port,
                          descriptor: WorkerDescriptor(id: "remote-\(spec)", models: [ModelID(model.modelId)],
                                      batchWidth: 12, effectiveParallelism: 1, estTokensPerSecPerStream: 12))
    }
    let pool: TracePool = remoteWorkers.isEmpty ? localPool : FanOutPool(workers: [localPool] + remoteWorkers)
    if !remoteWorkers.isEmpty {
        log("CLUSTER: generation fans across \(1 + remoteWorkers.count) workers (this box + \(remoteWorkers.count) remote)")
    }

    // Env knobs (calibration wants N≈16–50; reasoning bases need room + R1 sampling):
    let n = env["FLYWHEEL_N"].flatMap { Int($0) } ?? 16
    let maxTok = env["FLYWHEEL_MAXTOK"].flatMap { Int($0) } ?? (think ? 8192 : 1500)
    let ks = env["FLYWHEEL_KS"]?.split(separator: ",").compactMap { Int($0) }.sorted() ?? [1, 4, 8, 16]
    let passTemp = env["FLYWHEEL_TEMP"].flatMap { Float($0) } ?? 0.0
    let sampleTemp = env["FLYWHEEL_TEMP"].flatMap { Float($0) } ?? 0.7
    let topP = env["FLYWHEEL_TOPP"].flatMap { Float($0) } ?? 0.95
    let repPen = env["FLYWHEEL_REPPEN"].flatMap { Float($0) } ?? 1.0   // OFF by default (R1-safe)
    func curve(_ m: [Int: Double]) -> String {
        ks.map { "@\($0)=" + String(format: "%.0f%%", (m[$0] ?? 0) * 100) }.joined(separator: " ")
    }

    guard !tasks.isEmpty else { log("✗ no active domain tasks (need a leak-stripped prompt)."); exit(1) }
    log("\(mode) calibration: \(tasks.count) active task(s) [\(tasks.map(\.id).joined(separator: ", "))], base=\(baseName)")
    log("warming verifier (shared module cache) ...")
    let prep = DomainVerifier.prepare(repo: DomainEvalSuite.llmFromScratch)
    guard prep.passed else { log("✗ verifier warm failed: \(prep.diagnostics)"); exit(1) }
    log("verifier ready (\(prep.diagnostics))")

    log("\(mode): task-quality gate ...")
    for t in tasks {
        let q = DomainVerifier.qualityGate(task: t)
        log("  [\(t.id)] quality \(q.ok ? "OK" : "FAIL") — " + q.report.replacingOccurrences(of: "\n", with: " ; "))
    }

    // DOMAIN_DEBUG=N — dump N raw rollouts + extracted body + verifier stage/diagnostics, then
    // exit. Distinguishes "compiles-but-wrong-gradient" (real capability gap) from "doesn't compile"
    // (API-name mismatch → prompt/harness problem, NOT a gap).
    if let dbg = env["DOMAIN_DEBUG"] {
        let nDump = Int(dbg) ?? 3
        for t in tasks {
            log("DEBUG: \(nDump) rollouts for \(t.id) (T=\(sampleTemp)) ...")
            for i in 1...nDump {
                let out = (try? await model.generate(t.prompt ?? "", maxTokens: maxTok,
                           temperature: sampleTemp, topP: topP, repetitionPenalty: repPen)) ?? ""
                let res = DomainVerifier.check(candidate: out, task: t)
                print("\n=== [\(t.id)] sample \(i): \(res.passed ? "PASS" : "FAIL @ \(res.stage.rawValue)") (raw \(out.count) chars) ===")
                print("RAW: " + String(out.prefix(700)).replacingOccurrences(of: "\n", with: "⏎"))
                print("--- extracted body ---\n" + String(DomainVerifier.extractBody(out).prefix(500)))
                if !res.passed { print("--- diag ---\n" + String(res.diagnostics.prefix(800))) }
            }
        }
        return
    }

    let splitNote = heldOut
        ? "HELD-OUT split: train[\(trainTasks.map(\.id).joined(separator: ","))] → eval[\(evalTasks.map(\.id).joined(separator: ","))]"
        : "in-distribution (train == eval)"
    log("\(mode): \(splitNote)")
    log("\(mode): BEFORE pass@1 on eval (\(evalSamples)-sample, T=\(passTemp), think=\(think), maxTok=\(maxTok)) ...")
    let g1 = await model.domainPass(tasks: evalTasks, think: think, samples: evalSamples, temperature: passTemp,
                                    topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    var passK: [Int: Double] = [:]
    for k in ks {
        log("\(mode): eval pass@\(k) (sampled T=\(sampleTemp)) ...")
        passK[k] = await model.domainPassAtK(tasks: evalTasks, k: k, think: think, temperature: sampleTemp,
                                             topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    }
    // DPO generates PREFERENCE PAIRS in its own branch — skip the (separate) verified-trace gen then.
    let isDPO = env["DOMAIN_DISTILL"] != nil && env["DOMAIN_DPO"] != nil
    let gen: (traces: [TrainPair], passed: Int, attempted: Int, truncated: Int, perTask: [TaskYield])
    if isDPO {
        gen = ([], 0, 0, 0, [])
    } else {
        log("\(mode): best-of-\(n) verified traces from TRAIN tasks ...")
        gen = try await model.bestOfNDomainTraces(tasks: trainTasks, think: think, n: n, temperature: sampleTemp,
                                                  topP: topP, repetitionPenalty: repPen, maxTokens: maxTok, pool: pool)
    }

    let kMax = ks.max() ?? n

    // ── Calibration-only (default): classify the gap band and report. ────────────────────────────
    guard env["DOMAIN_DISTILL"] != nil else {
        let pN = passK[kMax] ?? 0
        // per-task difficulty-calibration table — which of the (now broad) task set sit in the
        // capability-gap band (0<won<n). This is the auto-discovery payoff: many in-band tasks.
        if !gen.perTask.isEmpty {
            log("\(mode): per-task yield over \(gen.perTask.count) train tasks (n=\(n)) —")
            for y in gen.perTask.sorted(by: { $0.won > $1.won }) {
                log("    [\(y.band == "in-band" ? "✅" : "  ")] \(y.id): \(y.won)/\(y.n) verified  (\(y.band))")
            }
            let nb = gen.perTask.filter { $0.band == "in-band" }.count
            let nd = gen.perTask.filter { $0.band == "dry" }.count
            let ns = gen.perTask.filter { $0.band == "saturated" }.count
            log("\(mode): BAND SUMMARY — \(nb) in-band / \(nd) dry / \(ns) saturated of \(gen.perTask.count) tasks")
        }
        let band: String
        if pN <= 0.0001 && gen.traces.isEmpty {
            band = "❌ TOO HARD — pass@\(kMax)=0 and best-of-\(n) yielded 0 verified traces; no bootstrap. Try a stronger base (R1-32B) or activate easier fns."
        } else if g1 >= 0.9999 {
            band = "❌ SATURATED — pass@1=100%; no gap to distill. Pick a harder fn (LayerNorm/Attention)."
        } else {
            band = "✅ IN THE CAPABILITY-GAP BAND — pass@1<100% AND verified traces exist ⇒ distillable. Run DOMAIN_DISTILL=1 for step 5."
        }
        print("""

        ============ \(mode) CALIBRATION — \(cycleId) ============
        split           : \(splitNote)
        eval pass@1     : \(fmt(g1))   (\(evalSamples)-sample)
        eval pass@k     : \(curve(passK))
        base model      : \(baseName)
        sampling        : passT=\(passTemp) sampleT=\(sampleTemp) topP=\(topP) repPen=\(repPen) maxTok=\(maxTok) think=\(think)
        train best-of-\(n): \(gen.passed)/\(gen.attempted) rollouts verified\(think ? " (\(gen.truncated) truncated mid-think)" : "") → \(gen.traces.count) unique verified traces
        capability gap  : \(band)
        ============================================================
        """)
        return
    }

    // ── DPO path (step 5b): preference learning on (verified, failed) pairs — the v2a YIELD fix. ──
    // SFT-on-positives starves (3 unique); DPO uses the abundant FAILED rollouts as negatives and
    // targets "pass@k holds, greedy mis-selects". Eval lift on the HELD-OUT fn = genuine transfer.
    if env["DOMAIN_DPO"] != nil {
        log("\(mode): DPO — best-of-\(n) → (verified, failed) preference pairs from TRAIN tasks ...")
        let pg = try await model.bestOfNDomainPairs(tasks: trainTasks, think: think, n: n,
                    temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok, pool: pool)
        // ── difficulty-calibration report (baked in): per-task yield → which tasks are in the
        // capability-gap band (0<won<n, the ones that actually contribute training signal). ──
        let inBand = pg.perTask.filter { $0.band == "in-band" }
        let dry = pg.perTask.filter { $0.band == "dry" }, sat = pg.perTask.filter { $0.band == "saturated" }
        log("\(mode): per-task yield (n=\(n)) —")
        for y in pg.perTask.sorted(by: { $0.won > $1.won }) {
            log("    [\(y.band == "in-band" ? "✅" : "  ")] \(y.id): \(y.won)/\(y.n) verified, \(y.pairs) pairs  (\(y.band))")
        }
        log("\(mode): CALIBRATION — \(inBand.count) in-band / \(dry.count) dry / \(sat.count) saturated of \(pg.perTask.count) train tasks → \(pg.verified) verified + \(pg.failed) failed → \(pg.pairs.count) preference pairs")
        guard pg.pairs.count >= 4 else { log("✗ too few preference pairs (\(pg.pairs.count))."); exit(1) }

        var cfg = MLXLanguageModel.LoRAConfig()
        cfg.iterations = Int(env["DOMAIN_ITERS"] ?? "") ?? 100
        cfg.learningRate = Float(env["DOMAIN_LR"] ?? "") ?? 1e-5   // DPO wants a LOW lr
        // DPO retains BOTH chosen+rejected forward graphs at once → batchSize 1 keeps a 32B in memory
        // (= the SFT distill's batch-2 single-forward footprint, which is known to complete). maxSeqLen
        // caps the long FAILED rejected traces (they ran to maxTok). Both overridable for smaller bases.
        cfg.batchSize = Int(env["DOMAIN_BATCH"] ?? "") ?? 1
        cfg.maxSeqLen = Int(env["DOMAIN_MAXLEN"] ?? "") ?? 2048
        let beta = Float(env["DOMAIN_BETA"] ?? "") ?? 0.1
        let probes = curriculum.heldout.prefix(2).map { userTurn($0) }
        var beforeProbe: [String] = []
        for p in probes { beforeProbe.append((try? await model.generate(p, maxTokens: 300, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) ?? "") }

        let r = try await model.trainDPO(pairs: pg.pairs, heldout: curriculum.heldout,
                    retention: RetentionSet.examples, adapterDir: adapterDir, config: cfg, beta: beta)

        log("\(mode): measuring AFTER pass-rates on eval ...")
        let g1After = await model.domainPass(tasks: evalTasks, think: think, samples: evalSamples, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
        var passKAfter: [Int: Double] = [:]
        for k in ks { passKAfter[k] = await model.domainPassAtK(tasks: evalTasks, k: k, think: think, temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) }
        let g1Ablate: Double? = think ? await model.domainPass(tasks: evalTasks, think: false, samples: evalSamples, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) : nil
        var afterProbe: [String] = []
        for p in probes { afterProbe.append((try? await model.generate(p, maxTokens: 300, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) ?? "") }

        let lift = g1After > g1 + 0.02
        let diversityFloor = (passKAfter[kMax] ?? 0) >= (passK[kMax] ?? 0) - 0.001
        let loadBearing = !think || (g1Ablate.map { g1After > $0 + 0.02 } ?? true)
        let gatePass = lift && r.retentionHolds && diversityFloor && loadBearing
        let timestamp = ISO8601DateFormatter().string(from: Date())
        try registry.appendHistory(.init(cycleId: cycleId, promoted: false,
            beforeHeldout: r.beforeHeldout, afterHeldout: r.afterHeldout,
            beforeRetention: r.beforeRetention, afterRetention: r.afterRetention,
            trainCount: pg.pairs.count, heldoutCount: curriculum.heldout.count, timestamp: timestamp))
        print("""

        ============ \(mode) DPO (step 5b) — \(cycleId) ============
        split           : \(splitNote)\(heldOut ? "  ← held-out lift = GENUINE transfer" : "")
        base model      : \(baseName)
        trained on      : \(pg.pairs.count) pref pairs (\(pg.verified) verified + \(pg.failed) failed), β=\(beta), lr=\(cfg.learningRate)
        EVAL pass@1     : \(fmt(g1)) → \(fmt(g1After))   (\(evalSamples)-sample)   want ↑  (THE LIFT NUMBER)
        EVAL pass@k     : BEFORE \(curve(passK))
                          AFTER  \(ks.map { "@\($0)=" + String(format: "%.0f%%", (passKAfter[$0] ?? 0) * 100) }.joined(separator: " "))
        diversity floor : after-pass@\(kMax) \(fmt(passKAfter[kMax])) vs before \(fmt(passK[kMax]))   \(diversityFloor ? "✅" : "❌")
        \(think ? "load-bearing  : think-pass@1 \(fmt(g1After)) vs no-think \(fmt(g1Ablate))   \(loadBearing ? "✅" : "❌ decorative")" : "")
        retention loss  : \(String(format: "%.4f", r.beforeRetention)) → \(String(format: "%.4f", r.afterRetention))   (\(pct(r.beforeRetention, r.afterRetention)))
        gate            : lift \(lift ? "✅" : "❌") · retention \(r.retentionHolds ? "✅" : "⚠️") · diversity \(diversityFloor ? "✅" : "❌")\(think ? " · load-bearing \(loadBearing ? "✅" : "❌")" : "")
        verdict         : \(gatePass ? "✅ GATE PASS — DPO lifted \(heldOut ? "HELD-OUT" : "in-distribution") pass@1" : "❌ gate fail")
        adapter         : \(adapterDir.path)
        ============================================================
        """)
        print("\n— behavioral A/B (held-out chat) —")
        for (i, p) in probes.enumerated() {
            func one(_ s: String) -> String { String(s.prefix(180)).replacingOccurrences(of: "\n", with: " ") }
            print("\nPROMPT \(i + 1): \(one(p))\n  BEFORE: \(one(beforeProbe[i]))\n  AFTER : \(one(afterProbe[i]))")
        }
        return
    }

    // ── SFT DISTILL (step 5): best-of-N traces from TRAIN tasks → LoRA distill → measure lift. ─────
    // With a held-out split, eval lift is GENUINE skill transfer (the distill never saw the eval fn).
    guard gen.traces.count >= 2 else { log("✗ too few verified traces (\(gen.traces.count)) to distill."); exit(1) }
    // Cap oversampling at ~2× unique (the prior run's 12×-of-2-traces was pure memorization).
    let target = min(Int(env["DOMAIN_TARGET"] ?? "") ?? 48, max(8, gen.traces.count * 2))
    var distill: [TrainPair] = []
    while distill.count < target && !gen.traces.isEmpty { distill += gen.traces }
    distill = Array(distill.prefix(target))
    let anchorN = Int(env["DOMAIN_ANCHOR"] ?? "") ?? 48
    let chatAnchor = Array(curriculum.trainPairs.filter { !$0.assistant.contains("<tool_call>") }.prefix(anchorN))
    let trainSet = distill + chatAnchor
    log("\(mode): distilling on \(distill.count) verified (\(gen.traces.count) unique) + \(chatAnchor.count) chat-anchor = \(trainSet.count) ...")

    let probes = curriculum.heldout.prefix(2).map { userTurn($0) }
    var beforeProbe: [String] = []
    for p in probes { beforeProbe.append((try? await model.generate(p, maxTokens: 300, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) ?? "") }

    var cfg = MLXLanguageModel.LoRAConfig()
    cfg.iterations = Int(env["DOMAIN_ITERS"] ?? "") ?? 100
    cfg.learningRate = Float(env["DOMAIN_LR"] ?? "") ?? 5e-5
    let r = try await model.trainLoRA(
        train: trainSet, heldout: curriculum.heldout,
        retention: RetentionSet.examples, adapterDir: adapterDir, config: cfg)

    log("\(mode): measuring AFTER pass-rates on eval ...")
    let g1After = await model.domainPass(tasks: evalTasks, think: think, samples: evalSamples, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok)
    var passKAfter: [Int: Double] = [:]
    for k in ks { passKAfter[k] = await model.domainPassAtK(tasks: evalTasks, k: k, think: think, temperature: sampleTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) }
    // CAUSAL load-bearing ablation (THINK): same trained adapter, reasoning OFF — must drop.
    let g1Ablate: Double? = think ? await model.domainPass(tasks: evalTasks, think: false, samples: evalSamples, temperature: passTemp, topP: topP, repetitionPenalty: repPen, maxTokens: maxTok) : nil
    var afterProbe: [String] = []
    for p in probes { afterProbe.append((try? await model.generate(p, maxTokens: 300, temperature: passTemp, topP: topP, repetitionPenalty: repPen)) ?? "") }

    let lift = g1After > g1 + 0.02                                                // eval pass@1 up beyond noise
    let diversityFloor = (passKAfter[kMax] ?? 0) >= (passK[kMax] ?? 0) - 0.001    // pass@kMax not collapsed
    let loadBearing = !think || (g1Ablate.map { g1After > $0 + 0.02 } ?? true)    // reasoning still load-bearing
    let gatePass = lift && r.retentionHolds && diversityFloor && loadBearing
    let timestamp = ISO8601DateFormatter().string(from: Date())
    try registry.appendHistory(.init(
        cycleId: cycleId, promoted: false,   // base differs from the 7B champion → never auto-swap here
        beforeHeldout: r.beforeHeldout, afterHeldout: r.afterHeldout,
        beforeRetention: r.beforeRetention, afterRetention: r.afterRetention,
        trainCount: trainSet.count, heldoutCount: curriculum.heldout.count, timestamp: timestamp))

    print("""

    ============ \(mode) DISTILL (step 5) — \(cycleId) ============
    split           : \(splitNote)\(heldOut ? "  ← held-out lift = GENUINE transfer" : "  ← in-distribution (partly memorization)")
    base model      : \(baseName)
    distilled on    : \(distill.count) verified (\(gen.traces.count) unique) + \(chatAnchor.count) chat-anchor
    train rollouts  : \(gen.passed)/\(gen.attempted) verified\(think ? " (\(gen.truncated) truncated mid-think)" : "")
    EVAL pass@1     : \(fmt(g1)) → \(fmt(g1After))   (\(evalSamples)-sample, T=\(passTemp))   want ↑  (THE LIFT NUMBER)
    EVAL pass@k     : BEFORE \(curve(passK))
                      AFTER  \(ks.map { "@\($0)=" + String(format: "%.0f%%", (passKAfter[$0] ?? 0) * 100) }.joined(separator: " "))
    diversity floor : after-pass@\(kMax) \(fmt(passKAfter[kMax])) vs before \(fmt(passK[kMax]))   \(diversityFloor ? "✅" : "❌ COLLAPSE")
    \(think ? "load-bearing  : think-pass@1 \(fmt(g1After)) vs no-think \(fmt(g1Ablate))   \(loadBearing ? "✅ reasoning helps" : "❌ decorative")" : "")
    retention loss  : \(String(format: "%.4f", r.beforeRetention)) → \(String(format: "%.4f", r.afterRetention))   (\(pct(r.beforeRetention, r.afterRetention)))
    gate            : lift \(lift ? "✅" : "❌") · retention \(r.retentionHolds ? "✅" : "⚠️") · diversity \(diversityFloor ? "✅" : "❌")\(think ? " · load-bearing \(loadBearing ? "✅" : "❌")" : "")
    verdict         : \(gatePass ? "✅ GATE PASS — flywheel lifted on \(heldOut ? "HELD-OUT" : "in-distribution") domain task(s)" : "❌ gate fail") (champion NOT swapped — R1-32B adapter ≠ 7B champion base; cross-base promotion deferred)
    adapter         : \(adapterDir.path)
    ============================================================
    """)
    print("\n— behavioral A/B (held-out chat prompts; freeform didn't collapse?) —")
    for (i, p) in probes.enumerated() {
        func one(_ s: String) -> String { String(s.prefix(200)).replacingOccurrences(of: "\n", with: " ") }
        print("\nPROMPT \(i + 1): \(one(p))")
        print("  BEFORE: \(one(beforeProbe[i]))")
        print("  AFTER : \(one(afterProbe[i]))")
    }
}
