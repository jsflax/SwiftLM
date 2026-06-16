import Foundation
import MLXLMCommon
import SelfImprove
import Orchestration

// ── v2a: the DOMAIN flywheel (model-in-the-loop). Best-of-N rollouts that REIMPLEMENT a real
// function from the user's own repo, each verified by that repo's OWN gradient-check test via
// `DomainVerifier` (clonefile → splice → build → seatbelt test). This is where the capability-gap
// principle predicts genuine LIFT: the base is weak at hand-deriving backward passes AND the test
// is an unspoofable oracle — unlike the saturated toy-LeetCode surface of the code axis.
//
// Mirrors Flywheel.swift's `bestOfNCodeTraces` but the verifier reimplements in-place and runs the
// real test. `domainPass`/`domainPassAtK` are the capability-gap instruments: run them on the
// FROZEN base to confirm a task sits in the 5–60% band (≥1 verified trace to bootstrap, real gap
// remaining) before investing a distill iteration.

/// Per-task best-of-N calibration record. `band` classifies the difficulty: a task only contributes
/// DPO pairs (and is the useful training signal) when 0 < won < n — i.e. the model can SOMETIMES
/// solve it but not always (the capability-gap band). won==0 is too hard (no bootstrap); won==n is
/// saturated (the model already knows it → distilling adds greedy noise, the capability-gap law).
public struct TaskYield: Sendable {
    public let id: String; public let won: Int; public let lost: Int; public let n: Int; public let pairs: Int
    public var band: String { won == 0 ? "dry" : (won >= n ? "saturated" : "in-band") }
    public init(id: String, won: Int, lost: Int, n: Int, pairs: Int) {
        self.id = id; self.won = won; self.lost = lost; self.n = n; self.pairs = pairs }
}

extension MLXLanguageModel {
    /// The model-facing prompt for a task (leak-stripped; only `active` tasks have one). In `think`
    /// mode we ask for reasoning first — the verifier grades only the post-`</think>` answer.
    private func domainPrompt(_ t: DomainTask, think: Bool) -> String {
        guard let p = t.prompt else { return "" }
        return think
            ? p + "\n\nFirst reason about the shapes and the chain rule; then give your final answer."
            : p
    }

    /// A reasoning rollout is TRUNCATED if it opened `<think>` but never closed it (hit the token
    /// budget before the answer) — a harness artifact, not a wrong answer.
    private func isTruncatedThink(_ out: String, think: Bool) -> Bool {
        think && out.contains("<think>") && !out.contains("</think>")
    }

    /// pass@1 estimate: the mean per-task pass RATE over `samples` draws, averaged across tasks. At
    /// temperature>0 a SINGLE draw is a coin flip (the noise bug that made the first step-5 run's
    /// headline meaningless), so the eval uses `samples≈8` for a stable estimate; `samples=1` is the
    /// cheap greedy headline. Pass an explicit `tasks` list (caller splits train vs held-out eval).
    public func domainPass(tasks: [DomainTask], think: Bool = false, samples: Int = 1,
                           temperature: Float = 0.0, topP: Float = 1.0, repetitionPenalty: Float = 1.0,
                           maxTokens: Int = 1200) async -> Double {
        let active = tasks.filter { $0.prompt != nil }
        guard let repo = active.first?.repo else { return 0 }
        DomainVerifier.prepare(repo: repo)
        let draws = max(1, samples)
        var total = 0.0
        for t in active {
            let outs = await batchGenerate(domainPrompt(t, think: think), n: draws,
                                           maxTokens: maxTokens, temperature: temperature)
            let pass = outs.filter { DomainVerifier.check(candidate: $0, task: t).passed }.count
            total += Double(pass) / Double(draws)
        }
        return active.isEmpty ? 0 : total / Double(active.count)
    }

    /// pass@k: a task counts solved if ANY of k sampled rollouts passes. The compounding instrument —
    /// run before/after a distill iteration; lift = pass@1 ↑ OR k-for-fixed-pass ↓.
    public func domainPassAtK(tasks: [DomainTask], k: Int, think: Bool = false, temperature: Float = 0.7,
                              topP: Float = 0.95, repetitionPenalty: Float = 1.0,
                              maxTokens: Int = 1200) async -> Double {
        let active = tasks.filter { $0.prompt != nil }
        guard let repo = active.first?.repo else { return 0 }
        DomainVerifier.prepare(repo: repo)
        var solved = 0
        for t in active {
            let outs = await batchGenerate(domainPrompt(t, think: think), n: k,
                                           maxTokens: maxTokens, temperature: temperature)
            if outs.contains(where: { DomainVerifier.check(candidate: $0, task: t).passed }) { solved += 1 }
        }
        return active.isEmpty ? 0 : Double(solved) / Double(active.count)
    }

    /// Best-of-N + the in-repo verifier → verified TrainPairs (STaR: in `think` mode the WHOLE
    /// `<think>…</think>` + clean body is distilled). Returns counts so a budget-truncated reasoning
    /// rollout is distinguished from a genuinely wrong answer.
    public func bestOfNDomainTraces(
        tasks: [DomainTask], think: Bool = false, n: Int = 8, temperature: Float = 0.7,
        topP: Float = 0.95, repetitionPenalty: Float = 1.0, maxTokens: Int = 1200, pool: TracePool
    ) async throws -> (traces: [TrainPair], passed: Int, attempted: Int, truncated: Int, perTask: [TaskYield]) {
        let active = tasks.filter { $0.prompt != nil }
        guard let repo = active.first?.repo else { return ([], 0, 0, 0, []) }
        DomainVerifier.prepare(repo: repo)
        // Distributed generation (one best-of-N job per task across the pool); verification local.
        let mid = ModelID(modelId)
        let jobs = active.map { GenJob(model: mid, prompt: domainPrompt($0, think: think), n: n,
                                       maxTokens: maxTokens, temperature: temperature, topP: topP,
                                       repetitionPenalty: repetitionPenalty) }
        let rolloutsPerTask = await pool.generateMany(jobs)
        var winners: [TrainPair] = []
        var seen = Set<String>()
        var passed = 0, attempted = 0, truncated = 0
        var perTask: [TaskYield] = []
        for (t, rollouts) in zip(active, rolloutsPerTask) {
            let prompt = domainPrompt(t, think: think)
            var tWon = 0
            for out in rollouts {
                attempted += 1
                if isTruncatedThink(out, think: think) { truncated += 1 }
                guard DomainVerifier.check(candidate: out, task: t).passed else { continue }
                passed += 1; tWon += 1
                let body = DomainVerifier.extractBody(out)
                var assistant = body
                if think, let ts = out.range(of: "<think>"), let te = out.range(of: "</think>"),
                   ts.lowerBound < te.lowerBound {
                    assistant = String(out[ts.lowerBound..<te.upperBound]) + "\n" + body
                }
                if seen.insert(t.id + "::" + assistant).inserted {
                    winners.append(TrainPair(user: prompt, assistant: assistant))
                }
            }
            perTask.append(TaskYield(id: t.id, won: tWon, lost: rollouts.count - tWon, n: n, pairs: 0))
        }
        return (winners, passed, attempted, truncated, perTask)
    }

    /// Format a rollout into the distill/preference assistant string: the full `<think>…</think>` +
    /// extracted body (STaR), or just the body. Used for BOTH chosen (verified) and rejected (failed)
    /// so the DPO contrast is same-format, differing only in correctness.
    private func formatTrace(_ out: String, think: Bool) -> String {
        let body = DomainVerifier.extractBody(out)
        if think, let ts = out.range(of: "<think>"), let te = out.range(of: "</think>"),
           ts.lowerBound < te.lowerBound {
            return String(out[ts.lowerBound..<te.upperBound]) + "\n" + body
        }
        return body
    }

    /// Best-of-N → DPO PREFERENCE PAIRS (chosen = verified, rejected = failed) per task. The v2a fix:
    /// instead of discarding the ~majority failed rollouts, use them as negatives — 3 positives ×
    /// ~9 negatives = many pairs, so DPO isn't starved the way SFT-on-positives is.
    public func bestOfNDomainPairs(
        tasks: [DomainTask], think: Bool = false, n: Int = 12, temperature: Float = 0.6,
        topP: Float = 0.95, repetitionPenalty: Float = 1.0, maxTokens: Int = 4096,
        maxPairsPerTask: Int = 24, pool: TracePool
    ) async throws -> (pairs: [DPOTraining.Pair], verified: Int, failed: Int, perTask: [TaskYield]) {
        let active = tasks.filter { $0.prompt != nil }
        guard let repo = active.first?.repo else { return ([], 0, 0, []) }
        DomainVerifier.prepare(repo: repo)
        // One best-of-N GENERATION job per task → distributed across the pool's workers (cluster
        // fan-out; a local-only pool runs them serially). Verification stays HERE on the coordinator,
        // which alone holds the repo + sandbox — workers ship back only the candidate strings.
        let mid = ModelID(modelId)
        let jobs = active.map { GenJob(model: mid, prompt: domainPrompt($0, think: think), n: n,
                                       maxTokens: maxTokens, temperature: temperature, topP: topP,
                                       repetitionPenalty: repetitionPenalty) }
        let rolloutsPerTask = await pool.generateMany(jobs)

        var pairs: [DPOTraining.Pair] = []
        var nVer = 0, nFail = 0
        var perTask: [TaskYield] = []
        for (t, rollouts) in zip(active, rolloutsPerTask) {
            let prompt = domainPrompt(t, think: think)
            var won: [String] = [], lost: [String] = []
            for out in rollouts {
                if DomainVerifier.check(candidate: out, task: t).passed { won.append(formatTrace(out, think: think)) }
                else { lost.append(formatTrace(out, think: think)) }
            }
            nVer += won.count; nFail += lost.count
            var made = 0
            outer: for w in won where !w.isEmpty {
                for l in lost where !l.isEmpty {
                    pairs.append(.init(chosen: TrainPair(user: prompt, assistant: w),
                                       rejected: TrainPair(user: prompt, assistant: l)))
                    made += 1
                    if made >= maxPairsPerTask { break outer }
                }
            }
            // Calibration record: band = 0<won<n contributes pairs; won==0 dry; won==n saturated.
            perTask.append(TaskYield(id: t.id, won: won.count, lost: lost.count, n: n, pairs: made))
        }
        return (pairs, nVer, nFail, perTask)
    }
}
