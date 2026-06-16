import Foundation
import MLXLMCommon
import SelfImprove

// Phase 4½ — the verifier-gated flywheel (core mechanism). Instead of training on raw
// harvested transcripts (noisy, contaminated), GENERATE candidate traces, VERIFY them
// with an objective check (here: tool-pass — did the model call the expected tool?),
// KEEP the winners, and distill THOSE. Best-of-N + verifier → clean self-play data.
//
// First version uses the tool-pass verifier (what we have). Real quality lift needs a
// larger task pool + a code-execution verifier (compile/test) — that's the next layer.

extension MLXLanguageModel {
    /// One rollout: return the FIRST tool call the model emits for `prompt` (name + args JSON),
    /// or nil if it called no tool. Sampled at `temperature` for best-of-N diversity.
    func firstToolCall(
        _ prompt: String, host: MCPHost, instructions: String?, temperature: Float
    ) async throws -> (name: String, argsJSON: String)? {
        let specs = await host.specs
        var params = GenerateParameters(maxTokens: 200, temperature: temperature)
        params.repetitionPenalty = 1.15
        params.repetitionContextSize = 20
        let session = ChatSession(container, instructions: instructions, generateParameters: params, tools: specs)
        for try await g in session.streamDetails(to: prompt) {
            if let tc = g.toolCall {
                let any = tc.function.arguments.mapValues { $0.anyValue }
                let argsJSON = (try? JSONSerialization.data(withJSONObject: any, options: [.sortedKeys]))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                return (tc.function.name, argsJSON)
            }
        }
        return nil
    }

    /// pass@k on a held-out case set: for each case, draw `k` sampled rollouts and count it
    /// solved if ANY draw calls the expected tool. This is the flywheel's compounding
    /// instrument — run before and after a distill iteration; the bet is that pass@1 rises OR
    /// the k needed for a fixed pass-rate drops (pass@4 after ≈ pass@8 before).
    public func toolPassAtK(
        cases: [ToolCase], host: MCPHost, k: Int, temperature: Float = 0.7
    ) async -> Double {
        let instructions = "You are a helpful agent with tools. Call the appropriate tool, then answer."
        var solved = 0
        for c in cases {
            var hit = false
            for _ in 0..<k {
                let res = (try? await firstToolCall(
                    c.prompt, host: host, instructions: instructions, temperature: temperature)).flatMap { $0 }
                if res?.name == c.expectedTool { hit = true; break }
            }
            if hit { solved += 1 }
        }
        return cases.isEmpty ? 0 : Double(solved) / Double(cases.count)
    }

    // ── Code-execution axis (Phase 4½ v2): the most reliable verifier tier. Generate Swift
    // solutions, COMPILE + RUN them against the task's hidden test, distill what passes. Unlike
    // the 5 saturated MCP tools, the 7B is NOT at ceiling here, so this is where real LIFT shows.

    /// The prompt for a code task — reasoning or direct, per `think`.
    private func codePrompt(_ t: CodeTask, think: Bool) -> String { think ? t.thinkPrompt : t.prompt }

    /// A trace is TRUNCATED if it opened a `<think>` but never closed it — the reasoning hit the
    /// token budget before emitting the answer (a harness artifact, NOT a wrong answer).
    private func isTruncated(_ out: String, think: Bool) -> Bool {
        think && out.contains("<think>") && !out.contains("</think>")
    }

    /// Sampled exec-pass@1 (or greedy if temperature 0): fraction of tasks whose solution compiles
    /// + passes. NOTE for R1-distill: use temperature ≈0.6 / topP 0.95 / repPen off — greedy
    /// degenerates. The verifier grades only the post-</think> answer.
    public func codePass(tasks: [CodeTask], think: Bool = false, temperature: Float = 0.0,
                         topP: Float = 1.0, repetitionPenalty: Float = 1.15, maxTokens: Int = 700) async -> Double {
        var pass = 0
        for t in tasks {
            let out = (try? await generate(codePrompt(t, think: think), maxTokens: maxTokens,
                                           temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty)) ?? ""
            if CodeVerifier.check(solution: out, task: t).passed { pass += 1 }
        }
        return tasks.isEmpty ? 0 : Double(pass) / Double(tasks.count)
    }

    /// Execution-pass@k: a task counts solved if ANY of k sampled solutions passes.
    public func codePassAtK(tasks: [CodeTask], k: Int, think: Bool = false, temperature: Float = 0.7,
                            topP: Float = 1.0, repetitionPenalty: Float = 1.15, maxTokens: Int = 700) async -> Double {
        var solved = 0
        for t in tasks {
            var hit = false
            for _ in 0..<k {
                let out = (try? await generate(codePrompt(t, think: think), maxTokens: maxTokens,
                                               temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty)) ?? ""
                if CodeVerifier.check(solution: out, task: t).passed { hit = true; break }
            }
            if hit { solved += 1 }
        }
        return tasks.isEmpty ? 0 : Double(solved) / Double(tasks.count)
    }

    /// Best-of-N + execution verifier. With `think`, the distilled trace is the FULL
    /// `<think>…</think>` + clean answer (STaR). Returns truncated count so a budget-truncated
    /// reasoning rollout is distinguished from a genuine wrong answer.
    public func bestOfNCodeTraces(
        tasks: [CodeTask], think: Bool = false, n: Int = 8, temperature: Float = 0.7,
        topP: Float = 1.0, repetitionPenalty: Float = 1.15, maxTokens: Int = 700
    ) async throws -> (traces: [TrainPair], passed: Int, attempted: Int, truncated: Int) {
        var winners: [TrainPair] = []
        var seen = Set<String>()
        var passed = 0, attempted = 0, truncated = 0
        for task in tasks {
            let prompt = codePrompt(task, think: think)
            for _ in 0..<n {
                attempted += 1
                let out = (try? await generate(prompt, maxTokens: maxTokens, temperature: temperature,
                                               topP: topP, repetitionPenalty: repetitionPenalty)) ?? ""
                if isTruncated(out, think: think) { truncated += 1 }
                guard CodeVerifier.check(solution: out, task: task).passed else { continue }
                passed += 1
                let code = CodeVerifier.extractSwift(out)
                var assistant = code
                if think, let ts = out.range(of: "<think>"), let te = out.range(of: "</think>"),
                   ts.lowerBound < te.lowerBound {
                    assistant = String(out[ts.lowerBound..<te.upperBound]) + "\n" + code
                }
                if seen.insert(task.id + "::" + assistant).inserted {
                    winners.append(TrainPair(user: prompt, assistant: assistant))
                }
            }
        }
        return (winners, passed, attempted, truncated)
    }

    /// Best-of-N + verifier: for each task, sample `n` rollouts; keep the ones that call the
    /// expected tool; emit each winner as a clean (user, assistant) TrainPair. The trainer
    /// renders it via applyChatTemplate, so the assistant side is the bare tool-call payload
    /// (NO hand-rolled <|im_start|> markers — those caused the control-token leak). Deduped.
    public func bestOfNToolTraces(
        tasks: [ToolCase], host: MCPHost, n: Int = 8, temperature: Float = 0.7
    ) async throws -> [TrainPair] {
        let instructions = "You are a helpful agent with tools. Call the appropriate tool, then answer."
        var winners = Set<TrainPair>()
        for task in tasks {
            for _ in 0..<n {
                guard let tc = try await firstToolCall(
                    task.prompt, host: host, instructions: instructions, temperature: temperature),
                      tc.name == task.expectedTool else { continue }   // VERIFY
                winners.insert(TrainPair(
                    user: task.prompt,
                    assistant: "<tool_call>\n{\"name\": \"\(tc.name)\", \"arguments\": \(tc.argsJSON)}\n</tool_call>"))
            }
        }
        return Array(winners)
    }
}
