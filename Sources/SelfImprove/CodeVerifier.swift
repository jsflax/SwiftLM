import Foundation

/// A coding task with a HIDDEN test (the test comes from the TASK, never from the rollout being
/// graded — the core Goodhart guard for an execution verifier). The model is asked to emit a
/// function with `signature`; the verifier appends `test` (which calls it and prints ALL_PASS on
/// success / exits non-zero on failure), compiles, runs, and checks the outcome.
public struct CodeTask: Sendable, Hashable {
    public let id: String
    public let signature: String   // the exact Swift signature the solution must match
    public let desc: String        // natural-language behavior description
    public let test: String        // Swift harness: top-level checks → print("ALL_PASS") or exit(1)
    public init(id: String, signature: String, desc: String, test: String) {
        self.id = id; self.signature = signature; self.desc = desc; self.test = test
    }

    /// Direct prompt — answer only, no reasoning (the answer-only flywheel uses this).
    public var prompt: String {
        "Write a Swift function with the exact signature `\(signature)` that \(desc). "
            + "Respond with ONLY the Swift function definition — no explanation, no markdown fences."
    }

    /// Reasoning prompt — think first, then answer (the CoT/STaR flywheel uses this). Does NOT
    /// instruct the model to open its own `<think>` tag: R1-distill bases prefill `<think>` via
    /// the chat template and reason natively, so a second in-prompt `<think>` would double up.
    /// The verifier extracts and grades ONLY the post-</think> answer (outcome reward).
    public var thinkPrompt: String {
        "Write a Swift function with the exact signature `\(signature)` that \(desc). "
            + "Reason about the approach and edge cases, then give your final answer as ONLY the "
            + "Swift function definition (no markdown fences, no commentary after the function)."
    }
}

public struct CodeCheckResult: Sendable {
    public let passed: Bool
    public let diagnostics: String   // compiler errors / runtime failure text (feeds refine-on-error)
    public init(passed: Bool, diagnostics: String) { self.passed = passed; self.diagnostics = diagnostics }
}

/// Executes model-generated Swift against a task's hidden test in a scratch dir. This is the
/// most reliable verifier tier (unit-test pass/fail ≫ compile ≫ tool-success). Tasks are pure
/// algorithmic functions (no file/network side-effects), so exit-code + ALL_PASS is trustworthy.
/// NOTE: this compiles & runs untrusted (model) code on-device — kept to benign pure-function
/// tasks, in an isolated temp dir, with a hard timeout.
public enum CodeVerifier {
    /// swiftc path (Xcode toolchain). Override via SWIFTC env for non-default toolchains.
    static var swiftc: String {
        ProcessInfo.processInfo.environment["SWIFTC"] ?? "/usr/bin/swiftc"
    }

    /// Extract a Swift code body from a model response: prefer a ```swift fenced block, then any
    /// ``` block, else from the first `import`/`func` line onward (drops leading prose).
    public static func extractSwift(_ rawOutput: String) -> String {
        // Outcome-only gating: if the model reasoned in a <think>…</think> span, grade ONLY the
        // final answer after the last </think>. (The reasoning is distilled but never verified.)
        let output: String
        if let r = rawOutput.range(of: "</think>", options: .backwards) {
            output = String(rawOutput[r.upperBound...])
        } else {
            output = rawOutput
        }
        func fenced(_ tag: String) -> String? {
            guard let open = output.range(of: "```\(tag)") else { return nil }
            let after = output[open.upperBound...]
            guard let close = after.range(of: "```") else { return nil }
            return String(after[..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let s = fenced("swift"), !s.isEmpty { return s }
        if let s = fenced(""), !s.isEmpty { return s }
        // No fences: drop any leading prose before the first code-looking line.
        let lines = output.components(separatedBy: "\n")
        if let i = lines.firstIndex(where: {
            let t = $0.trimmingCharacters(in: .whitespaces)
            return t.hasPrefix("import ") || t.hasPrefix("func ") || t.hasPrefix("public func ")
                || t.hasPrefix("extension ") || t.hasPrefix("struct ") || t.hasPrefix("enum ")
        }) {
            return lines[i...].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Compile `solution` + the task's hidden test and run it. passed iff it compiles, exits 0,
    /// and prints ALL_PASS. Diagnostics carry the compiler/runtime failure for refine-on-error.
    public static func check(solution rawSolution: String, task: CodeTask, timeout: TimeInterval = 20) -> CodeCheckResult {
        let solution = extractSwift(rawSolution)
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("swiftlm-verify-\(UUID().uuidString)")
        do { try fm.createDirectory(at: dir, withIntermediateDirectories: true) }
        catch { return CodeCheckResult(passed: false, diagnostics: "scratch dir failed: \(error)") }
        defer { try? fm.removeItem(at: dir) }

        let src = dir.appendingPathComponent("main.swift")
        let bin = dir.appendingPathComponent("prog")
        let program = "import Foundation\n\n" + solution + "\n\n" + task.test + "\n"
        do { try program.write(to: src, atomically: true, encoding: .utf8) }
        catch { return CodeCheckResult(passed: false, diagnostics: "write failed: \(error)") }

        let comp = runProc(swiftc, ["-O", "-o", bin.path, src.path], cwd: dir, timeout: timeout)
        guard comp.code == 0 else {
            return CodeCheckResult(passed: false, diagnostics: "COMPILE ERROR:\n" + comp.output.suffix(1500))
        }
        // Untrusted model code: run it UNDER the no-net seatbelt (network denied) — the same wrapper
        // DomainVerifier.runTest uses, never bare like before (red-team P0.4). The compile above is
        // trusted; the RUN is what we confine. (FS-scope tightening of the profile is a separate P0.)
        let run = runProc("/usr/bin/sandbox-exec",
                          ["-f", DomainVerifier.sandboxProfilePath, bin.path],
                          cwd: dir, timeout: timeout)
        let passed = run.code == 0 && run.output.contains("ALL_PASS")
        return CodeCheckResult(
            passed: passed,
            diagnostics: passed ? "" : "RUNTIME FAIL (rc=\(run.code)):\n" + run.output.suffix(1500))
    }

    /// Run a process with output captured to a file (no pipe-buffer deadlock) and a hard timeout.
    /// Internal (not private) so `DomainVerifier` reuses the same exec primitive.
    static func runProc(
        _ launch: String, _ args: [String], cwd: URL, timeout: TimeInterval
    ) -> (code: Int32, output: String) {
        let fm = FileManager.default
        let outURL = cwd.appendingPathComponent("\(UUID().uuidString).out")
        fm.createFile(atPath: outURL.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: outURL) else { return (126, "no out file") }
        defer { try? fh.close(); try? fm.removeItem(at: outURL) }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        p.currentDirectoryURL = cwd
        p.standardOutput = fh
        p.standardError = fh
        do { try p.run() } catch { return (127, "spawn failed: \(error)") }

        let deadline = Date().addingTimeInterval(timeout)
        while p.isRunning && Date() < deadline { usleep(40_000) }
        if p.isRunning { p.terminate(); usleep(200_000); if p.isRunning { p.interrupt() }
            return (124, "TIMEOUT after \(Int(timeout))s") }

        let data = (try? Data(contentsOf: outURL)) ?? Data()
        return (p.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }
}
