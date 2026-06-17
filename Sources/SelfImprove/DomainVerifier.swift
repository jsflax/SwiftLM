import Foundation

// ── v2a: the DOMAIN execution verifier (step 2). A drop-in sibling of `CodeVerifier`, but instead
// of compiling a standalone snippet it reimplements a function IN-PLACE in a real repo and runs
// that repo's own gradient-check test. This is the most trustworthy verifier tier we have: a
// numerical gradient check (numeric vs analytic, fixed seed) is an UNSPOOFABLE oracle.
//
// Per-rollout flow (each step a validated fix for a reproduced blocker):
//   1. clonefile the warm golden  (cp -Rc — APFS COW, shares the warm .build, ~0.9s)
//   2. capture the test binary's mtime BEFORE the build   (relink sentinel)
//   3. splice the candidate body into ONLY the blanked range
//   4. swift build --build-tests with a SHARED module cache OUTSIDE the clone   (Blocker-1 fix:
//      a cloned .build's ModuleCache embeds the golden's absolute path → build breaks; a shared
//      cache sidesteps it). Read the real exit code directly — NO pipe (the pipe-status trap is
//      the root of the stale-bundle false-PASS).
//   5. FAIL-CLOSED on any non-zero build exit   (a broken stub must never reach a test run)
//   6. relink guard: the test binary must exist AND its mtime must be > the pre-build mtime
//      (Blocker-2 fix: a failed/no-op build does not relink, so a stale prior bundle can't be
//      mistaken for a pass; per-rollout PRE>POST avoids the false-FAIL a global build-start clock
//      causes on legit no-op builds).
//   7. run the covering test(s) UNDER sandbox-exec (network denied) — model-written code is
//      untrusted. NEVER `swift test` under seatbelt (SwiftPM recompiles the manifest to a temp
//      dir → sandbox EPERM); build trusted, run the .xctest BINARY untrusted.
//   8. PASS_TO_PASS sibling tests must STILL pass (anti-deletion: the candidate can't earn a pass
//      by gutting shared code).
//
// Measured on llm-from-scratch (Swift 6.3.2, Apple Silicon): clone 0.95s, build ~2.0s,
// ~3.3s/rollout steady-state ⇒ best-of-16 ≈ 53s.

public struct DomainCheckResult: Sendable {
    /// The stage at which the rollout terminated — `passed` only if every gate cleared.
    public enum Stage: String, Sendable {
        case clone, splice, build, relink, coveringTest, siblingTest, passed
    }
    public let passed: Bool
    public let stage: Stage
    public let diagnostics: String   // compiler / runtime failure text (feeds refine-on-error)
    public init(passed: Bool, stage: Stage, diagnostics: String) {
        self.passed = passed; self.stage = stage; self.diagnostics = diagnostics
    }
    static func fail(_ s: Stage, _ d: String) -> DomainCheckResult {
        DomainCheckResult(passed: false, stage: s, diagnostics: d)
    }
}

public enum DomainVerifier {
    /// `swift` driver (Xcode toolchain). Override via SWIFT env for non-default toolchains.
    static var swift: String { ProcessInfo.processInfo.environment["SWIFT"] ?? "/usr/bin/swift" }
    /// Module cache shared by every clone — the Blocker-1 fix. Lives OUTSIDE any clone.
    static let sharedCache = ProcessInfo.processInfo.environment["SWIFTLM_MCACHE"] ?? "/tmp/swiftlm-shared-mcache"

    /// Seatbelt profile: allow everything except the network. Written once.
    private static let sandboxProfile = "(version 1)\n(allow default)\n(deny network*)\n"
    static var sandboxProfilePath: String {
        let p = "/tmp/swiftlm-no-net.sb"
        if !FileManager.default.fileExists(atPath: p) {
            try? sandboxProfile.write(toFile: p, atomically: true, encoding: .utf8)
        }
        return p
    }

    // ── Warming (run ONCE before a batch of rollouts) ─────────────────────────────────────────
    /// Populate the shared module cache so per-rollout clones build in ~2s. Idempotent: a non-empty
    /// `sharedCache` is treated as warm. Uses the validated recipe: clone the golden, REMOVE the
    /// clone's stale in-tree ModuleCache (so the compiler repopulates SHARED cleanly), then build.
    private static let warmLock = NSLock()
    nonisolated(unsafe) private static var warmedRepos: Set<String> = []
    /// Warm the shared module cache for EACH distinct repo in `tasks` (the multi-repo flywheel: tasks
    /// can span llm-from-scratch AND swift-transformers). Idempotent per repo per process.
    public static func prepareAll(_ tasks: [DomainTask], timeout: TimeInterval = 600) {
        for r in Set(tasks.map(\.repo)) { _ = prepare(repo: r, timeout: timeout) }
    }
    @discardableResult
    public static func prepare(repo: DomainRepo, timeout: TimeInterval = 600) -> DomainCheckResult {
        let fm = FileManager.default
        warmLock.lock(); let already = warmedRepos.contains(repo.path); warmLock.unlock()
        if already {
            return DomainCheckResult(passed: true, stage: .passed,
                                     diagnostics: "module cache already warm for \(repo.testProduct)")
        }
        // Per-repo warm path so multiple repos don't clobber each other.
        let warm = "/tmp/swiftlm-domain-warm-\(repo.testProduct)"
        try? fm.removeItem(atPath: warm)
        let clone = CodeVerifier.runProc("/bin/cp", ["-Rc", repo.path, warm],
                                         cwd: URL(fileURLWithPath: "/tmp"), timeout: timeout)
        guard clone.code == 0 else {
            return .fail(.clone, "warm clone failed (rc=\(clone.code)): " + clone.output.suffix(400))
        }
        // Blocker-1: drop the stale, golden-pathed ModuleCache from the clone for this first build only.
        try? fm.removeItem(atPath: "\(warm)/.build/\(repo.archTriple)/\(repo.config)/ModuleCache")
        let build = CodeVerifier.runProc(
            swift, ["build", "--build-tests", "-Xswiftc", "-module-cache-path", "-Xswiftc", sharedCache],
            cwd: URL(fileURLWithPath: warm), timeout: timeout)
        try? fm.removeItem(atPath: warm)
        guard build.code == 0 else {
            return .fail(.build, "warm build failed (rc=\(build.code)):\n" + build.output.suffix(1500))
        }
        warmLock.lock(); warmedRepos.insert(repo.path); warmLock.unlock()
        return DomainCheckResult(passed: true, stage: .passed, diagnostics: "warmed module cache for \(repo.testProduct)")
    }

    // ── Task-quality gate (step 3) ──────────────────────────────────────────────────────────────
    public struct TaskQuality: Sendable { public let ok: Bool; public let report: String }

    /// Vet a task before it enters the flywheel (needs `prepare` first). Two cheap, decisive checks:
    ///  • DETERMINISM — the covering tests pass on the PRISTINE golden across N isolated runs (a
    ///    flaky/non-deterministic test would forge or destroy traces at random).
    ///  • MUTATION SANITY — a `fatalError` stub body must FAIL the covering test. If a stub still
    ///    passes, the test doesn't actually exercise the function (glue/getter) → reject the task.
    ///    `fatalError` returns `Never`, so it type-checks for any return type — a universal stub.
    public static func qualityGate(task: DomainTask, determinismRuns: Int = 3,
                                   timeout: TimeInterval = 180) -> TaskQuality {
        let fm = FileManager.default
        var lines: [String] = []
        var ok = true

        // (1) determinism on the pristine golden
        let wt = "/tmp/swiftlm-wt-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: wt) }
        let clone = CodeVerifier.runProc("/bin/cp", ["-Rc", task.repo.path, wt],
                                         cwd: URL(fileURLWithPath: "/tmp"), timeout: timeout)
        guard clone.code == 0 else { return TaskQuality(ok: false, report: "determinism: clone failed") }
        let build = CodeVerifier.runProc(
            swift, ["build", "--build-tests", "-Xswiftc", "-module-cache-path", "-Xswiftc", sharedCache],
            cwd: URL(fileURLWithPath: wt), timeout: timeout)
        guard build.code == 0 else {
            return TaskQuality(ok: false, report: "determinism: golden build failed:\n" + build.output.suffix(800))
        }
        let bundle = task.repo.xctestBundle(in: wt)
        var detOk = true
        for i in 1...determinismRuns {
            for f in task.coveringTests {
                let r = runTest(f, bundle: bundle, cwd: wt, timeout: timeout)
                if r.code != 0 { detOk = false; lines.append("  run \(i) [\(f)] FAILED rc=\(r.code)") }
            }
        }
        lines.append("determinism: \(detOk ? "PASS" : "FAIL") (\(determinismRuns)× isolated, \(task.coveringTests.count) test(s))")
        ok = ok && detOk

        // (2) mutation sanity — stub must fail
        let mut = check(candidate: "fatalError(\"stub\")", task: task, timeout: timeout)
        let mutOk = !mut.passed
        lines.append("mutation-sanity: \(mutOk ? "PASS" : "FAIL") — stub body → "
                     + (mut.passed ? "test PASSED (BAD: test does not constrain the fn)"
                                   : "test failed at \(mut.stage.rawValue) (good)"))
        ok = ok && mutOk

        return TaskQuality(ok: ok, report: lines.joined(separator: "\n"))
    }

    // ── Per-rollout check ─────────────────────────────────────────────────────────────────────
    /// Compile `candidate` into `task`'s blanked range and run the covering + sibling tests.
    /// `passed` iff the build succeeds, the binary is relinked, and ALL covering + sibling tests
    /// exit 0. `candidate` may be a bare body OR a full function (the signature is stripped back to
    /// its body before splicing).
    public static func check(candidate: String, task: DomainTask, timeout: TimeInterval = 180) -> DomainCheckResult {
        let repo = task.repo
        let fm = FileManager.default
        let wt = "/tmp/swiftlm-wt-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: wt) }

        // 1. clone the warm golden
        let clone = CodeVerifier.runProc("/bin/cp", ["-Rc", repo.path, wt],
                                         cwd: URL(fileURLWithPath: "/tmp"), timeout: timeout)
        guard clone.code == 0 else {
            return .fail(.clone, "clone failed (rc=\(clone.code)): " + clone.output.suffix(400))
        }

        // 2. relink sentinel BEFORE the build
        let binary = repo.xctestBinary(in: wt)
        let pre = mtime(binary)

        // 3. splice the candidate body into the blanked range. A unique trailing comment makes the
        //    spliced file content distinct every rollout, so even a byte-identical-to-golden answer
        //    forces a recompile+relink — otherwise a no-op build leaves the binary un-relinked and
        //    the (correct) relink guard in step 6 would false-FAIL it.
        let body = extractBody(candidate) + "\n        // rollout \(UUID().uuidString)"
        do {
            try splice(fileAt: wt + "/" + task.targetFileRel,
                       start: task.bodyStartLine, end: task.bodyEndLine, body: body)
        } catch {
            return .fail(.splice, "splice failed: \(error)")
        }

        // 4. build with the shared module cache (no pipe — exit code read directly)
        let build = CodeVerifier.runProc(
            swift, ["build", "--build-tests", "-Xswiftc", "-module-cache-path", "-Xswiftc", sharedCache],
            cwd: URL(fileURLWithPath: wt), timeout: timeout)
        // 5. FAIL-CLOSED on a build error
        guard build.code == 0 else {
            return .fail(.build, "BUILD FAIL (rc=\(build.code)):\n" + build.output.suffix(1500))
        }

        // 6. relink guard — the binary must exist AND have been relinked by THIS rollout's build
        guard fm.isExecutableFile(atPath: binary) else {
            return .fail(.relink, "no test binary after a 'successful' build (\(binary))")
        }
        let post = mtime(binary)
        guard post > pre else {
            return .fail(.relink, "stale bundle: test binary not relinked (pre=\(pre) post=\(post))")
        }

        // 7. covering test(s) — model code is untrusted, network denied
        let bundle = repo.xctestBundle(in: wt)
        for filter in task.coveringTests {
            let r = runTest(filter, bundle: bundle, cwd: wt, timeout: timeout)
            guard r.code == 0 else {
                return .fail(.coveringTest, "covering test FAIL [\(filter)] rc=\(r.code):\n" + r.output.suffix(1200))
            }
        }
        // 8. PASS_TO_PASS siblings must still pass
        for filter in task.siblingTests {
            let r = runTest(filter, bundle: bundle, cwd: wt, timeout: timeout)
            guard r.code == 0 else {
                return .fail(.siblingTest, "PASS_TO_PASS sibling FAIL [\(filter)] rc=\(r.code):\n" + r.output.suffix(1200))
            }
        }
        return DomainCheckResult(passed: true, stage: .passed, diagnostics: "")
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────────────────

    /// Run ONE test filter against the built `.xctest` bundle, sandboxed (network denied). The
    /// filter is `ClassName/method` (e.g. `LinearTests/testLinearGradients`) — NOT target-qualified.
    static func runTest(_ filter: String, bundle: String, cwd: String,
                        timeout: TimeInterval) -> (code: Int32, output: String) {
        CodeVerifier.runProc(
            "/usr/bin/sandbox-exec",
            ["-f", sandboxProfilePath, "/usr/bin/xcrun", "xctest", "-XCTest", filter, bundle],
            cwd: URL(fileURLWithPath: cwd), timeout: timeout)
    }

    /// Modification time of a file as epoch seconds (0 if missing). Sub-second resolution is fine —
    /// a relink always bumps it; a no-op/failed build leaves it identical (→ `post > pre` is false).
    static func mtime(_ path: String) -> TimeInterval {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let date = attrs[.modificationDate] as? Date else { return 0 }
        return date.timeIntervalSince1970
    }

    /// Reduce a raw model output to the statements that belong inside the function body: strip
    /// `<think>`/fences/prose (via `CodeVerifier.extractSwift`), then — if the model disobeyed and
    /// emitted a whole `func` — pull out the inner brace-balanced body. Otherwise the cleaned text
    /// IS the body.
    public static func extractBody(_ raw: String) -> String {
        let code = CodeVerifier.extractSwift(raw)
        guard let funcRange = code.range(of: "func ") else { return code }
        let tail = code[funcRange.lowerBound...]
        guard let open = tail.firstIndex(of: "{") else { return code }
        var depth = 0
        var bodyStart: String.Index?
        var i = open
        while i < tail.endIndex {
            let ch = tail[i]
            if ch == "{" {
                depth += 1
                if depth == 1 { bodyStart = tail.index(after: i) }
            } else if ch == "}" {
                depth -= 1
                if depth == 0, let s = bodyStart {
                    return String(tail[s..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
            i = tail.index(after: i)
        }
        return code   // unbalanced — fall back to the cleaned text
    }

    enum DVError: Error { case badRange(String) }

    /// Replace 1-indexed inclusive lines `[start, end]` of the file with `body`.
    static func splice(fileAt path: String, start: Int, end: Int, body: String) throws {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        var lines = content.components(separatedBy: "\n")
        guard start >= 1, start <= end, end <= lines.count else {
            throw DVError.badRange("start=\(start) end=\(end) lineCount=\(lines.count) in \(path)")
        }
        let head = Array(lines[0..<(start - 1)])     // lines 1..<start
        let rest = Array(lines[end...])              // line end+1.. (end is 1-indexed inclusive)
        lines = head + [body] + rest
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
}
