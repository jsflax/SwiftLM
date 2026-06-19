import Testing
@testable import SelfImprove

/// Regression guard for the P0.4 fix: CodeVerifier runs untrusted model code UNDER sandbox-exec. This
/// confirms the sandbox wrapper doesn't break normal execution (a correct solution still passes). The
/// network-denial of the wrapper itself is validated separately (a probe under the no-net profile);
/// here we just lock in that wrapping the run didn't regress the happy path.
///
/// NOTE: integration test — invokes swiftc + sandbox-exec at runtime (a few seconds); runs under
/// `swift test`, not part of a fast pure-logic pass.
struct CodeVerifierTests {
    @Test func correctSolutionPassesUnderSandbox() {
        let task = CodeTask(
            id: "add",
            signature: "func add(_ a: Int, _ b: Int) -> Int",
            desc: "returns a + b",
            test: #"if add(2, 3) == 5 && add(-1, 1) == 0 { print("ALL_PASS") } else { exit(1) }"#)
        let r = CodeVerifier.check(solution: "func add(_ a: Int, _ b: Int) -> Int { a + b }", task: task)
        #expect(r.passed, "should pass under the no-net sandbox: \(r.diagnostics)")
    }

    @Test func wrongSolutionFailsUnderSandbox() {
        let task = CodeTask(
            id: "add",
            signature: "func add(_ a: Int, _ b: Int) -> Int",
            desc: "returns a + b",
            test: #"if add(2, 3) == 5 { print("ALL_PASS") } else { exit(1) }"#)
        let r = CodeVerifier.check(solution: "func add(_ a: Int, _ b: Int) -> Int { a - b }", task: task)
        #expect(!r.passed)  // wrong impl → exit(1) under the sandbox, not a false pass
    }
}
