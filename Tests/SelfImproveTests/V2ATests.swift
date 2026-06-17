import XCTest
@testable import SelfImprove

/// Unit tests for the v2a domain-flywheel pure logic: the leakage fence (`Decontaminator`), the
/// verifier's text handling (`extractBody`/`splice`), and the frozen eval-set manifest. These cover
/// the parts that were previously only manually validated; the clone+build+seatbelt mechanics are
/// exercised by the live runs, not here (they need the real repo + toolchain).
final class DecontaminatorTests: XCTestCase {
    let decon = Decontaminator()

    func testCleanTextNotFlagged() {
        XCTAssertNil(decon.match(in: "The regime flip is real — risk-off after a melt-up."))
        // Forward-pass code that is NOT an eval-target body must not false-positive.
        XCTAssertNil(decon.match(in: "let y = x.matmul(W.value).addRowVector(b.value)"))
    }

    func testEvalBodyFlagged() {
        XCTAssertNotNil(decon.match(in: "        W.grad = W.grad + x.transposed.matmul(dOut)   // dL/dW"))
        XCTAssertNotNil(decon.match(in: "var dS = a.softmaxRowsBackward(dA)"))
    }

    func testWhitespaceInsensitive() {
        // Respaced / reflowed code (as transcripts often are) must still match.
        XCTAssertNotNil(decon.match(in: "W.grad=W.grad+x.transposed.matmul(dOut)"))
        XCTAssertNotNil(decon.match(in: "return    dOut.matmul( W.value.transposed )"))
    }

    func testContaminatedPair() {
        XCTAssertTrue(decon.isContaminatedPair("how do I write Linear.backward?",
                                               "return dOut.matmul(W.value.transposed)"))
        XCTAssertFalse(decon.isContaminatedPair("what's the weather", "it is sunny today"))
    }

    func testAssertCleanThrows() {
        XCTAssertThrowsError(try decon.assertClean(["fine", "b.grad = b.grad + dOut.colSums()"]))
        XCTAssertNoThrow(try decon.assertClean(["a perfectly ordinary sentence", "another one"]))
    }

    func testProvenanceQuarantine() {
        let evalEdit = JSONValue.object([
            "file_path": .string(DomainEvalSuite.llmFromScratch.path + "/Sources/MiniLLM/Layers/Linear.swift")])
        let otherEdit = JSONValue.object(["file_path": .string("/Users/jason/Documents/TraderKit/Foo.swift")])
        XCTAssertTrue(decon.isQuarantinedEdit(toolName: "Edit", input: evalEdit))
        XCTAssertFalse(decon.isQuarantinedEdit(toolName: "Edit", input: otherEdit))
        // A non-mutating tool is never quarantined, even on the eval repo.
        XCTAssertFalse(decon.isQuarantinedEdit(toolName: "Read", input: evalEdit))
    }
}

final class ExtractBodyTests: XCTestCase {
    func testBareBodyPassthrough() {
        let body = "W.grad = W.grad + x.transposed.matmul(dOut)\nreturn dOut.matmul(W.value.transposed)"
        XCTAssertEqual(DomainVerifier.extractBody(body), body)
    }

    func testFullFunctionWithFencesAndProse() {
        let raw = """
        Here is the implementation:
        ```swift
        public func backward(_ dOut: Matrix) -> Matrix {
            guard let x else { fatalError("nope") }
            return dOut.matmul(W.value.transposed)
        }
        ```
        """
        let body = DomainVerifier.extractBody(raw)
        XCTAssertTrue(body.contains("guard let x else"))
        XCTAssertTrue(body.contains("return dOut.matmul(W.value.transposed)"))
        XCTAssertFalse(body.contains("func backward"))   // signature stripped
        XCTAssertFalse(body.contains("```"))             // fences stripped
    }

    func testThinkBlockStripped() {
        let raw = "<think>\nlet me reason about shapes...\n</think>\nreturn dOut.matmul(W.value.transposed)"
        let body = DomainVerifier.extractBody(raw)
        XCTAssertFalse(body.contains("<think>"))
        XCTAssertFalse(body.contains("reason about shapes"))
        XCTAssertTrue(body.contains("return dOut.matmul(W.value.transposed)"))
    }

    func testNestedBracesBalanced() {
        // A full func whose body contains a for-loop must extract the WHOLE inner body.
        let raw = """
        func backward(_ dOut: Matrix) {
            for (i, t) in ids.enumerated() {
                weight.grad.data[t] += dOut.data[i]
            }
        }
        """
        let body = DomainVerifier.extractBody(raw)
        XCTAssertTrue(body.contains("for (i, t) in ids.enumerated()"))
        XCTAssertTrue(body.contains("weight.grad.data[t] += dOut.data[i]"))
        XCTAssertFalse(body.hasPrefix("func"))
    }
}

final class SpliceTests: XCTestCase {
    func testLineRangeReplacement() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appendingPathComponent("t.swift")
        // 6 lines; replace lines 3...4 (1-indexed inclusive) with "X".
        try "l1\nl2\nl3\nl4\nl5\nl6".write(to: f, atomically: true, encoding: .utf8)
        try DomainVerifier.splice(fileAt: f.path, start: 3, end: 4, body: "X")
        let out = try String(contentsOfFile: f.path, encoding: .utf8)
        XCTAssertEqual(out, "l1\nl2\nX\nl5\nl6")
    }

    func testBadRangeThrows() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appendingPathComponent("t.swift")
        try "a\nb".write(to: f, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try DomainVerifier.splice(fileAt: f.path, start: 1, end: 99, body: "X"))
    }
}

/// Verify the leakage fence is actually WIRED into the harvest path (not just the unit) — this is
/// the security-critical guarantee: a transcript containing an eval-target body must never become a
/// training pair. Drives synthetic transcript `Record`s through `Harvester.pairs`.
final class HarvesterDecontamTests: XCTestCase {
    private func human(_ uuid: String, _ text: String) -> Record {
        Record(type: "user", uuid: uuid, parentUuid: nil, isSidechain: false, isMeta: false,
               message: Message(role: "user", content: .string(text)))
    }
    private func assistant(_ uuid: String, parent: String, _ text: String) -> Record {
        Record(type: "assistant", uuid: uuid, parentUuid: parent, isSidechain: false, isMeta: false,
               message: Message(role: "assistant",
                                content: .blocks([Block(type: "text", text: text, name: nil, input: nil)])))
    }

    func testCleanPairHarvested() {
        let h = human("h1", "How do I structure a backward pass for a fully-connected layer in this repo?")
        let pad = String(repeating: "Sure — here is a clear explanation of the approach. ", count: 4)
        let a = assistant("a1", parent: "h1", pad)   // clean, ≥120 chars
        XCTAssertEqual(Harvester().pairs(from: [h, a]).count, 1)
    }

    func testContaminatedPairDropped() {
        let h = human("h2", "How do I write Linear.backward in this codebase?")
        // Same length/shape, but the answer embeds an eval-target body line → must be dropped.
        let body = "Here is the gradient code you need for the layer's backward pass implementation: "
                 + "W.grad = W.grad + x.transposed.matmul(dOut) and then return the input gradient."
        let a = assistant("a2", parent: "h2", body)
        XCTAssertEqual(Harvester().pairs(from: [h, a]).count, 0, "eval-target body leaked into a training pair")
    }
}

final class ManifestTests: XCTestCase {
    func testTaskCounts() {
        XCTAssertEqual(DomainEvalSuite.tasks.count, 50)          // 9 core + 23 llm-from-scratch auto + 18 swift-transformers
        XCTAssertEqual(DomainEvalSuite.evalBodyHashes.count, DomainEvalSuite.tasks.count)   // all distinct
        XCTAssertFalse(DomainEvalSuite.evalSubstrings.isEmpty)
    }

    func testActiveTasksHavePrompts() {
        let active = DomainEvalSuite.active
        XCTAssertEqual(active.count, 43, "25 (llm-from-scratch) + 18 (swift-transformers) active")
        // the original hand-authored five must remain active
        XCTAssertTrue(Set(active.map(\.id)).isSuperset(of:
                        ["Linear.backward", "GELU.backward", "Embedding.backward",
                         "MLP.backward", "SoftmaxCrossEntropy.backward"]))
        for t in active {
            XCTAssertNotNil(t.prompt)
            XCTAssertGreaterThan(t.prompt!.count, 200, "\(t.id) prompt is substantive")
            // It must state the signature/intent but instruct body-only output (no fences).
            XCTAssertTrue(t.prompt!.contains("ONLY the"), "\(t.id) prompt requests body-only output")
        }
        // NOTE: `distinctiveSubstrings` are DECONTAM keys (catch leaked transcripts), NOT
        // prompt-forbidden strings — some overlap with chain-rule scaffolding legitimately given in
        // the prompt (e.g. GELU's `return dOut * grad`). Leak-freedom of the WITHHELD formula was
        // validated by the prompt-authoring workflow's compile + manual leak check.
    }

    func testTaskWellFormed() {
        for t in DomainEvalSuite.tasks {
            XCTAssertLessThanOrEqual(t.bodyStartLine, t.bodyEndLine, "\(t.id) range")
            XCTAssertFalse(t.coveringTests.isEmpty, "\(t.id) has a covering test")
            XCTAssertTrue(t.targetFileRel.hasPrefix("Sources/"), "\(t.id) path")   // 2 repos now
        }
    }
}
