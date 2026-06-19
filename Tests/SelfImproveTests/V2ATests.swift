import Testing
import Foundation
@testable import SelfImprove

/// Unit tests for the v2a domain-flywheel pure logic: the leakage fence (`Decontaminator`), the
/// verifier's text handling (`extractBody`/`splice`), and the frozen eval-set manifest. These cover
/// the parts that were previously only manually validated; the clone+build+seatbelt mechanics are
/// exercised by the live runs, not here (they need the real repo + toolchain).
struct DecontaminatorTests {
    let decon = Decontaminator()

    @Test func cleanTextNotFlagged() {
        #expect(decon.match(in: "The regime flip is real — risk-off after a melt-up.") == nil)
        // Forward-pass code that is NOT an eval-target body must not false-positive.
        #expect(decon.match(in: "let y = x.matmul(W.value).addRowVector(b.value)") == nil)
    }

    @Test func evalBodyFlagged() {
        #expect(decon.match(in: "        W.grad = W.grad + x.transposed.matmul(dOut)   // dL/dW") != nil)
        #expect(decon.match(in: "var dS = a.softmaxRowsBackward(dA)") != nil)
    }

    @Test func whitespaceInsensitive() {
        // Respaced / reflowed code (as transcripts often are) must still match.
        #expect(decon.match(in: "W.grad=W.grad+x.transposed.matmul(dOut)") != nil)
        #expect(decon.match(in: "return    dOut.matmul( W.value.transposed )") != nil)
    }

    @Test func contaminatedPair() {
        #expect(decon.isContaminatedPair("how do I write Linear.backward?",
                                         "return dOut.matmul(W.value.transposed)"))
        #expect(!decon.isContaminatedPair("what's the weather", "it is sunny today"))
    }

    @Test func assertCleanThrows() {
        #expect(throws: (any Error).self) { try decon.assertClean(["fine", "b.grad = b.grad + dOut.colSums()"]) }
        #expect(throws: Never.self) { try decon.assertClean(["a perfectly ordinary sentence", "another one"]) }
    }

    @Test func provenanceQuarantine() {
        let evalEdit = JSONValue.object([
            "file_path": .string(DomainEvalSuite.llmFromScratch.path + "/Sources/MiniLLM/Layers/Linear.swift")])
        let otherEdit = JSONValue.object(["file_path": .string("/Users/jason/Documents/TraderKit/Foo.swift")])
        #expect(decon.isQuarantinedEdit(toolName: "Edit", input: evalEdit))
        #expect(!decon.isQuarantinedEdit(toolName: "Edit", input: otherEdit))
        // A non-mutating tool is never quarantined, even on the eval repo.
        #expect(!decon.isQuarantinedEdit(toolName: "Read", input: evalEdit))
    }
}

struct ExtractBodyTests {
    @Test func bareBodyPassthrough() {
        let body = "W.grad = W.grad + x.transposed.matmul(dOut)\nreturn dOut.matmul(W.value.transposed)"
        #expect(DomainVerifier.extractBody(body) == body)
    }

    @Test func fullFunctionWithFencesAndProse() {
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
        #expect(body.contains("guard let x else"))
        #expect(body.contains("return dOut.matmul(W.value.transposed)"))
        #expect(!body.contains("func backward"))   // signature stripped
        #expect(!body.contains("```"))             // fences stripped
    }

    @Test func thinkBlockStripped() {
        let raw = "<think>\nlet me reason about shapes...\n</think>\nreturn dOut.matmul(W.value.transposed)"
        let body = DomainVerifier.extractBody(raw)
        #expect(!body.contains("<think>"))
        #expect(!body.contains("reason about shapes"))
        #expect(body.contains("return dOut.matmul(W.value.transposed)"))
    }

    @Test func nestedBracesBalanced() {
        // A full func whose body contains a for-loop must extract the WHOLE inner body.
        let raw = """
        func backward(_ dOut: Matrix) {
            for (i, t) in ids.enumerated() {
                weight.grad.data[t] += dOut.data[i]
            }
        }
        """
        let body = DomainVerifier.extractBody(raw)
        #expect(body.contains("for (i, t) in ids.enumerated()"))
        #expect(body.contains("weight.grad.data[t] += dOut.data[i]"))
        #expect(!body.hasPrefix("func"))
    }
}

struct SpliceTests {
    @Test func lineRangeReplacement() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appendingPathComponent("t.swift")
        // 6 lines; replace lines 3...4 (1-indexed inclusive) with "X".
        try "l1\nl2\nl3\nl4\nl5\nl6".write(to: f, atomically: true, encoding: .utf8)
        try DomainVerifier.splice(fileAt: f.path, start: 3, end: 4, body: "X")
        let out = try String(contentsOfFile: f.path, encoding: .utf8)
        #expect(out == "l1\nl2\nX\nl5\nl6")
    }

    @Test func badRangeThrows() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let f = dir.appendingPathComponent("t.swift")
        try "a\nb".write(to: f, atomically: true, encoding: .utf8)
        #expect(throws: (any Error).self) {
            try DomainVerifier.splice(fileAt: f.path, start: 1, end: 99, body: "X")
        }
    }
}

/// Verify the leakage fence is actually WIRED into the harvest path (not just the unit) — this is
/// the security-critical guarantee: a transcript containing an eval-target body must never become a
/// training pair. Drives synthetic transcript `Record`s through `Harvester.pairs`.
struct HarvesterDecontamTests {
    private func human(_ uuid: String, _ text: String) -> Record {
        Record(type: "user", uuid: uuid, parentUuid: nil, isSidechain: false, isMeta: false,
               message: Message(role: "user", content: .string(text)))
    }
    private func assistant(_ uuid: String, parent: String, _ text: String) -> Record {
        Record(type: "assistant", uuid: uuid, parentUuid: parent, isSidechain: false, isMeta: false,
               message: Message(role: "assistant",
                                content: .blocks([Block(type: "text", text: text, name: nil, input: nil)])))
    }

    @Test func cleanPairHarvested() {
        let h = human("h1", "How do I structure a backward pass for a fully-connected layer in this repo?")
        let pad = String(repeating: "Sure — here is a clear explanation of the approach. ", count: 4)
        let a = assistant("a1", parent: "h1", pad)   // clean, ≥120 chars
        #expect(Harvester().pairs(from: [h, a]).count == 1)
    }

    @Test func contaminatedPairDropped() {
        let h = human("h2", "How do I write Linear.backward in this codebase?")
        // Same length/shape, but the answer embeds an eval-target body line → must be dropped.
        let body = "Here is the gradient code you need for the layer's backward pass implementation: "
                 + "W.grad = W.grad + x.transposed.matmul(dOut) and then return the input gradient."
        let a = assistant("a2", parent: "h2", body)
        #expect(Harvester().pairs(from: [h, a]).count == 0, "eval-target body leaked into a training pair")
    }
}

struct ManifestTests {
    @Test func taskCounts() {
        #expect(DomainEvalSuite.tasks.count == 50)          // 9 core + 23 llm-from-scratch auto + 18 swift-transformers
        #expect(DomainEvalSuite.evalBodyHashes.count == DomainEvalSuite.tasks.count)   // all distinct
        #expect(!DomainEvalSuite.evalSubstrings.isEmpty)
    }

    @Test func activeTasksHavePrompts() {
        let active = DomainEvalSuite.active
        #expect(active.count == 43, "25 (llm-from-scratch) + 18 (swift-transformers) active")
        // the original hand-authored five must remain active
        #expect(Set(active.map(\.id)).isSuperset(of:
                    ["Linear.backward", "GELU.backward", "Embedding.backward",
                     "MLP.backward", "SoftmaxCrossEntropy.backward"]))
        for t in active {
            #expect(t.prompt != nil)
            #expect((t.prompt?.count ?? 0) > 200, "\(t.id) prompt is substantive")
            // It must state the signature/intent but instruct body-only output (no fences).
            #expect(t.prompt?.contains("ONLY the") == true, "\(t.id) prompt requests body-only output")
        }
    }

    @Test func taskWellFormed() {
        for t in DomainEvalSuite.tasks {
            #expect(t.bodyStartLine <= t.bodyEndLine, "\(t.id) range")
            #expect(!t.coveringTests.isEmpty, "\(t.id) has a covering test")
            #expect(t.targetFileRel.hasPrefix("Sources/"), "\(t.id) path")   // 2 repos now
        }
    }
}
