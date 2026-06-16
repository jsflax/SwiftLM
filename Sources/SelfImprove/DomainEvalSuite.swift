import Foundation

// ── v2a: the DOMAIN flywheel eval set (step 0 — the frozen manifest + leakage fence source).
//
// The capability-gap principle says lift lives where the base is WEAK *and* VERIFIABLE — the
// user's own repos, not toy LeetCode. The v1 target is `llm-from-scratch` (pure-Swift, zero-dep
// MiniLLM): every layer's hand-derived `backward(_:)` is proven by a numerical gradient-check
// test (numeric vs analytic, relativeError < 1e-2, fixed SplitMix64 seed → deterministic,
// pure-CPU, UNSPOOFABLE). The task shape is "reimplement against the existing test": blank a
// backward body (keep the signature), let the model regenerate it, compile the repo + run the
// covering test → ALL pass ⇒ a harvestable verified trace.
//
// This file is the FROZEN manifest of those tasks: exact body line-ranges, the canonical
// held-out body hash (the leakage-fence key), the covering + PASS_TO_PASS tests, and the
// distinctive code substrings the `Decontaminator` blocklists. Line ranges/hashes were produced
// by the step-0 inventory and must be re-validated if `llm-from-scratch` changes.
//
// LEAKAGE NOTE: the `distinctiveSubstrings` below ARE fragments of the held-out answers. That is
// unavoidable for a blocklist — but it is self-protecting: any harvested transcript line that
// contains one of these fragments (including a transcript of THIS very build session) is matched
// by the `Decontaminator` and dropped. Pair that with file-level quarantine of the MiniLLM repo.

/// Repo-wide constants for one golden SwiftPM package. The `.xctest` bundle lives at
/// `<repo>/.build/<archTriple>/<config>/<testProduct>.xctest`; the runnable binary is inside it.
public struct DomainRepo: Sendable, Hashable {
    public let path: String          // absolute path to the golden (pristine) repo
    public let testProduct: String   // SwiftPM test product, e.g. "LLMFromScratchPackageTests"
    public let archTriple: String    // e.g. "arm64-apple-macosx"
    public let config: String        // "debug" | "release"

    public init(path: String, testProduct: String,
                archTriple: String = "arm64-apple-macosx", config: String = "debug") {
        self.path = path; self.testProduct = testProduct
        self.archTriple = archTriple; self.config = config
    }

    /// `.xctest` bundle path inside a given working copy (clone or golden).
    public func xctestBundle(in workdir: String) -> String {
        "\(workdir)/.build/\(archTriple)/\(config)/\(testProduct).xctest"
    }
    /// The runnable test binary inside the bundle (what `stat -f %m` watches for the relink guard).
    public func xctestBinary(in workdir: String) -> String {
        "\(xctestBundle(in: workdir))/Contents/MacOS/\(testProduct)"
    }
}

/// One reimplement-against-tests task: blank `[bodyStartLine, bodyEndLine]` of `targetFileRel`,
/// regenerate it, then compile + run `coveringTests` (must pass) and `siblingTests` (PASS_TO_PASS
/// anti-deletion — must STILL pass, proving the candidate didn't gut shared code).
public struct DomainTask: Sendable, Hashable {
    public let id: String                       // e.g. "Linear.backward"
    public let repo: DomainRepo
    public let targetFileRel: String            // path relative to repo.path
    public let bodyStartLine: Int               // 1-indexed inclusive: first body line to replace
    public let bodyEndLine: Int                 // 1-indexed inclusive: last body line to replace
    public let coveringTests: [String]          // ClassName/method filters for `xctest -XCTest`
    public let siblingTests: [String]           // PASS_TO_PASS guards (different tests, must hold)
    public let docCommentLeaksFormula: Bool     // true ⇒ the file's comments spell out the answer
    public let bodyHash: String                 // sha256(body)[..16] — the held-out integrity key
    public let distinctiveSubstrings: [String]  // leakage-fence blocklist source
    public let prompt: String?                  // model-facing, leak-STRIPPED; nil ⇒ not yet active

    public init(id: String, repo: DomainRepo, targetFileRel: String,
                bodyStartLine: Int, bodyEndLine: Int,
                coveringTests: [String], siblingTests: [String] = [],
                docCommentLeaksFormula: Bool, bodyHash: String,
                distinctiveSubstrings: [String], prompt: String? = nil) {
        self.id = id; self.repo = repo; self.targetFileRel = targetFileRel
        self.bodyStartLine = bodyStartLine; self.bodyEndLine = bodyEndLine
        self.coveringTests = coveringTests; self.siblingTests = siblingTests
        self.docCommentLeaksFormula = docCommentLeaksFormula; self.bodyHash = bodyHash
        self.distinctiveSubstrings = distinctiveSubstrings; self.prompt = prompt
    }

    public var targetFileAbs: String { repo.path + "/" + targetFileRel }
}

public enum DomainEvalSuite {
    /// v1 golden: pure-Swift, zero-dep, deterministic gradient-check oracle. clonefile-friendly
    /// (no macros/Cxx/Metal), ~2s/rollout warm. Heavy repos (TraderKit/Engram/SwiftLM) are NOT v1.
    public static let llmFromScratch = DomainRepo(
        path: "/Users/jason/localdev/llm-from-scratch",
        testProduct: "LLMFromScratchPackageTests")

    /// The model-facing prompt for `Linear.backward`. Gives the signature, the touched types, the
    /// shapes, and the EXACT (bespoke) `Matrix`/`Parameter` API a reimplementer would read from the
    /// repo — but NEVER the gradient formulas (the held-out answer), so the gradient-check stays a
    /// closed-book oracle. The API block is essential: the base model has never seen this custom
    /// `Matrix` and otherwise guesses non-existent methods (`.transposed()`, `+=`) → false 0%.
    private static let linearPrompt = """
    Implement the BODY of this method on a fully-connected layer y = x·W + b:

        public func backward(_ dOut: Matrix) -> Matrix

    State & shapes you may rely on:
    • `forward(_:)` already ran and cached its input as `self.x`, type `Matrix?` (shape N × in).
    • `W` is a `Parameter` of shape (in × out); `b` is a `Parameter` of shape (1 × out).
    • `dOut` is the upstream gradient dL/dy, shape (N × out).

    The EXACT API of the bespoke types (use these names precisely):
        struct Matrix {
            func matmul(_ other: Matrix) -> Matrix     // matrix product
            var transposed: Matrix                     // a PROPERTY — write `m.transposed`, NOT m.transposed()
            func colSums() -> Matrix                   // sums down each column → a (1 × cols) row
            func addRowVector(_ v: Matrix) -> Matrix   // adds a (1 × cols) row to every row
            static func + (Matrix, Matrix) -> Matrix   // elementwise add
            static func * (Matrix, Matrix) -> Matrix   // elementwise multiply
        }
        final class Parameter {
            var value: Matrix    // the parameter tensor
            var grad: Matrix     // its gradient
        }
    Gotchas: `Matrix` has NO `+=` and NO `-=` — accumulate with `p.grad = p.grad + <delta>`.
    `transposed` is a property (no parentheses). The caller already zeroed `.grad`, so you must
    ACCUMULATE into it (a parameter used by several forward passes sums its gradients).

    Requirements: accumulate the weight gradient into `W.grad`, the bias gradient into `b.grad`, and
    RETURN the input gradient dL/dx (shape N × in). Unwrap `self.x` first (it is non-nil after
    `forward`): `guard let x = self.x else { fatalError("backward before forward") }`.

    Respond with ONLY the Swift statements that belong inside the function body — do NOT repeat the
    signature or the surrounding braces, no markdown fences, no commentary.
    """

    /// All 9 tasks (the frozen eval set + decontam source). Only `Linear.backward` is ACTIVE for
    /// v1 (`prompt != nil`); the rest carry their ranges/hashes/substrings so the leakage fence
    /// covers the whole MiniLLM backward surface and they can be activated incrementally.
    public static let coreTasks: [DomainTask] = [
        DomainTask(
            id: "Linear.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/Linear.swift",
            bodyStartLine: 61, bodyEndLine: 66,
            coveringTests: ["LinearTests/testLinearGradients"],
            siblingTests: ["LinearTests/testGradientsAccumulate"],
            docCommentLeaksFormula: true, bodyHash: "30309e5a5ca6019e",
            distinctiveSubstrings: [
                "W.grad = W.grad + x.transposed.matmul(dOut)",
                "b.grad = b.grad + dOut.colSums()",
                "return dOut.matmul(W.value.transposed)",
            ],
            prompt: linearPrompt),
        DomainTask(
            id: "GELU.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/GELU.swift",
            bodyStartLine: 46, bodyEndLine: 54,
            coveringTests: ["LayersTests/testGELUGradient"],
            docCommentLeaksFormula: false, bodyHash: "9cde60a18aab0c48",
            distinctiveSubstrings: [
                "let t = Foundation.tanh(inner)",
                "grad.data[i] = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * innerPrime",
                "return dOut * grad",
            ],
            prompt: DomainPrompts.geluBackward),
        DomainTask(
            id: "LayerNorm.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/LayerNorm.swift",
            bodyStartLine: 84, bodyEndLine: 112,
            coveringTests: ["LayersTests/testLayerNormGradients"],
            docCommentLeaksFormula: true, bodyHash: "8716f4433c3e7594",
            distinctiveSubstrings: [
                "let dh = dOut.data[base + c] * gamma.value.data[c]",
                "meanDxhatXhat += dh * xhat.data[base + c]",
                "dx[base + c] = istd * (dxhat[c] - meanDxhat - xhat.data[base + c] * meanDxhatXhat)",
            ],
            prompt: DomainPrompts.layerNormBackward),
        DomainTask(
            id: "SoftmaxCrossEntropy.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/SoftmaxCrossEntropy.swift",
            bodyStartLine: 69, bodyEndLine: 81,
            coveringTests: ["LayersTests/testCrossEntropyGradient"],
            docCommentLeaksFormula: true, bodyHash: "8dcb5e5a13cac981",
            distinctiveSubstrings: [
                "d[r * v + targets[r]] -= 1",
                "if targets[r] == ignoreIndex",
                "for c in 0..<v { d[r * v + c] *= inv }",
            ],
            prompt: DomainPrompts.softmaxCEBackward),
        DomainTask(
            id: "Embedding.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/Embedding.swift",
            bodyStartLine: 56, bodyEndLine: 61,
            coveringTests: ["LayersTests/testEmbeddingGradient"],
            docCommentLeaksFormula: false, bodyHash: "0cedfd762ffce1b3",
            distinctiveSubstrings: [
                "for (i, t) in ids.enumerated()",
                "let dst = t * dModel",
                "weight.grad.data[dst + c] += dOut.data[src + c]",
            ],
            prompt: DomainPrompts.embeddingBackward),
        DomainTask(
            id: "MLP.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Layers/MLP.swift",
            bodyStartLine: 33, bodyEndLine: 33,
            coveringTests: ["TransformerTests/testMLPGradients"],
            docCommentLeaksFormula: false, bodyHash: "5843b84a1f3e4581",
            distinctiveSubstrings: [
                "fc.backward(act.backward(proj.backward(dOut)))",
            ],
            prompt: DomainPrompts.mlpBackward),
        DomainTask(
            id: "MultiHeadSelfAttention.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Attention/MultiHeadSelfAttention.swift",
            bodyStartLine: 95, bodyEndLine: 128,
            coveringTests: ["AttentionTests/testAttentionGradients",
                            "AttentionTests/testSoftmaxBackward",
                            "AttentionTests/testKeyBiasGradientIsZero"],
            docCommentLeaksFormula: true, bodyHash: "3e4c8e703a95d71a",
            distinctiveSubstrings: [
                "let dA = dO.matmul(v.transposed)",
                "var dS = a.softmaxRowsBackward(dA)",
                "zeroUpperTriangle(&dS)",
                "let dKh = dS.transposed.matmul(q)",
            ],
            prompt: DomainPrompts.multiHeadSelfAttentionBackward),
        DomainTask(
            id: "TransformerBlock.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Transformer/TransformerBlock.swift",
            bodyStartLine: 52, bodyEndLine: 58,
            coveringTests: ["TransformerTests/testTransformerBlockGradients"],
            docCommentLeaksFormula: false, bodyHash: "590ebd3d962386d6",
            distinctiveSubstrings: [
                "let dA = dY + ln2.backward(mlp.backward(dY))",
                "let dX = dA + ln1.backward(attn.backward(dA))",
            ],
            prompt: DomainPrompts.transformerBlockBackward),
        DomainTask(
            id: "GPT.backward", repo: llmFromScratch,
            targetFileRel: "Sources/MiniLLM/Transformer/GPT.swift",
            bodyStartLine: 95, bodyEndLine: 101,
            coveringTests: ["TransformerTests/testGPTGradients",
                            "TransformerTests/testGPTOverfitsSingleBatch",
                            "TrainingTests/testTrainingReducesLoss"],
            docCommentLeaksFormula: false, bodyHash: "6eed8e2f4b53f21f",
            distinctiveSubstrings: [
                "var d = head.backward(dLogits)",
                "for block in blocks.reversed() { d = block.backward(d) }",
                "tokEmb.backward(d)",
            ],
            prompt: DomainPrompts.gPTBackward),
    ]

    /// All v2a tasks: the 9 hand-inventoried backward passes + the auto-discovered breadth set
    /// (forward passes + algorithmic fns) in `autoTasks` (DomainEvalSuite2.swift).
    public static let tasks: [DomainTask] = coreTasks + autoTasks

    /// Tasks the flywheel may currently generate rollouts for (have a leak-stripped prompt).
    public static var active: [DomainTask] { tasks.filter { $0.prompt != nil } }

    /// Every eval-target distinctive substring — the `Decontaminator` blocklist source.
    public static var evalSubstrings: [String] { tasks.flatMap(\.distinctiveSubstrings) }

    /// The canonical held-out body hashes — the integrity-gate key set.
    public static var evalBodyHashes: Set<String> { Set(tasks.map(\.bodyHash)) }
}
