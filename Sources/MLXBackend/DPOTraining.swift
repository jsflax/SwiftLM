import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLLM
import MLXLMCommon
import SelfImprove

// DPO (Direct Preference Optimization) for the domain flywheel.
//
// The v2a negative: SFT on the few VERIFIED traces starves (3 unique) and skews greedy without
// adding capability (pass@k holds, greedy drops). DPO uses the ABUNDANT FAILED rollouts as
// negatives instead of discarding them — per task, (chosen = verified, rejected = failed) pairs push
// the policy TOWARD the correct derivation and AWAY from the wrong one. That directly targets the
// real failure mode ("the capability exists — pass@k=100% — greedy just mis-selects it"), and it is
// NOT starved: 3 positives × ~9 negatives = many preference pairs.
//
// Loss:  −log σ( β [ (logπθ(y_w) − logπ_ref(y_w)) − (logπθ(y_l) − logπ_ref(y_l)) ] )
// Reference = the frozen base (LoRA at B=0, a no-op) — captured ONCE before any update; policy =
// base + LoRA. Assistant-only masking (same renderOne as SFT) so logprobs cover the completion only.
public enum DPOTraining {
    public struct Pair: Sendable { public let chosen: TrainPair; public let rejected: TrainPair
        public init(chosen: TrainPair, rejected: TrainPair) { self.chosen = chosen; self.rejected = rejected } }

    static func train(
        model: Module, pairs rawPairs: [Pair], optimizer: Optimizer,
        tokenizer: any MLXLMCommon.Tokenizer, iterations: Int, batchSize: Int, beta: Float = 0.1,
        maxLen: Int = 2048, report: (Int, Float) -> Void = { _, _ in }
    ) {
        // Keep only pairs whose BOTH sides render (so batched rows never drop out of alignment).
        let pairs = rawPairs.filter { renderOne($0.chosen, tokenizer) != nil && renderOne($0.rejected, tokenizer) != nil }
        guard !pairs.isEmpty else { return }

        // Reference seq-logprobs (frozen base = LoRA B=0 no-op), computed ONCE before training.
        // MUST use the SAME maxLen truncation as the policy batches below, else lp and ref cover
        // different token spans and the DPO logit (lp−ref) is garbage.
        func dlog(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
        dlog("DPO: computing reference logprobs for \(pairs.count) pairs (\(pairs.count * 2) forwards)…")
        let refC = pairs.map { refLogProb(model, $0.chosen, tokenizer, maxLen) }
        let refR = pairs.map { refLogProb(model, $0.rejected, tokenizer, maxLen) }
        dlog("DPO: reference logprobs done — starting \(iterations) iters (batch=\(batchSize), \(pairs.count) pairs)")

        let lossVG = valueAndGrad(model: model) { (m: Module, a: [MLXArray]) -> [MLXArray] in
            let lpC = seqLogProb(m, inputs: a[0], targets: a[1], mask: a[2])   // [B]
            let lpR = seqLogProb(m, inputs: a[3], targets: a[4], mask: a[5])   // [B]
            let x = beta * ((lpC - a[6]) - (lpR - a[7]))                       // [B] DPO logit
            let loss = -(log(sigmoid(x) + 1e-6)).mean()
            return [loss]
        }

        var cursor = 0
        for iter in 0..<iterations {
            var cs: [TrainPair] = [], rs: [TrainPair] = [], rc: [Float] = [], rr: [Float] = []
            for _ in 0..<batchSize {
                let j = cursor % pairs.count
                cs.append(pairs[j].chosen); rs.append(pairs[j].rejected)
                rc.append(refC[j]); rr.append(refR[j]); cursor += 1
            }
            guard let cb = makeBatch(cs, tokenizer: tokenizer, maxLen: maxLen),
                  let rb = makeBatch(rs, tokenizer: tokenizer, maxLen: maxLen) else { continue }
            let (res, grad) = lossVG(model, [cb.inputs, cb.targets, cb.mask, rb.inputs, rb.targets, rb.mask,
                                             MLXArray(rc), MLXArray(rr)])
            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, res[0])
            if (iter + 1) % 5 == 0 || iter == 0 {
                let l = res[0].item(Float.self)
                dlog("DPO: iter \(iter + 1)/\(iterations)  loss=\(String(format: "%.4f", l))")
                report(iter + 1, l)
            }
        }
    }

    /// Sum of completion-token logprobs per example: −(crossEntropy · mask).sum over time → [B].
    private static func seqLogProb(_ model: Module, inputs: MLXArray, targets: MLXArray, mask: MLXArray) -> MLXArray {
        let logits = (model as! any LLMModel)(inputs, cache: nil).asType(.float32)
        let ce = crossEntropy(logits: logits, targets: targets, reduction: .none)   // [B, T]
        return -(ce * mask).sum(axis: 1)                                            // [B]
    }

    private static func refLogProb(_ model: Module, _ p: TrainPair, _ tok: any MLXLMCommon.Tokenizer, _ maxLen: Int) -> Float {
        guard let b = makeBatch([p], tokenizer: tok, maxLen: maxLen) else { return 0 }
        let lp = seqLogProb(model, inputs: b.inputs, targets: b.targets, mask: b.mask)
        eval(lp)
        return lp.item(Float.self)
    }

    // ── render + batch (assistant-only masking; mirrors MaskedTraining) ─────────────────────────
    private struct Batch { let inputs: MLXArray; let targets: MLXArray; let mask: MLXArray }

    private static func renderOne(_ p: TrainPair, _ tokenizer: any MLXLMCommon.Tokenizer) -> (full: [Int], start: Int)? {
        // TRAIN==SERVE fast path: a pre-rendered (prefix + completion + im_end) row already matches the
        // owned-render serve format (system + tool schemas + reasoning + xmlFunction). Score it directly;
        // `makeBatch`'s start>=2 / start<count guards still apply. (Populated by RoleDisciplineTrainer.)
        if let toks = p.renderedTokens, let s = p.completionStart, toks.count >= 2 {
            return (toks.map(Int.init), s)
        }
        let user: [String: any Sendable] = ["role": "user", "content": p.user]
        guard let prefix = try? tokenizer.applyChatTemplate(messages: [user]), prefix.count >= 2 else { return nil }
        let content = tokenizer.encode(text: p.assistant, addSpecialTokens: false)
        guard !content.isEmpty, let imEnd = tokenizer.convertTokenToId("<|im_end|>") ?? tokenizer.eosTokenId else { return nil }
        return (prefix + content + [imEnd], prefix.count)
    }

    private static func makeBatch(_ pairs: [TrainPair], tokenizer: any MLXLMCommon.Tokenizer, maxLen: Int) -> Batch? {
        var inRows: [[Int32]] = [], tgtRows: [[Int32]] = [], maskRows: [[Float]] = []
        for p in pairs {
            guard let r = renderOne(p, tokenizer), r.start >= 2, r.start < r.full.count else { continue }
            // Length cap (DPO OOM guard): keep the prefix + as much completion as fits, truncate the
            // tail. Skip if the prefix alone fills the window (no completion tokens left to score).
            let capped = r.full.count > maxLen ? Array(r.full.prefix(maxLen)) : r.full
            guard r.start < capped.count else { continue }
            let inp = capped.dropLast().map(Int32.init)
            let tgt = capped.dropFirst().map(Int32.init)
            let msk = (0..<inp.count).map { Float(($0 + 1) >= r.start ? 1 : 0) }
            inRows.append(Array(inp)); tgtRows.append(Array(tgt)); maskRows.append(msk)
        }
        guard !inRows.isEmpty else { return nil }
        let maxL = max(inRows.map(\.count).max() ?? 1, 1)
        func padI(_ rows: [[Int32]]) -> [Int32] { rows.flatMap { $0 + Array(repeating: Int32(0), count: maxL - $0.count) } }
        func padF(_ rows: [[Float]]) -> [Float] { rows.flatMap { $0 + Array(repeating: Float(0), count: maxL - $0.count) } }
        let b = inRows.count
        return Batch(inputs: MLXArray(padI(inRows)).reshaped([b, maxL]),
                     targets: MLXArray(padI(tgtRows)).reshaped([b, maxL]),
                     mask: MLXArray(padF(maskRows)).reshaped([b, maxL]))
    }
}
