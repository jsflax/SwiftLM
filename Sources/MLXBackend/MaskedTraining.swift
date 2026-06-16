import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLLM
import MLXLMCommon
import SelfImprove

// Assistant-only loss masking, trained on the tokenizer's REAL chat template so the
// training format matches inference exactly (the hand-rolled format caused the model to
// emit <|im_start|>). Each example is rendered with applyChatTemplate; the assistant
// mask boundary is the common-prefix length of the full render vs the same conversation
// with empty assistant content — exact, immune to BPE-join slack. Mirrors LoRATrain's
// valueAndGrad → optimizer.update → eval loop, masking everything but the assistant turn.
enum MaskedTraining {
    static func train(
        model: Module,
        data: [TrainPair],
        optimizer: Optimizer,
        tokenizer: any MLXLMCommon.Tokenizer,
        iterations: Int,
        batchSize: Int,
        report: (Int, Float) -> Void = { _, _ in }
    ) {
        let lossVG = valueAndGrad(model: model) { (model: Module, arrays: [MLXArray]) -> [MLXArray] in
            let logits = (model as! any LLMModel)(arrays[0], cache: nil).asType(.float32)
            let ce = crossEntropy(logits: logits, targets: arrays[1], reduction: .none)
            let mask = arrays[2]
            let ntoks = mask.sum()
            let loss = (ce * mask).sum() / ntoks
            return [loss, ntoks]
        }

        var cursor = 0
        for iteration in 0..<iterations {
            var batch: [TrainPair] = []
            for _ in 0..<batchSize { batch.append(data[cursor % data.count]); cursor += 1 }
            guard let b = makeBatch(batch, tokenizer: tokenizer) else { continue }
            let (result, grad) = lossVG(model, [b.inputs, b.targets, b.mask])
            optimizer.update(model: model, gradients: grad)
            eval(model, optimizer, result[0])
            if (iteration + 1) % 50 == 0 { report(iteration + 1, result[0].item(Float.self)) }
        }
    }

    private struct Batch { let inputs: MLXArray; let targets: MLXArray; let mask: MLXArray }

    /// Render one (user, assistant) pair into the token sequence the model should learn, plus
    /// the index where the assistant turn begins (the loss boundary). Built as
    ///   prefix = applyChatTemplate([user])   // ends with "<|im_start|>assistant\n"
    ///   full   = prefix + encode(assistant) + <|im_end|>
    /// so the generation prompt sits INSIDE the masked prefix and the trained region is exactly
    /// the assistant content + its terminating <|im_end|>. (The convenience applyChatTemplate
    /// hardcodes addGenerationPrompt:true and would otherwise append a spurious second
    /// "<|im_start|>assistant\n" after the answer — that taught the model to emit <|im_start|>.)
    private static func renderOne(
        _ p: TrainPair, tokenizer: any MLXLMCommon.Tokenizer
    ) -> (full: [Int], start: Int)? {
        let user: [String: any Sendable] = ["role": "user", "content": p.user]
        guard let prefix = try? tokenizer.applyChatTemplate(messages: [user]), prefix.count >= 2
        else { return nil }
        let content = tokenizer.encode(text: p.assistant, addSpecialTokens: false)
        guard !content.isEmpty,
              let imEnd = tokenizer.convertTokenToId("<|im_end|>") ?? tokenizer.eosTokenId
        else { return nil }
        return (prefix + content + [imEnd], prefix.count)
    }

    /// Diagnostic: for each pair, return the assistant-content boundary plus the FULL render
    /// and the exact TRAINED region (tokens the loss mask keeps). If the trained region isn't
    /// precisely the assistant turn + a single trailing <|im_end|>, the masking is wrong.
    public static func debugRender(
        _ pairs: [TrainPair], tokenizer: any MLXLMCommon.Tokenizer
    ) -> [(start: Int, total: Int, full: String, trained: String)] {
        var out: [(start: Int, total: Int, full: String, trained: String)] = []
        for p in pairs {
            guard let r = renderOne(p, tokenizer: tokenizer) else { continue }
            let trainedToks = Array(r.full[r.start...])
            out.append((r.start, r.full.count,
                        tokenizer.decode(tokenIds: r.full),
                        tokenizer.decode(tokenIds: trainedToks)))
        }
        return out
    }

    /// Render each pair (prefix + assistant content + <|im_end|>) and mask everything before
    /// the assistant turn, so the loss covers only the assistant content + its terminator.
    private static func makeBatch(_ pairs: [TrainPair], tokenizer: any MLXLMCommon.Tokenizer) -> Batch? {
        var inRows: [[Int32]] = [], tgtRows: [[Int32]] = [], maskRows: [[Float]] = []
        for p in pairs {
            guard let r = renderOne(p, tokenizer: tokenizer),
                  r.start >= 2, r.start < r.full.count else { continue }
            let inp = r.full.dropLast().map(Int32.init)
            let tgt = r.full.dropFirst().map(Int32.init)
            // position i predicts full[i+1]; train iff that target is in the assistant turn.
            let msk = (0..<inp.count).map { Float(($0 + 1) >= r.start ? 1 : 0) }
            inRows.append(Array(inp)); tgtRows.append(Array(tgt)); maskRows.append(msk)
        }
        guard !inRows.isEmpty else { return nil }
        let maxL = max(inRows.map(\.count).max() ?? 1, 1)
        func padI(_ rows: [[Int32]]) -> [Int32] {
            rows.flatMap { $0 + Array(repeating: Int32(0), count: maxL - $0.count) }
        }
        func padF(_ rows: [[Float]]) -> [Float] {
            rows.flatMap { $0 + Array(repeating: Float(0), count: maxL - $0.count) }
        }
        let b = inRows.count
        return Batch(
            inputs: MLXArray(padI(inRows)).reshaped([b, maxL]),
            targets: MLXArray(padI(tgtRows)).reshaped([b, maxL]),
            mask: MLXArray(padF(maskRows)).reshaped([b, maxL]))
    }
}
