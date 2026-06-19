import Foundation
import MLX
import MLXLMCommon
import SwiftLM
import MiniBPE
import JSONSchema
import Serving

// ── SLICE 4 — PER-ROW grammar-constrained decoding in a BATCHED pass (the fan-out × constrained-tool-call
// collision the sub-agent batching needs resolved).
//
// `GrammarLogitProcessor` masks one stream with ONE tracker (its additive mask broadcasts the SAME row to
// every batch element — wrong for fan-out, where each row is mid-DIFFERENT JSON/tool). `BatchedGrammar`
// holds B INDEPENDENT `JSONSchemaStateTracker`s (cheap value structs) and builds a PER-ROW `[B, vocab]`
// additive mask each step. The open question — answered by `grammarBatchBench`, not asserted — is whether
// the per-step host-side mask build (B × vocab, ~1.2M floats at B=8) erodes the batched-decode win enough
// that constrained rows are better off decoded solo. The decode itself stays SLICE 1b (separate-prefill +
// merge); only the per-step sampling gains a mask + per-row tracker advance.
final class BatchedGrammar {
    private var trackers: [JSONSchemaStateTracker]
    private var decoded: [[Int]]
    /// Per current-batch-row: has this row's JSON closed?
    private(set) var complete: [Bool]

    init(count: Int, fields: [SchemaProperty], tokenizer: any GrammarTokenizer,
         runtimeEnums: [String: [String]]? = nil) {
        let t = JSONSchemaStateTracker(fields: fields, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
        self.trackers = Array(repeating: t, count: count)   // value structs → B independent copies
        self.decoded = Array(repeating: [], count: count)
        self.complete = Array(repeating: false, count: count)
    }

    /// Additive mask: per row, `0` for grammar-valid tokens and `-inf` for the rest, added to `[B, vocab]`
    /// logits. A row whose grammar is finished (empty valid set) passes through unmasked so its `argMax`
    /// stays well-defined (used by the fixed-length benchmark; the live path evicts completed rows).
    func maskedLogits(_ logits: MLXArray) -> MLXArray {
        let B = trackers.count
        let dim = logits.shape.last ?? 0
        guard B > 0, dim > 0 else { return logits }
        var scalars = [Float](repeating: -Float.greatestFiniteMagnitude, count: B * dim)
        for r in 0..<B {
            let base = r * dim
            let valid = trackers[r].validTokens()
            if valid.isEmpty {
                for i in base..<(base + dim) { scalars[i] = 0 }          // passthrough (no constraint)
            } else {
                for v in valid where v < dim { scalars[base + v] = 0 }
            }
        }
        return logits + MLXArray(scalars).reshaped([B, dim])
    }

    /// Advance each row's tracker with the token it just sampled, refreshing `complete`.
    func advance(_ sampled: [Int]) {
        for r in 0..<trackers.count where r < sampled.count {
            trackers[r].updateState(with: sampled[r], &decoded[r])
            complete[r] = trackers[r].isComplete
        }
    }

    /// Keep only the given current-batch rows (mirrors `BatchedKVCache.evict`).
    func evict(keep: [Int]) {
        trackers = keep.map { trackers[$0] }
        decoded = keep.map { decoded[$0] }
        complete = keep.map { complete[$0] }
    }
}

extension MLXLanguageModel {
    /// SLICE 4 — decode B DIFFERENT prompts in one batched pass with PER-ROW grammar constraints (each row's
    /// output is valid-by-construction JSON for `fields`). Reuses the SLICE 1b prefill+merge; adds a per-step
    /// `[B, vocab]` mask + per-row tracker advance. `stopOnComplete` evicts a row when its JSON closes;
    /// `stopOnComplete: false` (benchmark) runs all rows to `maxTokens` for a clean throughput comparison.
    func batchDecodeConstrained(model: any MLXLMCommon.LanguageModel, promptRows: [[Int32]], fields: [SchemaProperty],
                                tokenizer: any GrammarTokenizer, maxTokens: Int, temperature: Float,
                                stops: Set<Int>, stopOnEOS: Bool, stopOnComplete: Bool,
                                runtimeEnums: [String: [String]]? = nil) -> [[Int]] {
        let n = promptRows.count
        guard let (merged, firstLogits, _) = prefillAndMerge(model: model, promptRows: promptRows) else {
            return Array(repeating: [], count: n)
        }
        let grammar = BatchedGrammar(count: n, fields: fields, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
        var next = sampleBatched(grammar.maskedLogits(firstLogits), temperature: temperature)  // [B]
        eval(next)

        var out = Array(repeating: [Int](), count: n)
        var streamOf = Array(0..<n)
        for _ in 0..<maxTokens {
            let toks = next.asArray(Int.self)
            grammar.advance(toks)                                    // advance trackers with what we sampled
            var keep: [Int] = []
            for r in 0..<streamOf.count {
                if stopOnEOS && stops.contains(toks[r]) { continue } // evict on a hard stop (don't append)
                out[streamOf[r]].append(toks[r])
                if stopOnComplete && grammar.complete[r] { continue }// JSON closed → evict after appending
                keep.append(r)
            }
            if keep.isEmpty { break }
            if keep.count < streamOf.count {
                streamOf = keep.map { streamOf[$0] }
                for c in merged { (c as! BatchedKVCache).evict(keep: keep) }
                grammar.evict(keep: keep)
                next = next[MLXArray(keep.map { Int32($0) })]
            }
            let logits = model(next.reshaped([streamOf.count, 1]), cache: merged)[0..., -1, 0...]  // [B,vocab]
            next = sampleBatched(grammar.maskedLogits(logits), temperature: temperature)
            eval(next)
        }
        return out
    }

    /// Public entry: decode B distinct prompts in one batched pass, each row's output a valid-by-construction
    /// JSON for `fields` (evicts a row when its JSON closes). This is the batched analogue of
    /// `generateConstrained` — the constrained-fan-out the sub-agent path needs.
    public func batchGenerateConstrained(_ prompts: [String], fields: [SchemaProperty],
                                         tokenizer: any GrammarTokenizer, maxTokens: Int = 128,
                                         runtimeEnums: [String: [String]]? = nil) async -> [String] {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let rows: [[Int32]] = prompts.map {
                ((try? tok.applyChatTemplate(messages: [["role": "user", "content": $0]])) ?? []).map { Int32($0) }
            }
            let outIds = self.batchDecodeConstrained(
                model: ctx.model, promptRows: rows, fields: fields, tokenizer: tokenizer,
                maxTokens: maxTokens, temperature: 0, stops: self.batchStops(tok),
                stopOnEOS: true, stopOnComplete: true, runtimeEnums: runtimeEnums)
            return outIds.map { tok.decode(tokenIds: $0) }
        }
    }

    /// SLICE 4 MEASUREMENT — is per-row grammar in a batched pass worth it? Times, on the SAME B prompts +
    /// `fields`, fixed-length (no early stop) greedy decode for: (a) batched UNconstrained, (b) batched
    /// CONSTRAINED, (c) constrained SOLO (B sequential 1-row constrained runs, same masking code at B=1).
    /// Returns the three wall-times (seconds). Verdict: batched grammar is worth it iff (c)/(b) ≫ 1 while
    /// (b)/(a) — the per-step mask-build overhead — stays modest.
    public func grammarBatchBench(prompts: [String], fields: [SchemaProperty],
                                  tokenizer: any GrammarTokenizer, maxTokens: Int = 64) async
        -> (batchedUnconstrained: Double, batchedConstrained: Double, soloConstrained: Double, batch: Int) {
        await container.perform { ctx in
            let tok = ctx.tokenizer
            let rows: [[Int32]] = prompts.map {
                ((try? tok.applyChatTemplate(messages: [["role": "user", "content": $0]])) ?? []).map { Int32($0) }
            }
            let noStops: Set<Int> = []

            // warm-up (build kernels / page weights) so the first timed run isn't penalized
            _ = self.batchDecodeDistinct(model: ctx.model, promptRows: rows, maxTokens: 4,
                                         temperature: 0, stops: noStops, stopOnEOS: false)

            func time(_ body: () -> Void) -> Double {
                let t0 = Date(); body(); return Date().timeIntervalSince(t0)
            }

            // (a) batched, unconstrained — the batching baseline.
            let tA = time {
                _ = self.batchDecodeDistinct(model: ctx.model, promptRows: rows, maxTokens: maxTokens,
                                             temperature: 0, stops: noStops, stopOnEOS: false)
            }
            // (b) batched, per-row grammar constrained.
            let tB = time {
                _ = self.batchDecodeConstrained(model: ctx.model, promptRows: rows, fields: fields,
                                                tokenizer: tokenizer, maxTokens: maxTokens, temperature: 0,
                                                stops: noStops, stopOnEOS: false, stopOnComplete: false)
            }
            // (c) constrained, SOLO — each prompt run alone (B=1), sequentially; same masking code path.
            let tC = time {
                for row in rows {
                    _ = self.batchDecodeConstrained(model: ctx.model, promptRows: [row], fields: fields,
                                                    tokenizer: tokenizer, maxTokens: maxTokens, temperature: 0,
                                                    stops: noStops, stopOnEOS: false, stopOnComplete: false)
                }
            }
            return (tA, tB, tC, rows.count)
        }
    }
}
