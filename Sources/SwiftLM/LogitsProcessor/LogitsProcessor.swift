import Foundation
import CoreML

extension MLTensor {
    /// Apply repetition penalty to logits based on previously generated tokens.
    /// This is the standard implementation matching HuggingFace Transformers.
    ///
    /// For each token in `generatedTokenIds`:
    /// - If logit > 0: divide by penalty (reduces probability of repeating)
    /// - If logit < 0: multiply by penalty (makes it even less likely)
    ///
    /// - Parameters:
    ///   - penalty: The repetition penalty factor (typically 1.0-1.5, where 1.0 = no penalty)
    ///   - generatedTokenIds: Array of token IDs that have been generated so far
    /// - Returns: Modified logits tensor with penalties applied
    func applyRepetitionPenalty(_ penalty: Float, generatedTokenIds: [Int]) async -> MLTensor {
        guard penalty != 1.0, !generatedTokenIds.isEmpty else {
            return self
        }

        // Get unique token IDs and vocab size from tensor shape
        let uniqueTokenIds = Array(Set(generatedTokenIds))
        let logitsArray = await self.shapedArray(of: Float.self)
        let vocabSize = logitsArray.scalarCount

        // Build multiplier array on CPU, apply on GPU
        // For positive logits at repeated positions: multiply by 1/penalty
        // For negative logits at repeated positions: multiply by penalty
        var multipliers = [Float](repeating: 1.0, count: vocabSize)

        for tokenId in uniqueTokenIds {
            guard tokenId >= 0 && tokenId < vocabSize else { continue }
            let logit = logitsArray[scalarAt: tokenId]
            if logit > 0 {
                multipliers[tokenId] = 1.0 / penalty
            } else if logit < 0 {
                multipliers[tokenId] = penalty
            }
        }

        // Single GPU multiply operation
        let multiplierTensor = MLTensor(shape: [vocabSize], scalars: multipliers)
        return self * multiplierTensor
    }
}
