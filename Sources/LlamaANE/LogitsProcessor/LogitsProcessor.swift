import Foundation
import CoreML

extension MLTensor {
//    func topP(_ p: Float16, indices: MLTensor) async -> (values: MLTensor, indices: MLTensor) {
//        // Step 2: Compute softmax probabilities
//        let softmaxProbs = self.softmax(alongAxis: 0)
//        
//        // Step 3: Sort probabilities in descending order and get sorted indices
//        let sortedProbs = softmaxProbs.argsort(descendingOrder: true)
//        let softmaxProbsArray = await softmaxProbs.shapedArray(of: Float.self)
//        let sortedProbsArray = await sortedProbs.shapedArray(of: Int32.self)
//        let indicesArray = await indices.shapedArray(of: Int32.self)
//        let logitsArray = await self.shapedArray(of: Float.self)
//        let sortedProbsTensor = MLTensor(sortedProbsArray.map {
//            softmaxProbsArray[scalarAt: Int($0.scalar!)]
//        })
//        let sortedIndicesTensor = MLTensor(sortedProbsArray.map {
//            indicesArray[scalarAt: Int($0.scalar!)]
//        })
//        let sortedLogitsTensor = MLTensor(sortedProbsArray.map {
//            logitsArray[scalarAt: Int($0.scalar!)]
//        })
//        // Step 4: Gather sorted probabilities and corresponding sorted indices
////        let sortedProbsTensor = softmaxProbs.gathering(atIndices: sortedProbs, alongAxis: -1)
////        let sortedIndicesTensor = indices.gathering(atIndices: sortedProbs, alongAxis: -1)
////        let sortedLogitsTensor = self.gathering(atIndices: sortedProbs, alongAxis: -1)
//        
//        // Step 5: Compute cumulative sum of sorted probabilities
//        let cumsumProbsTensor = sortedProbsTensor.cumulativeSum()
//        // Step 6: Compare cumulative sum with threshold p to create a condition tensor
//        let conditionTensor = cumsumProbsTensor .> p
//        
//        // Step 7: Convert condition tensor to shaped array of Int32 (1 for true, 0 for false)
//        let conditionArray = await conditionTensor.cast(to: Int32.self).shapedArray(of: Int32.self)
//        
//        // Step 8: Find the first index where the cumulative sum exceeds p
//        let conditionScalars = conditionArray.scalars
//        guard var cutoffIndex = conditionScalars.firstIndex(of: 1) else {
//            // If no cutoff index found, include all
////            let allIndices = await sortedIndicesTensor.shapedArray(of: Int32.self)
////            let allProbs = await sortedProbsTensor.shapedArray(of: Float16.self)
//            return (sortedProbsTensor, sortedIndicesTensor)
//        }
//        if cutoffIndex == 0 {
//            cutoffIndex = logitsArray.count - 1
//        }
//        // Step 9: Slice the sorted indices and probabilities up to the cutoff index
//        let slicedIndicesTensor = sortedIndicesTensor[0...cutoffIndex]
//        let slicedProbsTensor = sortedLogitsTensor[0...cutoffIndex]
//        return (slicedProbsTensor, slicedIndicesTensor)
//    }
    /// Nucleus (top-p) sampling filter.
    ///
    /// This function filters tokens to keep only those whose cumulative probability
    /// is within the top-p threshold. It returns LOGITS (not probabilities) so the
    /// caller can apply softmax once at the end for sampling.
    ///
    /// - Parameters:
    ///   - p: The cumulative probability threshold (0.0-1.0)
    ///   - indices: Token indices corresponding to the logits
    /// - Returns: Filtered (logits, indices) tuple, sorted by probability descending
    func topP(_ p: Float16, indices: MLTensor) async -> (values: MLTensor, indices: MLTensor) {
        // For small tensors (post-topK), use fast array-based implementation
        let logitsArray = await self.shapedArray(of: Float.self).scalars
        let indicesArray = await indices.shapedArray(of: Int32.self).scalars

        guard !logitsArray.isEmpty else {
            return (self, indices)
        }

        // Compute softmax probabilities
        let maxLogit = logitsArray.max() ?? 0
        let exps = logitsArray.map { Foundation.exp($0 - maxLogit) }
        let expSum = exps.reduce(0, +)
        let probs = exps.map { $0 / expSum }

        // Create indexed tuples and sort by probability descending
        var indexed = zip(0..<logitsArray.count, zip(logitsArray, zip(indicesArray, probs)))
            .map { ($0, $1.0, $1.1.0, $1.1.1) }  // (origIdx, logit, tokenIdx, prob)
        indexed.sort { $0.3 > $1.3 }  // Sort by prob descending

        // Find cutoff where cumulative probability exceeds p
        var cumsum: Float = 0
        var cutoff = indexed.count
        for (i, (_, _, _, prob)) in indexed.enumerated() {
            cumsum += prob
            if cumsum > Float(p) {
                cutoff = i + 1
                break
            }
        }
        cutoff = Swift.max(cutoff, 1)  // Keep at least 1 token

        // Extract filtered results
        let filtered = Array(indexed.prefix(cutoff))
        let filteredLogits = filtered.map { $0.1 }
        let filteredIndices = filtered.map { $0.2 }

        // Return as tensors with same shape as input
        let shape = self.shape
        if shape.count == 1 {
            return (
                MLTensor(shape: [cutoff], scalars: filteredLogits),
                MLTensor(shape: [cutoff], scalars: filteredIndices)
            )
        } else {
            return (
                MLTensor(shape: [1, cutoff], scalars: filteredLogits),
                MLTensor(shape: [1, cutoff], scalars: filteredIndices)
            )
        }
    }
    
    func penalizeRepetition(_ penalty: Float16, atIndices indices: MLTensor) async -> (logits: MLTensor,
                                                                                       indices: MLTensor) {

        // Step 2: Create a mask where gathered logits are negative
        let negativeMask = self .< 0
        let positiveMask = self .> 0
        let penalized = self.replacing(with: self * penalty, where: negativeMask)
        return (penalized.replacing(with: self / penalty, where: positiveMask), indices)
    }

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
