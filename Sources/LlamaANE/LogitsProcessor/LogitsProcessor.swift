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
    func topP(_ p: Float16, indices: MLTensor) async -> (values: MLTensor, indices: MLTensor) {
        // Step 2: Compute softmax probabilities
        let softmaxProbs = self.softmax()
        
        // Step 3: Sort probabilities in descending order and get sorted indices
        let sortedProbs = softmaxProbs.argsort(descendingOrder: true)
        
        // Step 4: Gather sorted probabilities and corresponding sorted indices
        let sortedProbsTensor = softmaxProbs.gathering(atIndices: sortedProbs, alongAxis: -1)
        let sortedIndicesTensor = indices.gathering(atIndices: sortedProbs, alongAxis: -1)
        let sortedLogitsTensor = self.gathering(atIndices: sortedProbs, alongAxis: -1)
        
        // Step 5: Compute cumulative sum of sorted probabilities
        let cumsumProbsTensor = sortedProbsTensor.cumulativeSum(alongAxis: -1)
        
        // Step 6: Compare cumulative sum with threshold p to create a condition tensor
        let conditionTensor = cumsumProbsTensor .> p
        
        // Step 7: Convert condition tensor to shaped array of Int32 (1 for true, 0 for false)
        let conditionArray = await conditionTensor.cast(to: Int32.self).shapedArray(of: Int32.self)
        
        // Step 8: Find the first index where the cumulative sum exceeds p
        let conditionScalars = conditionArray.scalars
        guard let cutoffIndex = conditionScalars.firstIndex(of: 1) else {
            // If no cutoff index found, include all
//            let allIndices = await sortedIndicesTensor.shapedArray(of: Int32.self)
//            let allProbs = await sortedProbsTensor.shapedArray(of: Float16.self)
            return (sortedProbsTensor, sortedIndicesTensor)
        }
        
        // Step 9: Slice the sorted indices and probabilities up to the cutoff index
        if sortedIndicesTensor.rank == 1 {
            let slicedIndicesTensor = sortedIndicesTensor[0...cutoffIndex]
            let slicedProbsTensor = sortedLogitsTensor[0...cutoffIndex]
            return (slicedProbsTensor, slicedIndicesTensor)
        } else {
            let slicedIndicesTensor = sortedIndicesTensor[..., 0...cutoffIndex]
            let slicedProbsTensor = sortedLogitsTensor[..., 0...cutoffIndex]
            return (slicedProbsTensor, slicedIndicesTensor)
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

        // Get unique token IDs
        let uniqueTokenIds = Array(Set(generatedTokenIds))
        let vocabSize = await self.shapedArray(of: Float.self).scalarCount

        // Create a penalty mask tensor (1.0 for non-repeated tokens, penalty factor for repeated)
        // For positive logits: divide by penalty (use 1/penalty)
        // For negative logits: multiply by penalty
        // We'll create two masks and apply them based on sign

        var penaltyMask = [Float](repeating: 1.0, count: vocabSize)
        let logitsArray = await self.shapedArray(of: Float.self)

        for tokenId in uniqueTokenIds {
            guard tokenId >= 0 && tokenId < vocabSize else { continue }
            let logit = logitsArray[scalarAt: tokenId]
            if logit > 0 {
                penaltyMask[tokenId] = 1.0 / penalty
            } else if logit < 0 {
                penaltyMask[tokenId] = penalty
            }
        }

        let penaltyTensor = MLTensor(shape: [vocabSize], scalars: penaltyMask)
        return self * penaltyTensor
    }
}
