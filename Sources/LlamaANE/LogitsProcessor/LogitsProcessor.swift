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
}
