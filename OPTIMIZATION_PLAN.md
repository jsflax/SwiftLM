# LlamaANE Optimization Analysis

Analysis of suggested optimizations against current codebase state.

---

## 1. Speculative Decoding ⭐ HIGH IMPACT - NOT IMPLEMENTED

**Status**: Not implemented. Would require significant architecture changes.

**What it does**: Use a small draft model (e.g., Qwen 0.5B) to propose multiple tokens, then verify them in a single forward pass of the main model.

**Implementation complexity**: High
- Need to export and load two models
- Verify tokens in parallel
- Handle rejection/acceptance logic

**Estimated speedup**: 2-3x for longer generations

**Recommendation**: Worth implementing as a future feature, but requires careful design.

---

## 2. Causal Mask Rebuilt Every Token ⭐ HIGH IMPACT - FIXABLE

**Status**: STILL AN ISSUE in `LlamaSession.swift:72-84`

**Current code**:
```swift
if requiresCausal {
    queryLen = min(totalBuffer.count, maxContextSize)
    let endStepDim = queryLen
    let shape = [1, 1, queryLen, endStepDim]  // Full square mask every time!
    let onesTensor = MLTensor(ones: shape, scalarType: Float16.self)
    let causalMask = await onesTensor.bandPart(lowerBandCount: -1, upperBandCount: 0)
```

**Problem**: For incremental decoding (single new token), we rebuild a full NxN mask when we only need a 1xN mask.

**Fix**: For single-token generation, use a simple all-ones row:
```swift
if requiresCausal {
    let kvLen = min(totalBuffer.count, maxContextSize)
    let queryLen = min(tokens.count, maxContextSize)

    let causalMask: MLShapedArray<Float16>
    if queryLen == 1 {
        // Incremental: new token attends to all previous
        causalMask = MLShapedArray<Float16>(
            repeating: 1.0,
            shape: [1, 1, 1, kvLen]
        )
    } else {
        // Prefill: full lower triangular
        let shape = [1, 1, kvLen, kvLen]
        let onesTensor = MLTensor(ones: shape, scalarType: Float16.self)
        causalMask = await onesTensor.bandPart(lowerBandCount: -1, upperBandCount: 0)
            .shapedArray(of: Float16.self)
    }
}
```

**Estimated speedup**: 10-20% for token generation loop (mask creation is async and involves tensor ops)

**Recommendation**: IMPLEMENT - Simple fix with measurable impact.

---

## 3. Batch Grammar Token Filtering ⭐ MEDIUM IMPACT - FIXABLE

**Status**: STILL AN ISSUE in `Grammar.swift:420-435`

**Current code**:
```swift
func applyPenalty(_ tensor: MLTensor) async -> MLTensor {
    let xShaped = await tensor.shapedArray(of: Float.self)  // Expensive await every token
    let disallowedIndices = xShaped.enumerated().compactMap { ... }
```

**Problem**: Converts tensor to shaped array on every token, iterates through all vocab indices.

**Fix**: Pre-compute valid token masks per grammar state:
```swift
private var cachedMasks: [GrammarState: MLTensor] = [:]

func getOrCreateMask(for state: GrammarState) -> MLTensor {
    if let cached = cachedMasks[state] { return cached }
    let validTokens = getValidTokens(for: state)
    // Create additive mask: 0 for valid, -inf for invalid
    var maskData = [Float](repeating: -Float.greatestFiniteMagnitude, count: vocabSize)
    for token in validTokens { maskData[token] = 0 }
    let mask = MLTensor(shape: [vocabSize], scalars: maskData)
    cachedMasks[state] = mask
    return mask
}
```

**Estimated speedup**: 5-15% for grammar-constrained generation

**Recommendation**: IMPLEMENT - Grammar states are finite, caching is effective.

---

## 4. Use MLTensor Operations Instead of Swift Loops ⭐ MEDIUM IMPACT - PARTIALLY IMPLEMENTED

**Status**: PARTIALLY ADDRESSED in `LlamaANE.swift:823-834`

**Current code**:
```swift
if mlTensor.shape[1] > 1 {
    var logits = await mlTensor.shapedArray(of: Float.self).scalars  // CPU pull
    var indices = Array(logits.indices)
    (indices, logits) = TopKLogitsWarper(k: config.topK).warp(...)  // CPU sort
} else {
    (mlTensor, otherIndices) = mlTensor.topK(Int(config.topK))  // GPU topK
}
```

**Analysis**: When `shape[1] > 1` (prefill), it falls back to CPU. For incremental (shape[1] == 1), it uses MLTensor.topK which is better.

**The branch condition is confusing**: `shape[1]` is sequence length, not batch. After prefill, we always have shape[1] == 1 for incremental tokens, so the GPU path IS used for most generation.

**Remaining issue**: The `topP` and `penalizeRepetition` still involve `await` and tensor conversions.

**Recommendation**: LOW PRIORITY - The hot path already uses MLTensor.topK. Further optimization would require custom Metal kernels.

---

## 5. Reduce Model Size / Block Size ⚠️ MARGINAL IMPACT

**Status**: Already using INT4 with block_size=32

**Current code** in `export.py`:
```python
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_block",
    block_size=32,
)
```

**Suggestion**: Try block_size=16 for better ANE scheduling.

**Analysis**: Block size 16 would increase metadata overhead. The performance difference is likely minimal and model-dependent.

**Recommendation**: LOW PRIORITY - Test with a specific model if curious, but don't expect significant gains.

---

## 6. Prompt Caching for System Prompts ⭐ HIGH IMPACT - NOT IMPLEMENTED

**Status**: NOT IMPLEMENTED - System prompts are re-processed every conversation.

**Current behavior**: Each `makeSession()` starts fresh and re-encodes the system prompt.

**Fix**: Cache KV state after processing system prompt:
```swift
actor CachedSession {
    private var systemPromptKVState: MLState?
    private var systemPromptTokenCount: Int = 0

    func initializeWithSystemPrompt(_ prompt: String) async throws {
        // Process system prompt once
        let tokens = tokenizer.encode(text: formatSystemPrompt(prompt))
        systemPromptTokenCount = tokens.count
        let input = await input(from: tokens, totalBuffer: tokens)
        _ = try await model.prediction(from: input, using: kvCache)
        systemPromptKVState = kvCache  // Save state
    }

    func resetConversation() {
        // Restore to cached state instead of re-processing
        // Note: MLState may not support copying - need to verify API
    }
}
```

**Challenge**: Need to verify if `MLState` supports cloning/copying.

**Estimated speedup**: Saves 20-50 tokens of processing per conversation reset.

**Recommendation**: IMPLEMENT IF FEASIBLE - Check MLState API for copy capability.

---

## 7. Model Warmup ⭐ MEDIUM IMPACT - NOT IMPLEMENTED

**Status**: NOT IMPLEMENTED

**Problem**: First inference is slow due to ANE cold start.

**Fix**: Add warmup method:
```swift
extension LanguageModel {
    public func warmup() async throws {
        let dummyTokens = [tokenizer.bosTokenId ?? 1]
        let input = try await makeInput(from: dummyTokens, totalBuffer: dummyTokens)
        if let kvCache = model.makeState() {
            _ = try await model.prediction(from: input, using: kvCache)
        }
    }
}
```

**Recommendation**: IMPLEMENT - Simple addition, improves perceived latency.

---

## 8. Model Recommendations

**Current**: Llama 3.2 3B INT4, Qwen 2.5 1.5B

**For JSON/structured output**: Qwen tends to follow schemas better than Llama at same size.

**Size recommendations**:
| Use Case | Model | Size | RAM |
|----------|-------|------|-----|
| Fast responses | Qwen 2.5 0.5B INT4 | ~300MB | ~400MB |
| Balanced | Qwen 2.5 1.5B INT4 | ~900MB | ~1.2GB |
| Best quality | Qwen 2.5 3B INT4 | ~1.8GB | ~2.5GB |

---

## Priority Summary

| # | Optimization | Impact | Effort | Recommend |
|---|-------------|--------|--------|-----------|
| 2 | Fix causal mask rebuild | High | Low | ✅ DO NOW |
| 7 | Model warmup | Medium | Low | ✅ DO NOW |
| 3 | Cache grammar masks | Medium | Medium | ✅ NEXT |
| 6 | System prompt caching | High | Medium | ✅ IF FEASIBLE |
| 1 | Speculative decoding | Very High | High | 🔮 FUTURE |
| 4 | MLTensor ops | Low | High | ⏸️ SKIP |
| 5 | Block size tuning | Marginal | Low | ⏸️ SKIP |
