// AUTO-GENERATED (swift-transformers 2nd flywheel repo, Jun 17 2026) by taskscan-st.swift +
// the prompt-authoring workflow. 18 tokenizer/text-algorithm tasks (a DIFFERENT domain from
// the math repo) → trace diversity to push held-out lift off the noise floor. Do NOT hand-edit.
import Foundation

extension DomainEvalSuite {
    /// 2nd golden: HuggingFace swift-transformers (your fork; XCTest harness). Pure SwiftPM,
    /// ~0.76s incremental rebuild. Tokenizer/text algorithms (Trie, BPE split, normalizers, etc.).
    public static let swiftTransformers = DomainRepo(
        path: "/Users/jason/Documents/swift-transformers",
        testProduct: "swift-transformersPackageTests")

    public static let autoTasks2: [DomainTask] = [
        DomainTask(
            id: "Trie.insert", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Trie.swift",
            bodyStartLine: 23, bodyEndLine: 33,
            coveringTests: ["TrieTests/testTrieBuilding"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "5f35683340892564",
            distinctiveSubstrings: [
                "if let child = node.children[item] {",
                "node.children[item] = child",
                "for item in element {",
            ],
            prompt: DomainPrompts.trieInsert),
        DomainTask(
            id: "Trie.commonPrefixSearch", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Trie.swift",
            bodyStartLine: 43, bodyEndLine: 54,
            coveringTests: ["TrieTests/testTrieCommonPrefixSearch"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "98f93233210275af",
            distinctiveSubstrings: [
                "var seqs: [[T]] = []",
                "for item in text {",
                "var seq: [T] = []",
            ],
            prompt: DomainPrompts.trieCommonPrefixSearch),
        DomainTask(
            id: "String.splitByBehavior", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PreTokenizer.swift",
            bodyStartLine: 348, bodyEndLine: 394,
            coveringTests: ["SplitTests/testSplitBehaviorMergedWithPrevious"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "43477af81868fc8e",
            distinctiveSubstrings: [
                "func mergedWithPrevious(ranges: [Range<String.Index>]) -> [Range<String.Index>] {",
                "func mergedWithNext(ranges: [Range<String.Index>]) -> [Range<String.Index>] {",
                "return split(by: string, options: options, includeSeparators: false)",
            ],
            prompt: DomainPrompts.stringSplitByBehavior),
        DomainTask(
            id: "ByteLevelPreTokenizer.preTokenize", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PreTokenizer.swift",
            bodyStartLine: 188, bodyEndLine: 197,
            coveringTests: ["PreTokenizerTests/testByteLevelPreTokenizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "6eda9c78263945b4",
            distinctiveSubstrings: [
                "let tokens = useRegex ? text.ranges(of: RE).map({ String(text[$0]) }) : [text]",
                "return Array(token.utf8).map { byteEncoder[$0]! }.joined()",
                "if addPrefixSpace && !token.hasPrefix(\" \") {",
            ],
            prompt: DomainPrompts.byteLevelPreTokenizerPreTokenize),
        DomainTask(
            id: "DigitsPreTokenizer.preTokenize", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PreTokenizer.swift",
            bodyStartLine: 223, bodyEndLine: 223,
            coveringTests: ["PreTokenizerTests/testDigitsPreTokenizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "df6e0c448285cfbf",
            distinctiveSubstrings: [
                "return text.ranges(of: re).map { String(text[$0]) }",
            ],
            prompt: DomainPrompts.digitsPreTokenizerPreTokenize),
        DomainTask(
            id: "MetaspacePreTokenizer.preTokenize", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PreTokenizer.swift",
            bodyStartLine: 149, bodyEndLine: 171,
            coveringTests: ["PreTokenizerTests/testMetaspacePreTokenizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "5d067edf68e76d53",
            distinctiveSubstrings: [
                "return (prepend + normalized).split(by: replacement, behavior: .mergedWithNext)",
                "let normalized = text.replacingOccurrences(of: \" \", with: stringReplacement)",
                "if prependScheme == .first && options.contains(.firstSection) {",
            ],
            prompt: DomainPrompts.metaspacePreTokenizerPreTokenize),
        DomainTask(
            id: "SplitPreTokenizer.preTokenize", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PreTokenizer.swift",
            bodyStartLine: 237, bodyEndLine: 238,
            coveringTests: ["PreTokenizerTests/testSplitPreTokenizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "66d4d20feee3ff67",
            distinctiveSubstrings: [
                "return pattern.split(text, invert: invert)",
            ],
            prompt: DomainPrompts.splitPreTokenizerPreTokenize),
        DomainTask(
            id: "BertNormalizer.cleanText", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Normalizer.swift",
            bodyStartLine: 178, bodyEndLine: 192,
            coveringTests: ["NormalizerTests/testBertNormalizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "d65edbcf4969501c",
            distinctiveSubstrings: [
                "if scalar.value == 0x009 || scalar.value == 0x00A || scalar.value == 0x000D {",
                "scalar.value != 0xFFFD,",
                "else { return \"\\(c)\" }",
            ],
            prompt: DomainPrompts.bertNormalizerCleanText),
        DomainTask(
            id: "BertNormalizer.handleChineseChars", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Normalizer.swift",
            bodyStartLine: 211, bodyEndLine: 218,
            coveringTests: ["NormalizerTests/testBertNormalizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "e5a4a94593cc8063",
            distinctiveSubstrings: [
                "if let scalar = c.unicodeScalars.first, Utils.isChineseChar(scalar) {",
                "text.map { c in",
            ],
            prompt: DomainPrompts.bertNormalizerHandleChineseChars),
        DomainTask(
            id: "BertNormalizer.stripAccents", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Normalizer.swift",
            bodyStartLine: 222, bodyEndLine: 227,
            coveringTests: ["NormalizerTests/testBertNormalizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "e9b76d267bf3f669",
            distinctiveSubstrings: [
                "!(0x0300 <= scalar.value && scalar.value <= 0x036F)",
                "text.decomposedStringWithCanonicalMapping",
                "$0.unicodeScalars.allSatisfy { scalar in",
            ],
            prompt: DomainPrompts.bertNormalizerStripAccents),
        DomainTask(
            id: "PrecompiledNormalizer.normalize", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Normalizer.swift",
            bodyStartLine: 236, bodyEndLine: 269,
            coveringTests: ["NormalizerTests/testPrecompiledNormalizer"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "c895c7f11e97f75a",
            distinctiveSubstrings: [
                "case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B...0x200F, 0x2028, 0x2029, 0x2581,",
                "case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F:",
                "return output.precomposedStringWithCompatibilityMapping",
            ],
            prompt: DomainPrompts.precompiledNormalizerNormalize),
        DomainTask(
            id: "RobertaProcessing.postProcess", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PostProcessor.swift",
            bodyStartLine: 110, bodyEndLine: 129,
            coveringTests: ["PostProcessorTests/testRobertaProcessing"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "f03da2df2440fbcf",
            distinctiveSubstrings: [
                "tokensPair = tokensPair?.map({ $0.trimmingCharacters(in: .whitespaces) })",
                "outTokens = outTokens.map({ $0.trimmingCharacters(in: .whitespaces) })",
                "tokensPair = tokensPair?.map({ trimExtraSpaces(token: $0) })",
            ],
            prompt: DomainPrompts.robertaProcessingPostProcess),
        DomainTask(
            id: "RobertaProcessing.trimExtraSpaces", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/PostProcessor.swift",
            bodyStartLine: 135, bodyEndLine: 139,
            coveringTests: ["PostProcessorTests/testRobertaProcessing"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "0936ceb1b0150d8c",
            distinctiveSubstrings: [
                "let suffixIndex = token.index(token.startIndex, offsetBy: token.count - suffixOffset)",
                "let prefixIndex = token.index(token.startIndex, offsetBy: prefixOffset)",
                "let prefixOffset = findPrefixIndex(text: token)",
            ],
            prompt: DomainPrompts.robertaProcessingTrimExtraSpaces),
        DomainTask(
            id: "MetaspaceDecoder.decode", repo: swiftTransformers,
            targetFileRel: "Sources/Tokenizers/Decoder.swift",
            bodyStartLine: 233, bodyEndLine: 239,
            coveringTests: ["DecoderTests/testMetaspaceDecoder"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "23f59d0901e912d9",
            distinctiveSubstrings: [
                "if addPrefixSpace && replaced.first?.starts(with: \" \") ?? false {",
                "token.replacingOccurrences(of: replacement, with: \" \")",
                "var replaced = tokens.map { token in",
            ],
            prompt: DomainPrompts.metaspaceDecoderDecode),
        DomainTask(
            id: "TopKLogitsWarper.warp", repo: swiftTransformers,
            targetFileRel: "Sources/TensorUtils/LogitsWarper/TopKLogitsWarper.swift",
            bodyStartLine: 17, bodyEndLine: 57,
            coveringTests: ["LogitsWarperTests/testTopKLogitsWarper"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "f42d7b76a2386636",
            distinctiveSubstrings: [
                "let topkIndices = bestIndices.data!.withMemoryRebound(to: Int32.self, capacity: k) { ptr in",
                "let topkLogits = bestValues.data!.withMemoryRebound(to: Float.self, capacity: k) { ptr in",
                "return (indices: topkIndices.map { indices[Int($0)] }, logits: topkLogits)",
            ],
            prompt: DomainPrompts.topKLogitsWarperWarp),
        DomainTask(
            id: "TopPLogitsWarper.warp", repo: swiftTransformers,
            targetFileRel: "Sources/TensorUtils/LogitsWarper/TopPLogitsWarper.swift",
            bodyStartLine: 14, bodyEndLine: 35,
            coveringTests: ["LogitsWarperTests/testTopPLogitsWarper"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "5bc380348a4669b2",
            distinctiveSubstrings: [
                "let toppIndices = indexLogitProb[0 ... sliceIndex].map { indices[$0.index] }",
                "indexLogitProb.append((index: index, logit: data.0, prob: data.1))",
                "var indexLogitProb = [(index: Int, logit: Float, prob: Float)]()",
            ],
            prompt: DomainPrompts.topPLogitsWarperWarp),
        DomainTask(
            id: "RepetitionPenaltyWarper.warp", repo: swiftTransformers,
            targetFileRel: "Sources/TensorUtils/LogitsWarper/RepetitionPenaltyWarper.swift",
            bodyStartLine: 14, bodyEndLine: 23,
            coveringTests: ["LogitsWarperTests/testRepetitionPenaltyWarper"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "cd2873cd65334b11",
            distinctiveSubstrings: [
                "for index in indices.indices {",
                "logits[index] *= penalty",
                "logits[index] /= penalty",
            ],
            prompt: DomainPrompts.repetitionPenaltyWarperWarp),
        DomainTask(
            id: "Math.cumsum", repo: swiftTransformers,
            targetFileRel: "Sources/TensorUtils/Math.swift",
            bodyStartLine: 74, bodyEndLine: 83,
            coveringTests: ["TensorUtilsTests/testCumsum"],
            siblingTests: [],
            docCommentLeaksFormula: false, bodyHash: "3aa5640274581be6",
            distinctiveSubstrings: [
                "var result: [Float] = Array(repeating: 0.0, count: arr.count)",
                "vDSP_vsadd(result, 1, &firstItem, &result, 1, arrCount)",
                "vDSP_vrsum(arr, 1, &weight, &result, 1, arrCount)",
            ],
            prompt: DomainPrompts.mathCumsum),
    ]
}
