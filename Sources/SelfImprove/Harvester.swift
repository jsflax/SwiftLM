import Foundation

/// A clean (human instruction -> assistant final answer) pair, ready to format.
struct Pair: Hashable { let human: String; let assistant: String }

/// Extracts clean chat pairs from transcripts: for each assistant turn that produced
/// final `text`, walk parentUuid up to the originating bare-string human prompt.
/// Filters noise (sidechains, meta, slash-command/synthetic/system-reminder wrappers).
struct Harvester {
    var minAssistant = 120          // assistant answer must have substance
    var maxHuman = 4000             // skip giant pastes as the "instruction"
    var maxAssistant = 6000         // cap answer length (token budget)

    /// v2a leakage fence: drops examples containing held-out eval-set bodies + quarantines edits to
    /// the eval repo. `Redactor` handles secrets; this handles eval contamination (separate concern).
    let decon = Decontaminator()

    /// Chat-template / FIM control tokens that must NEVER appear in training *content*
    /// (they're structural). Our own transcripts discuss the tokenizer and literally
    /// contain "<|im_start|>" etc. — training on those taught the model to emit control
    /// tokens (the observed garbled-output leak). `<tool_call>` is intentionally allowed
    /// (the tool-trace stream teaches the model to emit it). Drop any contaminated example.
    static let controlTokens = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|fim_", "<|object_ref",
        "<|box_", "<|quad_", "<|vision_", "<|image_pad|>", "<|video_pad|>",
        "<|repo_name|>", "<|file_sep|>",
    ]
    func hasControlToken(_ s: String) -> Bool { Self.controlTokens.contains { s.contains($0) } }

    /// Strip harness wrapper blocks from a human turn; return nil if nothing human remains.
    private func cleanHuman(_ s: String) -> String? {
        var t = s
        // Remove system-reminder / local-command / command-* / synthetic wrapper blocks.
        for pat in [
            #"<system-reminder>[\s\S]*?</system-reminder>"#,
            #"<local-command-[\s\S]*?</local-command-[a-z]+>"#,
            #"<command-[a-z]+>[\s\S]*?</command-[a-z]+>"#,
            #"<synthetic>[\s\S]*?</synthetic>"#,
            #"<bash-(input|stdout|stderr)>[\s\S]*?</bash-(input|stdout|stderr)>"#,
        ] {
            t = t.replacingOccurrences(of: pat, with: " ", options: .regularExpression)
        }
        let trimmed = t.trimmingCharacters(in: .whitespacesAndNewlines)
        // Drop turns that were *only* a slash-command / caveat / pasted noise.
        if trimmed.count < 8 { return nil }
        if trimmed.hasPrefix("Caveat:") { return nil }
        if trimmed.hasPrefix("[Request interrupted") { return nil }
        return trimmed
    }

    /// Extract pairs from one file's records.
    func pairs(from records: [Record]) -> [Pair] {
        // Index by uuid for parent walking.
        var byUuid: [String: Record] = [:]
        for r in records { if let u = r.uuid { byUuid[u] = r } }

        // Find the nearest ancestor that is a bare-string human prompt.
        func originHuman(of rec: Record) -> String? {
            var cur: Record? = rec
            var hops = 0
            while let c = cur, hops < 40 {
                if c.message?.role == "user", let content = c.message?.content,
                   content.isBareString, let s = content.texts.first {
                    return s
                }
                cur = c.parentUuid.flatMap { byUuid[$0] }
                hops += 1
            }
            return nil
        }

        var out: [Pair] = []
        var seen = Set<Pair>()
        for r in records {
            guard r.type == "assistant", r.isSidechain != true, r.isMeta != true else { continue }
            let answer = (r.message?.content?.texts ?? [])
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard answer.count >= minAssistant, answer.count <= maxAssistant else { continue }
            guard let rawHuman = originHuman(of: r), rawHuman.count <= maxHuman,
                  let human = cleanHuman(rawHuman) else { continue }
            guard !hasControlToken(human), !hasControlToken(answer) else { continue }  // data hygiene
            guard !decon.isContaminatedPair(human, answer) else { continue }           // v2a leakage fence
            let p = Pair(human: human, assistant: answer)
            if seen.insert(p).inserted { out.append(p) }
        }
        return out
    }

    /// Extract tool-call traces: for each assistant turn that emitted a `tool_use`, pair the
    /// originating request with the model's tool call in Qwen `<tool_call>` format. Teaches the
    /// model to EMIT tool calls (the signal chat-pairs lack — without it, training kills tool-use).
    func toolTraces(from records: [Record]) -> [Pair] {
        var byUuid: [String: Record] = [:]
        for r in records { if let u = r.uuid { byUuid[u] = r } }
        func originHuman(of rec: Record) -> String? {
            var cur: Record? = rec
            var hops = 0
            while let c = cur, hops < 40 {
                if c.message?.role == "user", let content = c.message?.content,
                   content.isBareString, let s = content.texts.first { return s }
                cur = c.parentUuid.flatMap { byUuid[$0] }
                hops += 1
            }
            return nil
        }

        var out: [Pair] = []
        var seen = Set<Pair>()
        for r in records {
            guard r.type == "assistant", r.isSidechain != true, r.isMeta != true else { continue }
            // The first tool_use of the turn = the action the assistant chose.
            guard let tu = (r.message?.content?.blocks ?? []).first(where: { $0.type == "tool_use" }),
                  let name = tu.name else { continue }
            guard !decon.isQuarantinedEdit(toolName: tu.name, input: tu.input) else { continue }  // v2a provenance fence
            guard let rawHuman = originHuman(of: r), rawHuman.count <= maxHuman,
                  let request = cleanHuman(rawHuman) else { continue }
            let args = tu.input?.jsonString ?? "{}"
            guard args.count <= maxAssistant else { continue }   // skip giant inputs
            guard !hasControlToken(request), !hasControlToken(args) else { continue }  // data hygiene
            guard !decon.isContaminatedPair(request, args) else { continue }           // v2a leakage fence
            let toolCall = "<tool_call>\n{\"name\": \(JSONValue.string(name).jsonString), "
                + "\"arguments\": \(args)}\n</tool_call>"
            let p = Pair(human: request, assistant: toolCall)
            if seen.insert(p).inserted { out.append(p) }
        }
        return out
    }
}

/// Format a pair into a Qwen2.5 chat-template string (consistent train/heldout markup).
func formatQwen(_ p: Pair) -> String {
    "<|im_start|>user\n\(p.human)<|im_end|>\n<|im_start|>assistant\n\(p.assistant)<|im_end|>"
}
