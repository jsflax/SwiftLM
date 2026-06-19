import Foundation

// ── Serve-time abstention (build step 1, the freeform correctness story's second half).
//
// Retrieval grounding handles "inject what we know"; abstention handles "admit when we don't." The
// red-team's point: the freeform ~73% of turns have NO execution verifier, so the only guard against
// confident-but-wrong answers is (a) ground them and (b) HEDGE when a question asks for specific
// recalled facts the agent has no grounding for — instead of confabulating.
//
// v1 signal = retrieval-grounded: it's nearly free (grounding already computed `hasRelevant`). The
// `needsGrounding` predicate is injected so it can be upgraded later (a small classifier, or asking
// the model itself, or a self-consistency check) without touching the policy structure.

public enum AbstentionDecision: Sendable, Equatable {
    /// Answer normally — either grounded, or a turn that doesn't need recalled facts (code, chitchat).
    case proceed
    /// The turn asks for specific/recalled facts but no relevant grounding was found → answer with
    /// explicit uncertainty (the agent prepends a hedge / instructs the model not to fabricate).
    case hedge(reason: String)
}

public struct AbstentionPolicy: Sendable {
    let needsGrounding: @Sendable (String) -> Bool

    public init(needsGrounding: @escaping @Sendable (String) -> Bool = AbstentionPolicy.defaultNeedsGrounding) {
        self.needsGrounding = needsGrounding
    }

    /// Decide whether to answer confidently or hedge, given the grounding the retriever produced.
    public func decide(prompt: String, grounding: Grounding) -> AbstentionDecision {
        guard needsGrounding(prompt) else { return .proceed }        // doesn't need recalled facts
        return grounding.hasRelevant
            ? .proceed                                                // grounded → answer
            : .hedge(reason: "asks for specific/recalled facts but memory returned nothing relevant")
    }

    /// The hedge instruction injected into the model's system context when abstention fires (used by
    /// `EngramAdviseHook`). Lives here because the hedge is abstention's concern, not the orchestrator's.
    public static func hedgeInstruction(_ reason: String) -> String {
        "\nNOTE: you have no memory relevant to this question (\(reason)). Do NOT fabricate specifics, "
        + "names, numbers, or past decisions. Say plainly what you don't have, and answer only what you "
        + "can in general terms.\n"
    }

    /// v1 heuristic: a prompt "needs grounding" when it asks for specific/recalled facts — an
    /// interrogative seeking particulars, or a reference to personal/prior context — as opposed to a
    /// generic instruction, code request, or chitchat the base can answer from its own weights.
    /// Deliberately coarse and REPLACEABLE; the policy structure is what's load-bearing.
    public static let defaultNeedsGrounding: @Sendable (String) -> Bool = { prompt in
        let p = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let interrogatives = ["what", "when", "where", "who", "which", "whose", "how many", "how much"]
        let startsInterrogative = interrogatives.contains { p.hasPrefix($0 + " ") || p.hasPrefix($0 + "'") }
        let isQuestion = p.contains("?")
        let hasInterrogative = interrogatives.contains { p.contains(" \($0) ") || p.hasPrefix("\($0) ") }
        let asksFact = startsInterrogative || (isQuestion && hasInterrogative)
        // references to personal / prior-session context that only memory could supply
        let contextRefs = ["my ", "our ", "we discussed", "we decided", "last time", "earlier",
                           "remember", "you said", "previously", "the issue with"]
        let refsContext = contextRefs.contains { p.contains($0) }
        return asksFact || refsContext
    }
}
