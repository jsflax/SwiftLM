import Foundation

// ── A0 decode tripwire — the UNCONDITIONAL bound on a degenerate decode loop. The owned-render decode
// (MLXBackend.streamFromTokens) once spiraled into whitespace: the same low-entropy token (or a tiny
// whitespace cycle) emitted thousands of times, never reaching EOS, until the 900s turn watchdog killed it
// with NO telemetry. A repetition penalty (mlx's RepetitionContext, now wired into the owned-render sampler)
// suppresses MOST such loops — but it is not a guarantee: a dominant logit spike survives a single ÷penalty,
// and a cycle longer than the rep window escapes it entirely. THIS is the guarantee: a degenerate run aborts
// the decode in milliseconds instead of running to the runaway ceiling. Pure + MLX-free so the safety bound is
// unit-tested in the fast suite, with no model.

/// Watches the decoded token stream for a degenerate (non-terminating) run and reports when one is confirmed.
/// Two INDEPENDENT signals, either of which trips:
///   • identical-token run — the SAME token id repeated `identicalRunLimit` times (the classic stuck-token loop)
///   • whitespace-only run  — the decoded suffix is whitespace-only for `whitespaceRunLimit` consecutive steps
///     (catches a small whitespace CYCLE — "\n", " ", "\n", " ", … — that the identical-token check misses)
///
/// Both limits are set high enough that legitimate output never trips: real text always interleaves a
/// non-whitespace, non-repeated token long before the limit (even deep indentation or a `====` rule is broken
/// up by content tokens). The detector is deliberately NOT a raw "repeated content" check — that would clip a
/// model legitimately emitting a repetitive structure; it fires only on whitespace or exact-token degeneracy.
public struct DegenerateRunDetector {
    /// The same token id repeated this many times in a row trips the wire. 0 disables this signal.
    public let identicalRunLimit: Int
    /// This many consecutive whitespace-only decoded suffixes trips the wire. 0 disables this signal.
    public let whitespaceRunLimit: Int

    private var lastToken: Int? = nil
    private var identicalRun = 0
    private var whitespaceRun = 0

    /// Why the wire tripped — surfaced for HONEST, visible telemetry (never a silent cap).
    public enum Trip: Equatable, CustomStringConvertible {
        case identicalRun(token: Int, length: Int)
        case whitespaceRun(length: Int)
        public var description: String {
            switch self {
            case let .identicalRun(token, length): return "identical token \(token) ×\(length)"
            case let .whitespaceRun(length): return "whitespace-only ×\(length)"
            }
        }
    }

    /// Defaults are tuned to never clip legitimate output (48 in a row is already pathological) and are
    /// env-overridable for the battle-test (`SWIFTLM_TRIPWIRE_IDENTICAL` / `SWIFTLM_TRIPWIRE_WHITESPACE`;
    /// set to 0 to disable a signal). Explicit init args win over the env so tests are hermetic.
    public init(identicalRunLimit: Int? = nil, whitespaceRunLimit: Int? = nil) {
        let env = ProcessInfo.processInfo.environment
        self.identicalRunLimit = identicalRunLimit
            ?? env["SWIFTLM_TRIPWIRE_IDENTICAL"].flatMap(Int.init) ?? 48
        self.whitespaceRunLimit = whitespaceRunLimit
            ?? env["SWIFTLM_TRIPWIRE_WHITESPACE"].flatMap(Int.init) ?? 48
    }

    /// Observe ONE decoded step. `token` is the sampled id; `suffix` is the text this token decoded to (the
    /// newly-emitted piece, possibly empty when a multi-token grapheme is still being built). Returns a `Trip`
    /// the first time a degenerate run reaches its limit, else nil. Mutating — the caller threads ONE detector
    /// across the whole decode loop.
    public mutating func observe(token: Int, suffix: String) -> Trip? {
        // identical-token run
        if token == lastToken { identicalRun += 1 } else { identicalRun = 1; lastToken = token }
        if identicalRunLimit > 0 && identicalRun >= identicalRunLimit {
            return .identicalRun(token: token, length: identicalRun)
        }
        // whitespace-only run: a NON-EMPTY suffix that is all whitespace extends the run; a real
        // (non-whitespace) token breaks it; an EMPTY suffix (partial grapheme) is neutral — it neither extends
        // nor resets, so a multi-token whitespace cycle can't be masked by an intermediate empty decode.
        if !suffix.isEmpty {
            if suffix.allSatisfy({ $0.isWhitespace }) {
                whitespaceRun += 1
                if whitespaceRunLimit > 0 && whitespaceRun >= whitespaceRunLimit {
                    return .whitespaceRun(length: whitespaceRun)
                }
            } else {
                whitespaceRun = 0
            }
        }
        return nil
    }
}
