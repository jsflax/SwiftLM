import Foundation

// ── The interleaving gate for constrained tool-call emission.
//
// A real agent free-texts normally and only SOMETIMES calls a tool. So the JSON grammar must be OFF during
// prose and flip ON exactly when the model begins a tool call (e.g. Qwen emits `<tool_call>` then the JSON
// then `</tool_call>`). This gate is the pure decision logic: feed it the decoded text of each sampled
// token; it watches for the trigger marker (which may span several tokens) and reports when to switch into
// constrained mode. The MLX `ConditionalGrammarProcessor` wraps it — passthrough while `.freeText`, mask
// via the inner schema grammar while `.constrained`. No MLX here → unit-tested directly.
public struct GrammarActivationGate: Sendable, Equatable {
    public enum Mode: Sendable, Equatable { case freeText; case constrained }

    private let trigger: String
    private var buffer = ""
    public private(set) var mode: Mode = .freeText

    /// `trigger` is the marker that opens a constrained span (default Qwen's `<tool_call>`).
    public init(trigger: String = "<tool_call>") { self.trigger = trigger }

    /// Feed the decoded text of one sampled token. Returns `true` on the token that flips us INTO
    /// constrained mode (so the caller knows the NEXT `process` should mask). No-op once constrained.
    @discardableResult
    public mutating func observe(_ tokenText: String) -> Bool {
        guard mode == .freeText else { return false }
        buffer += tokenText
        if buffer.hasSuffix(trigger) {
            mode = .constrained
            buffer = ""
            return true
        }
        // Keep only enough tail to still detect a trigger forming across token boundaries.
        let cap = max(trigger.count * 2, 8)
        if buffer.count > cap * 2 { buffer = String(buffer.suffix(cap)) }
        return false
    }

    /// Return to free text once the inner grammar has produced a complete value (a later turn may call
    /// another tool).
    public mutating func deactivate() {
        mode = .freeText
        buffer = ""
    }
}
