import Foundation

/// A held-out task with an objective correctness check. For now: tool-call pass@1 —
/// does the model select the expected tool? This is the cheapest correctness signal
/// (no judge, no compile sandbox) and directly catches an adapter that has become so
/// chatty it stops calling tools. (Compile/test-based checks come later.)
public struct ToolCase: Sendable {
    public let prompt: String
    public let expectedTool: String
    public init(prompt: String, expectedTool: String) {
        self.prompt = prompt
        self.expectedTool = expectedTool
    }
}

public enum EvalSuite {
    /// Frozen tool-selection cases (claude-utils tools). The model passes a case iff its
    /// tool loop calls `expectedTool`.
    public static let toolCases: [ToolCase] = [
        ToolCase(prompt: "What is the current date and time right now? Use your tools.", expectedTool: "current_time"),
        ToolCase(prompt: "Generate a fresh random UUID for me.", expectedTool: "uuid"),
        ToolCase(prompt: "How many CPU cores does this machine have? Check the system.", expectedTool: "system_info"),
        ToolCase(prompt: "What's this machine's network info — hostname and interfaces?", expectedTool: "network_info"),
    ]

    /// Flywheel TRAINING tasks — DISJOINT phrasings from `toolCases` (the held-out eval set),
    /// covering the same claude-utils tool surface plus `clipboard`. The flywheel samples
    /// best-of-N over these, keeps the rollouts that call the right tool, and distills them;
    /// generalization is then measured on the unseen `toolCases` phrasings. Phrasing + arg
    /// variety (UTC/epoch/timezone, read/peek) gives the verified-trace stream real diversity
    /// even though the underlying tools are few.
    public static let flywheelTrainTasks: [ToolCase] = [
        // current_time
        ToolCase(prompt: "What time is it right now?", expectedTool: "current_time"),
        ToolCase(prompt: "Give me the current timestamp in UTC.", expectedTool: "current_time"),
        ToolCase(prompt: "What's today's date?", expectedTool: "current_time"),
        ToolCase(prompt: "Tell me the Unix epoch time at this moment.", expectedTool: "current_time"),
        ToolCase(prompt: "What's the current time in the Tokyo timezone?", expectedTool: "current_time"),
        // uuid
        ToolCase(prompt: "I need a unique identifier — make one.", expectedTool: "uuid"),
        ToolCase(prompt: "Create a random GUID for me.", expectedTool: "uuid"),
        ToolCase(prompt: "Spin up a fresh UUID please.", expectedTool: "uuid"),
        // system_info
        ToolCase(prompt: "What CPU and memory does this box have?", expectedTool: "system_info"),
        ToolCase(prompt: "Describe this machine's hardware specs.", expectedTool: "system_info"),
        ToolCase(prompt: "How much RAM is installed on this computer?", expectedTool: "system_info"),
        // network_info
        ToolCase(prompt: "List the network interfaces on this host.", expectedTool: "network_info"),
        ToolCase(prompt: "What's my local IP address and hostname?", expectedTool: "network_info"),
        // clipboard
        ToolCase(prompt: "Read what's currently on my clipboard.", expectedTool: "clipboard"),
        ToolCase(prompt: "What text is in the clipboard right now?", expectedTool: "clipboard"),
        ToolCase(prompt: "Show me the clipboard contents.", expectedTool: "clipboard"),
    ]
}
