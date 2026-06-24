import Foundation

/// One training example as a (user, assistant) message pair — NOT a pre-formatted
/// string. The trainer renders it via the tokenizer's real chat template so the
/// training format matches inference exactly (fixes the <|im_start|> leak).
public struct TrainPair: Sendable, Hashable {
    public let user: String
    public let assistant: String
    /// Pre-rendered (prefix + completion + im_end) token row for the TRAIN==SERVE path. When set, the DPO
    /// trainer scores THESE tokens directly and skips the bare `applyChatTemplate([user])` render — so training
    /// matches the owned-render serve format (system prompt + tool schemas + reasoning, the 122B's xmlFunction
    /// completion). `completionStart` is the prefix token count (the assistant-only masking boundary). Both nil ⇒
    /// the legacy bare-render path (the domain flywheel) is byte-for-byte unchanged.
    public let renderedTokens: [Int32]?
    public let completionStart: Int?
    public init(user: String, assistant: String,
                renderedTokens: [Int32]? = nil, completionStart: Int? = nil) {
        self.user = user; self.assistant = assistant
        self.renderedTokens = renderedTokens; self.completionStart = completionStart
    }
}

/// A training curriculum harvested from Claude Code transcripts: clean (user, assistant)
/// pairs split file-level (no within-session leakage). PII is redacted before anything is
/// retained. `trainPairs` feed the masked trainer (rendered via applyChatTemplate);
/// `heldout` stays as formatted strings for the (self-consistent) eval gate.
public struct Curriculum: Sendable {
    public let trainPairs: [TrainPair]
    public let heldout: [String]
    public let trainToolCount: Int        // how many of trainPairs are tool-traces (rest = chat)
    public let filesScanned: Int
    public let productiveFiles: Int
    public let toolTraceExamples: Int     // total tool-call traces harvested (before capping)
    public let redactionSummary: String   // type=count only, never values
}

/// Harvests transcripts into a `Curriculum`. Pure Foundation, on-device.
public enum TranscriptHarvester {
    /// Projects under ~/.claude/projects (basenames). Default: the user's own code work.
    public static let defaultProjects = [
        "-Users-jason-localdev",
        "-Users-jason-localdev-Engram",
        "-Users-jason-localdev-Lattice",
        "-Users-jason-localdev-llm-from-scratch",
        "-Users-jason-Documents-Trader",
        "-Users-jason-Documents-TraderKit",
        "-Users-jason-Documents-Portfolio",
        "-Users-jason-Documents-Lattice",
    ]

    public static func harvest(
        projects: [String] = defaultProjects,
        maxFilesPerProject: Int = 80,
        trainCap: Int = 800,
        heldoutCap: Int = 80,
        heldoutFileEvery: Int = 4
    ) -> Curriculum {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let root = home.appending(path: ".claude/projects")
        let fm = FileManager.default

        // Gather files recursively, most-recent N per project.
        struct Scanned { let url: URL; let mtime: Date }
        var files: [Scanned] = []
        for name in projects {
            let dir = root.appending(path: name)
            guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else { continue }
            var here: [Scanned] = []
            for case let u as URL in en where u.pathExtension == "jsonl" {
                let mt = (try? u.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                here.append(Scanned(url: u, mtime: mt))
            }
            here.sort { $0.mtime > $1.mtime }
            files.append(contentsOf: here.prefix(maxFilesPerProject))
        }
        files.sort { $0.url.path < $1.url.path }   // stable order → deterministic split

        // Harvest + redact, keeping only productive files. Tag each pair with its stream
        // (chat vs tool-trace) so the two can be balanced — tool-traces vastly outnumber chat
        // pairs, and if they dominate the train cap they crush freeform ability (the model
        // learns only to emit <tool_call> and stop → garbled chat). Keep them separate.
        let harvester = Harvester()
        var redactor = Redactor()
        struct Tagged { let pair: Pair; let isTool: Bool }
        var productive: [[Tagged]] = []
        var toolTraceCount = 0
        for sc in files {
            let recs = loadRecords(sc.url)
            let chat = harvester.pairs(from: recs).map {
                Tagged(pair: Pair(human: redactor.redact($0.human),
                                  assistant: redactor.redact($0.assistant)), isTool: false)
            }
            let tools = harvester.toolTraces(from: recs).map {   // stream 2: tool-call emission
                Tagged(pair: Pair(human: redactor.redact($0.human),
                                  assistant: redactor.redact($0.assistant)), isTool: true)
            }
            toolTraceCount += tools.count
            let all = chat + tools
            if !all.isEmpty { productive.append(all) }
        }

        // File-level split across productive sessions (no within-session leakage), keeping
        // the chat and tool streams in separate buckets for balanced capping.
        var chatTrain: [TrainPair] = [], toolTrain: [TrainPair] = [], heldout: [String] = []
        var heldoutSet = Set<Pair>(), trainSet = Set<Pair>()
        for (i, tagged) in productive.enumerated() {
            let isHeldout = (i % heldoutFileEvery == 0)
            for t in tagged {
                let p = t.pair
                if isHeldout {
                    if heldoutSet.insert(p).inserted { heldout.append(formatQwen(p)) }
                } else if !heldoutSet.contains(p), trainSet.insert(p).inserted {
                    let tp = TrainPair(user: p.human, assistant: p.assistant)
                    if t.isTool { toolTrain.append(tp) } else { chatTrain.append(tp) }
                }
            }
        }

        // Balanced compose: tool-traces capped at ~half the budget so chat survives, but a
        // scarce stream lets the other fill the remainder (don't waste budget).
        let toolCap = Int(Double(trainCap) * 0.5)
        var tools = Array(toolTrain.prefix(toolCap))
        var chat = Array(chatTrain.prefix(trainCap - tools.count))
        if chat.count + tools.count < trainCap {       // chat scarce → top up tools
            tools = Array(toolTrain.prefix(trainCap - chat.count))
        }
        let trainPairs = chat + tools                  // sampled cyclically downstream; order is fine
        if heldout.count > heldoutCap { heldout = Array(heldout.prefix(heldoutCap)) }

        return Curriculum(
            trainPairs: trainPairs, heldout: heldout, trainToolCount: tools.count,
            filesScanned: files.count, productiveFiles: productive.count,
            toolTraceExamples: toolTraceCount, redactionSummary: redactor.summary)
    }
}
