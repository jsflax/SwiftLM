import Foundation

// ── Serve-time retrieval grounding (build step 1 of the "useful agent first" product).
//
// The red-team measured that ~73% of real agent turns are freeform/read — the slice the execution
// verifier CANNOT grade. For that slice the correctness story is RETRIEVAL GROUNDING (+ abstention):
// before the model answers, pull relevant memory and inject it as context, so freeform answers are
// grounded in what's actually known rather than free-associated.
//
// The retrieval BACKEND is injected as a closure (`recall`) — the Engram MCP `recall` tool, a local
// index, whatever — so this type carries no MLX/MCP/network dependency and is fully unit-testable. The
// agent executable closes `recall` over the real Engram call; tests close it over a fixture. Same
// closure-injection pattern that keeps `LocalPool` pure-Swift.

/// One retrieved snippet. `distance` is a relevance signal where LOWER is closer (Engram returns vector
/// distance; 0 = identical). `RetrievalGrounding` filters and ranks on it.
public struct RetrievedItem: Sendable, Equatable {
    public let id: String
    public let text: String
    public let distance: Double
    public init(id: String, text: String, distance: Double) {
        self.id = id; self.text = text; self.distance = distance
    }
}

/// The result of grounding a prompt: the kept (relevant) items + a formatted block to inject, plus
/// `hasRelevant` — the signal abstention consumes (a freeform answer with NO relevant grounding is a
/// hedge candidate, not a confident assertion).
public struct Grounding: Sendable, Equatable {
    public let items: [RetrievedItem]
    public let contextBlock: String
    public init(items: [RetrievedItem], contextBlock: String) {
        self.items = items; self.contextBlock = contextBlock
    }
    public var hasRelevant: Bool { !items.isEmpty }
}

public struct RetrievalGrounding: Sendable {
    public struct Config: Sendable {
        public var maxItems: Int        // cap injected items (context budget)
        public var maxDistance: Double  // drop items LESS relevant than this (Engram distance)
        public var header: String
        public init(
            maxItems: Int = 5,
            maxDistance: Double = 0.45,
            header: String = "Relevant context from your memory (may be incomplete — ground your answer in it and do NOT invent facts beyond it):"
        ) {
            self.maxItems = maxItems; self.maxDistance = maxDistance; self.header = header
        }
    }

    let config: Config
    let recall: @Sendable (_ query: String, _ limit: Int) async -> [RetrievedItem]

    /// `recall(query, limit)` performs the actual retrieval (injected; Agent wires Engram's MCP recall).
    public init(
        config: Config = Config(),
        recall: @escaping @Sendable (_ query: String, _ limit: Int) async -> [RetrievedItem]
    ) {
        self.config = config
        self.recall = recall
    }

    /// Retrieve for `prompt`, keep only items at/under the distance threshold, rank best-first, cap to
    /// the budget, and format an injectable context block. Returns an empty `Grounding` (hasRelevant
    /// false, blank block) when nothing relevant is found — the caller injects nothing and abstention
    /// sees the gap.
    public func ground(_ prompt: String) async -> Grounding {
        let raw = await recall(prompt, max(config.maxItems * 2, config.maxItems))
        let kept = raw
            .filter { $0.distance <= config.maxDistance }
            .sorted { $0.distance < $1.distance }
            .prefix(config.maxItems)
        let items = Array(kept)
        guard !items.isEmpty else { return Grounding(items: [], contextBlock: "") }
        let lines = items.map { "- \($0.text)" }.joined(separator: "\n")
        return Grounding(items: items, contextBlock: "\(config.header)\n\(lines)\n")
    }
}
