import Foundation

/// On-device registry of self-improvement cycles: the current champion adapter and a
/// history of promote/reject decisions. Lives under ~/.swiftlm (chmod 700). The frozen
/// base never changes; only the adapter pointer moves (hot-swap = pointer flip).
public struct Registry: Sendable {
    public let root: URL

    public struct Champion: Codable, Sendable {
        public var cycleId: String
        public var adapterPath: String
        public var heldoutLoss: Float
        public var promotedAt: String
        public init(cycleId: String, adapterPath: String, heldoutLoss: Float, promotedAt: String) {
            self.cycleId = cycleId
            self.adapterPath = adapterPath
            self.heldoutLoss = heldoutLoss
            self.promotedAt = promotedAt
        }
    }

    public struct CycleRecord: Codable, Sendable {
        public var cycleId: String
        public var promoted: Bool
        public var beforeHeldout: Float
        public var afterHeldout: Float
        public var beforeRetention: Float
        public var afterRetention: Float
        public var trainCount: Int
        public var heldoutCount: Int
        public var timestamp: String
        public init(cycleId: String, promoted: Bool, beforeHeldout: Float, afterHeldout: Float,
                    beforeRetention: Float, afterRetention: Float, trainCount: Int,
                    heldoutCount: Int, timestamp: String) {
            self.cycleId = cycleId
            self.promoted = promoted
            self.beforeHeldout = beforeHeldout
            self.afterHeldout = afterHeldout
            self.beforeRetention = beforeRetention
            self.afterRetention = afterRetention
            self.trainCount = trainCount
            self.heldoutCount = heldoutCount
            self.timestamp = timestamp
        }
    }

    public init(root: URL? = nil) throws {
        let base = root ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".swiftlm")
        self.root = base
        try FileManager.default.createDirectory(
            at: base.appending(path: "adapters"), withIntermediateDirectories: true)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: base.path)
    }

    var championURL: URL { root.appending(path: "champion.json") }
    var historyURL: URL { root.appending(path: "history.jsonl") }
    /// Adapter *directory* for a cycle (holds adapter_config.json + adapters.safetensors).
    public func adapterURL(cycleId: String) -> URL {
        root.appending(path: "adapters/\(cycleId)")
    }

    public func currentChampion() -> Champion? {
        guard let data = try? Data(contentsOf: championURL) else { return nil }
        return try? JSONDecoder().decode(Champion.self, from: data)
    }

    public func promote(_ champion: Champion) throws {
        try JSONEncoder().encode(champion).write(to: championURL, options: .atomic)
    }

    public func appendHistory(_ record: CycleRecord) throws {
        let line = String(data: try JSONEncoder().encode(record), encoding: .utf8)! + "\n"
        if !FileManager.default.fileExists(atPath: historyURL.path) {
            FileManager.default.createFile(atPath: historyURL.path, contents: nil)
        }
        let h = try FileHandle(forWritingTo: historyURL)
        defer { try? h.close() }
        try h.seekToEnd()
        h.write(Data(line.utf8))
    }
}
