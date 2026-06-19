import Foundation

public enum RegistryError: Error, Sendable {
    /// A champion's adapter dir resolved OUTSIDE the registry's adapters/ tree (symlink/`..` escape).
    case adapterPathEscape(String)
}

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
        /// The base model the LoRA was fit to. A LoRA delta is only valid against the exact base it
        /// trained on, so a base swap silently corrupts every adapter — `baseMatches` guards the
        /// hot-swap. `nil` = legacy champion recorded before this field existed (allowed, not pinned).
        public var baseModelId: String?
        public init(cycleId: String, adapterPath: String, heldoutLoss: Float, promotedAt: String,
                    baseModelId: String? = nil) {
            self.cycleId = cycleId
            self.adapterPath = adapterPath
            self.heldoutLoss = heldoutLoss
            self.promotedAt = promotedAt
            self.baseModelId = baseModelId
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
        public var baseModelId: String?   // the base this cycle trained against (nil = legacy record)
        public init(cycleId: String, promoted: Bool, beforeHeldout: Float, afterHeldout: Float,
                    beforeRetention: Float, afterRetention: Float, trainCount: Int,
                    heldoutCount: Int, timestamp: String, baseModelId: String? = nil) {
            self.cycleId = cycleId
            self.promoted = promoted
            self.beforeHeldout = beforeHeldout
            self.afterHeldout = afterHeldout
            self.beforeRetention = beforeRetention
            self.afterRetention = afterRetention
            self.trainCount = trainCount
            self.heldoutCount = heldoutCount
            self.timestamp = timestamp
            self.baseModelId = baseModelId
        }
    }

    public init(root: URL? = nil) throws {
        let base = root ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: ".swiftlm")
        self.root = base
        try FileManager.default.createDirectory(
            at: base.appending(path: "adapters"), withIntermediateDirectories: true)
        // The dir tree holds the champion pointer + adapters — keep it owner-only, not the world-readable
        // default; a flippable champion.json is a privilege-escalation path (any user-process could point
        // serving/export at an attacker-staged adapter).
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: base.path)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700],
                                               ofItemAtPath: base.appending(path: "adapters").path)
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
        try validateAdapterPath(champion.adapterPath)   // refuse a pointer that escapes the registry
        try JSONEncoder().encode(champion).write(to: championURL, options: .atomic)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: championURL.path)
    }

    public func appendHistory(_ record: CycleRecord) throws {
        let line = String(data: try JSONEncoder().encode(record), encoding: .utf8)! + "\n"
        if !FileManager.default.fileExists(atPath: historyURL.path) {
            FileManager.default.createFile(atPath: historyURL.path, contents: nil,
                                           attributes: [.posixPermissions: 0o600])
        }
        let h = try FileHandle(forWritingTo: historyURL)
        defer { try? h.close() }
        try h.seekToEnd()
        h.write(Data(line.utf8))
    }

    /// Reject a champion whose adapter dir resolves outside `root/adapters` (defeats a `..`/symlink
    /// pointer-flip to an attacker-staged adapter). Resolves symlinks + `..` on both sides before comparing.
    func validateAdapterPath(_ path: String) throws {
        let adaptersRoot = root.appending(path: "adapters")
            .standardizedFileURL.resolvingSymlinksInPath().path
        let resolved = URL(fileURLWithPath: path)
            .standardizedFileURL.resolvingSymlinksInPath().path
        guard resolved == adaptersRoot || resolved.hasPrefix(adaptersRoot + "/") else {
            throw RegistryError.adapterPathEscape(path)
        }
    }

    /// Whether `champion` may hot-swap onto `currentBase`. A recorded base must match exactly (a LoRA
    /// is only valid against the base it trained on); an unrecorded (legacy) base is allowed but unpinned.
    public static func baseMatches(_ champion: Champion, currentBase: String) -> Bool {
        guard let recorded = champion.baseModelId else { return true }
        return recorded == currentBase
    }
}
