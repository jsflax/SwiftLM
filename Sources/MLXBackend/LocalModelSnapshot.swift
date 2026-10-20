import Foundation

enum LocalModelSnapshotError: LocalizedError {
    case notCached(String)

    var errorDescription: String? {
        switch self {
        case .notCached(let modelID):
            "\(modelID) is not downloaded on this device. Download it in Local Models before starting a response."
        }
    }
}

/// Resolves only local files. A partial snapshot will fail during loading; it never triggers a download.
enum LocalModelSnapshot {
    static func resolve(modelID: String,
                        environment: [String: String] = ProcessInfo.processInfo.environment,
                        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL? {
        let hub: URL
        if let path = environment["HF_HUB_CACHE"], !path.isEmpty {
            hub = URL(fileURLWithPath: path)
        } else if let path = environment["HF_HOME"], !path.isEmpty {
            hub = URL(fileURLWithPath: path).appendingPathComponent("hub")
        } else {
            hub = homeDirectory.appendingPathComponent(".cache/huggingface/hub")
        }
        let root = hub.appendingPathComponent("models--" + modelID.replacingOccurrences(of: "/", with: "--"))
        let snapshots = root.appendingPathComponent("snapshots")
        let fm = FileManager.default
        func hasConfiguration(_ directory: URL) -> Bool {
            fm.fileExists(atPath: directory.appendingPathComponent("config.json").path)
        }
        if let ref = try? String(contentsOf: root.appendingPathComponent("refs/main"), encoding: .utf8) {
            let revision = ref.trimmingCharacters(in: .whitespacesAndNewlines)
            if !revision.isEmpty, revision != ".", revision != "..", !revision.contains("/"),
               hasConfiguration(snapshots.appendingPathComponent(revision)) {
                return snapshots.appendingPathComponent(revision)
            }
        }
        guard let candidates = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        return candidates.filter(hasConfiguration).sorted {
            let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return left == right ? $0.lastPathComponent < $1.lastPathComponent : left > right
        }.first
    }
}
