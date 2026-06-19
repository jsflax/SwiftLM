import Testing
import Foundation
@testable import SelfImprove

/// P0.6 + P1.12 registry hardening. All tests use a UNIQUE temp root (never the real ~/.swiftlm), so
/// they can run concurrently and never touch the live registry.
struct RegistryHardeningTests {
    private func tempRoot() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "reg-\(UUID().uuidString)")
    }
    private func mode(_ url: URL) throws -> Int? {
        try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
    }

    @Test func promoteSetsOwnerOnlyPermsAndPinsBase() throws {
        let root = tempRoot()
        let reg = try Registry(root: root)
        defer { try? FileManager.default.removeItem(at: root) }
        let adapterDir = reg.adapterURL(cycleId: "c1")
        try FileManager.default.createDirectory(at: adapterDir, withIntermediateDirectories: true)

        try reg.promote(.init(cycleId: "c1", adapterPath: adapterDir.path,
                              heldoutLoss: 1.0, promotedAt: "t", baseModelId: "base-x"))

        #expect(try mode(root.appending(path: "champion.json")) == 0o600)
        #expect(reg.currentChampion()?.baseModelId == "base-x")  // base pin round-trips
    }

    @Test func promoteRejectsAdapterPathEscape() throws {
        let root = tempRoot()
        let reg = try Registry(root: root)
        defer { try? FileManager.default.removeItem(at: root) }
        // an adapter dir OUTSIDE root/adapters → must be refused (no pointer-flip to a staged adapter)
        #expect(throws: RegistryError.self) {
            try reg.promote(.init(cycleId: "evil", adapterPath: "/tmp/not-in-registry",
                                  heldoutLoss: 1.0, promotedAt: "t"))
        }
    }

    @Test func legacyChampionWithoutBaseDecodesAndIsAllowed() throws {
        let root = tempRoot()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // champion.json under the OLD schema (no baseModelId) must still decode (optional → nil)
        let legacy = #"{"cycleId":"c0","adapterPath":"/x","heldoutLoss":2.0,"promotedAt":"t"}"#
        try Data(legacy.utf8).write(to: root.appending(path: "champion.json"))
        let reg = try Registry(root: root)
        let champ = try #require(reg.currentChampion())
        #expect(champ.baseModelId == nil)
        #expect(Registry.baseMatches(champ, currentBase: "anything"))  // unpinned legacy → allowed
    }

    @Test func baseMismatchBlocksHotSwap() {
        let c = Registry.Champion(cycleId: "c", adapterPath: "/x", heldoutLoss: 1,
                                  promotedAt: "t", baseModelId: "base-a")
        #expect(Registry.baseMatches(c, currentBase: "base-a"))
        #expect(!Registry.baseMatches(c, currentBase: "base-b"))  // a base swap is caught
    }

    @Test func historyFileIsOwnerOnly() throws {
        let root = tempRoot()
        let reg = try Registry(root: root)
        defer { try? FileManager.default.removeItem(at: root) }
        try reg.appendHistory(.init(cycleId: "c1", promoted: false, beforeHeldout: 1, afterHeldout: 1,
                                    beforeRetention: 0, afterRetention: 0, trainCount: 0, heldoutCount: 0,
                                    timestamp: "t"))
        #expect(try mode(root.appending(path: "history.jsonl")) == 0o600)
    }
}
