import Foundation

// ── The cluster observability surface (the "I shouldn't need lsof to know if box B is working" fix).
//
// Before this, answering "is the remote worker getting jobs?" meant: grep two differently-named stderr
// loggers (`log`/`clog`, both stderr but named as if they differ), KNOW that the per-job remote line
// only appears during the trace-gen phase (eval is coordinator-local), and `lsof` the process to confirm
// stderr even reaches the log. That's a forensic exercise for a yes/no question — a design smell.
//
// `ClusterStatus` makes liveness first-class and queryable. It is a process-wide singleton both the
// coordinator and each worker box update:
//   • phase banners that NAME whether fan-out is active for the current phase;
//   • a per-round dispatch summary (jobs → per-worker split, served vs empty/failed);
//   • a heartbeat tick so a long silent best-of-N job (12 tok/s × 4k tokens = minutes) ≠ a hang;
//   • an atomically-written JSON status file → `cat ~/.swiftlm/cluster-status.json` answers everything
//     with zero grep/lsof. The coordinator and box B each write their own (one process per box).
//
// Accounting lives at ONE point (the FanOutPool, which sees every worker incl. the local box) so there
// is no double-counting; RemotePool/WorkerServer keep their detailed per-job `clog` lines as secondary
// trace. All cluster log lines are `[tag]`-prefixed so the log is greppable by subsystem.
public actor ClusterStatus {
    public static let shared = ClusterStatus()

    public struct WorkerStat: Sendable {
        public var label: String
        public var host: String
        public var served = 0          // jobs that returned ≥1 non-empty completion
        public var emptyOrFailed = 0   // jobs that returned nothing (RPC failure or all-empty rollouts)
        public var lastServedEpoch: Double?
    }

    private var role = "coordinator"            // or "worker"
    private var phaseName = "(startup)"
    private var fanOutActive = false
    private var round = 0
    private var workers: [String: WorkerStat] = [:]   // keyed by label
    private let statusPath: String
    private let startEpoch: Double
    private var heartbeat: Task<Void, Never>?
    private var heartbeatSec: UInt64 = 15

    private init() {
        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/tmp"
        let dir = home + "/.swiftlm"
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        self.statusPath = dir + "/cluster-status.json"
        self.startEpoch = Date().timeIntervalSince1970
    }

    /// Path a human can `cat` to see live cluster state (per box).
    public func statusFilePath() -> String { statusPath }

    /// Force an immediate status-file write (heartbeat flushes on a timer; `served` only updates
    /// in-memory). Use at a checkpoint when you want the file current right now.
    public func flush() { writeStatus() }

    /// Identify this process and start the heartbeat. Call once at startup on each box.
    public func start(role: String, heartbeatSeconds: UInt64 = 15) {
        self.role = role
        self.heartbeatSec = max(2, heartbeatSeconds)
        clog("[cluster] \(role) up — status file: \(statusPath)")
        startHeartbeat()
        writeStatus()
    }

    /// Pre-register a worker so it shows in the status file before its first job lands (so a 0-served
    /// worker is visibly KNOWN-but-idle, not absent).
    public func registerWorker(label: String, host: String) {
        if workers[label] == nil { workers[label] = WorkerStat(label: label, host: host) }
        clog("[cluster] worker registered: \(label) @ \(host)")
        writeStatus()
    }

    /// Announce a phase and whether fan-out is active in it. Makes "no remote jobs yet" self-explanatory.
    public func enterPhase(_ name: String, fanOut: Bool) {
        phaseName = name
        fanOutActive = fanOut
        let nW = workers.count
        let how = fanOut ? "fan-out ACTIVE across \(max(nW, 1)) worker(s)"
                         : "coordinator-local (no fan-out — runs on this box only)"
        clog("[phase] \(name) — \(how)")
        writeStatus()
    }

    /// One line per fan-out round: the job→worker split. The single best answer to "did it fan out?".
    public func dispatched(round r: Int, split: [(label: String, count: Int)]) {
        round = r
        let total = split.reduce(0) { $0 + $1.count }
        let desc = split.map { "[\($0.label)]=\($0.count)" }.joined(separator: " ")
        clog("[dispatch] round \(r): \(total) jobs → \(desc)")
        writeStatus()
    }

    /// Record one job's outcome against its worker. Cheap (in-memory); flushed by heartbeat + dispatch.
    public func served(label: String, host: String = "?", ok: Bool) {
        var w = workers[label] ?? WorkerStat(label: label, host: host)
        if ok { w.served += 1 } else { w.emptyOrFailed += 1 }
        w.lastServedEpoch = Date().timeIntervalSince1970
        workers[label] = w
    }

    /// A compact, sorted summary string (also used by the heartbeat line).
    private func summaryLine(now: Double) -> String {
        workers.values.sorted { $0.label < $1.label }.map { w in
            let ago = w.lastServedEpoch.map { String(format: "%.0fs", now - $0) } ?? "never"
            return "\(w.label):ok=\(w.served)/fail=\(w.emptyOrFailed)/last=\(ago)"
        }.joined(separator: " · ")
    }

    private func startHeartbeat() {
        guard heartbeat == nil else { return }
        let sec = heartbeatSec
        heartbeat = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: sec * 1_000_000_000)
                await self?.tick()
            }
        }
    }

    private func tick() {
        writeStatus()
        let now = Date().timeIntervalSince1970
        clog("[heartbeat] \(role) phase=\(phaseName) fanout=\(fanOutActive) round=\(round) · "
             + (workers.isEmpty ? "no workers" : summaryLine(now: now)))
    }

    /// Atomic write (temp + rename) so a concurrent `cat` never sees a half-written file.
    private func writeStatus() {
        let now = Date().timeIntervalSince1970
        let workerObjs: [[String: Any]] = workers.values.sorted { $0.label < $1.label }.map { w in
            [
                "label": w.label,
                "host": w.host,
                "served": w.served,
                "emptyOrFailed": w.emptyOrFailed,
                "lastServedAgoSec": w.lastServedEpoch.map { Int(now - $0) } ?? NSNull(),
            ]
        }
        let payload: [String: Any] = [
            "role": role,
            "phase": phaseName,
            "fanOutActive": fanOutActive,
            "round": round,
            "uptimeSec": Int(now - startEpoch),
            "updatedEpoch": now,
            "workers": workerObjs,
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? data.write(to: URL(fileURLWithPath: statusPath), options: .atomic)
    }
}
