import Foundation
import Network

// ── The cluster control plane (the "most of ClusterPool" piece the recon flagged).
//
// A from-scratch SwiftLM RPC service over Network.framework (TCP, Thunderbolt-IP friendly; optional
// Bonjour `_swiftlm._tcp` for discovery). `WorkerServer` runs on each box and serves its `LocalPool`
// over the wire; `RemotePool` is a `TracePool` that RPCs best-of-N jobs to a peer's WorkerServer.
// Drop `RemotePool`s into a `FanOutPool` and the trace-volume fan-out spans machines — same coordinator
// code, only the worker conformer swaps (the ComputePool thesis, lifted to placement).
//
// Wire framing: 4-byte big-endian length prefix + JSON payload. One job request → one job response,
// pipelined per connection. Loopback-testable with no second machine.

public enum ClusterError: Error, Sendable { case connectionClosed, badFrame, timeout }

/// Unbuffered stderr log — the worker daemon parks, so `print()` (block-buffered stdout) never flushes;
/// cluster events MUST go to stderr to be visible live.
func clog(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

/// Wire messages. A GenJob in, the completions out (or an error string).
struct JobRequest: Codable, Sendable { let job: GenJob }
struct JobResponse: Codable, Sendable { let id: String; let completions: [String]; let error: String? }

/// Async, length-prefixed Codable channel over one NWConnection.
final class FramedChannel: @unchecked Sendable {
    private let conn: NWConnection
    init(_ conn: NWConnection) { self.conn = conn }

    func send<T: Encodable>(_ msg: T) async throws {
        let payload = try JSONEncoder().encode(msg)
        var len = UInt32(payload.count).bigEndian
        var frame = Data(bytes: &len, count: 4)
        frame.append(payload)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: frame, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    func receive<T: Decodable>(_ type: T.Type) async throws -> T {
        let header = try await recvExactly(4)
        let len = UInt32(bigEndian: header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        guard len > 0, len < 64_000_000 else { throw ClusterError.badFrame }
        let payload = try await recvExactly(Int(len))
        return try JSONDecoder().decode(T.self, from: payload)
    }

    private func recvExactly(_ n: Int) async throws -> Data {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            conn.receive(minimumIncompleteLength: n, maximumLength: n) { data, _, isComplete, error in
                if let error { cont.resume(throwing: error); return }
                if let data, data.count == n { cont.resume(returning: data); return }
                cont.resume(throwing: ClusterError.connectionClosed)   // short read ⇒ peer closed
            }
        }
    }

    func cancel() { conn.cancel() }
}

/// Serves a local `TracePool` (this box's LocalPool) over TCP. One server per box.
public actor WorkerServer {
    private let pool: TracePool
    private let serviceName: String?
    private var listener: NWListener?

    public init(pool: TracePool, serviceName: String? = nil) {
        self.pool = pool
        self.serviceName = serviceName
    }

    /// Bind (auto-assigning a port unless `port` given), optionally advertise via Bonjour, and start
    /// accepting. Returns the bound port. Resolves once the listener is ready.
    public func start(port: UInt16? = nil) async throws -> UInt16 {
        let params = NWParameters.tcp
        let listener = port.flatMap { NWEndpoint.Port(rawValue: $0) }
            .map { try? NWListener(using: params, on: $0) } ?? (try? NWListener(using: params))
        guard let l = listener ?? (try? NWListener(using: params)) else { throw ClusterError.connectionClosed }
        self.listener = l
        if let serviceName { l.service = NWListener.Service(name: serviceName, type: "_swiftlm._tcp") }

        let pool = self.pool
        l.newConnectionHandler = { conn in
            conn.start(queue: .global())
            Task { await WorkerServer.serve(conn, pool: pool) }
        }

        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<UInt16, Error>) in
            let resumed = ResumeOnce()
            l.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if resumed.fire() { cont.resume(returning: l.port?.rawValue ?? 0) }
                case .failed(let e):
                    if resumed.fire() { cont.resume(throwing: e) }
                default: break
                }
            }
            l.start(queue: .global())
        }
    }

    /// Read job requests, run them on the local pool, write responses — pipelined until the peer closes.
    private static func serve(_ conn: NWConnection, pool: TracePool) async {
        let chan = FramedChannel(conn)
        clog("[worker] ← connection accepted")
        while true {
            do {
                let req = try await chan.receive(JobRequest.self)
                clog("[worker] ← job \(req.job.id.prefix(8)) (n=\(req.job.n), maxTok=\(req.job.maxTokens)) — generating ...")
                let comps = await pool.generate(req.job)
                clog("[worker] → job \(req.job.id.prefix(8)): \(comps.filter { !$0.isEmpty }.count)/\(comps.count) completions sent")
                try await chan.send(JobResponse(id: req.job.id, completions: comps, error: nil))
            } catch {
                chan.cancel(); return
            }
        }
    }

    public func stop() { listener?.cancel(); listener = nil }
}

/// A `TracePool` that runs jobs on a REMOTE `WorkerServer`. Lazily connects; drops the channel on any
/// error so the next call reconnects (the failure-detection the recon said the control plane must own).
public actor RemotePool: TracePool {
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let descriptor: WorkerDescriptor
    private var channel: FramedChannel?

    public init(host: String, port: UInt16, descriptor: WorkerDescriptor) {
        self.host = NWEndpoint.Host(host)
        self.port = NWEndpoint.Port(rawValue: port) ?? .any
        self.descriptor = descriptor
    }

    private func connect() async throws -> FramedChannel {
        if let channel { return channel }
        let conn = NWConnection(host: host, port: port, using: .tcp)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let resumed = ResumeOnce()
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready: if resumed.fire() { cont.resume() }
                // ECONNREFUSED / unreachable surfaces as `.waiting` (NWConnection retries connectivity
                // by default and NEVER reaches `.failed`) — for fast failure-detection we treat it as
                // a connect failure so a dead peer doesn't hang the caller forever.
                case .waiting(let e): if resumed.fire() { conn.cancel(); cont.resume(throwing: e) }
                case .failed(let e): if resumed.fire() { cont.resume(throwing: e) }
                case .cancelled: if resumed.fire() { cont.resume(throwing: ClusterError.connectionClosed) }
                default: break
                }
            }
            conn.start(queue: .global())
        }
        let chan = FramedChannel(conn)
        channel = chan
        return chan
    }

    public func generate(_ job: GenJob) async -> [String] {
        do {
            let chan = try await connect()
            try await chan.send(JobRequest(job: job))
            let resp = try await chan.receive(JobResponse.self)
            clog("[coordinator] remote \(descriptor.id) served job \(job.id.prefix(8)) → \(resp.completions.count) completions")
            return resp.completions
        } catch {
            clog("[coordinator] remote \(descriptor.id) FAILED job \(job.id.prefix(8)): \(error)")
            channel = nil   // failure detection: drop so the next job reconnects
            return []
        }
    }

    public func capabilities() async -> [WorkerDescriptor] { [descriptor] }
}

/// Thread-safe one-shot guard so a Network.framework state handler resumes its continuation exactly
/// once (`.ready` then later `.failed` must not double-resume → crash).
final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func fire() -> Bool { lock.lock(); defer { lock.unlock() }; if done { return false }; done = true; return true }
}
