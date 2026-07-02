import Foundation
import Network
import Orchestration
import MLXLMCommon
import MLX

// M4d — the FOLLOWER side of distributed inference. The leader (rank 0, the orbital-loop --room driver)
// recruits this peer; the peer loads its pipeline shard (rank N-1) and MIRRORS the leader's decode, fed the
// per-round input token rows over the reused cluster side-channel. The peer never runs tools, never touches
// Lattice — lockstep is held by the ring (the forward collectives + the per-token broadcast in
// BatchedGenerate.streamFromTokens). Orbital stays agnostic: this all lives in SwiftLM/MLX.

private func distLog(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

/// Leader → follower handshake. Carries the ring topology + the EXACT decode params so the follower's
/// `streamFromTokens` stays byte-identical to the leader's (the user chose full sampling support, so all
/// sampler fields ride along, not just temp).
public struct Recruit: Codable, Sendable {
    public let modelId: String
    public let ringHosts: [String]
    public let rank: Int
    public let boundary: Int?
    public let generation: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float?
    public let repetitionContextSize: Int
    public let prefillStepSize: Int

    public init(modelId: String, ringHosts: [String], rank: Int, boundary: Int?, generation: Int,
                params: GenerateParameters) {
        self.modelId = modelId; self.ringHosts = ringHosts; self.rank = rank; self.boundary = boundary
        self.generation = generation
        self.temperature = params.temperature; self.topP = params.topP
        self.topK = params.topK; self.minP = params.minP
        self.repetitionPenalty = params.repetitionPenalty
        self.repetitionContextSize = params.repetitionContextSize
        self.prefillStepSize = params.prefillStepSize
    }

    public func generateParameters() -> GenerateParameters {
        GenerateParameters(
            temperature: temperature, topP: topP, topK: topK, minP: minP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
            prefillStepSize: prefillStepSize)
    }
}

/// One round's input for the follower: the rendered prompt / tool-result token row + how many tokens to
/// decode. `reset` ⇒ the leader reset its KV this round (compaction) so the follower must too. Without it
/// both KV boxes track identically (same row + same broadcast tokens ⇒ same `isReusablePrefix` decision).
public struct FollowerRoundInput: Codable, Sendable {
    public let tokens: [Int32]
    public let maxTokens: Int
    public let reset: Bool
    /// The LEADER's FULL per-round decode params. The follower MUST decode byte-identically to the leader: it
    /// recomputes the SAME logits (the pipeline all-gathers rank-0's hidden so every rank runs norm+lm_head) and
    /// then samples — so it must apply the SAME logit processing (rep-pen) and the SAME sampler. The load-time
    /// `Recruit` carries only DEFAULT params (`loadDistributedLeader` gets none), and the per-turn agent params
    /// (rep-pen/top-p/temp from the model's adapter) live HERE. BUG-E (Jun 30 2026): syncing only `temperature`
    /// left the follower on DEFAULT rep-pen while the leader applied the adapter's — so in greedy lockstep the
    /// ranks matched for a few tokens (rep-pen inactive) then DIVERGED once repetition built → pipeline desync
    /// (one rank stops/tripwires, the other waits its next collective) → ~45s GPU watchdog crash. All sampler
    /// fields now ride along per round. (`prefillStepSize` stays the follower's load-time value.)
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float?
    public let repetitionContextSize: Int
    public init(tokens: [Int32], maxTokens: Int, reset: Bool, params: GenerateParameters) {
        self.tokens = tokens; self.maxTokens = maxTokens; self.reset = reset
        self.temperature = params.temperature; self.topP = params.topP
        self.topK = params.topK; self.minP = params.minP
        self.repetitionPenalty = params.repetitionPenalty
        self.repetitionContextSize = params.repetitionContextSize
    }
    /// The leader's per-round decode params, applied over the follower's load-time `base` (keeps `base`'s
    /// `prefillStepSize`/`maxTokens` shape; overrides every sampler field that drives token selection).
    public func decodeParams(over base: GenerateParameters) -> GenerateParameters {
        var p = base
        p.temperature = temperature; p.topP = topP; p.topK = topK; p.minP = minP
        p.repetitionPenalty = repetitionPenalty; p.repetitionContextSize = repetitionContextSize
        return p
    }
}

/// Run the follower loop on an accepted side-channel: receive Recruit → load the shard (rank N-1) → mirror
/// the leader's decode for each `FollowerRoundInput` until the channel closes. Discards all output. v1 is
/// base-model only (no resident-trait LoRA) — the bank-off default — so both ranks forward base and stay
/// byte-identical; `activeTraits` is empty here on purpose.
public func serveFollower(channel: FramedChannel) async throws {
    let recruit = try await channel.receive(Recruit.self)
    distLog("[follower] recruited \(recruit.modelId) rank \(recruit.rank)/\(recruit.ringHosts.count) "
        + "boundary=\(recruit.boundary.map(String.init) ?? "even")")
    // Ring-topology breadcrumb (one per recruit): the EXACT ringHosts this rank received + who it dials,
    // diffable against the leader's "[leader] ring …" line when a rendezvous ever misbehaves again.
    distLog("[follower] ringHosts=\(recruit.ringHosts) — rank \(recruit.rank) will connect to rank "
        + "\((recruit.rank + 1) % max(1, recruit.ringHosts.count)) = "
        + "\(recruit.ringHosts.indices.contains((recruit.rank + 1) % max(1, recruit.ringHosts.count)) ? recruit.ringHosts[(recruit.rank + 1) % recruit.ringHosts.count] : "?")")
    let (model, rank, size) = try await MLXLanguageModel.loadDistributedFromHosts(
        modelId: recruit.modelId, hosts: recruit.ringHosts, rank: recruit.rank, boundary: recruit.boundary)
    // BUG-E (Jun 30 2026): the follower MUST compute the SAME stop set as the leader, or in greedy lockstep a
    // token that's a stop for one rank but not the other makes one rank `break` while the other enters its next
    // ring collective → deadlock → 45s GPU watchdog. The leader's stops come from the CONFIG-DRIVEN `localAdapter`
    // (the model's own generation_config eos_token_id + stop_strings); the old family `ModelFamilyDetector.profile`
    // uses hardcoded per-family eos that differ. Use the config-driven adapter here too — the follower has the same
    // model files, so it resolves byte-identical stops.
    let profile = await model.localAdapter
    let params = recruit.generateParameters()
    let kvBox = OwnedKVCacheBox()
    distLog("[follower] ring formed rank \(rank)/\(size); serving round inputs …")
    while true {
        let input: FollowerRoundInput
        do { input = try await channel.receive(FollowerRoundInput.self) }
        catch { break }   // channel closed ⇒ leader done/gone ⇒ release the shard
        if input.reset { kvBox.reset() }
        // Decode with the LEADER's FULL per-round params (rep-pen + sampler), not just temperature — otherwise the
        // ranks apply DIFFERENT logit processing and diverge in greedy lockstep (BUG-E). `base` (Recruit params)
        // contributes only the non-sampler shape (prefillStepSize).
        let roundParams = input.decodeParams(over: params)
        for try await _ in model.streamFromTokens(
            input.tokens, maxTokens: input.maxTokens, adapter: profile, params: roundParams, kvBox: kvBox) {}
    }
    distLog("[follower] channel closed; shard released.")
}

/// Carried on the LEADER's `MLXLanguageModel`: the ring group + the side-channel to the follower. The turn
/// loop feeds each round's input row through `sendRoundInput`. `@unchecked Sendable` — the group/channel are
/// driven only from the (serialized) turn path.
public final class DistributedContext: @unchecked Sendable {
    public let group: DistributedGroup
    public let channel: FramedChannel
    public let generation: Int
    public init(group: DistributedGroup, channel: FramedChannel, generation: Int) {
        self.group = group; self.channel = channel; self.generation = generation
    }
    /// Feed the follower one round's input row (the leader hook in `CompactingSession.ownedRound`).
    public func sendRoundInput(_ input: FollowerRoundInput) async throws {
        try await channel.send(input)
    }
    /// Drop the side-channel (the follower sees the close and releases its shard).
    public func teardown() { channel.cancel() }
}

extension MLXLanguageModel {
    /// LEADER (rank 0) load: recruit the follower over the side-channel, form the ring, load this shard,
    /// and carry the `DistributedContext` so the turn loop can feed the follower. The follower must already
    /// be listening (`orbital-loop --follower`); it's discovered via the M4a roster. `ringHosts` is rank
    /// order (rank 0 = this leader, first); the follower is the last rank.
    public static func loadDistributedLeader(
        modelId: String, ringHosts: [String], boundary: Int?,
        followerHost: String, followerPort: UInt16,
        params: GenerateParameters = GenerateParameters(), generation: Int = 0
    ) async throws -> (model: MLXLanguageModel, rank: Int, size: Int) {
        let channel = try await connectFollower(host: followerHost, port: followerPort)
        let followerRank = ringHosts.count - 1
        try await channel.send(Recruit(
            modelId: modelId, ringHosts: ringHosts, rank: followerRank,
            boundary: boundary, generation: generation, params: params))
        // Ring-topology breadcrumb (one per load): the EXACT ringHosts the leader binds + sent in the Recruit,
        // diffable against the follower's "[follower] ringHosts=" line when a rendezvous ever misbehaves again.
        // (A silent SYN_SENT hang here historically meant the FOLLOWER lost macOS Local-Network access — an
        // orphaned launch; the launcher must keep it `ssh -tt` PTY-attached. See DISTRIBUTED_ROOM_HANDOFF bug D.)
        distLog("[leader] ring rank 0/\(ringHosts.count) hosts=\(ringHosts) — binds \(ringHosts.first ?? "?"), "
            + "rendezvous with \(ringHosts.last ?? "?") …")
        // Form the ring as rank 0 (rendezvous with the follower's loadDistributedFromHosts) + load.
        let group = try DistributedGroup(ringHosts: ringHosts, rank: 0)
        let ctx = DistributedContext(group: group, channel: channel, generation: generation)
        let model = try await loadDistributed(
            modelId: modelId, group: group, boundary: boundary, distributedContext: ctx)
        return (model, group.rank, group.size)
    }

    /// Open the recruit side-channel to the follower (reuses the cluster PSK/plaintext policy). Fails fast
    /// if the follower isn't listening (`.waiting`), so the feasibility gate can fall back / stay paused.
    static func connectFollower(host: String, port: UInt16) async throws -> FramedChannel {
        let psk = ProcessInfo.processInfo.environment["SWIFTLM_CLUSTER_PSK"]
        guard let p = NWEndpoint.Port(rawValue: port) else { throw ClusterError.connectionClosed }
        let conn = NWConnection(host: NWEndpoint.Host(host), port: p, using: clusterParameters(psk: psk))
        let once = OnceGuard()
        let q = DispatchQueue(label: "io.orbital.leader-recruit")   // Network.start(queue:) only
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready: once.run { cont.resume() }
                case .failed(let e): once.run { cont.resume(throwing: e) }
                case .waiting(let e): once.run { cont.resume(throwing: e) }
                default: break
                }
            }
            conn.start(queue: q)
        }
        return FramedChannel(conn)
    }

    /// M5 Phase-0 spike: probe whether THIS process can reach a follower's recruit listener, returning the
    /// terminal NWConnection state as a string — WITHOUT loading a model or sending a Recruit (serveFollower
    /// only loads a shard AFTER a Recruit, `DistributedTurn.serveFollower`, so the follower stays clean). The
    /// connection is cancelled before returning so the follower's accept→receive(Recruit) doesn't dangle.
    /// Used to settle the headless-leader macOS Local-Network-Privacy question (does a DETACHED orbital-loop
    /// keep local-network access?) before building any ensureFollower machinery. `.ready` ⇒ reachable.
    public static func recruitProbe(host: String, port: UInt16, timeoutSec: Double = 6.0) async -> String {
        let psk = ProcessInfo.processInfo.environment["SWIFTLM_CLUSTER_PSK"]
        guard let p = NWEndpoint.Port(rawValue: port) else { return "bad-port" }
        let conn = NWConnection(host: NWEndpoint.Host(host), port: p, using: clusterParameters(psk: psk))
        let once = OnceGuard()
        let q = DispatchQueue(label: "io.orbital.recruit-probe")   // Network.start(queue:) mandate only
        let result: String = await withCheckedContinuation { (cont: CheckedContinuation<String, Never>) in
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:          once.run { cont.resume(returning: "ready") }
                case .failed(let e):  once.run { cont.resume(returning: "failed: \(e)") }
                case .waiting(let e): once.run { cont.resume(returning: "waiting: \(e)") }
                case .cancelled:      once.run { cont.resume(returning: "cancelled") }
                default: break
                }
            }
            conn.start(queue: q)
            // A TCC Local-Network DENY can manifest as a silent packet drop with NO terminal state — which
            // would hang forever. Time it out on the same mandated net queue (probe-only; not production logic).
            q.asyncAfter(deadline: .now() + timeoutSec) {
                once.run { cont.resume(returning:
                    "timeout(\(timeoutSec)s — no terminal state; consistent with a silent Local-Network drop)") }
            }
        }
        conn.cancel()
        return result
    }
}

/// One-shot guard so a multi-firing NWConnection state handler resumes its continuation exactly once.
final class OnceGuard: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func run(_ body: () -> Void) {
        lock.lock(); defer { lock.unlock() }
        if !done { done = true; body() }
    }
}

/// A listener that accepts recruiters (the leader) and runs `serveFollower` per connection. One per
/// `--follower` box. Reuses the cluster PSK/plaintext policy (`SWIFTLM_CLUSTER_PSK`).
public actor FollowerServer {
    private var listener: NWListener?
    /// ONLY because `NWListener/NWConnection.start(queue:)` mandates a `DispatchQueue`; no logic runs here.
    private let netQueue = DispatchQueue(label: "io.orbital.follower-net")

    public init() {}

    /// Bind on `port` and serve. Resolves once bound; the caller keeps the process alive (accepted
    /// connections run `serveFollower` in their own tasks).
    public func start(port: UInt16) throws {
        let psk = ProcessInfo.processInfo.environment["SWIFTLM_CLUSTER_PSK"]
        let params = clusterParameters(psk: psk)
        guard let port = NWEndpoint.Port(rawValue: port),
              let l = try? NWListener(using: params, on: port) else {
            throw ClusterError.connectionClosed
        }
        let q = netQueue
        l.newConnectionHandler = { conn in
            conn.start(queue: q)
            Task { try? await serveFollower(channel: FramedChannel(conn)) }
        }
        l.start(queue: q)
        listener = l
        let sec = (psk?.isEmpty == false) ? "TLS-PSK" : "plaintext over private TB"
        distLog("[follower] listening on :\(port.rawValue) (\(sec))")
    }

    public func stop() { listener?.cancel(); listener = nil }
}
