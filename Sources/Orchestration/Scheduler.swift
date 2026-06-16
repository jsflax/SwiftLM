import Foundation

/// Sits in front of a `ComputePool`: an admission queue tagged by priority that caps how many
/// requests run concurrently (the cap ≈ the pool's batch width, so the admitted concurrent streams
/// are exactly what a continuous-batching `LocalPool` fuses into one forward pass). Higher-priority
/// work is admitted first — a live `interactive` turn never waits behind a `background` flywheel
/// batch. FIFO within a priority. The actual batching/placement is the pool's concern; the Scheduler
/// only decides *what is allowed to run now*.
public actor Scheduler {
    private let pool: ComputePool
    public let maxConcurrent: Int
    private var inflight = 0
    private var seqCounter = 0

    private struct Waiter { let priority: Priority; let seq: Int; let cont: CheckedContinuation<Void, Never> }
    private var waiters: [Waiter] = []

    public init(pool: ComputePool, maxConcurrent: Int) {
        self.pool = pool
        self.maxConcurrent = max(1, maxConcurrent)
    }

    // ── Admission ───────────────────────────────────────────────────────────────────────────────
    private func acquire(_ priority: Priority) async {
        if inflight < maxConcurrent { inflight += 1; return }
        seqCounter += 1
        let seq = seqCounter
        await withCheckedContinuation { cont in
            waiters.append(Waiter(priority: priority, seq: seq, cont: cont))
        }
        // Resumed by `release()`, which already incremented `inflight` on our behalf.
    }

    private func release() {
        inflight -= 1
        guard !waiters.isEmpty else { return }
        // Admit the highest-priority, then earliest (FIFO-within-priority) waiter.
        var best = 0
        for i in waiters.indices {
            let a = waiters[i], b = waiters[best]
            if a.priority > b.priority || (a.priority == b.priority && a.seq < b.seq) { best = i }
        }
        let w = waiters.remove(at: best)
        inflight += 1
        w.cont.resume()
    }

    // ── Submit ──────────────────────────────────────────────────────────────────────────────────
    /// Enqueue (respecting priority + the concurrency cap), then stream the pool's tokens, releasing
    /// the slot when the generation finishes so the next waiter is admitted.
    public func submit(_ req: InferenceRequest) async -> AsyncStream<Token> {
        await acquire(req.priority)
        let upstream = await pool.submit(req)
        return AsyncStream<Token> { continuation in
            let task = Task {
                for await tok in upstream { continuation.yield(tok) }
                continuation.finish()
                await self.release()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Convenience: run a request to completion under scheduling, returning its full text.
    public func complete(_ req: InferenceRequest) async -> String {
        var out = ""
        for await tok in await submit(req) { out += tok.text }
        return out
    }

    // ── Introspection (tests / observability) ───────────────────────────────────────────────────
    public var inflightCount: Int { inflight }
    public var queuedCount: Int { waiters.count }
}
