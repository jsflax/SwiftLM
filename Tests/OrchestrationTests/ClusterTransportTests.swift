import Testing
@testable import Orchestration

struct ClusterTransportTests {
    private func desc(_ id: String) -> WorkerDescriptor {
        WorkerDescriptor(id: id, models: [ModelID("m")], batchWidth: 8,
                         effectiveParallelism: 1, estTokensPerSecPerStream: 40)
    }
    /// A server whose LocalPool tags each completion with `tag` so the test can see which box ran it.
    private func server(tag: String) -> WorkerServer {
        WorkerServer(pool: LocalPool(descriptor: desc(tag)) { j in
            (0..<j.n).map { "\(tag):\(j.prompt):\($0)" }
        })
    }
    private func job(_ p: String, n: Int = 1) -> GenJob { GenJob(model: ModelID("m"), prompt: p, n: n) }

    @Test func loopbackRPCRoundTrip() async throws {
        let srv = server(tag: "srv")
        let port = try await srv.start()
        #expect(port > 0)
        defer { Task { await srv.stop() } }

        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("remote"))
        let out = await remote.generate(job("hello", n: 3))
        #expect(out == ["srv:hello:0", "srv:hello:1", "srv:hello:2"])
    }

    @Test func remotePoolPipelinesManyJobsOnOneConnection() async throws {
        let srv = server(tag: "srv")
        let port = try await srv.start()
        defer { Task { await srv.stop() } }

        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("remote"))
        for i in 0..<5 {
            let out = await remote.generate(job("p\(i)"))
            #expect(out == ["srv:p\(i):0"])
        }
    }

    /// The whole cluster path: FanOutPool round-robins jobs across TWO network-separated workers.
    @Test func fanOutAcrossTwoRemoteWorkers() async throws {
        let s0 = server(tag: "s0"); let p0 = try await s0.start()
        let s1 = server(tag: "s1"); let p1 = try await s1.start()
        defer { Task { await s0.stop(); await s1.stop() } }

        let fan = FanOutPool(workers: [
            RemotePool(host: "127.0.0.1", port: p0, descriptor: desc("r0")),
            RemotePool(host: "127.0.0.1", port: p1, descriptor: desc("r1")),
        ])
        let res = await fan.generateMany((0..<4).map { job("j\($0)") })
        #expect(res.count == 4)
        #expect(res[0][0].hasPrefix("s0:"))  // round-robin: j0→s0
        #expect(res[1][0].hasPrefix("s1:"))  // j1→s1
        #expect(res[2][0].hasPrefix("s0:"))  // j2→s0
        #expect(res[3][0].hasPrefix("s1:"))  // j3→s1
        let caps = await fan.capabilities()
        #expect(Set(caps.map(\.id)) == ["r0", "r1"])
    }

    /// Failure detection: a RemotePool to a dead port returns [] (and doesn't hang/crash).
    @Test func remotePoolToDeadPortReturnsEmpty() async {
        let remote = RemotePool(host: "127.0.0.1", port: 1, descriptor: desc("dead"))
        let out = await remote.generate(job("x"))
        #expect(out == [])
    }

    /// TLS-PSK auth: a matching pre-shared key completes the handshake and serves the job.
    @Test func pskMatchConnectsAndServes() async throws {
        let srv = WorkerServer(pool: LocalPool(descriptor: desc("s")) { j in (0..<j.n).map { "ok:\($0)" } },
                               psk: "shared-secret")
        let port = try await srv.start()
        defer { Task { await srv.stop() } }
        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("r"), psk: "shared-secret")
        let out = await remote.generate(job("hi", n: 2))
        #expect(out == ["ok:0", "ok:1"])  // matching PSK → TLS handshake succeeds, job served
    }

    /// TLS-PSK auth: a mismatched key fails the handshake; RemotePool returns [] (no hang, no serve).
    @Test func pskMismatchReturnsEmpty() async throws {
        let srv = WorkerServer(pool: LocalPool(descriptor: desc("s")) { j in (0..<j.n).map { "ok:\($0)" } },
                               psk: "shared-secret")
        let port = try await srv.start()
        defer { Task { await srv.stop() } }
        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("r"), psk: "WRONG-KEY")
        let out = await remote.generate(job("hi"))
        #expect(out == [])  // mismatched PSK → handshake fails → []
    }
}
