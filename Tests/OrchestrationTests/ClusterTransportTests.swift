import XCTest
@testable import Orchestration

final class ClusterTransportTests: XCTestCase {
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

    func testLoopbackRPCRoundTrip() async throws {
        let srv = server(tag: "srv")
        let port = try await srv.start()
        XCTAssertGreaterThan(port, 0)
        defer { Task { await srv.stop() } }

        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("remote"))
        let out = await remote.generate(job("hello", n: 3))
        XCTAssertEqual(out, ["srv:hello:0", "srv:hello:1", "srv:hello:2"])
    }

    func testRemotePoolPipelinesManyJobsOnOneConnection() async throws {
        let srv = server(tag: "srv")
        let port = try await srv.start()
        defer { Task { await srv.stop() } }

        let remote = RemotePool(host: "127.0.0.1", port: port, descriptor: desc("remote"))
        for i in 0..<5 {
            let out = await remote.generate(job("p\(i)"))
            XCTAssertEqual(out, ["srv:p\(i):0"])
        }
    }

    /// The whole cluster path: FanOutPool round-robins jobs across TWO network-separated workers.
    func testFanOutAcrossTwoRemoteWorkers() async throws {
        let s0 = server(tag: "s0"); let p0 = try await s0.start()
        let s1 = server(tag: "s1"); let p1 = try await s1.start()
        defer { Task { await s0.stop(); await s1.stop() } }

        let fan = FanOutPool(workers: [
            RemotePool(host: "127.0.0.1", port: p0, descriptor: desc("r0")),
            RemotePool(host: "127.0.0.1", port: p1, descriptor: desc("r1")),
        ])
        let res = await fan.generateMany((0..<4).map { job("j\($0)") })
        XCTAssertEqual(res.count, 4)
        XCTAssertTrue(res[0][0].hasPrefix("s0:"), "\(res[0])")  // round-robin: j0→s0
        XCTAssertTrue(res[1][0].hasPrefix("s1:"), "\(res[1])")  // j1→s1
        XCTAssertTrue(res[2][0].hasPrefix("s0:"), "\(res[2])")  // j2→s0
        XCTAssertTrue(res[3][0].hasPrefix("s1:"), "\(res[3])")  // j3→s1
        // and capabilities aggregate across both remote workers
        let caps = await fan.capabilities()
        XCTAssertEqual(Set(caps.map(\.id)), ["r0", "r1"])
    }

    /// Failure detection: a RemotePool to a dead port returns [] (and doesn't hang/crash).
    func testRemotePoolToDeadPortReturnsEmpty() async {
        let remote = RemotePool(host: "127.0.0.1", port: 1, descriptor: desc("dead"))
        let out = await remote.generate(job("x"))
        XCTAssertEqual(out, [])
    }
}
