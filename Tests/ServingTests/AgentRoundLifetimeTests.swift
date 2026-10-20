import Foundation
import Testing
@testable import Serving

@Suite("Owned agent round lifetime", .serialized)
struct AgentRoundLifetimeTests {
    @Test(arguments: [RoundExit.cancel, .finish, .failure])
    func streamTerminationDoesNotReleaseProducerOwnership(exit: RoundExit) async throws {
        let backend = RoundLifetimeBackend(exit: exit)
        let turn = streamAgentTurnWithTask(prompt: "synthetic", instructions: nil, hooks: nil,
            config: .init(maxRounds: 8), backend: backend)
        let completed = RoundLifetimeState()
        let consumer = Task { () -> Bool in
            do { for try await _ in turn.stream {} ; return false }
            catch { return error is RoundLifetimeError }
        }
        let waiter = Task { await turn.task.value; completed.setCompleted() }
        do {
            try await waitFor { backend.state.snapshot.started == 1 }
            if exit == .cancel { turn.task.cancel() }
            try await waitFor { backend.state.snapshot.terminated >= 1 }
            // The producer's explicit cleanup gate is still closed. Stream termination is
            // already observed, so outer completion here would be a false drain receipt.
            try await Task.sleep(for: .milliseconds(30))
            let beforeRelease = backend.state.snapshot
            let outerFinishedEarly = completed.snapshot.completed
            await backend.cleanup.open()
            await waiter.value
            let failed = await consumer.value
            await backend.joinAll()
            #expect(beforeRelease.started == 1)
            #expect(beforeRelease.active == 1)
            #expect(!outerFinishedEarly)
            #expect(backend.state.snapshot.active == 0)
            #expect(backend.state.snapshot.started == 1)
            #expect(failed == (exit == .failure))
            #expect(backend.state.snapshot.dispatched == 0)
        } catch {
            turn.task.cancel(); await backend.cleanup.open()
            await waiter.value; _ = await consumer.value; await backend.joinAll()
            throw error
        }
    }

    @Test func nextRoundWaitsForPreviousCleanupAndKeepsNormalNudge() async throws {
        let backend = RoundLifetimeBackend(exit: .empty)
        let turn = streamAgentTurnWithTask(prompt: "synthetic", instructions: nil, hooks: nil,
            config: .init(maxRounds: 8), backend: backend)
        let consumer = Task { () throws -> [AgentStreamEvent] in
            var events: [AgentStreamEvent] = []
            for try await event in turn.stream { events.append(event) }
            return events
        }
        do {
            try await waitFor { backend.state.snapshot.terminated >= 1 }
            try await Task.sleep(for: .milliseconds(30))
            let beforeRelease = backend.state.snapshot
            await backend.cleanup.open()
            let events = try await consumer.value
            await turn.task.value; await backend.joinAll()
            #expect(beforeRelease.started == 1)
            #expect(beforeRelease.active == 1)
            #expect(backend.state.snapshot.started == 2)
            #expect(backend.state.snapshot.maximumActive == 1)
            #expect(backend.state.snapshot.nudged)
            guard case .result(let text, _, _, let isError)? = events.last else {
                Issue.record("Expected ordinary final answer after the nudge"); return
            }
            #expect(text == "Finished.")
            #expect(!isError)
        } catch {
            turn.task.cancel(); await backend.cleanup.open()
            _ = try? await consumer.value; await turn.task.value; await backend.joinAll()
            throw error
        }
    }
}

enum RoundExit: Sendable, Equatable { case cancel, finish, failure, empty }
private struct RoundLifetimeError: Error {}
private struct RoundLifetimeTimeout: Error {}
private actor RoundCleanupGate {
    private var opened = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiting.append($0) }
    }
    func open() {
        opened = true
        let callbacks = waiting; waiting.removeAll()
        callbacks.forEach { $0.resume() }
    }
}
private final class RoundLifetimeState: @unchecked Sendable {
    struct Snapshot {
        var started = 0
        var active = 0
        var maximumActive = 0
        var terminated = 0
        var dispatched = 0
        var completed = false
        var nudged = false
    }
    private let lock = NSLock()
    private var value = Snapshot()
    private var tasks: [Task<Void, Never>] = []
    var snapshot: Snapshot { lock.withLock { value } }
    func start(resume: [ResumeMessage]) -> Int {
        lock.withLock {
            value.started += 1; value.active += 1
            value.maximumActive = max(value.maximumActive, value.active)
            if case .user(let text)? = resume.first { value.nudged = text.contains("Continue") }
            return value.started
        }
    }
    func end() { lock.withLock { value.active -= 1 } }
    func terminate() { lock.withLock { value.terminated += 1 } }
    func dispatch() { lock.withLock { value.dispatched += 1 } }
    func setCompleted() { lock.withLock { value.completed = true } }
    func retain(_ task: Task<Void, Never>) { lock.withLock { tasks.append(task) } }
    func allTasks() -> [Task<Void, Never>] { lock.withLock { tasks } }
}
private final class RoundLifetimeBackend: AgentTurnBackend, @unchecked Sendable {
    let state = RoundLifetimeState()
    let cleanup = RoundCleanupGate()
    let exit: RoundExit
    init(exit: RoundExit) { self.exit = exit }
    func round(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool)
        -> AsyncThrowingStream<GenStep, Error> {
        roundWithTask(instructions: instructions, prompt: prompt, resume: resume,
                      toolsEnabled: toolsEnabled, images: []).stream
    }
    func roundWithTask(instructions: String?, prompt: String, resume: [ResumeMessage], toolsEnabled: Bool, images: [URL])
        -> (stream: AsyncThrowingStream<GenStep, Error>, task: Task<Void, Never>?) {
        let number = state.start(resume: resume)
        let (stream, continuation) = AsyncThrowingStream<GenStep, Error>.makeStream()
        let task = Task {
            if number > 1 || exit == .finish { continuation.yield(.chunk("Finished.")) }
            if exit == .failure { continuation.finish(throwing: RoundLifetimeError()) }
            else if exit != .cancel { continuation.finish() }
            await cleanup.wait() // Intentionally ignores cancellation until real cleanup is released.
            state.end()
            continuation.finish()
        }
        state.retain(task)
        continuation.onTermination = { [state] _ in state.terminate(); task.cancel() }
        return (stream, task)
    }
    func dispatch(name: String, argsJSON: String) async -> (result: String, isError: Bool) {
        state.dispatch(); return ("Unexpected", true)
    }
    func joinAll() async { for task in state.allTasks() { await task.value } }
}
private func waitFor(_ condition: @escaping @Sendable () -> Bool) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(3))
    while !condition() {
        guard ContinuousClock.now < end else { throw RoundLifetimeTimeout() }
        try await Task.sleep(for: .milliseconds(2))
    }
}
