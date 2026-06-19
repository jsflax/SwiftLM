import Foundation

// ── A wall-clock bound on a single tool dispatch.
//
// Native tools self-bound (bash has its own timeout, web_fetch 30s), but an MCP tool dispatch
// (`client.callTool`) waits for the server's response with NO timeout — so a wedged/slow/deadlocked MCP
// tool hangs the ENTIRE agent turn indefinitely (observed: a `clipboard` read that never returned stalled
// the loop for 8+ minutes). The round budget and completion-stop run at round BOUNDARIES, so they can't
// help while the loop is blocked INSIDE a dispatch. This bounds the dispatch itself: the wedged tool is
// abandoned after `seconds` and reported as "did not respond" so the turn proceeds.

/// Default per-dispatch wall-clock bound. Generous — above the native tools' own timeouts (bash 60s) so it
/// only catches genuine hangs, not slow-but-legit work.
public let defaultToolDispatchTimeoutSeconds: Double = 90

/// Run `operation` with a wall-clock bound. Returns its result, or `nil` if it didn't finish within
/// `seconds` (the abandoned task is cancelled; a dispatch that ignores cancellation simply runs detached to
/// completion while the caller moves on). `seconds <= 0` disables the bound (always awaits the operation).
public func withToolTimeout<T: Sendable>(
    seconds: Double, _ operation: @escaping @Sendable () async -> T
) async -> T? {
    guard seconds > 0 else { return await operation() }
    return await withTaskGroup(of: T?.self) { group in
        group.addTask { await operation() }
        group.addTask {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            return nil                                  // timeout sentinel
        }
        let first = await group.next() ?? nil           // whichever finishes first: result, or nil on timeout
        group.cancelAll()
        return first
    }
}
