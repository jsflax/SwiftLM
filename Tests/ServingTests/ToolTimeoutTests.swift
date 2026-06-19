import Testing
import Foundation
@testable import Serving

struct ToolTimeoutTests {
    @Test func returnsResultWhenOperationFinishesInTime() async {
        let r = await withToolTimeout(seconds: 5) { "done" }
        #expect(r == "done")
    }

    @Test func returnsNilWhenOperationExceedsBound() async {
        let r = await withToolTimeout(seconds: 0.05) { () async -> String in
            try? await Task.sleep(nanoseconds: 3_000_000_000)   // would take 3s; cancelled at 0.05s
            return "late"
        }
        #expect(r == nil)   // timed out
    }

    @Test func zeroDisablesTheBound() async {
        let r = await withToolTimeout(seconds: 0) { "always" }
        #expect(r == "always")
    }
}
