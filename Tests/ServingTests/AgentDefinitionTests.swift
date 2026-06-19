import Testing
import Foundation
@testable import Serving

struct AgentDefinitionTests {
    // Mirrors the real ~/.claude/agents/session-learner.md shape: `---` frontmatter + markdown body.
    let sessionLearner = """
    ---
    name: session-learner
    description: Learns from coding sessions by analyzing what happened and storing key insights as memories.
    model: sonnet
    maxTurns: 15
    ---

    You are a session learning agent. Your job is to review what happened in a coding session and store
    the key insights as memories for future recall.

    Run `recall(query: "project overview")` then `remember` the insights.
    """

    @Test func parsesFrontmatterAndBody() throws {
        let def = try #require(AgentDefinitionLoader.parse(sessionLearner, fallbackName: "x"))
        #expect(def.name == "session-learner")
        #expect(def.model == "sonnet")
        #expect(def.maxTurns == 15)
        #expect(def.tools == nil)                                  // no `tools:` → inherit all
        #expect(def.description.contains("Learns from coding sessions"))
        #expect(def.systemPrompt.hasPrefix("You are a session learning agent"))
        #expect(def.systemPrompt.contains("`recall("))             // body preserved verbatim
        #expect(!def.systemPrompt.contains("---"))                 // frontmatter stripped
    }

    @Test func toolsFieldVariants() {
        #expect(AgentDefinitionLoader.parseTools(nil) == nil)
        #expect(AgentDefinitionLoader.parseTools("*") == nil)
        #expect(AgentDefinitionLoader.parseTools("All tools") == nil)
        #expect(AgentDefinitionLoader.parseTools("mcp__memory__remember, mcp__memory__recall")
                == ["mcp__memory__remember", "mcp__memory__recall"])
    }

    @Test func noFrontmatterReturnsNil() {
        #expect(AgentDefinitionLoader.parse("just a plain markdown file\nno frontmatter", fallbackName: "x") == nil)
    }

    @Test func nameFallsBackToFilename() throws {
        let def = try #require(AgentDefinitionLoader.parse("---\nmodel: haiku\n---\nbody", fallbackName: "reviewer"))
        #expect(def.name == "reviewer")        // no `name:` field → filename
        #expect(def.model == "haiku")
    }

    @Test func loadsRealAgentsDirIfPresent() {
        // Non-fatal: if the user's ~/.claude/agents exists, session-learner should load cleanly.
        let defs = AgentDefinitionLoader.load()
        if let sl = defs["session-learner"] {
            #expect(!sl.systemPrompt.isEmpty)
            #expect(sl.systemPrompt.lowercased().contains("memor"))   // it's about memories
        }
    }
}
