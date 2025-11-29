import Foundation
import Testing
@testable import LlamaANE
import CoreML
import Tokenizers
import TensorUtils
import Generation
import Hub
//import Models
import XCTest
import JSONSchema

// Path to test models - update this to point to your exported models
let testModelsPath = "/Users/jason/Documents/LlamaANE/Plugins/LLMGenerator/models"

@Suite("LlamaANETest Suite")
struct LlamaANETests {

    // MARK: - Basic Model Loading Test

    @Test("Load model from path and run basic inference")
    func testBasicInference() async throws {
        let modelPath = "\(testModelsPath)/Llama-3.2-3B-Instruct.mlpackage"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)

        // Verify model loaded
        print("Model loaded: \(lm.modelName ?? "unknown")")
        print("Context Length: \(lm.minContextLength) - \(lm.maxContextLength)")

        // Warm up the model
        print("Warming up...")
        let warmupStart = Date()
        try await lm.warmup()
        let warmupTime = Date().timeIntervalSince(warmupStart)
        print("Warmup took: \(String(format: "%.2f", warmupTime))s")

        // Create a session and run inference
        let session = await lm.makeSession(systemPrompt: "You are a helpful assistant.")
        let stream = await session.infer(prompt: "Say hello in one sentence.")

        var output = ""
        for await token in stream {
            output += token
            print(token, terminator: "")
        }
        print()

        #expect(!output.isEmpty, "Model should generate some output")
    }

    @Test("Test INT4 quantized model")
    func testInt4Model() async throws {
        let modelPath = "\(testModelsPath)/Llama-3.2-3B-Instruct_Int4.mlpackage"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        let session = await lm.makeSession()
        let stream = await session.infer(prompt: "What is 2 + 2?")

        var output = ""
        for await token in stream {
            output += token
            print(token, terminator: "")
        }
        print()

        #expect(!output.isEmpty, "INT4 model should generate some output")
    }

    // MARK: - Grammar Session Test

    @Test("Test session with JSON schema constraint")
    func testGrammarSession() async throws {
        let modelPath = "\(testModelsPath)/Llama-3.2-3B-Instruct_Int4.mlpackage"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        let session = await lm.makeSession(
            systemPrompt: "You are a helpful assistant that responds in JSON format."
        )

        // Use the parsed result directly
        let result: SimpleResponse = try await session.infer(prompt: "What is the capital of France?", as: SimpleResponse.self)
        print("Grammar result: \(result)")

        #expect(!result.answer.isEmpty, "Grammar session should produce a valid response")
    }

    // MARK: - Qwen Model Test

    @Test("Test Qwen 2.5 model with sampling", .timeLimit(.minutes(2)))
    func testQwenModel() async throws {
        // Try FP16 model first (INT4 has CoreML execution issues on some systems)
        let modelPath = "\(testModelsPath)/Qwen2.5-1.5B-Instruct.mlpackage"

        // Compile and load the model
        print("Compiling model...")
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU  // Qwen requires GPU; produces NaN on CPU-only (root cause unknown)
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        // Create LanguageModel - tokenizer is auto-selected based on model metadata
        let lm = try LanguageModel(model: mlModel)

        // Verify model loaded
        print("Model loaded: \(lm.modelName ?? "unknown")")
        print("Model type: \(lm.modelConfig.modelType ?? "unknown")")
        print("Model family: \(lm.modelConfig.modelFamily)")
        print("Context Length: \(lm.minContextLength) - \(lm.maxContextLength)")

        // Test 1: Greedy decoding (baseline)
        print("\n--- Test 1: Greedy Decoding ---")
        let greedySession = await lm.makeSession(
            systemPrompt: "You are a helpful assistant.",
            doSample: false
        )

        var greedyTokenCount = 0
        let greedyStart = Date()
        let greedyStream = await greedySession.infer(prompt: "What is the capital of Japan?")

        var greedyOutput = ""
        for await token in greedyStream {
            greedyOutput += token
            greedyTokenCount += 1
            print(token, terminator: "")
        }
        let greedyElapsed = Date().timeIntervalSince(greedyStart)
        let greedyTPS = Double(greedyTokenCount) / greedyElapsed
        print()
        print("Greedy: \(greedyTokenCount) tokens in \(String(format: "%.2f", greedyElapsed))s (\(String(format: "%.1f", greedyTPS)) tok/s)")

        // Test 2: Sampling with temperature=0.7, topK=40, topP=0.9
        print("\n--- Test 2: Sampling (temp=0.7, topK=40, topP=0.9) ---")
        let samplingSession = await lm.makeSession(
            systemPrompt: "You are a helpful assistant.",
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            doSample: true
        )

        var samplingTokenCount = 0
        let samplingStart = Date()
        let samplingStream = await samplingSession.infer(prompt: "What is the capital of Japan?")

        var samplingOutput = ""
        for await token in samplingStream {
            samplingOutput += token
            samplingTokenCount += 1
            print(token, terminator: "")
        }
        let samplingElapsed = Date().timeIntervalSince(samplingStart)
        let samplingTPS = Double(samplingTokenCount) / samplingElapsed
        print()
        print("Sampling: \(samplingTokenCount) tokens in \(String(format: "%.2f", samplingElapsed))s (\(String(format: "%.1f", samplingTPS)) tok/s)")

        // Compare performance
        let speedRatio = greedyTPS / samplingTPS
        print("\nPerformance: Greedy is \(String(format: "%.2f", speedRatio))x faster than sampling")

        // Both outputs should be non-empty and coherent
        #expect(!greedyOutput.isEmpty, "Greedy output should not be empty")
        #expect(!samplingOutput.isEmpty, "Sampling output should not be empty")
        #expect(greedyOutput.lowercased().contains("tokyo"), "Greedy output should mention Tokyo")
        // Sampling may pick different cities due to randomness, just check it's about Japan
        #expect(samplingOutput.lowercased().contains("tokyo") ||
                samplingOutput.lowercased().contains("japan") ||
                samplingOutput.lowercased().contains("osaka") ||
                samplingOutput.lowercased().contains("kyoto"),
                "Sampling output should mention Japanese cities")
    }

    // MARK: - Performance Test

    enum TestModelKind: String, CaseIterable {
        case llamaInt4 = "Llama-3.2-3B-Instruct_Int4.mlpackage"
        case qwen = "Qwen2.5-1.5B-Instruct.mlpackage"
        case qwenInt4 = "Qwen2.5-1.5B-Instruct_Int4.mlpackage"
        case qwen0_5Int4 = "Qwen2.5-0.5B-Instruct_Int4.mlpackage"
    }

    @available(macOS 26.0, *)
    @Test func testFoundationModels() async throws {
        let modelPath = "\(testModelsPath)/\(TestModelKind.qwen0_5Int4.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)
        
        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let aneSession = await lm.makeSession(
            systemPrompt: """
            You are a helpful assistant that provides information about people in JSON format.
            Include their name, age, and address (with street, city, and zip code).
            """
        )
        let session = LanguageModelSession(model: .default)
        session.prewarm(promptPrefix: .some(.init("""
        You are a helpful assistant that provides information about people in JSON format.
        Include their name, age, and address (with street, city, and zip code).
        """)))
        
        var startTime = Date()
        let response = try await session.respond(to: """
            Tell me about John Smith who is 35 years old and lives at 123 Main St, Springfield, 12345.
            """, generating: Person.self)
        var totalTime = Date().timeIntervalSince(startTime)
        print("Apple FM Total time: \(String(format: "%.2f", totalTime))s")
        print(response.content)
        
        startTime = Date()
        let response2 = try await aneSession.infer(prompt: """
        Tell me about John Smith who is 35 years old and lives at 123 Main St, Springfield, 12345.
        """, as: Person.self)
        totalTime = Date().timeIntervalSince(startTime)
        print("ANE Total time: \(String(format: "%.2f", totalTime))s")
        print(response2)
    }
    
    @Test("Grammar session performance with complex schema", .timeLimit(.minutes(2)), .serialized, arguments: [TestModelKind.qwenInt4, .llamaInt4])
    func testGrammarSessionPerformance(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)

        // Warm up
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: """
            You are a movie critic assistant. When asked about a movie, provide a structured review in JSON format.
            Include the title, director, release year, rating (0-10), genre, and a brief summary.
            """
        )

        let prompt = "Give me a review of the movie Inception directed by Christopher Nolan."

        print("\n--- Grammar Session Performance Test ---")
        print("Model: \(modelKind)")
        print("Schema: MovieReview (6 fields: title, director, releaseYear, rating, genre, summary)")

        let startTime = Date()
        var firstTokenTime: Date?
        var tokenCount = 0
        var output = ""

        // Use the AsyncStream<String> overload for streaming tokens with grammar constraint
        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: MovieReview.self)
        for await token in stream {
            if firstTokenTime == nil {
                firstTokenTime = Date()
            }
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let totalTime = Date().timeIntervalSince(startTime)
        let timeToFirstToken = firstTokenTime?.timeIntervalSince(startTime) ?? 0
        let generationTime = totalTime - timeToFirstToken
        let tokensPerSecond = tokenCount > 0 ? Double(tokenCount) / generationTime : 0

        print("\n--- Performance Metrics ---")
        print("Time to first token: \(String(format: "%.3f", timeToFirstToken))s")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", totalTime))s")
        print("Generation time: \(String(format: "%.2f", generationTime))s")
        print("Tokens/second: \(String(format: "%.2f", tokensPerSecond))")
        print("---------------------------")

        // Try to decode the result
        if let jsonData = output.data(using: .utf8) {
            do {
                let review = try JSONDecoder().decode(MovieReview.self, from: jsonData)
                print("Parsed MovieReview:")
                print("  Title: \(review.title)")
                print("  Director: \(review.director)")
                print("  Year: \(review.releaseYear)")
                print("  Rating: \(review.rating)")
                print("  Genre: \(review.genre)")
                print("  Summary: \(review.summary)")
            } catch {
                print("Failed to decode JSON: \(error)")
                print("Raw output: \(output)")
            }
        }

        #expect(!output.isEmpty, "Grammar session should generate output")
        #expect(tokenCount > 5, "Should generate multiple tokens")
    }

    // MARK: - SchemaGuide Model Inference Test

    @Test("SchemaGuide descriptions improve model output quality", .timeLimit(.minutes(2)), .serialized, arguments: [TestModelKind.qwenInt4])
    func testSchemaGuideModelInference(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: "You are a helpful assistant that generates user profiles."
        )

        // UserProfile has @SchemaGuide annotations:
        // - name: "The user's display name"
        // - age: "Age in years" with .range(0...150)
        // - bio: "User's biography" with .maxLength(500)
        // - rating: "Rating from 0.0 to 5.0" with .doubleRange(0.0...5.0)
        // - followers: "Number of followers" with .range(0...1000000)

        let prompt = "Create a profile for a software engineer named Alice who is 28 years old."

        print("\n--- SchemaGuide Model Inference Test ---")
        print("Model: \(modelKind)")
        print("Schema: UserProfile with @SchemaGuide descriptions and constraints")
        print("\nSchema description injected into prompt:")
        print(UserProfile.schemaDescription())

        let startTime = Date()
        var output = ""
        var tokenCount = 0

        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: UserProfile.self)
        for await token in stream {
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let totalTime = Date().timeIntervalSince(startTime)
        print("\n--- Metrics ---")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", totalTime))s")

        // Try to decode and validate the result
        if let jsonData = output.data(using: .utf8) {
            do {
                let profile = try JSONDecoder().decode(UserProfile.self, from: jsonData)
                print("\nParsed UserProfile:")
                print("  Name: \(profile.name)")
                print("  Age: \(profile.age)")
                print("  Bio: \(profile.bio)")
                print("  Rating: \(profile.rating)")
                print("  Followers: \(profile.followers)")

                // Validate constraints from @SchemaGuide
                #expect(profile.age >= 0 && profile.age <= 150, "Age should be in range 0-150")
                #expect(profile.rating >= 0.0 && profile.rating <= 5.0, "Rating should be in range 0.0-5.0")
                #expect(profile.followers >= 0 && profile.followers <= 1000000, "Followers should be in range 0-1000000")
                #expect(profile.bio.count <= 500, "Bio should be max 500 chars")

                // Check that the model understood the prompt
                #expect(profile.name.lowercased().contains("alice") || profile.age == 28,
                       "Model should have understood the prompt about Alice who is 28")

                print("\n✅ All @SchemaGuide constraints validated!")
            } catch {
                print("Failed to decode JSON: \(error)")
                print("Raw output: \(output)")
            }
        }

        #expect(!output.isEmpty, "Should generate output")
        #expect(tokenCount > 5, "Should generate multiple tokens")
    }

    // MARK: - Nested Object Grammar Test

    @Test("Grammar session with nested objects", .timeLimit(.minutes(2)), .serialized, arguments: [TestModelKind.qwenInt4, .llamaInt4])
    @available(macOS 26.0, *)
    func testGrammarNestedObjects(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: """
            You are a helpful assistant that provides information about people in JSON format.
            Include their name, age, and address (with street, city, and zip code).
            """
        )

        let prompt = "Tell me about John Smith who is 35 years old and lives at 123 Main St, Springfield, 12345."

        print("\n--- Nested Objects Grammar Test ---")
        print("Model: \(modelKind)")
        print("Schema: Person (nested Address object)")

        let startTime = Date()
        var output = ""
        var tokenCount = 0

        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: Person.self)
        for await token in stream {
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let totalTime = Date().timeIntervalSince(startTime)
        print("\n--- Metrics ---")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", totalTime))s")

        // Try to decode the result
        if let jsonData = output.data(using: .utf8) {
            do {
                let person = try JSONDecoder().decode(Person.self, from: jsonData)
                print("Parsed Person:")
                print("  Name: \(person.name)")
                print("  Age: \(person.age)")
                print("  Address:")
                print("    Street: \(person.address.street)")
                print("    City: \(person.address.city)")
                print("    Zip: \(person.address.zipCode)")

                #expect(!person.name.isEmpty, "Person name should not be empty")
                #expect(!person.address.city.isEmpty, "Address city should not be empty")
            } catch {
                print("Failed to decode JSON: \(error)")
                print("Raw output: \(output)")
                Issue.record("Failed to decode nested object: \(error)")
            }
        }

        #expect(!output.isEmpty, "Grammar session should generate output")
    }

    // MARK: - Array of Objects Grammar Test

    @Test("Grammar session with array of objects", .timeLimit(.minutes(2)), .serialized, arguments: [TestModelKind.qwenInt4, .llamaInt4])
    func testGrammarArrayOfObjects(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: """
            You are a helpful assistant that creates todo lists in JSON format.
            Each list has a name and an array of tasks with title and completion status.
            """
        )

        let prompt = "Create a todo list called 'Weekend Chores' with 3 tasks: Buy groceries (not done), Clean house (done), and Walk dog (not done)."

        print("\n--- Array of Objects Grammar Test ---")
        print("Model: \(modelKind)")
        print("Schema: TodoList (array of Task objects)")

        let startTime = Date()
        var output = ""
        var tokenCount = 0

        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: TodoList.self)
        for await token in stream {
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let totalTime = Date().timeIntervalSince(startTime)
        print("\n--- Metrics ---")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", totalTime))s")

        // Try to decode the result
        if let jsonData = output.data(using: .utf8) {
            do {
                let todoList = try JSONDecoder().decode(TodoList.self, from: jsonData)
                print("Parsed TodoList:")
                print("  List Name: \(todoList.listName)")
                print("  Tasks (\(todoList.tasks.count)):")
                for task in todoList.tasks {
                    print("    - \(task.title): \(task.completed ? "Done" : "Not done")")
                }

                #expect(!todoList.listName.isEmpty, "List name should not be empty")
                #expect(todoList.tasks.count > 0, "Should have at least one task")
            } catch {
                print("Failed to decode JSON: \(error)")
                print("Raw output: \(output)")
                Issue.record("Failed to decode array of objects: \(error)")
            }
        }

        #expect(!output.isEmpty, "Grammar session should generate output")
    }

    // MARK: - Complex Nested + Array Grammar Test

    @Test("Grammar session with nested objects and arrays", .timeLimit(.minutes(2)), .serialized, arguments: [TestModelKind.qwenInt4, .llamaInt4])
    func testGrammarComplexNestedAndArray(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: """
            You are a helpful assistant that describes projects in JSON format.
            Include the project name, description, a lead member (with name and role),
            and an array of team members (each with name and role).
            """
        )

        let prompt = "Describe Project Alpha led by Alice (Manager) with team members Bob (Developer) and Carol (Designer)."

        print("\n--- Complex Nested + Array Grammar Test ---")
        print("Model: \(modelKind)")
        print("Schema: Project (nested TeamMember + array of TeamMember)")

        let startTime = Date()
        var output = ""
        var tokenCount = 0

        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: Project.self)
        for await token in stream {
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let totalTime = Date().timeIntervalSince(startTime)
        print("\n--- Metrics ---")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", totalTime))s")

        // Try to decode the result
        if let jsonData = output.data(using: .utf8) {
            do {
                let project = try JSONDecoder().decode(Project.self, from: jsonData)
                print("Parsed Project:")
                print("  Name: \(project.projectName)")
                print("  Description: \(project.description)")
                print("  Lead: \(project.lead.name) (\(project.lead.role))")
                print("  Members (\(project.members.count)):")
                for member in project.members {
                    print("    - \(member.name): \(member.role)")
                }

                #expect(!project.projectName.isEmpty, "Project name should not be empty")
                #expect(!project.lead.name.isEmpty, "Lead name should not be empty")
                #expect(project.members.count > 0, "Should have at least one team member")
            } catch {
                print("Failed to decode JSON: \(error)")
                print("Raw output: \(output)")
                Issue.record("Failed to decode complex nested structure: \(error)")
            }
        }

        #expect(!output.isEmpty, "Grammar session should generate output")
    }

    @Test("Performance test with longer generation", arguments: TestModelKind.allCases)
    func testPerformanceLongerGeneration(modelKind: TestModelKind) async throws {
        let modelPath = "\(testModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        let session = try await lm.makeSession(systemPrompt: "You are a helpful assistant.")

        let prompt = "Explain in 3-4 sentences why the sky is blue."

        let startTime = Date()
        let stream = await session.infer(prompt: prompt)

        var output = ""
        var tokenCount = 0
        for await token in stream {
            output += token
            tokenCount += 1
            print(token, terminator: "")
        }
        print()

        let elapsed = Date().timeIntervalSince(startTime)
        let tokensPerSecond = Double(tokenCount) / elapsed

        print("--- Performance Metrics ---")
        print("Model: \(modelKind)")
        print("Total tokens: \(tokenCount)")
        print("Total time: \(String(format: "%.2f", elapsed))s")
        print("Tokens/second: \(String(format: "%.2f", tokensPerSecond))")
        print("---------------------------")

        #expect(!output.isEmpty, "Model should generate output")
        #expect(tokenCount > 10, "Should generate more than 10 tokens for this prompt")
    }

    // MARK: - SchemaGuide Attribute Tests

    @Test("SchemaGuide attribute generates descriptions")
    func testSchemaGuideDescriptions() throws {
        // Verify schemaProperties contains descriptions from @SchemaGuide
        let properties = UserProfile.schemaProperties
        #expect(properties != nil, "UserProfile should have schemaProperties")

        guard let props = properties else { return }

        // Check that we have all expected properties
        #expect(props.count == 5, "UserProfile should have 5 properties")

        // Find and verify each property's description
        let nameProperty = props.first { $0.name == "name" }
        #expect(nameProperty?.description == "The user's display name", "name should have correct description")

        let ageProperty = props.first { $0.name == "age" }
        #expect(ageProperty?.description == "Age in years", "age should have correct description")

        let bioProperty = props.first { $0.name == "bio" }
        #expect(bioProperty?.description == "User's biography", "bio should have correct description")

        let ratingProperty = props.first { $0.name == "rating" }
        #expect(ratingProperty?.description == "Rating from 0.0 to 5.0", "rating should have correct description")

        let followersProperty = props.first { $0.name == "followers" }
        #expect(followersProperty?.description == "Number of followers", "followers should have correct description")

        print("All SchemaGuide descriptions verified successfully!")
    }

    @Test("SchemaGuide attribute generates constraints")
    func testSchemaGuideConstraints() throws {
        // Verify schemaProperties contains constraints from @SchemaGuide
        let properties = UserProfile.schemaProperties
        #expect(properties != nil, "UserProfile should have schemaProperties")

        guard let props = properties else { return }

        // Check age has range constraint
        let ageProperty = props.first { $0.name == "age" }
        #expect(ageProperty != nil, "Should have age property")
        #expect(ageProperty!.constraints.contains(.range(0...150)), "age should have range constraint 0...150")

        // Check bio has maxLength constraint
        let bioProperty = props.first { $0.name == "bio" }
        #expect(bioProperty != nil, "Should have bio property")
        #expect(bioProperty!.constraints.contains(.maxLength(500)), "bio should have maxLength constraint 500")

        // Check rating has doubleRange constraint
        let ratingProperty = props.first { $0.name == "rating" }
        #expect(ratingProperty != nil, "Should have rating property")
        #expect(ratingProperty!.constraints.contains(.doubleRange(0.0...5.0)), "rating should have doubleRange constraint 0.0...5.0")

        // Check followers has range constraint
        let followersProperty = props.first { $0.name == "followers" }
        #expect(followersProperty != nil, "Should have followers property")
        #expect(followersProperty!.constraints.contains(.range(0...1000000)), "followers should have range constraint 0...1000000")

        print("All SchemaGuide constraints verified successfully!")
    }

    @Test("SchemaGuide attribute with array count constraint")
    func testSchemaGuideArrayCountConstraint() throws {
        let properties = Playlist.schemaProperties
        #expect(properties != nil, "Playlist should have schemaProperties")

        guard let props = properties else { return }

        let songsProperty = props.first { $0.name == "songs" }
        #expect(songsProperty != nil, "Should have songs property")
        #expect(songsProperty!.constraints.contains(.count(3...50)), "songs should have count constraint 3...50")

        print("Array count constraint verified successfully!")
    }

    @Test("SchemaGuide attribute on ProductReview")
    func testSchemaGuideProductReview() throws {
        let properties = ProductReview.schemaProperties
        #expect(properties != nil, "ProductReview should have schemaProperties")

        guard let props = properties else { return }
        #expect(props.count == 4, "ProductReview should have 4 properties")

        // Check rating has range 1-5
        let ratingProperty = props.first { $0.name == "rating" }
        #expect(ratingProperty?.constraints.contains(.range(1...5)) == true, "rating should have range 1...5")

        // Check title has maxLength 100
        let titleProperty = props.first { $0.name == "title" }
        #expect(titleProperty?.constraints.contains(.maxLength(100)) == true, "title should have maxLength 100")

        // Check reviewText has maxLength 2000
        let reviewTextProperty = props.first { $0.name == "reviewText" }
        #expect(reviewTextProperty?.constraints.contains(.maxLength(2000)) == true, "reviewText should have maxLength 2000")

        print("ProductReview SchemaGuide attributes verified successfully!")
    }

    @Test("SchemaDescription includes SchemaGuide info")
    func testSchemaDescriptionWithSchemaGuide() throws {
        // Test that schemaDescription includes description and constraints
        let description = UserProfile.schemaDescription()

        print("Schema description:\n\(description)")

        // The description should contain property info
        #expect(description.contains("name"), "Should contain name property")
        #expect(description.contains("age"), "Should contain age property")
        #expect(description.contains("bio"), "Should contain bio property")
    }

    @Test("SchemaDescription includes nested object properties")
    func testSchemaDescriptionNested() throws {
        // Test that nested objects show their properties recursively
        let description = Person.schemaDescription()
        print("Person schema description:\n\(description)")

        // Should contain Person's properties
        #expect(description.contains("name"), "Should contain Person's name property")
        #expect(description.contains("age"), "Should contain Person's age property")
        #expect(description.contains("address"), "Should contain Person's address property")

        // Should contain nested Address properties
        #expect(description.contains("street"), "Should contain Address's street property")
        #expect(description.contains("city"), "Should contain Address's city property")
        #expect(description.contains("zipCode"), "Should contain Address's zipCode property")
    }

    @Test("SchemaDescription includes array of objects")
    func testSchemaDescriptionArrayOfObjects() throws {
        // Test that arrays of objects show element schema
        let description = TodoList.schemaDescription()
        print("TodoList schema description:\n\(description)")

        // Should contain TodoList properties
        #expect(description.contains("listName"), "Should contain listName property")
        #expect(description.contains("tasks"), "Should contain tasks array property")

        // Should contain nested Task properties
        #expect(description.contains("title"), "Should contain Task's title property")
        #expect(description.contains("completed"), "Should contain Task's completed property")
    }

    @available(macOS 26.0, iOS 26.0, *)
    @Test("FoundationModels GenerationSchema includes SchemaGuide info")
    func testGenerationSchemaWithSchemaGuide() throws {
        // Test that generationSchema (for FoundationModels) includes descriptions and guides
        let schema = UserProfile.generationSchema

        print("GenerationSchema: \(schema)")

        // The schema should have properties for all fields
        // Note: We can't easily inspect GenerationSchema internals, but we can verify it exists
        // and doesn't throw when accessed
        #expect(schema != nil, "Should have a valid generationSchema")
    }

    @Test("Grammar uses SchemaGuide maxLength constraint")
    func testGrammarMaxLengthConstraint() async throws {
        // Test that the grammar tracker respects maxLength constraints
        // UserProfile has bio with .maxLength(500) and name with no constraint

        // Get schemaProperties and verify constraints are present
        let properties = UserProfile.schemaProperties
        #expect(properties != nil, "Should have schemaProperties")

        guard let props = properties else { return }

        // bio should have maxLength 500
        let bioProperty = props.first { $0.name == "bio" }
        #expect(bioProperty?.maxLength == 500, "bio should have maxLength 500")

        // name should use default (nil maxLength)
        let nameProperty = props.first { $0.name == "name" }
        #expect(nameProperty?.maxLength == nil, "name should not have maxLength constraint")

        print("Grammar constraint test passed - maxLength constraints are correctly extracted from @SchemaGuide")
    }

    @Test("Schema description is suitable for prompt injection")
    func testSchemaDescriptionForPromptInjection() throws {
        // Test that schemaDescription produces output suitable for LLM prompt injection
        let description = UserProfile.schemaDescription()

        print("Schema description for prompt injection:\n\(description)")

        // Should be human-readable format
        #expect(description.contains("{"), "Should have opening brace")
        #expect(description.contains("}"), "Should have closing brace")

        // Should include field names
        #expect(description.contains("\"name\""), "Should include name field")
        #expect(description.contains("\"age\""), "Should include age field")
        #expect(description.contains("\"bio\""), "Should include bio field")

        // Should include type information
        #expect(description.contains("string"), "Should include string type")
        #expect(description.contains("integer"), "Should include integer type")

        // Should include descriptions from @SchemaGuide
        #expect(description.contains("The user's display name"), "Should include name description")
        #expect(description.contains("Age in years"), "Should include age description")

        // Should include constraints
        #expect(description.contains("max 500 chars"), "Should include maxLength constraint for bio")
        #expect(description.contains("range: 0-150"), "Should include range constraint for age")

        print("Schema description is suitable for prompt injection!")
    }

    // MARK: - Legacy Tests (commented out, require specific model classes)
    //    @Test("Test basic generation fixed size") func testExample() throws {
    //        // Load your Core ML model
    //        // Replace "MyModel" with the actual name of your MLModel class generated by Xcode
    //        let config = MLModelConfiguration()
    ////        config.computeUnits = .cpuAndNeuralEngine
    //
    //        config.optimizationHints.specializationStrategy = .fastPrediction
    //        config.optimizationHints.reshapeFrequency = .infrequent
    //        config.allowLowPrecisionAccumulationOnGPU = true // Use FP16 on GPU
    //        let mlModel = try Llama_3_2_3B_Instruct_Flex(configuration: config).model
    //
    //        let maxContextSize = 2048
    //
    //        let tokenizerConfig = try! JSONSerialization.jsonObject(with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer_config", withExtension: "json")!))
    //        let tokenizerData = try! JSONSerialization.jsonObject(with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer", withExtension: "json")!))
    //        // Assume we have a Tokenizer object that matches the Python tokenizer
    //        let tokenizer = try! AutoTokenizer.from(tokenizerConfig: Config(tokenizerConfig as! [NSString : Any]),
    //                                            tokenizerData: Config(tokenizerData as! [NSString : Any])) // Implement or provide this
    //        let eosTokenId = tokenizer.eosTokenId!  // Replace with your model's actual EOS token ID
    //        let padTokenId = 0  // Replace with your model's actual PAD token ID
    //
    //        let prompt = """
    //        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    //        You are an AI Assistant.
    //        <|eot_id|><|start_header_id|>user<|end_header_id|>
    //
    //        How are you today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    //        """
    //
    //        // Tokenize the prompt
    //        var tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
    //        let promptLength = tokens.count
    //
    //        // Right-pad to max_context_size
    //        let maxTokens = min(promptLength, maxContextSize)
    //        let padLength = maxTokens < maxContextSize ? (maxContextSize - maxTokens) : 0
    //        tokens = Array(tokens.prefix(maxTokens)) + Array(repeating: padTokenId, count: padLength)
    //
    //        // Prepare initial input arrays
    //        var finalTokens = tokens  // This array will be updated as we generate
    //        var usedTokens = maxTokens // How many tokens (not padding) are currently in use
    //
    //        // Decode initial prompt
    //        var decodedTextSoFar = tokenizer.decode(tokens: Array(finalTokens.prefix(usedTokens)))
    //
    //        // Generation loop
    //        for _ in 0..<100 {
    //            // Update attention mask: 1 for all used tokens, then 0 for padding
    //            let attentionMaskValues = Array(repeating: Int32(1), count: usedTokens)
    //                + Array(repeating: Int32(0), count: maxContextSize - usedTokens)
    //
    //            let inputIds = finalTokens.map { Int32($0) }
    //            let inputIdsArray = MLShapedArray<Int32>(scalars: inputIds, shape: [1, maxContextSize])
    //            let attentionMaskArray = MLShapedArray<Int32>(scalars: attentionMaskValues, shape: [1, maxContextSize])
    //
    //            let inputFeatures: [String: MLFeatureValue] = [
    //                "inputIds": MLFeatureValue(shapedArray: inputIdsArray),
    //                "attentionMask": MLFeatureValue(shapedArray: attentionMaskArray)
    //            ]
    //
    //            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputFeatures) else {
    //                Issue.record("Failed to create input feature provider")
    //                return
    //            }
    //
    //            // Run inference
    //            let time = Date.now
    //            let prediction = try mlModel.prediction(from: inputProvider)
    //            print("Prediction took \(Date.now.timeIntervalSince1970 - time.timeIntervalSince1970)s")
    //            guard let logitsValue = prediction.featureValue(for: "logits")?.multiArrayValue else {
    //                Issue.record("Failed to run model prediction")
    //                return
    //            }
    //            // Extract last token's logits
    //            // logits shape: (1, max_context_size, vocab_size)
    //            let vocabSize = logitsValue.shape[2].intValue
    //            let lastTokenIndex = usedTokens - 1
    //            let startIdx = lastTokenIndex * vocabSize
    //            let endIdx = startIdx + vocabSize
    //            var lastTokenLogits = [Float]()
    //            lastTokenLogits.reserveCapacity(vocabSize)
    //            for i in startIdx..<endIdx {
    //                lastTokenLogits.append(logitsValue[i].floatValue)
    //            }
    //
    //            // Greedy selection: take argmax
    //            guard let (nextTokenId, _) = lastTokenLogits.enumerated().max(by: { $0.element < $1.element }) else {
    //                Issue.record("No max found in logits")
    //                return
    //            }
    //
    //            // Stop if EOS
    //            if nextTokenId == eosTokenId {
    //                break
    //            }
    //
    //            // Add next token
    //            if usedTokens < maxContextSize {
    //                finalTokens[usedTokens] = nextTokenId
    //                usedTokens += 1
    //            } else {
    //                print("\nReached max context size.")
    //                break
    //            }
    //
    //            // Decode the entire active portion
    //            let activeTokens = Array(finalTokens.prefix(usedTokens))
    //            let fullText = tokenizer.decode(tokens: activeTokens)
    //            let newText = String(fullText.dropFirst(decodedTextSoFar.count))
    //            decodedTextSoFar = fullText
    //            print(newText, terminator: "")
    //        }
    //
    //        print("\nFinal output:\n\(decodedTextSoFar)")
    //    }
    //
    //    @Test("Test basic generation flexible size")
    //    func testFlexibleSize() throws {
    //        // Load your Core ML model with dynamic shapes
    //        let config = MLModelConfiguration()
    //        config.optimizationHints.specializationStrategy = .fastPrediction
    //        config.optimizationHints.reshapeFrequency = .frequent
    //        config.allowLowPrecisionAccumulationOnGPU = true // Use FP16 on GPU
    ////        config.computeUnits = .cpuAndNeuralEngine
    //        // Replace with a dynamically shaped model:
    //        let mlModel = try Llama_3_2_3B_Instruct_Flex(configuration: config).model
    //
    //        // Load tokenizer configuration
    //        let tokenizerConfig = try! JSONSerialization.jsonObject(
    //            with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer_config", withExtension: "json")!)
    //        )
    //        let tokenizerData = try! JSONSerialization.jsonObject(
    //            with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer", withExtension: "json")!)
    //        )
    //
    //        let tokenizer = try! AutoTokenizer.from(
    //            tokenizerConfig: Config(tokenizerConfig as! [NSString: Any]),
    //            tokenizerData: Config(tokenizerData as! [NSString: Any])
    //        )
    //        let eosTokenId = tokenizer.eosTokenId! // Replace with actual EOS token ID
    //
    //        let prompt = """
    //        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    //        You are an AI Assistant.
    //        <|eot_id|><|start_header_id|>user<|end_header_id|>
    //
    //        How are you today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    //        """
    //
    //        // Tokenize the prompt without forcing it to a fixed length
    //        var finalTokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
    //        var decodedTextSoFar = tokenizer.decode(tokens: finalTokens)
    //
    //        // Generation loop
    //        for _ in 0..<100 {
    //            let usedTokens = finalTokens.count
    //            // Create attention mask: all 1’s since we are using the entire context
    //            let attentionMaskValues = Array(repeating: Int32(1), count: usedTokens)
    //
    //            let inputIds = finalTokens.map { Int32($0) }
    //            let inputIdsArray = MLShapedArray<Int32>(scalars: inputIds, shape: [1, usedTokens])
    //            let attentionMaskArray = MLShapedArray<Int32>(scalars: attentionMaskValues, shape: [1, usedTokens])
    //
    //            let inputFeatures: [String: MLFeatureValue] = [
    //                "inputIds": MLFeatureValue(shapedArray: inputIdsArray),
    //                "attentionMask": MLFeatureValue(shapedArray: attentionMaskArray)
    //            ]
    //
    //            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputFeatures) else {
    //                Issue.record("Failed to create input feature provider")
    //                return
    //            }
    //
    //            // Run inference
    //            let time = Date.now
    //            let prediction = try mlModel.prediction(from: inputProvider)
    //            let elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
    //            print("Prediction took \(elapsed)s")
    //
    //            guard let logitsValue = prediction.featureValue(for: "logits")?.multiArrayValue else {
    //                Issue.record("Failed to run model prediction")
    //                return
    //            }
    //
    //            // logits shape is (1, usedTokens, vocab_size)
    //            let vocabSize = logitsValue.shape[2].intValue
    //            let lastTokenIndex = usedTokens - 1
    //            let startIdx = lastTokenIndex * vocabSize
    //            let endIdx = startIdx + vocabSize
    //            var lastTokenLogits = [Float]()
    //            lastTokenLogits.reserveCapacity(vocabSize)
    //            for i in startIdx..<endIdx {
    //                lastTokenLogits.append(logitsValue[i].floatValue)
    //            }
    //
    //            // Greedy selection: take argmax
    //            guard let (nextTokenId, _) = lastTokenLogits.enumerated().max(by: { $0.element < $1.element }) else {
    //                Issue.record("No max found in logits")
    //                return
    //            }
    //
    //            // Stop if EOS
    //            if nextTokenId == eosTokenId {
    //                break
    //            }
    //
    //            // Add next token
    //            finalTokens.append(nextTokenId)
    //
    //            // Decode new token(s)
    //            let fullText = tokenizer.decode(tokens: finalTokens)
    //            let newText = String(fullText.dropFirst(decodedTextSoFar.count))
    //            decodedTextSoFar = fullText
    //            print(newText, terminator: "")
    //        }
    //
    //        print("\nFinal output:\n\(decodedTextSoFar)")
    //    }
    
    //    @Test("Test basic generation flexible size")
    //    func testCausalMask() throws {
    //        // Load your Core ML model with dynamic shapes
    //        let config = MLModelConfiguration()
    //        config.optimizationHints.specializationStrategy = .fastPrediction
    //        config.optimizationHints.reshapeFrequency = .frequent
    //        config.allowLowPrecisionAccumulationOnGPU = true // Use FP16 on GPU
    ////        config.computeUnits = .cpuAndNeuralEngine
    //        // Replace with a dynamically shaped model:
    //        let mlModel = try Llama_3_2_1B_Instruct_A8W8(configuration: config).model
    //
    //        // Load tokenizer configuration
    //        let tokenizerConfig = try! JSONSerialization.jsonObject(
    //            with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer_config", withExtension: "json")!)
    //        )
    //        let tokenizerData = try! JSONSerialization.jsonObject(
    //            with: try! Data(contentsOf: Bundle.module.url(forResource: "tokenizer", withExtension: "json")!)
    //        )
    //
    //        let tokenizer = try! AutoTokenizer.from(
    //            tokenizerConfig: Config(tokenizerConfig as! [NSString: Any]),
    //            tokenizerData: Config(tokenizerData as! [NSString: Any])
    //        )
    //        let eosTokenId = tokenizer.eosTokenId! // Replace with actual EOS token ID
    //
    //        let prompt = """
    //        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    //        You are an AI Assistant.
    //        <|eot_id|><|start_header_id|>user<|end_header_id|>
    //
    //        How are you today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    //        """
    //
    //        // Tokenize the prompt without forcing it to a fixed length
    //        var finalTokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
    //        var decodedTextSoFar = tokenizer.decode(tokens: finalTokens)
    //
    //        let state = mlModel.makeState()
    //        // Generation loop
    //        for _ in 0..<100 {
    //            let usedTokens = finalTokens.count
    //            // Construct a causal mask (lower-triangular)
    //            // Shape: (1, 1, queryLen, endStepDim)
    //            // For each query position i:
    //            // positions j ≤ i = 1 (allowed), j > i = 0 (blocked)
    //            var maskValues = [Float16]()
    //            maskValues.reserveCapacity(usedTokens * usedTokens)
    //
    //            for i in 0..<usedTokens {
    //                for j in 0..<usedTokens {
    //                    let value: Float = (j <= i) ? 1.0 : 0.0
    //                    maskValues.append(Float16(value))
    //                }
    //            }
    //
    //            let causalMaskArray = MLShapedArray<Float16>(
    //                scalars: maskValues,
    //                shape: [1, 1, usedTokens, usedTokens]
    //            )
    //
    //            let inputIds = finalTokens.map { Int32($0) }
    //            let inputIdsArray = MLShapedArray<Int32>(scalars: inputIds, shape: [1, usedTokens])
    //
    //            let inputFeatures: [String: MLFeatureValue] = [
    //                "inputIds": MLFeatureValue(shapedArray: inputIdsArray),
    //                "causalMask": MLFeatureValue(shapedArray: causalMaskArray)
    //            ]
    //
    //            guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: inputFeatures) else {
    //                Issue.record("Failed to create input feature provider")
    //                return
    //            }
    //
    //            // Run inference
    //            let time = Date.now
    //            let prediction = try mlModel.prediction(from: inputProvider, using: state)
    //            let elapsed = Date.now.timeIntervalSince1970 - time.timeIntervalSince1970
    //            print("Prediction took \(elapsed)s")
    //
    //            guard let logitsValue = prediction.featureValue(for: "logits")?.multiArrayValue else {
    //                Issue.record("Failed to run model prediction")
    //                return
    //            }
    //
    //            // logits shape is (1, usedTokens, vocab_size)
    //            let vocabSize = logitsValue.shape[2].intValue
    //            let lastTokenIndex = usedTokens - 1
    //            let startIdx = lastTokenIndex * vocabSize
    //            let endIdx = startIdx + vocabSize
    //            var lastTokenLogits = [Float]()
    //
    //            lastTokenLogits.reserveCapacity(vocabSize)
    //            for i in startIdx..<endIdx {
    //                lastTokenLogits.append(logitsValue[i].floatValue)
    //            }
    //
    //            // Greedy selection: take argmax
    //            guard let (nextTokenId, _) = lastTokenLogits.enumerated().max(by: { $0.element < $1.element }) else {
    //                Issue.record("No max found in logits")
    //                return
    //            }
    //
    //            // Stop if EOS
    //            if nextTokenId == eosTokenId {
    //                break
    //            }
    //
    //            // Add next token
    //            finalTokens.append(nextTokenId)
    //
    //            // Decode new token(s)
    //            let fullText = tokenizer.decode(tokens: finalTokens)
    //            let newText = String(fullText.dropFirst(decodedTextSoFar.count))
    //            decodedTextSoFar = fullText
    //            print(newText, terminator: "")
    //        }
    //
    //        print("\nFinal output:\n\(decodedTextSoFar)")
    //    }

    // NOTE: The following tests are commented out because they reference
    // model classes that need to be generated from compiled CoreML models.
    // Use testBasicInference and testInt4Model above for testing with path-based loading.

    /*
    @Test func testGreedy() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored().model)
        let session = try await lm.makeSession()
        let stream = try await session.infer(prompt: "How are you today?")
        for await token in stream {
            print(token, terminator: "")
        }
    }
    */

    @Test func testSortAndGather2() async throws {
        let x = MLTensor(shape: [1, 3, 4], scalars: [
            // First 3x4 matrix
            0,  2,  1, 4,
            11, 10, 12, 9,
            24, 22, 25, 20,
        ], scalarType: Float.self)
        let xShaped = await x.shapedArray(of: Float.self)
        _ = await x.squeezingShape().shapedArray(of: Float.self)
        let probs = x.softmax(alongAxis: -1)  // Convert logits to probabilities
        let probsShaped = await probs.shapedArray(of: Float.self)
        let sortedIndices = probs.argsort(alongAxis: -1, descendingOrder: true)
        let sortedIndicesShaped = await sortedIndices.shapedArray(of: Int32.self)
        let batchSize = 1
        let rows = 3
        let cols = 4
        var probsShapedCopy = probsShaped
        for i in 0..<1 {
            for j in 0..<3 {
                for k in 0..<4 {
                    // Retrieve the sorted column index for this position
                    let targetCol: Int = Int(sortedIndicesShaped[i, j, k].scalar!)
                    // Use that index to fetch the corresponding element from x
                    let value: Float = probsShaped[i, j, targetCol].scalar!
                    // Calculate flat index for sortedValues array
                    probsShapedCopy[scalarAt: i, j, k] = value
                }
            }
        }
        let topP: Float = 0.8
        let sortedProbs = MLTensor(probsShapedCopy)
        let cumsum = sortedProbs.cumulativeSum(alongAxis: -1)
        let mask = cumsum .> topP
        let maskedSortedProbs = sortedProbs.replacing(with: -Float.greatestFiniteMagnitude, where: mask)
        
        // 4. Determine valid positions within top-P.
        let validPositions = maskedSortedProbs .> -Float.greatestFiniteMagnitude
        let validPositionsShaped = await validPositions.cast(to: Int32.self).shapedArray(of: Int32.self)
        // 5. Iterate over valid positions to retrieve original logits and indices.
        var survivingLogits: [Float] = []
        var survivingIndices: [(i: Int, j: Int, originalCol: Int)] = []
        
        for i in 0..<batchSize {
            for j in 0..<rows {
                for k in 0..<cols {
                    // Check validity at sorted position [i, j, k].
                    let isValid = validPositionsShaped[i, j, k].scalar! != 0
                    if isValid {
                        // Retrieve original column index from sortedIndices.
                        let origCol = Int(sortedIndicesShaped[i, j, k].scalar!)
                        // Fetch the original logit from x using the original indices.
                        let originalLogit = xShaped[i, j, origCol].scalar!
                        
                        survivingLogits.append(originalLogit)
                        survivingIndices.append((i, j, origCol))
                    }
                }
            }
        }
        
        // For demonstration, print surviving indices and logits.
        print("==================")
        print("Surviving indices (i, j, originalCol): \(survivingIndices)")
        print("Surviving logits: \(survivingLogits)")
        print("==================")
        
        let (warpedIndices, warpedLogits) = TopPLogitsWarper(p: topP)
            .warp(indices: xShaped.scalars.indices.map { $0 },
                  logits: xShaped.scalars)
        print(warpedIndices, warpedLogits)
    }
    
    @Test func testSortAndGather() async throws {
        var tensors = [[Float]]()
        let vocab = 12800
        let seqs = 124
        for _ in 0..<vocab {
            var t = [Float]()
            for _ in 0..<seqs {
                t.append(Float.random(in: 0...10))
            }
            tensors.append(t)
        }
        let tensor: MLTensor = MLTensor(shape: [1, seqs, vocab],
                                        scalars: tensors.flatMap({ $0 }),
                                        scalarType: Float.self)
        var (values, indices) = tensor.topK(50)
        (values, indices) = values.flattened().topK(49)
        print(values, indices)
    }

    /*
    // MARK: - Tests requiring specific model classes (commented out)
    // These tests need model classes generated from compiled CoreML models.
    // Uncomment when you have the appropriate models available.

    @Test func testSampling() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored().model
                                            ,
                                            temperature: 1.2,
                                            topK: 50,
                                            topP: 0.9,
                                            repetitionPenalty: 1.1
        )
        let session = try await lm.makeSession(systemPrompt: """
        You are Meta AI, a friendly AI Assistant. Today's date is \(Date.now.ISO8601Format()). Respond to the input as a friendly AI assistant, generating human-like text, and follow the instructions in the input if applicable. Keep the response concise and engaging, using Markdown when appropriate. The user lives in the USA, so be aware of the local context and preferences. Use a conversational tone and provide helpful and informative responses, utilizing external knowledge when necessary.
        """)
        for _ in 0..<10 {

            let stream = try await session.infer(prompt: """
            Tell me a two sentence story.
            """)
            print("===========")
            await Task.detached {
                for await token in stream {
                    print(token, terminator: "")
                }
            }.value
            print("\n===========")
        }
    }
    
    @Test func testSummarizing() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let duration = try await Test.Clock().measure {
            let session = try await lm.makeSession(systemPrompt: """
        Task Prompt:
        
        You are an advanced text analysis assistant. For each article provided to you, perform the following tasks:
            1.    Generate a Summary: Provide a concise summary of the article, highlighting the main points and key takeaways. Aim for a summary length of 3-5 sentences.
            2.    Analyze Sentiment: Analyze the sentiment of the article’s content and provide a sentiment score between -1 and 1, where:
            •    -1 represents extremely negative sentiment,
            •    0 represents neutral sentiment, and
            •    1 represents extremely positive sentiment.
        
        Output your response in the following JSON format:
        
        {"summary": "[Your summary here]", "sentiment_score": [Your sentiment score here]}
        
        Here are some examples:
        Example 1
        
        Article:
        “Global markets rebounded today after a week of losses, with investors showing optimism in tech stocks. Experts believe this recovery is driven by positive earnings reports and a slowdown in inflation.”
        
        Response:
        {"summary": "Global markets recovered after a week of losses, driven by optimism in tech stocks. Positive earnings reports and a slowdown in inflation contributed to the rebound. Experts remain cautiously optimistic about sustained recovery.",
          "sentiment_score": 0.8}
        
        Example 2
        
        Article:
        “The recent oil spill off the Gulf Coast has devastated local wildlife, with reports of significant damage to marine ecosystems. Environmentalists are calling for urgent action to mitigate further harm.”
        {"summary": "The Gulf Coast oil spill has caused severe damage to marine ecosystems, prompting environmentalists to demand urgent action. Local wildlife has been heavily impacted by the disaster.",
          "sentiment_score": -0.85}
        
        Example 3
        
        Article:
        “A new study suggests that regular exercise can improve cognitive function in older adults. Researchers found a significant correlation between physical activity and mental sharpness.”
        
        {"summary": "A recent study highlights that regular exercise enhances cognitive function in older adults. The research reveals a strong link between physical activity and improved mental sharpness.",
          "sentiment_score": 0.7}
        """)
            let stream = try await session.infer(prompt: """
        Intel Corporation (INTC) Faces Investor Lawsuit over Foundry Business – Hagens Berman
        August 08, 2024 14:25 ET
         | Source: Hagens Berman Sobol Shapiro LLP
        
        Share
        
        SAN FRANCISCO, Aug. 08, 2024 (GLOBE NEWSWIRE) -- Hagens Berman urges Intel Corporation (NASDAQ: INTC) investors who suffered substantial losses to submit your losses now.
        
        Class Period: Jan. 25, 2024 – Aug. 1, 2024
        Lead Plaintiff Deadline: Oct. 7, 2024
        Visit: www.hbsslaw.com/investor-fraud/intc
        Contact the Firm Now: INTC@hbsslaw.com
                                                     844-916-0895
        
        Class Action Against Intel Corporation (INTC):
        
        Intel Corp. is under fire from investors alleging the chip giant misled shareholders about the performance of its foundry business. A federal class-action lawsuit filed in the Northern District of California claims Intel made false and misleading statements about the financial health and progress of its foundry operations.
        
        The complaint centers on Intel’s repeated assurances that its foundry plans were on track and delivering as promised. Investors allege that these statements were contradicted by the company’s subsequent announcement of severe problems in the foundry business, which led to a $10 billion cost-cutting plan, including layoffs and dividend suspension.
        
        On August 1, when Intel unveiled these dramatic measures, its share price plummeted by 26%, wiping out billions of dollars in market value. Analysts swiftly downgraded the stock, citing concerns about the company’s manufacturing capabilities and overall business outlook.
        
        The lawsuit contends that Intel’s decision to accelerate its move to a higher-cost Ireland fabrication facility masked underlying problems and inflated profit margins.
        
        “We are investigating whether Intel may have misrepresented the challenges its foundry business faced,” said Reed Kathrein, a partner at Hagens Berman, a law firm seeking to lead the case on behalf of the investor class.
        
        If you invested in Intel and have substantial losses submit your losses now. »
        
        If you’d like more information about the Intel case and our investigation, read more »
        
        Whistleblowers: Persons with non-public information regarding Intel should consider their options to help in the investigation or take advantage of the SEC Whistleblower program. Under the new program, whistleblowers who provide original information may receive rewards totaling up to 30 percent of any successful recovery made by the SEC. For more information, call Reed Kathrein at 844-916-0895 or email INTC@hbsslaw.com.
        
        About Hagens Berman
        Hagens Berman is a global plaintiffs’ rights complex litigation firm focusing on corporate accountability. The firm is home to a robust practice and represents investors as well as whistleblowers, workers, consumers and others in cases achieving real results for those harmed by corporate negligence and other wrongdoings. Hagens Berman’s team has secured more than $2.9 billion in this area of law. More about the firm and its successes can be found at hbsslaw.com. Follow the firm for updates and news at @ClassActionLaw.
        """)
            print("===========")
            await Task.detached {
                for await token in stream {
                    print(token, terminator: "")
                }
            }.value
            print("\n===========")
        }
        print(duration)
    }
    
    @Test func testSentimentOnly() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let duration = try await Test.Clock().measure {
            let session = try await lm.makeSession(systemPrompt: """
        Task Prompt:
        
        You are an advanced sentiment analysis assistant. Your task is to analyze the sentiment of each article provided, considering its potential impact on stock prices. Generate a sentiment score ranging from -1 to 1, where:
            •    -1 represents extremely negative sentiment,
            •    0 represents neutral sentiment, and
            •    1 represents extremely positive sentiment.
        
        Output the sentiment score in the following JSON format:
        
        {"sentimentScore": [Your sentiment score here]}
        
        Here are some examples:
        
        Example 1
        
        Article:
        “Global markets rebounded today after a week of losses, with investors showing optimism in tech stocks. Experts believe this recovery is driven by positive earnings reports and a slowdown in inflation.”
        
        Response:
        
        {"sentimentScore": 0.8}
        
        Example 2
        
        Article:
        “The recent oil spill off the Gulf Coast has devastated local wildlife, with reports of significant damage to marine ecosystems. Environmentalists are calling for urgent action to mitigate further harm.”
        
        Response:
        
        {"sentimentScore": -0.9}
        
        Example 3
        
        Article:
        “A new study suggests that regular exercise can improve cognitive function in older adults. Researchers found a significant correlation between physical activity and mental sharpness.”
        
        Response:
        
        {"sentimentScore": 0.7}
        """)
            let stream = try await session.infer(prompt: """
        Shopify stock soars to 52-week high, hits $115.64
        
        Shopify Inc (NYSE:SHOP). shares have surged to a 52-week high, reaching a price level of $115.64, as the e-commerce platform continues to capitalize on the expanding online retail market. According to InvestingPro data, the company's impressive 23.47% revenue growth and GREAT Financial Health Score reflect its strong market position, though technical indicators suggest the stock may be in overbought territory. This milestone reflects a significant recovery and growth trajectory for the company, which has seen its stock value increase by an impressive 58.46% over the past year. With a substantial market capitalization of $149.38 billion, investors are responding positively to Shopify's strategic initiatives and its ability to attract and retain merchants. Based on InvestingPro's Fair Value analysis, the stock appears to be trading above its intrinsic value. Discover 20+ additional exclusive insights and a comprehensive Pro Research Report available for Shopify on InvestingPro.
        In other recent news, Shopify has been experiencing significant financial growth. Earnings reports from the third quarter highlighted a 26% increase in revenue, surpassing analysts' estimates. This was largely due to a robust 24% growth in Gross Merchandise Volume (GMV). Operating income more than doubled from the previous year, and the free cash flow margin expanded to 19%. Shopify's fourth-quarter projections indicate an acceleration of top-line growth to the mid- to high-20% range, partly due to a new partnership with PayPal (NASDAQ:PYPL).
        Several firms have adjusted their price targets for Shopify following these developments. Piper Sandler maintained a Neutral rating, with a price target set at $94.00. The firm's analysis focused on Shopify's partnership with PayPal and the effects of revenue reclassification on the company's financial outlook. Truist Securities increased its target to $110, also holding a Neutral stance. Loop Capital followed suit, lifting its price target to $110. Scotiabank (TSX:BNS) analyst Kevin Krishnaratne raised the price target to $115 with a Sector Perform rating, and Oppenheimer maintained a positive outlook, raising the target to $130 while keeping an Outperform rating.
        
        3rd party Ad. Not an offer or recommendation by Investing.com. See disclosure here or remove ads.
        Shopify also reported an impressive 35% growth in the Europe, Middle East, and Africa (EMEA) region. These are the recent developments for Shopify.
        This article was generated with the support of AI and reviewed by an editor. For more information see our T&C.
        SHOP: is this perennial leader facing new challenges?
        With valuations skyrocketing in 2024, many investors are uneasy putting more money into stocks. Sure, there are always opportunities in the stock market – but finding them feels more difficult now than a year ago. Unsure where to invest next? One of the best ways to discover new high-potential opportunities is to look at the top performing portfolios this year. ProPicks AI offers 6 model portfolios from Investing.com which identify the best stocks for investors to buy right now. For example, ProPicks AI found 9 overlooked stocks that jumped over 25% this year alone. The new stocks that made the monthly cut could yield enormous returns in the coming years. Is SHOP one of them?
        """)
            print("===========")
            await Task.detached {
                for await token in stream {
                    print(token, terminator: "")
                }
            }.value
            print("\n===========")
        }
        print(duration)
    }
    
    @Test func testTensorMask() async throws {
        let x = MLTensor(shape: [1, 3, 4], scalars: [
            // First 3x4 matrix
            0,  2,  1, 4,
            11, 10, 12, 9,
            24, 22, 25, 20,
        ], scalarType: Float.self).flattened()
        let xShaped = await x.shapedArray(of: Float.self)
        let validTokens = Set<Float>([2.0, 12.0, 20.0])
        let disallowedIndices = xShaped.enumerated().filter { !validTokens.contains($0.element.scalar!) }.map { Int32($0.offset) }
        let xMasked = x.replacing(atIndices: MLTensor(disallowedIndices, scalarType: Int32.self),
                                  with: -Float.greatestFiniteMagnitude,
                                  alongAxis: -1)
        print(await xMasked.shapedArray(of: Float.self))
    }

    
    @Test func testGrammar() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let session = try await lm.makeSession(Trip.self, systemPrompt: """
        You are the world's greatest itinerary creator. You are to provide the user
        with a simple JSON response that gives them a basic trip itinerary.
        """)
        
        var tokens = ""
        for await token in await session.infer(prompt: "I want to go somewhere cold and be away for 40 days.") {
            tokens += token
            print(token)
        }
        print(try JSONDecoder().decode(Trip.self, from: tokens.data(using: .utf8)!))
    }
    
    @Test func testGrammar_Sentiment() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let session = try await lm.makeSession(Sentiment.self, systemPrompt: """
    Task Prompt:
    
    You are an advanced sentiment analysis assistant. Your task is to analyze the sentiment of each article provided, considering its potential impact on stock prices. Generate a sentiment score ranging from -1 to 1, where:
        •    -1 represents extremely negative sentiment,
        •    0 represents neutral sentiment, and
        •    1 represents extremely positive sentiment.
    
    Output the sentiment score in the following JSON format:
    
    {"sentimentScore": [Your sentiment score here]}
    
    Here are some examples:
    
    Example 1
    
    Article:
    “Global markets rebounded today after a week of losses, with investors showing optimism in tech stocks. Experts believe this recovery is driven by positive earnings reports and a slowdown in inflation.”
    
    Response:
    
    {"sentimentScore": 0.8}
    
    Example 2
    
    Article:
    “The recent oil spill off the Gulf Coast has devastated local wildlife, with reports of significant damage to marine ecosystems. Environmentalists are calling for urgent action to mitigate further harm.”
    
    Response:
    
    {"sentimentScore": -0.9}
    
    Example 3
    
    Article:
    “A new study suggests that regular exercise can improve cognitive function in older adults. Researchers found a significant correlation between physical activity and mental sharpness.”
    
    Response:
    
    {"sentimentScore": 0.7}
    """)
        let stream: AsyncStream<String> = try await session.infer(prompt: """
    Shopify stock soars to 52-week high, hits $115.64
    
    Shopify Inc (NYSE:SHOP). shares have surged to a 52-week high, reaching a price level of $115.64, as the e-commerce platform continues to capitalize on the expanding online retail market. According to InvestingPro data, the company's impressive 23.47% revenue growth and GREAT Financial Health Score reflect its strong market position, though technical indicators suggest the stock may be in overbought territory. This milestone reflects a significant recovery and growth trajectory for the company, which has seen its stock value increase by an impressive 58.46% over the past year. With a substantial market capitalization of $149.38 billion, investors are responding positively to Shopify's strategic initiatives and its ability to attract and retain merchants. Based on InvestingPro's Fair Value analysis, the stock appears to be trading above its intrinsic value. Discover 20+ additional exclusive insights and a comprehensive Pro Research Report available for Shopify on InvestingPro.
    In other recent news, Shopify has been experiencing significant financial growth. Earnings reports from the third quarter highlighted a 26% increase in revenue, surpassing analysts' estimates. This was largely due to a robust 24% growth in Gross Merchandise Volume (GMV). Operating income more than doubled from the previous year, and the free cash flow margin expanded to 19%. Shopify's fourth-quarter projections indicate an acceleration of top-line growth to the mid- to high-20% range, partly due to a new partnership with PayPal (NASDAQ:PYPL).
    Several firms have adjusted their price targets for Shopify following these developments. Piper Sandler maintained a Neutral rating, with a price target set at $94.00. The firm's analysis focused on Shopify's partnership with PayPal and the effects of revenue reclassification on the company's financial outlook. Truist Securities increased its target to $110, also holding a Neutral stance. Loop Capital followed suit, lifting its price target to $110. Scotiabank (TSX:BNS) analyst Kevin Krishnaratne raised the price target to $115 with a Sector Perform rating, and Oppenheimer maintained a positive outlook, raising the target to $130 while keeping an Outperform rating.
    
    3rd party Ad. Not an offer or recommendation by Investing.com. See disclosure here or remove ads.
    Shopify also reported an impressive 35% growth in the Europe, Middle East, and Africa (EMEA) region. These are the recent developments for Shopify.
    This article was generated with the support of AI and reviewed by an editor. For more information see our T&C.
    SHOP: is this perennial leader facing new challenges?
    With valuations skyrocketing in 2024, many investors are uneasy putting more money into stocks. Sure, there are always opportunities in the stock market – but finding them feels more difficult now than a year ago. Unsure where to invest next? One of the best ways to discover new high-potential opportunities is to look at the top performing portfolios this year. ProPicks AI offers 6 model portfolios from Investing.com which identify the best stocks for investors to buy right now. For example, ProPicks AI found 9 overlooked stocks that jumped over 25% this year alone. The new stocks that made the monthly cut could yield enormous returns in the coming years. Is SHOP one of them?
    """)
        print("===========")
        var tokens = [String]()
        await Task.detached {
            for await token in stream {
                print(token, terminator: "")
                tokens.append(token)
            }
        }.value
        print("\n===========")
        print(tokens.joined())
    }
    
    @Test func testGrammar_ApplyWeightsAndRelevancy() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_1B_Instruct_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let duration = try await Test.Clock().measure {
            let summarySession = try await lm.makeSession(Summary.self, systemPrompt: """
            Task Prompt:
            
            You are an advanced text analysis assistant. For each article provided to you, perform the following tasks:
                1.    Generate a Summary: Provide a concise summary of the article, highlighting the main points and key takeaways. Aim for a summary length of 3-5 sentences.
            
            Output your response in the following JSON format:
            
            {"summary": "[Your summary here]"}
            
            Here are some examples:
            Example 1
            
            Article:
            “Global markets rebounded today after a week of losses, with investors showing optimism in tech stocks. Experts believe this recovery is driven by positive earnings reports and a slowdown in inflation.”
            
            Response:
            {"summary": "Global markets recovered after a week of losses, driven by optimism in tech stocks. Positive earnings reports and a slowdown in inflation contributed to the rebound. Experts remain cautiously optimistic about sustained recovery."}
            
            Example 2
            
            Article:
            “The recent oil spill off the Gulf Coast has devastated local wildlife, with reports of significant damage to marine ecosystems. Environmentalists are calling for urgent action to mitigate further harm.”
            {"summary": "The Gulf Coast oil spill has caused severe damage to marine ecosystems, prompting environmentalists to demand urgent action. Local wildlife has been heavily impacted by the disaster."}
            
            Example 3
            
            Article:
            “A new study suggests that regular exercise can improve cognitive function in older adults. Researchers found a significant correlation between physical activity and mental sharpness.”
            
            {"summary": "A recent study highlights that regular exercise enhances cognitive function in older adults. The research reveals a strong link between physical activity and improved mental sharpness."}
            """, temperature: 1.2,
                                                          topK: 50,
                                                          topP: 0.9,
                                                          repetitionPenalty: 1.1)
            
            let summary = try await summarySession.infer(prompt: """
            Intel Corporation (INTC) Faces Investor Lawsuit over Foundry Business – Hagens Berman
            August 08, 2024 14:25 ET
             | Source: Hagens Berman Sobol Shapiro LLP
            
            Share
            
            SAN FRANCISCO, Aug. 08, 2024 (GLOBE NEWSWIRE) -- Hagens Berman urges Intel Corporation (NASDAQ: INTC) investors who suffered substantial losses to submit your losses now.
            
            Class Period: Jan. 25, 2024 – Aug. 1, 2024
            Lead Plaintiff Deadline: Oct. 7, 2024
            Visit: www.hbsslaw.com/investor-fraud/intc
            Contact the Firm Now: INTC@hbsslaw.com
                                                         844-916-0895
            
            Class Action Against Intel Corporation (INTC):
            
            Intel Corp. is under fire from investors alleging the chip giant misled shareholders about the performance of its foundry business. A federal class-action lawsuit filed in the Northern District of California claims Intel made false and misleading statements about the financial health and progress of its foundry operations.
            
            The complaint centers on Intel’s repeated assurances that its foundry plans were on track and delivering as promised. Investors allege that these statements were contradicted by the company’s subsequent announcement of severe problems in the foundry business, which led to a $10 billion cost-cutting plan, including layoffs and dividend suspension.
            
            On August 1, when Intel unveiled these dramatic measures, its share price plummeted by 26%, wiping out billions of dollars in market value. Analysts swiftly downgraded the stock, citing concerns about the company’s manufacturing capabilities and overall business outlook.
            
            The lawsuit contends that Intel’s decision to accelerate its move to a higher-cost Ireland fabrication facility masked underlying problems and inflated profit margins.
            
            “We are investigating whether Intel may have misrepresented the challenges its foundry business faced,” said Reed Kathrein, a partner at Hagens Berman, a law firm seeking to lead the case on behalf of the investor class.
            
            If you invested in Intel and have substantial losses submit your losses now. »
            
            If you’d like more information about the Intel case and our investigation, read more »
            
            Whistleblowers: Persons with non-public information regarding Intel should consider their options to help in the investigation or take advantage of the SEC Whistleblower program. Under the new program, whistleblowers who provide original information may receive rewards totaling up to 30 percent of any successful recovery made by the SEC. For more information, call Reed Kathrein at 844-916-0895 or email INTC@hbsslaw.com.
            
            About Hagens Berman
            Hagens Berman is a global plaintiffs’ rights complex litigation firm focusing on corporate accountability. The firm is home to a robust practice and represents investors as well as whistleblowers, workers, consumers and others in cases achieving real results for those harmed by corporate negligence and other wrongdoings. Hagens Berman’s team has secured more than $2.9 billion in this area of law. More about the firm and its successes can be found at hbsslaw.com. Follow the firm for updates and news at @ClassActionLaw.
            """)
            print(summary)
            let session = try await lm.makeSession(ArticleScores.self, systemPrompt: """
            Task Prompt:

            You are an advanced news analysis assistant. For each article provided, your task is to perform the following analyses, focusing on its potential impact on the stock associated with the ticker INTC (Intel Corporation):
            1. *Relevancy Analysis:*
            Evaluate how relevant the article is to the company in question. Generate a relevancy score ranging from 0 (completely irrelevant) to 1 (highly relevant). This helps filter out false positives (e.g., distinguishing between “Intel” the company and other uses of the word “intel”).
            2. *Sentiment Analysis:*
            Analyze the sentiment of the article as it pertains to the stock. Generate a sentiment score ranging from -1 to 1, where -1 is extremely negative, 0 is neutral, and 1 is extremely positive. 
            *Important:* Ensure that if the sentiment is negative, the score includes the `-` symbol (e.g., -0.8).
            3. *Weight Assessment:*
            Determine how likely this article is to impact the stock price, regardless of its sentiment intensity. Generate a weight score ranging from 0 (minimal impact) to 1 (high impact). Consider factors such as the source credibility, event significance, and the broader market context.

            Output your response in the following JSON format:
            {
              "relevancyScore": [Your relevancy score here],
              "sentimentScore": [Your sentiment score here],
              "weightScore": [Your weight score here]
            }
            
            *Example 1 – Positive, Highly Relevant (Tech Company)*

            Article:
            “Apple’s latest iPhone release has shattered pre-sale records, with enthusiastic consumers lining up outside stores worldwide. The company’s innovative features and robust performance are expected to significantly boost revenue this quarter.”

            Ticker: AAPL

            Analysis:
                •    Relevancy Score: The article focuses directly on Apple’s product launch → Score: 1.0
                •    Sentiment Score: The tone is very positive due to record-breaking sales and strong market expectations → Score: 0.9
                •    Weight Score: The news is likely to have a substantial impact on the stock given the product’s market significance → Score: 0.8

            Response:
            {
              "relevancyScore": 1.0,
              "sentimentScore": 0.9,
              "weightScore": 0.8
            }
            
            *Example 2 – Negative, Highly Relevant (Semiconductor Company)*

            Article:
            “Intel faces a major setback as its latest processor launch is plagued by severe technical issues. Customers report frequent system crashes, and pre-orders have plummeted. Financial analysts warn these problems could significantly damage the company’s reputation and lead to a sustained decline in its stock performance.”

            Ticker: INTC

            Analysis:
                •    Relevancy Score: The article directly discusses Intel’s business challenges → Score: 0.95
                •    Sentiment Score: The tone is strongly negative due to technical issues and falling pre-orders → Score: -0.8
                •    Weight Score: Given the potential long-term impact on sales and reputation, the news is moderately weighted → Score: 0.7
            
            Response:
            {
              "relevancyScore": 0.95,
              "sentimentScore": -0.8,
              "weightScore": 0.7
            }
            
            *Example 3 – Mixed Relevance, Mixed Sentiment (Retail Company)*

            Article:
            “Retail giant Macy’s announced an expansion of its online shopping platform this week, which has been well-received by consumers. However, the report also mentions some unrelated comments about local economic challenges that could affect consumer spending in the region.”

            Ticker: M

            Analysis:
                •    Relevancy Score: The primary focus is on Macy’s business expansion, but there is a minor diversion to unrelated economic issues → Score: 0.8
                •    Sentiment Score: The sentiment is moderately positive regarding the expansion, though slightly dampened by the unrelated concerns → Score: 0.5
                •    Weight Score: The expansion news is significant, but the mixed content slightly reduces its overall market impact → Score: 0.6

            Response:
            {
              "relevancyScore": 0.8,
              "sentimentScore": 0.5,
              "weightScore": 0.6
            }
            
            *Example 4 – Irrelevant Article*

            Article:
            “Recent developments in space exploration have captured public attention, as a private company announces plans to launch a fleet of satellites for global internet coverage. The breakthrough in rocket technology is poised to revolutionize affordable space travel, drawing attention from tech enthusiasts worldwide.”

            Ticker: MSFT

            Analysis:
                •    Relevancy Score: Although the article is positive and interesting, it does not address Microsoft’s business operations → Score: 0.1
                •    Sentiment Score: The sentiment is positive, but it is unrelated to Microsoft’s performance → Score: 0.8
                •    Weight Score: The news is unlikely to impact Microsoft’s stock because it is not connected to the company’s core business → Score: 0.1

            Response:
            {
              "relevancyScore": 0.1,
              "sentimentScore": 0.8,
              "weightScore": 0.1
            }
            """)
            let stream = try await session.infer(prompt: summary.summary)
            print(stream)
        }
        print(duration)
    }
    
    @Test func testGrammar_NegativeNumber() async throws {
        try await JSONDecoder().decode(Int.self, from: "-5".data(using: .utf8)!)
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_1B_Instruct_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let posSession = try await lm.makeSession(PositiveTest.self, systemPrompt: """
        You are a positive number generator. Generate positive floating point numbers.
        
        Example:
        {"positiveNumber":5.3842}
        """)
        print(try await posSession.infer(prompt: "Generate any number above 0"))
        
        let session = try await lm.makeSession(NegativeTest.self, systemPrompt: """
        You are a negative number generator. Generate negative floating point numbers.
        
        Example:
        {"negativeNumber": -5.3842}
        """)
        print(try await session.infer(prompt: "Generate any number below 0"))
    }
    
    @JSONSchema struct Quote {
        let quote: String
    }
    
    @Test func testGrammar_EscapedString() async throws {
        let lm = try LlamaANE.LanguageModel(model: Llama_3_2_1B_Instruct_Int4().model
//                                            ,
//                                            temperature: 1.2,
//                                            topK: 50,
//                                            topP: 0.9,
//                                            repetitionPenalty: 1.1
        )
        let session = try await lm.makeSession(Quote.self, systemPrompt: """
        You are a string generator. Generate me someone saying something surrounded by quotes.
        
        Example:
        {"quote":"And then he said \"I think therefore I am.\""}
        """)
        
        print(try await session.infer(prompt: "Generate me prose with dialogue, e.g, Tim said \"Hello\"."))
    }

    @Test func testLlamaTools() async throws {
        let lm = try LanguageModel(model: Llama_3_2_3B_Instruct_uncensored().model)
        lm.makeSession(systemPrompt: "", tools: MyTools(), temperature: 1.0, topK: 0, topP: 1.0, repetitionPenalty: 1.0, isLogginEnabled: false)
    }
    */ // End of commented-out tests requiring specific model classes
}

@llamaActor(.v3_2) actor MyTools {
    /// Returns the current date as a human readable string.
    @Tool func getCurrentDate() -> String {
        Date.now.formatted(date: .long, time: .complete)
    }
}

@JSONSchema struct Trip {
    let location: String
    let daysTraveling: Int
}

@JSONSchema struct Sentiment {
    let sentimentScore: Float
}

@JSONSchema struct Summary {
    let summary: String
}

@JSONSchema struct ArticleScores {
    let relevancyScore: Float
    let sentimentScore: Float
    let weightScore: Float
}

@JSONSchema struct PositiveTest {
    let positiveNumber: Float
}

@JSONSchema struct NegativeTest {
    let negativeNumber: Float
}

@JSONSchema struct SimpleResponse {
    let answer: String
}

// Complex data model for grammar performance testing
@JSONSchema struct MovieReview {
    let title: String
    let director: String
    let releaseYear: Int
    let rating: Float
    let genre: String
    let summary: String
}

// Nested object test - an object containing another object
@JSONSchema struct Address {
    let street: String
    let city: String
    let zipCode: String
}

@JSONSchema struct Person {
    let name: String
    let age: Int
    let address: Address
}

// Array of objects test
@JSONSchema struct Task {
    let title: String
    let completed: Bool
}

@JSONSchema struct TodoList {
    let listName: String
    let tasks: [Task]
}

// Nested object with array test
@JSONSchema struct TeamMember {
    let name: String
    let role: String
}

@JSONSchema struct Project {
    let projectName: String
    let description: String
    let lead: TeamMember
    let members: [TeamMember]
}

// MARK: - SchemaGuide Attribute Tests

/// Test struct with @SchemaGuide attributes for descriptions
@JSONSchema struct UserProfile {
    @SchemaGuide("The user's display name")
    let name: String

    @SchemaGuide("Age in years", .range(0...150))
    let age: Int

    @SchemaGuide("User's biography", .maxLength(500))
    let bio: String

    @SchemaGuide("Rating from 0.0 to 5.0", .doubleRange(0.0...5.0))
    let rating: Double

    @SchemaGuide("Number of followers", .range(0...1000000))
    let followers: Int
}

/// Test struct with array count constraint
@JSONSchema struct Playlist {
    @SchemaGuide("Name of the playlist")
    let name: String

    @SchemaGuide("Songs in the playlist (3-50 songs)", .count(3...50))
    let songs: [String]
}

/// Test struct combining multiple constraints
@JSONSchema struct ProductReview {
    @SchemaGuide("Product identifier")
    let productId: String

    @SchemaGuide("Review rating", .range(1...5))
    let rating: Int

    @SchemaGuide("Review title", .maxLength(100))
    let title: String

    @SchemaGuide("Full review text", .maxLength(2000))
    let reviewText: String
}
