import Foundation
import Testing
@testable import LlamaANE
import CoreML
import JSONSchema
import MapKit
import CoreLocation

// Path to test models (uses the one from LlamaANETests.swift via internal linkage)
private let tripTestModelsPath = "/Users/jason/Documents/LlamaANE/Plugins/LLMGenerator/models"

// MARK: - TripIntent Data Models (Ported from JoyJet)

/// Travel persona enum for trip planning
/// Note: Don't include CaseIterable here since @JSONSchema adds it
@JSONSchema enum TravelPersona: String, Codable, Sendable {
    case bohemian      // indie, street food, thrift, alt music
    case bougie        // fine dining, spa, designer shopping
    case classic       // landmarks, museums, cafés
    case outdoorsy     // hikes, beaches, nature, surf
    case nightOwl      // bars, live music, late eats
    case familyFriendly// zoos, aquariums, playgrounds
}

/// Parsed trip intent - complex nested structure for testing LLM structured output
@JSONSchema struct ParsedTripIntent: Sendable {
    @JSONSchema struct Destination: Sendable {
        @SchemaGuide("City or region name, e.g. 'Tokyo', 'Kyoto', 'Tuscany'", .maxLength(100))
        var name: String

        @SchemaGuide("Country name where this destination is located", .maxLength(100))
        var country: String?

        @SchemaGuide("Type of destination", .anyOf(["country", "city", "region"]))
        var kind: String?

        @SchemaGuide("Priority order, 1 = primary destination", .range(1...10))
        var priority: Int?

        @SchemaGuide("Number of days to spend at this destination", .range(1...90))
        var destinationDays: Int

        @SchemaGuide("Starting day index within the trip (0-based)", .range(0...365))
        var destinationStartDayIndex: Int

        var coordinateRegion: MKCoordinateRegion
    }

    @SchemaGuide("List of destinations to visit in order", .count(1...20))
    var destinations: [Destination]

    @SchemaGuide("Total trip duration in days", .range(1...365))
    var durationDays: Int?

    @SchemaGuide("User interests like 'food', 'urban exploration', 'surfing', 'history'", .count(1...20))
    var interests: [String]

    @SchemaGuide("Guide tags from: Foodie, Art & Culture, Nightlife, Outdoors, Family", .count(1...5))
    var guideTags: [String]?

    @SchemaGuide("Trip constraints like 'budget: $$', 'fast pace', 'avoid tour buses'")
    var constraints: [String]?

    @SchemaGuide("Travel style", .anyOf(["solo", "couple", "family", "group", "business"]))
    var travelStyle: String?

    @SchemaGuide("Additional freeform notes from the user", .maxLength(500))
    var notes: String?

    var travelPersona: TravelPersona?

    @SchemaGuide("Short descriptive name for this trip, e.g. '2 Weeks in Japan'", .maxLength(100))
    var name: String

    @SchemaGuide("City to return to at end of trip, if different from departure", .maxLength(100))
    var returnCity: String?
}

/// Input for the NL intent parser
@JSONSchema struct TripIntentInput: Sendable {
    @SchemaGuide("The user's natural language travel request")
    let utterance: String

    @SchemaGuide("Locale hint like 'en-US'")
    let localeHint: String?
}

extension MKCoordinateSpan {
    static let example = MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0)
}

/// Simpler trip intent for basic testing
@JSONSchema struct SimpleTripIntent: Sendable {
    @SchemaGuide("Trip name like '2 Weeks in Japan'", .maxLength(100))
    var name: String

    @SchemaGuide("Total days", .range(1...365))
    var durationDays: Int

    @SchemaGuide("List of city names to visit")
    var cities: [String]

    @SchemaGuide("User interests")
    var interests: [String]
}

// MARK: - Parallel Inference Sub-Schemas

/// Extract just the basic trip info (name, duration)
@JSONSchema struct TripBasicInfo: Sendable {
    @SchemaGuide("Short trip name like '2 Weeks in Japan'", .maxLength(100))
    var name: String

    @SchemaGuide("Total days (1 week=7, 2 weeks=14)", .range(1...365))
    var durationDays: Int

    @SchemaGuide("Infer travel style from context: adventure/surfing→solo or couple, kid activities→family, meetings→business", .anyOf(["solo", "couple", "family", "group", "business"]))
    var travelStyle: String
}

/// Extract interests from the user's message
@JSONSchema struct TripInterests: Sendable {
    @SchemaGuide("Activities from message: food, surfing, urban exploration, temples, art, etc", .count(1...10))
    var interests: [String]
}

/// Simpler persona extraction
@JSONSchema struct TripPersona: Sendable {
    var travelPersona: TravelPersona
}

/// Extract a single destination (no coordinates - will use MapKit lookup)
@JSONSchema struct TripDestination: Sendable {
    @SchemaGuide("City name like 'Tokyo', 'Kyoto'", .maxLength(100))
    var name: String

    @SchemaGuide("Country name", .maxLength(100))
    var country: String

    @SchemaGuide("Number of days at this destination", .range(1...90))
    var days: Int
}

/// Extract list of destinations
@JSONSchema struct TripDestinations: Sendable {
    @SchemaGuide("Cities to visit for a complete trip itinerary", .count(1...5))
    var destinations: [TripDestination]
}

// MARK: - Test Models Enum

enum TripTestModelKind: String, CaseIterable {
    case qwen05bInt4 = "Qwen2.5-0.5B-Instruct_Int4.mlpackage"
    case qwen15bInt4 = "Qwen2.5-1.5B-Instruct_Int4.mlpackage"
    case qwen3_06bInt4 = "Qwen3-0.6B_Int4.mlpackage"
    case llama31bInt4 = "Llama-3.2-1B-Instruct_Int4.mlpackage"
    case llama33bInt4 = "Llama-3.2-3B-Instruct_Int4.mlpackage"
}

// MARK: - Integration Tests

@Suite("Trip Intent Integration Tests", .serialized)
struct TripIntentIntegrationTests {

    // MARK: - Schema Description Tests

    @Test("ParsedTripIntent schema description shows nested Destination fields")
    func testTripIntentSchemaDescription() throws {
        let description = ParsedTripIntent.schemaDescription()
        print("ParsedTripIntent Schema Description:\n\(description)")

        // Should contain top-level fields
        #expect(description.contains("destinations"), "Should contain destinations field")
        #expect(description.contains("durationDays"), "Should contain durationDays field")
        #expect(description.contains("interests"), "Should contain interests field")
        #expect(description.contains("name"), "Should contain name field")

        // Should contain nested Destination fields
        #expect(description.contains("destinationDays"), "Should contain nested destinationDays")
        #expect(description.contains("coordinateRegion"), "Should contain nested coordinateRegion")
        #expect(description.contains("country"), "Should contain nested country")

        // Should contain constraint info
        #expect(description.contains("range:") || description.contains("max"), "Should contain constraint info")
    }

    @Test("ParsedTripIntent has correct schemaProperties")
    func testTripIntentSchemaProperties() throws {
        let props = ParsedTripIntent.schemaProperties
        #expect(props != nil, "Should have schemaProperties")

        guard let props = props else { return }

        // Check destinations array
        let destProp = props.first { $0.name == "destinations" }
        #expect(destProp != nil, "Should have destinations property")
        #expect(destProp?.isArray == true, "destinations should be an array")
        #expect(destProp?.countRange == 1...20, "destinations should have count constraint 1...20")

        // Check durationDays
        let durationProp = props.first { $0.name == "durationDays" }
        #expect(durationProp != nil, "Should have durationDays property")
        #expect(durationProp?.intRange == 1...365, "durationDays should have range 1...365")

        // Check nested Destination properties
        if let destProp = destProp, let elemType = destProp.elementType {
            let nestedProps = elemType.schemaProperties
            #expect(nestedProps != nil, "Destination should have schemaProperties")

            if let nestedProps = nestedProps {
                let daysProp = nestedProps.first { $0.name == "destinationDays" }
                #expect(daysProp?.intRange == 1...90, "destinationDays should have range 1...90")
            }
        }

        print("ParsedTripIntent schema properties verified!")
    }

    // MARK: - Model Inference Tests

    @Test("Parse trip intent with local model", arguments: [
        TripTestModelKind.qwen15bInt4, TripTestModelKind.llama31bInt4, TripTestModelKind.llama33bInt4])
    func testParseTripIntent(modelKind: TripTestModelKind) async throws {
        let modelPath = "\(tripTestModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let systemPrompt = """
        Extract travel intent as JSON. Output JSON only.
        Rules: "2 weeks"=14 days. interests=activities mentioned (food, surfing, urban exploration, etc).

        Input: "Thailand 1 week, eat street food, explore temples, go diving"
        Output: {"destinations":[{"name":"Bangkok","country":"Thailand","kind":"city","priority":1,"destinationDays":4,"destinationStartDayIndex":0,"coordinateRegion":{"center":{"latitude":13.7,"longitude":100.5},"span":{"latitudeDelta":0.5,"longitudeDelta":0.5}}},{"name":"Phuket","country":"Thailand","kind":"city","priority":2,"destinationDays":3,"destinationStartDayIndex":4,"coordinateRegion":{"center":{"latitude":7.9,"longitude":98.4},"span":{"latitudeDelta":0.5,"longitudeDelta":0.5}}}],"durationDays":7,"interests":["food","temples","diving"],"guideTags":["Foodie","Art & Culture","Outdoors"],"constraints":[],"travelStyle":"solo","notes":"","travelPersona":"outdoorsy","name":"1 Week in Thailand","returnCity":"Bangkok"}

        Input:
        """

        let session = await lm.makeSession(systemPrompt: systemPrompt)

        let userUtterance = "I wanna go to Japan for 2 weeks, eat great food, urban explore, and surf."

        print("\n--- Trip Intent Parsing Test ---")
        print("Model: \(modelKind)")
        print("User utterance: \(userUtterance)")
        print("\nSchema description:")
        print(ParsedTripIntent.schemaDescription())

        let startTime = Date()
        var output = ""
        var tokenCount = 0

        print("\n--- Generated Output ---")
        let stream: AsyncStream<String> = await session.infer(prompt: userUtterance, as: ParsedTripIntent.self)
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
        print("Tokens/sec: \(String(format: "%.1f", Double(tokenCount) / totalTime))")

        // Try to decode and validate
        print("\n--- Validation ---")
        guard let jsonData = output.data(using: .utf8) else {
            Issue.record("Failed to convert output to data")
            return
        }

        do {
            let tripIntent = try JSONDecoder().decode(ParsedTripIntent.self, from: jsonData)

            print("Successfully decoded ParsedTripIntent!")
            print("  Name: \(tripIntent.name)")
            print("  Duration: \(tripIntent.durationDays ?? 0) days")
            print("  Destinations: \(tripIntent.destinations.count)")
            for (i, dest) in tripIntent.destinations.enumerated() {
                print("    \(i+1). \(dest.name) (\(dest.destinationDays) days)")
                print("       Country: \(dest.country ?? "nil")")
                print("       Kind: \(dest.kind ?? "nil")")
                print("       Coordinates: \(dest.coordinateRegion.center.latitude), \(dest.coordinateRegion.center.longitude)")
            }
            print("  Interests: \(tripIntent.interests)")
            print("  Guide Tags: \(tripIntent.guideTags ?? [])")
            print("  Travel Style: \(tripIntent.travelStyle ?? "nil")")
            print("  Persona: \(tripIntent.travelPersona?.rawValue ?? "nil")")

            // Validate constraints
            #expect(!tripIntent.destinations.isEmpty, "Should have at least one destination")

            if let duration = tripIntent.durationDays {
                #expect(duration >= 1 && duration <= 365, "Duration should be in valid range")

                // Check that destination days sum to total duration
                let totalDestDays = tripIntent.destinations.reduce(0) { $0 + $1.destinationDays }
                print("  Total destination days: \(totalDestDays) (expected: \(duration))")

                // This is a known hard problem for LLMs - check but don't fail
                if totalDestDays != duration {
                    print("  ⚠️ WARNING: Destination days (\(totalDestDays)) != durationDays (\(duration))")
                }
            }

            #expect(!tripIntent.interests.isEmpty, "Should have at least one interest")
            #expect(tripIntent.name.count <= 100, "Name should respect maxLength constraint")

            print("\n✅ Trip intent parsed and validated successfully!")

        } catch {
            print("❌ Failed to decode: \(error)")
            print("Raw output:\n\(output)")
            Issue.record("JSON decoding failed: \(error)")
        }
    }

    @Test("Simple Japan trip parsing", arguments: [TripTestModelKind.qwen15bInt4])
    func testSimpleJapanTrip(modelKind: TripTestModelKind) async throws {
        // Simpler test with just essential fields to diagnose issues
        let modelPath = "\(tripTestModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let session = await lm.makeSession(
            systemPrompt: "You extract travel plans from user messages. Return valid JSON only."
        )

        let prompt = "I want to visit Japan for 2 weeks. I like food and temples."

        print("\n--- Simple Trip Intent Test ---")
        print("Model: \(modelKind)")
        print("Prompt: \(prompt)")
        print("\nSchema:")
        print(SimpleTripIntent.schemaDescription())

        var output = ""
        print("\n--- Output ---")
        let stream: AsyncStream<String> = await session.infer(prompt: prompt, as: SimpleTripIntent.self)
        for await token in stream {
            output += token
            print(token, terminator: "")
        }
        print()

        // Try to decode
        if let data = output.data(using: .utf8) {
            do {
                let intent = try JSONDecoder().decode(SimpleTripIntent.self, from: data)
                print("\n✅ Decoded: \(intent.name), \(intent.durationDays) days, cities: \(intent.cities)")
            } catch {
                print("\n❌ Decode error: \(error)")
                print("Raw: \(output)")
            }
        }
    }

    // MARK: - Parallel Inference Tests

    @Test("Parallel inference for trip intent", arguments: [TripTestModelKind.llama33bInt4])
    func testParallelTripIntent(modelKind: TripTestModelKind) async throws {
        let modelPath = "\(tripTestModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try LanguageModel(model: mlModel)
        try await lm.warmup()

        let userUtterance = "I wanna go to Japan for 2 weeks, eat great food, urban explore, and surf."

        print("\n--- Parallel Trip Intent Test ---")
        print("Model: \(modelKind)")
        print("User utterance: \(userUtterance)")

        let startTime = Date()

        // Run parallel inferences using TaskGroup
        async let basicInfoTask = extractBasicInfo(lm: lm, utterance: userUtterance)
        async let interestsTask = extractInterests(lm: lm, utterance: userUtterance)
        async let destinationsTask = extractDestinations(lm: lm, utterance: userUtterance)

        let (basicInfo, interests, destinations) = await (basicInfoTask, interestsTask, destinationsTask)

        let totalTime = Date().timeIntervalSince(startTime)

        print("\n--- Results ---")
        print("Total time: \(String(format: "%.2f", totalTime))s")

        if let basic = basicInfo {
            print("✅ Basic Info: name='\(basic.name)', duration=\(basic.durationDays) days, style=\(basic.travelStyle)")
        } else {
            print("❌ Basic Info: failed")
        }

        if let int = interests {
            print("✅ Interests: \(int.interests)")
        } else {
            print("❌ Interests: failed")
        }

        if let dest = destinations {
            print("✅ Destinations: \(dest.destinations.count) cities")
            for d in dest.destinations {
                print("   - \(d.name), \(d.country): \(d.days) days")
            }
        } else {
            print("❌ Destinations: failed")
        }

        // Validate results
        #expect(basicInfo != nil, "Should extract basic info")
        #expect(interests != nil, "Should extract interests")
        #expect(destinations != nil, "Should extract destinations")

        if let int = interests {
            #expect(!int.interests.isEmpty, "Should have at least one interest")
        }

        // Assemble the final ParsedTripIntent from parallel results
        if let basic = basicInfo, let int = interests, let dest = destinations {
            var dayIndex = 0
            let parsedDestinations = dest.destinations.map { d -> ParsedTripIntent.Destination in
                // TODO: Use MapKit to look up coordinates for city name
                // For now, use placeholder coordinates
                let destination = ParsedTripIntent.Destination(
                    name: d.name,
                    country: d.country,
                    kind: "city",
                    priority: 1,
                    destinationDays: d.days,
                    destinationStartDayIndex: dayIndex,
                    coordinateRegion: MKCoordinateRegion(
                        center: CLLocationCoordinate2D(latitude: 0, longitude: 0),
                        span: .example
                    )
                )
                dayIndex += d.days
                return destination
            }

            let finalIntent = ParsedTripIntent(
                destinations: parsedDestinations,
                durationDays: basic.durationDays,
                interests: int.interests,
                guideTags: nil,
                constraints: nil,
                travelStyle: basic.travelStyle,
                notes: nil,
                travelPersona: nil,
                name: basic.name,
                returnCity: nil
            )

            print("\n--- Final ParsedTripIntent ---")
            print("Name: \(finalIntent.name)")
            print("Duration: \(finalIntent.durationDays ?? 0) days")
            print("Travel Style: \(finalIntent.travelStyle ?? "not specified")")
            print("Interests: \(finalIntent.interests)")
            print("Destinations:")
            for d in finalIntent.destinations {
                print("  • \(d.name), \(d.country ?? "?"): \(d.destinationDays) days (starting day \(d.destinationStartDayIndex))")
                print("    Coordinates: (\(d.coordinateRegion.center.latitude), \(d.coordinateRegion.center.longitude))")
            }
        }
    }

    // Helper functions for parallel extraction
    private func extractBasicInfo(lm: LanguageModel, utterance: String) async -> TripBasicInfo? {
        let session = await lm.makeSession(systemPrompt: "Extract trip name and duration. Output JSON only.")
        var output = ""
        let stream: AsyncStream<String> = await session.infer(prompt: utterance, as: TripBasicInfo.self)
        for await token in stream { output += token }
        print("\n[BasicInfo] Output: \(output)")
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(TripBasicInfo.self, from: data)
    }

    private func extractInterests(lm: LanguageModel, utterance: String) async -> TripInterests? {
        // Chain-of-thought: First have model enumerate activities in a structured way
        // Using few-shot format to encourage the model to list all activities
        let cotSession = await lm.makeSession(systemPrompt: """
            You extract activities from travel messages. List each activity on a new line with a dash.

            Message: "I want to eat sushi and visit temples"
            Activities:
            - eating sushi
            - visiting temples

            Message: "beach vacation with snorkeling and relaxation"
            Activities:
            - snorkeling
            - relaxation
            - beach
            """)

        // Step 1: Get CoT reasoning (unconstrained)
        var reasoning = ""
        let cotPrompt = "Message: \"\(utterance)\"\nActivities:"
        let cotStream = await cotSession.infer(prompt: cotPrompt)
        for await token in cotStream {
            reasoning += token
            // Stop after getting enough or hitting double newlines
            if reasoning.count > 300 || (reasoning.count > 30 && reasoning.hasSuffix("\n\n")) { break }
        }
        print("[Interests CoT] '\(reasoning.trimmingCharacters(in: .whitespacesAndNewlines))'")

        // If CoT produced nothing, try direct extraction
        if reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("[Interests] CoT empty, using direct extraction")
            let directSession = await lm.makeSession(systemPrompt: """
                Extract activities/interests from the message. Output JSON with interests array.
                """)
            var output = ""
            let stream: AsyncStream<String> = await directSession.infer(prompt: utterance, as: TripInterests.self)
            for await token in stream { output += token }
            print("[Interests] Direct output: \(output)")
            guard let data = output.data(using: .utf8) else { return nil }
            return try? JSONDecoder().decode(TripInterests.self, from: data)
        }

        // Step 2: Parse the activities directly from CoT output - no need for second LLM call!
        // The CoT already produced a clean list, just split by newlines
        let activities = reasoning
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n")
            .map { line -> String in
                // Remove leading dashes/bullets if present
                var cleaned = line.trimmingCharacters(in: .whitespaces)
                if cleaned.hasPrefix("-") { cleaned = String(cleaned.dropFirst()).trimmingCharacters(in: .whitespaces) }
                if cleaned.hasPrefix("•") { cleaned = String(cleaned.dropFirst()).trimmingCharacters(in: .whitespaces) }
                return cleaned
            }
            .filter { !$0.isEmpty }

        print("[Interests] Extracted: \(activities)")
        return TripInterests(interests: activities)
    }

    private func extractDestinations(lm: LanguageModel, utterance: String) async -> TripDestinations? {
        // Use greedy - sampling with grammar constraints has unresolved issues
        let session = await lm.makeSession(
            systemPrompt: """
                Extract cities to visit. For multi-week trips, suggest 2-3 cities.
                Distribute days to fill the trip duration.
                Output JSON only.
                """
        )
        var output = ""
        let stream: AsyncStream<String> = await session.infer(prompt: utterance, as: TripDestinations.self)
        for await token in stream { output += token }
        print("\n[Destinations] Output: \(output)")
        guard let data = output.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(TripDestinations.self, from: data)
    }
}
