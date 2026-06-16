import Foundation
import Testing
@testable import SwiftLM
import CoreML
import JSONSchema
import MapKit
import CoreLocation

// Path to test models
private let tripTestModelsPath = "/Users/jason/Documents/SwiftLM/Plugins/LLMGenerator/models"

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
    @SchemaGuide("You're reasoning for picking the interests based on the prompt.")
    var reasoning: String
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
    @SchemaGuide("Cities to visit for a complete trip itinerary")
    var destinations: [TripDestination]
    @SchemaGuide("You're reasoning for picking the cities and days.")
    var reasoning: String
}

// MARK: - Test Models Enum

enum TripTestModelKind: String, CaseIterable {
    case qwen05bInt4 = "Qwen2.5-0.5B-Instruct_Int4.mlpackage"
    case qwen15bInt4 = "Qwen2.5-1.5B-Instruct_Int4.mlpackage"
    case qwen3_06bInt4 = "Qwen3-0.6B_Int4.mlpackage"
    case llama31bInt4 = "Llama-3.2-1B-Instruct_Int4.mlpackage"
    case llama33bInt4 = "Llama-3.2-3B-Instruct_Int4.mlpackage"
    case qwenTripIntentFinetuned = "Qwen-Trip-Intent-Finetuned_Int4.mlpackage"
    case qwenTripIntentProd = "Qwen-Trip-Intent-Prod_Int4.mlpackage"
    case qwenTripCombined = "Qwen-Trip-Combined_Int4.mlpackage"
    case qwen05bTripCombined = "Qwen-0.5B-Trip-Combined_Int4.mlpackage"
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

        let lm = try CoreMLLanguageModel(model: mlModel)
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

        let lm = try CoreMLLanguageModel(model: mlModel)
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

    @available(macOS 26.0, *)
    @Test("Parallel inference for trip intent", arguments: [TripTestModelKind.qwenTripCombined])
    func testParallelTripIntent(modelKind: TripTestModelKind) async throws {
        let modelPath = "\(tripTestModelsPath)/\(modelKind.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

//        let lm = try CoreMLLanguageModel(model: mlModel)
//        try await lm.warmup()

        let lm = try FoundationLanguageModel(model: .default)
        let userUtterance = "I wanna go to Japan for 2 weeks, eat great food, urban explore, and surf."

        print("\n--- Parallel Trip Intent Test ---")
        print("Model: \(modelKind)")
        print("User utterance: \(userUtterance)")

        let startTime = Date()

        // Run parallel inferences using TaskGroup
        async let basicInfoTask = extractBasicInfo(lm: lm, utterance: userUtterance)
        async let interestsTask = extractInterests(lm: lm, utterance: userUtterance)
        let (basicInfo, interests) = try await (basicInfoTask, interestsTask)
        let destinations = try await extractDestinationsIterative(lm: lm, tripInfo: basicInfo!, originLocation: "New York, USA", roundTrip: true)

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

        print("✅ Destinations: \(destinations.destinations.count) cities")
        for d in destinations.destinations {
            print("   - \(d.name), \(d.country): \(d.days) days")
        }

        // Validate results
        #expect(basicInfo != nil, "Should extract basic info")
        #expect(interests != nil, "Should extract interests")

        if let int = interests {
            #expect(!int.interests.isEmpty, "Should have at least one interest")
        }

        // Assemble the final ParsedTripIntent from parallel results
        if let basic = basicInfo, let int = interests {
            var dayIndex = 0
            var parsedDestinations: [ParsedTripIntent.Destination] = []
            for d in destinations.destinations {
                // TODO: Use MapKit to look up coordinates for city name
                // For now, use placeholder coordinates
                let placemarks = try? await CLGeocoder().geocodeAddressString("\(d.name), \(d.country)")
                let region = placemarks?.first?.location.map {
                    MKCoordinateRegion(
                        center: $0.coordinate,
                        span: .example
                    )
                } ?? MKCoordinateRegion(
                    center: CLLocationCoordinate2D(latitude: 0, longitude: 0),
                    span: .example
                )
                let destination = ParsedTripIntent.Destination(
                    name: d.name,
                    country: d.country,
                    kind: "city",
                    priority: 1,
                    destinationDays: d.days,
                    destinationStartDayIndex: dayIndex,
                    coordinateRegion: region
                )
                dayIndex += d.days
                parsedDestinations.append(destination)
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
    private func extractBasicInfo(lm: some LanguageModel, utterance: String) async throws -> TripBasicInfo? {
        let session = await lm.startConversation(systemPrompt: "Extract trip name and duration. Output JSON only.")
        let tripBasicInfo: TripBasicInfo = try await session.continue(input: utterance, expecting: TripBasicInfo.self)
        return tripBasicInfo
    }

    private func extractInterests(lm: some LanguageModel, utterance: String) async throws -> TripInterests? {
        // Use plain text extraction first to avoid single-element array issue
        let session = await lm.startConversation(
            systemPrompt: """
            List ALL activities from the message. 
            
            Utterance:
            """,
//            doSample: true
        )

        let interests = try await session.continue(input: utterance, expecting: TripInterests.self)

        print("Extracted interests: \(interests)")
        return interests
    }

    private func extractDestinations(lm: some LanguageModel, tripInfo: TripBasicInfo) async throws -> TripDestinations {
        // Iterative extraction: get one destination at a time until days are exhausted
        // This avoids the single-element array problem with grammar-constrained generation
        let session = await lm.startConversation(
            systemPrompt: """
            You are a travel planner for \(tripInfo.name). Suggest multiple cities to visit.
            Pick popular destinations that fit the trip style (\(tripInfo.travelStyle)).
            
            """,
//            doSample: true  // Greedy for consistent results
        )

        var daysRemaining = tripInfo.durationDays
        var destinations: TripDestinations = try await session.continue(input: """
        The trip is \(tripInfo.durationDays) days long. You MUST fill up the entire \(tripInfo.durationDays) days with destinations, but not exceed \(tripInfo.durationDays) days.
        Allocate reasonable days (3-5 for major cities, 2-3 for smaller ones).    
        """, expecting: TripDestinations.self)
        print(destinations)
        return destinations
    }
    
    private func extractDestinationsIterative(
        lm: some LanguageModel,
        tripInfo: TripBasicInfo,
        originLocation: String,
        roundTrip: Bool = false,
    ) async throws -> TripDestinations {
        // Iterative extraction: get one destination at a time until days are exhausted
        // This avoids the single-element array problem with grammar-constrained generation
        let session = await lm.startConversation(
            systemPrompt: """
            You are a travel planner for \(tripInfo.name). Suggest ONE city to visit.
            Pick popular destinations that fit the trip style (\(tripInfo.travelStyle)).
            Allocate reasonable days (3-5 for major cities, 2-3 for smaller ones).
            
            Origin location: \(originLocation). Consider whether or not the user has to fly from here.
            """,
//            temperature: 0.6,
//            doSample: true,
        )

        var daysRemaining = tripInfo.durationDays
        var destinations: [TripDestination] = []
        var visitedCities: [String] = []

        while daysRemaining > 0 {
            // Build context about what's already been planned
            let contextPrompt: String
            if visitedCities.isEmpty {
                contextPrompt = "First destination. \(daysRemaining) days total."
            } else {
                contextPrompt = "Already visiting: \(visitedCities.joined(separator: ", ")). \(daysRemaining) days left."
            }

            let destination = try await session.continue(
                input: contextPrompt,
                expecting: TripDestination.self
            )

            // Clamp days to remaining (in case model overshoots)
            var adjustedDestination = destination
            if destination.days > daysRemaining {
                adjustedDestination = TripDestination(
                    name: destination.name,
                    country: destination.country,
                    days: daysRemaining
                )
            }

            print("  → \(adjustedDestination.name), \(adjustedDestination.country): \(adjustedDestination.days) days")
            destinations.append(adjustedDestination)
            visitedCities.append(adjustedDestination.name)
            daysRemaining -= adjustedDestination.days
        }

        return TripDestinations(destinations: destinations, reasoning: "Iterative extraction")
    }

    // MARK: - Fine-tuned Model Tests

    @Test("Fine-tuned model intent parsing")
    func testFinetunedModelIntentParsing() async throws {
        let modelPath = "\(tripTestModelsPath)/\(TripTestModelKind.qwen05bTripCombined.rawValue)"
        let compiledURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        let lm = try CoreMLLanguageModel(model: mlModel)
        try await lm.warmup()

        // System prompt matching the fine-tuning data format
        let systemPrompt = """
        You are a travel planner AI. Parse the user's travel request and extract trip intent as JSON.
        Rules:
        - "2 weeks" = 14 days, "1 week" = 7 days
        - Infer interests from activities mentioned (food, urban exploration, surfing, temples, etc.)
        - Pick appropriate travel persona (bohemian, bougie, classic, outdoorsy, nightOwl, familyFriendly)
        - Use guideTags from: Foodie, Art & Culture, Nightlife, Outdoors, Family
        - Provide reasonable coordinates for each destination
        """

        let session = await lm.makeSession(systemPrompt: systemPrompt)

        let testCases = [
            "I want to visit Japan for 2 weeks. I love food and want to explore cities.",
            "Planning a romantic trip to Paris and Rome for 10 days",
            "Family vacation to Hawaii - beaches and snorkeling for a week",
        ]

        for utterance in testCases {
            print("\n--- Fine-tuned Model Test ---")
            print("Utterance: \(utterance)")

            let input = TripIntentInput(utterance: utterance, localeHint: "en-US")
            let inputJSON = try JSONEncoder().encode(input)
            let inputStr = String(data: inputJSON, encoding: .utf8)!

            let startTime = Date()
            var output = ""
            var tokenCount = 0

            print("\n--- Output ---")
            let stream: AsyncStream<String> = await session.infer(prompt: inputStr, as: ParsedTripIntent.self)
            for await token in stream {
                output += token
                tokenCount += 1
                print(token, terminator: "")
            }
            print()

            let totalTime = Date().timeIntervalSince(startTime)
            print("\n--- Metrics ---")
            print("Tokens: \(tokenCount), Time: \(String(format: "%.2f", totalTime))s, Speed: \(String(format: "%.1f", Double(tokenCount) / totalTime)) tok/s")

            // Try to decode
            if let data = output.data(using: .utf8) {
                do {
                    let intent = try JSONDecoder().decode(ParsedTripIntent.self, from: data)
                    print("✅ Decoded: \(intent.name)")
                    print("   Duration: \(intent.durationDays ?? 0) days")
                    print("   Destinations: \(intent.destinations.map { $0.name }.joined(separator: ", "))")
                    print("   Interests: \(intent.interests.joined(separator: ", "))")
                } catch {
                    print("❌ Decode error: \(error)")
                    print("Raw: \(output)")
                }
            }
        }
    }

    @Test("Compare fine-tuned vs base model")
    func testCompareFinetunedVsBase() async throws {
        // Load both models
        let basePath = "\(tripTestModelsPath)/\(TripTestModelKind.qwen15bInt4.rawValue)"
        let finetunedPath = "\(tripTestModelsPath)/\(TripTestModelKind.qwenTripIntentProd.rawValue)"

        let baseCompiled = try await MLModel.compileModel(at: URL(fileURLWithPath: basePath))
        let finetunedCompiled = try await MLModel.compileModel(at: URL(fileURLWithPath: finetunedPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU

        let baseModel = try MLModel(contentsOf: baseCompiled, configuration: mlConfig)
        let finetunedModel = try MLModel(contentsOf: finetunedCompiled, configuration: mlConfig)

        let baseLM = try CoreMLLanguageModel(model: baseModel)
        let finetunedLM = try CoreMLLanguageModel(model: finetunedModel)

        try await baseLM.warmup()
        try await finetunedLM.warmup()

        let systemPrompt = """
        Parse the user's travel request and extract trip intent as JSON.
        """

        let utterance = "I wanna go to Japan for 2 weeks, eat great food, urban explore, and surf."

        print("\n=== Base Model (Qwen 1.5B) ===")
        let baseSession = await baseLM.makeSession(systemPrompt: systemPrompt)
        var baseOutput = ""
        var baseTokens = 0
        let baseStart = Date()
        let baseStream: AsyncStream<String> = await baseSession.infer(prompt: utterance, as: SimpleTripIntent.self)
        for await token in baseStream {
            baseOutput += token
            baseTokens += 1
            print(token, terminator: "")
        }
        let baseTime = Date().timeIntervalSince(baseStart)
        print("\n[Base: \(baseTokens) tokens, \(String(format: "%.2f", baseTime))s]")

        print("\n=== Fine-tuned Model ===")
        let ftSession = await finetunedLM.makeSession(systemPrompt: systemPrompt)
        var ftOutput = ""
        var ftTokens = 0
        let ftStart = Date()
        let ftStream: AsyncStream<String> = await ftSession.infer(prompt: utterance, as: SimpleTripIntent.self)
        for await token in ftStream {
            ftOutput += token
            ftTokens += 1
            print(token, terminator: "")
        }
        let ftTime = Date().timeIntervalSince(ftStart)
        print("\n[Fine-tuned: \(ftTokens) tokens, \(String(format: "%.2f", ftTime))s]")

        // Compare decoding success
        print("\n=== Comparison ===")
        var baseSuccess = false
        var ftSuccess = false

        if let data = baseOutput.data(using: .utf8) {
            if let _ = try? JSONDecoder().decode(SimpleTripIntent.self, from: data) {
                baseSuccess = true
                print("Base model: ✅ Valid JSON")
            } else {
                print("Base model: ❌ Invalid JSON")
            }
        }

        if let data = ftOutput.data(using: .utf8) {
            if let intent = try? JSONDecoder().decode(SimpleTripIntent.self, from: data) {
                ftSuccess = true
                print("Fine-tuned:  ✅ Valid JSON - \(intent.name)")
            } else {
                print("Fine-tuned:  ❌ Invalid JSON")
            }
        }

        print("\nBase speed: \(String(format: "%.1f", Double(baseTokens) / baseTime)) tok/s")
        print("Fine-tuned speed: \(String(format: "%.1f", Double(ftTokens) / ftTime)) tok/s")
    }
    
    @Test
    func testTransport() async throws {
        let finetunedPath = "\(tripTestModelsPath)/\(TripTestModelKind.qwenTripIntentProd.rawValue)"
        let finetunedCompiled = try await MLModel.compileModel(at: URL(fileURLWithPath: finetunedPath))

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndGPU

        let finetunedModel = try MLModel(contentsOf: finetunedCompiled, configuration: mlConfig)
        
        let finetunedLM = try CoreMLLanguageModel(model: finetunedModel)
        try await finetunedLM.warmup()
        
        let start = Date()
        print("Start: \(start)")
        let session = await finetunedLM.makeSession(systemPrompt: """
        Choose the most sensible transport mode to get from the origin location to the destination location.
        Consider distance, rail system, and mode feasability.
        """)
        
        let mode = try await session.infer(input: TransportDeciderInput(fromCity: "Tokyo",
                                                                        toCity: "Osaka"),
                                           as: TransportDeciderOutput.self)
        let ftTime = Date().timeIntervalSince(start)
        print("Total inference time: \(String(format: "%.2f", ftTime))s")
        print(mode.transportMode)
    }

}

@JSONSchema struct TransportDeciderInput {
    let fromCity: String
    let toCity: String
}
@JSONSchema enum TransportMode: String, Codable {
    case plane, train, automobile
}
@JSONSchema struct TransportDeciderOutput {
    let transportMode: TransportMode
}
