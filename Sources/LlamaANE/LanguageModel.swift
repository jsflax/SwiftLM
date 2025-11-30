import Foundation
import FoundationModels


#if canImport(FoundationModels)
@available(macOS 26.0, *)
public typealias GenerableCompat = Generable
#else
@_marker protocol GenerableCompat {}
#endif

public protocol LanguageModelConversation {
    nonisolated(nonsending) func `continue`<Input: Encodable, Output>(
        input: Input,
        expecting: Output.Type
    ) async throws -> Output where Output: JSONSchemaConvertible
    
    @available(macOS 26.0, iOS 26.0, *)
    nonisolated(nonsending) func `continue`<Input: Encodable, Output>(
        input: Input,
        expecting: Output.Type
    ) async throws -> Output where Output: JSONSchemaConvertible & Generable
}

extension LanguageModelConversation {
    nonisolated(nonsending) func `continue`<Input: Codable, Output>(
        input: Input,
        expecting: Output.Type
    ) async throws -> Output where Output: JSONSchemaConvertible {
        try await self.continue(input: input, expecting: expecting)
    }
}

public protocol LanguageModel: Sendable {
    associatedtype Conversation: LanguageModelConversation
    
    func startConversation(systemPrompt: String) async -> Conversation
    
    func structuredAsk<Input, Output>(
        system: String,
        input: Input,
        output: Output.Type,
    ) async throws -> Output where Input: Encodable
    
    @available(macOS 26.0, iOS 26.0, *)
    func strusturedAsk<Input: Sendable, Output>(
        system: String,
        input: Input,
        output: Output.Type,
    ) async throws -> Output where Input: Encodable, Output: JSONSchemaConvertible, Output: Generable
}

extension Session: LanguageModelConversation {
    public typealias Generable = JSONSchemaConvertible
    
    public func `continue`<Output: Sendable>(
        input: String,
        expecting: Output.Type
    ) async throws -> Output {
        let expecting = expecting as! (JSONSchemaConvertible & Sendable).Type
        return try await self.infer(input: input, as: expecting) as! Output
    }
    
    nonisolated(nonsending) public func `continue`<Input: Encodable, Output>(
        input: Input,
        expecting: Output.Type
    ) async throws -> Output where Input: Sendable, Output: Sendable {
        let expecting = expecting as! (JSONSchemaConvertible & Sendable).Type
        return try await self.infer(input: input, as: expecting) as! Output
    }
}

extension CoreMLLanguageModel: LanguageModel {
    public func structuredAsk<Input, Output>(system: String, input: Input, output: Output.Type) async throws -> Output where Input : Encodable {
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let output = output as! (JSONSchemaConvertible & Sendable).Type
        return try await oneShot(prompt: system, input: input, output: output) as! Output
    }
    
    @available(macOS 26.0, *)
    public func strusturedAsk<Input, Output>(system: String, input: Input, output: Output.Type) async throws -> Output where Input : Encodable, Output : Generable, Output : JSONSchemaConvertible, Output : Sendable {
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let output = output as (JSONSchemaConvertible & Sendable).Type
        return try await oneShot(prompt: system, input: input, output: output) as! Output
    }
    
    public func startConversation(systemPrompt: String) async -> Session {
        await self.makeSession(systemPrompt: systemPrompt)
    }
}

@available(macOS 26.0, iOS 26.0, *)
extension LanguageModelSession: LanguageModelConversation {
    public typealias Generable = FoundationModels.Generable
    
    public func `continue`<Input, Output>(input: Input, expecting: Output.Type) async throws -> Output where Input : Encodable, Output : JSONSchemaConvertible {
        let expecting = expecting as! (any Generable & JSONSchemaConvertible).Type
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let content = try await self.respond(to: input, schema: Output.generationSchema)
        return try Output(content.content)
    }
    
    public func `continue`<Input, Output>(input: Input, expecting: Output.Type) async throws -> Output where Input : Encodable, Output: Generable & JSONSchemaConvertible {
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let content: LanguageModelSession.Response<Output> = try await respond(to: input, generating: Output.self)
        return content.content
    }
    
    
    
//    public nonisolated func `continue`<Output>(
//        input: String,
//        expecting: Output.Type
//    ) async throws -> Output where Output: Generable {
//        return try await self.respond(to: input, generating: expecting).content
//    }
//    
//    func test<Input: Encodable, Output: Generable>(input: Input, outputType: Output.Type) async throws -> Output {
//        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
//        let content: LanguageModelSession.Response<Output> = try await respond(to: input, generating: Output.self)
//        return content.content
//    }
//    
//    public func `continue`<Input: Codable, Output>(
//        input: Input,
//        expecting outputType: Output.Type
//    ) async throws -> Output where Output: Generable {
//        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
//        let content: LanguageModelSession.Response<Output> = try await respond(to: input, generating: Output.self)
//        return content.content
//    }
}

@available(macOS 26.0, *)
public struct FoundationLanguageModel: LanguageModel {
    private let model: SystemLanguageModel
    public init(model: SystemLanguageModel) {
        self.model = model
    }
    
    func test<Input: Encodable, Output: Generable>(system: String, input: Input, outputType: Output.Type) async throws -> Output {
        let session = await self.startConversation(systemPrompt: system)
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let content: LanguageModelSession.Response<Output> = try await session.respond(to: input, generating: Output.self)
        return content.content
    }
    
    public func structuredAsk<Input, Output>(system: String, input: Input, output: Output.Type) async throws -> Output where Input : Encodable {
        let session = await self.startConversation(systemPrompt: system)
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        let output = output as! (Generable & Sendable).Type
        return try await test(system: system, input: input, outputType: output) as! Output
    }
    
    public func strusturedAsk<Input, Output>(system: String, input: Input, output: Output.Type) async throws -> Output where Input : Encodable, Output : Generable, Output : JSONSchemaConvertible, Output : Sendable {
        let session = await self.startConversation(systemPrompt: system)
        let input = try String(data: JSONEncoder().encode(input), encoding: .utf8)!
        return try await session.respond(to: input, generating: output).content
    }
    
    public func startConversation(systemPrompt: String) async -> LanguageModelSession {
        let session = LanguageModelSession(model: model, instructions: systemPrompt)
        return session
    }
}
//struct Test {
//
//}
//if #available(macOS 26.0, *) {
//
//}
//@available(macOS 26.0, *)
//typealias _Generable = FoundationModels.Generable
//typealias _Generable = LanguageModel.JSONSchema
