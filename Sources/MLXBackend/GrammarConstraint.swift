import Foundation
import MLX
import MLXLMCommon
import SwiftLM
import MiniBPE
import JSONSchema
import Serving

// ── Grammar-constrained MLX decoding (the "valid-by-construction" tool-call upgrade).
//
// `GrammarLogitProcessor` is an mlx-swift-lm `LogitProcessor` that, at each step, masks the model's
// logits to ONLY the tokens the JSON schema's grammar allows in the current state — so the model can't
// emit malformed JSON (unbalanced braces, wrong enum, bad type). It reuses SwiftLM's battle-tested
// `JSONSchemaStateTracker` (CoreML-proven) over MLX `MLXArray` logits via the now-public `validTokens()`/
// `updateState()` surface — no sampler fork, no duplicated grammar.
//
// It's a CLASS (not a struct) on purpose: `TokenIterator` stores the processor by value and mutates its
// own copy (`processor?.didSample(...)`), so a struct's `isComplete` would never be visible to the caller.
// A class shares the reference, so `generateConstrained` can stop the loop the moment the JSON closes.
//
// Vocab alignment: the grammar tokenizer is a `MiniBPE` loaded from the SAME `tokenizer.json` the MLX
// model uses, so its token ids ARE the logit indices — no remapping. The masking logic is unit-tested via
// the tracker's public API (SwiftLMTests); the end-to-end constrained run is the model smoke test.
public final class GrammarLogitProcessor: LogitProcessor, @unchecked Sendable {
    private var tracker: JSONSchemaStateTracker
    private var decoded: [Int] = []
    private let vocabSize: Int

    public init(schema: JSONSchemaConvertible.Type, tokenizer: any GrammarTokenizer,
                runtimeEnums: [String: [String]]? = nil) {
        self.tracker = JSONSchemaStateTracker(schema: schema, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
        self.vocabSize = tokenizer.tokensToIds.count
    }

    /// Runtime-fields variant: constrain to a directly-supplied `[SchemaProperty]` (e.g. a tool's args
    /// built from its MCP inputSchema) — no compile-time `@JSONSchema` type needed.
    public init(fields: [SchemaProperty], tokenizer: any GrammarTokenizer,
                runtimeEnums: [String: [String]]? = nil) {
        self.tracker = JSONSchemaStateTracker(fields: fields, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
        self.vocabSize = tokenizer.tokensToIds.count
    }

    /// True once the grammar has produced a complete JSON value — the caller stops generation here.
    public var isComplete: Bool { tracker.isComplete }

    public func prompt(_ prompt: MLXArray) {}   // the constraint applies to generation only

    public func process(logits: MLXArray) -> MLXArray {
        // Additive mask: 0 for grammar-valid tokens, -inf for the rest; broadcasts over leading [batch] dims.
        // Size to the model's ACTUAL logit dim (a model's vocab can be padded beyond tokenizer.json's count;
        // a mismatched mask length would break the broadcast). Ids past the tokenizer's range stay masked.
        let dim = logits.shape.last ?? vocabSize
        var scalars = [Float](repeating: -Float.greatestFiniteMagnitude, count: dim)
        for v in tracker.validTokens() where v < dim { scalars[v] = 0 }
        return logits + MLXArray(scalars)
    }

    public func didSample(token: MLXArray) {
        let id = Int(token.asType(.int32).item(Int32.self))   // coerce dtype so any sampler's output works
        tracker.updateState(with: id, &decoded)
    }
}

// ── Interleaved constrained tool-call emission: free-text until the model opens a tool call, then mask to
// a valid tool-call JSON, then back to free text. Wraps `GrammarActivationGate` (pure trigger detection,
// unit-tested in Serving) around a `GrammarLogitProcessor`. While `.freeText` it's a passthrough (the
// model reasons/responds normally); the moment the gate sees the trigger (`<tool_call>`) it constrains the
// following JSON to the schema (with `name` ∈ the runtime tool list); when that JSON completes it resets
// to free text for a possible next call. This replaces MLXLMCommon's generate-then-regex tool parse with
// valid-by-construction tool calls.
public final class ConditionalGrammarProcessor: LogitProcessor, @unchecked Sendable {
    private let schema: JSONSchemaConvertible.Type
    private let tokenizer: any GrammarTokenizer
    private let runtimeEnums: [String: [String]]?
    private var gate: GrammarActivationGate
    private var inner: GrammarLogitProcessor

    public init(schema: JSONSchemaConvertible.Type, tokenizer: any GrammarTokenizer,
                runtimeEnums: [String: [String]]? = nil, trigger: String = "<tool_call>") {
        self.schema = schema
        self.tokenizer = tokenizer
        self.runtimeEnums = runtimeEnums
        self.gate = GrammarActivationGate(trigger: trigger)
        self.inner = GrammarLogitProcessor(schema: schema, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
    }

    public var isConstraining: Bool { gate.mode == .constrained }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        gate.mode == .constrained ? inner.process(logits: logits) : logits   // prose is unconstrained
    }

    public func didSample(token: MLXArray) {
        switch gate.mode {
        case .freeText:
            let id = Int(token.asType(.int32).item(Int32.self))
            gate.observe(tokenizer.idsToTokens[id] ?? "")   // the flip takes effect on the next process()
        case .constrained:
            inner.didSample(token: token)
            if inner.isComplete {
                gate.deactivate()
                // fresh grammar so a SECOND tool call later in the same turn is constrained too
                inner = GrammarLogitProcessor(schema: schema, tokenizer: tokenizer, runtimeEnums: runtimeEnums)
            }
        }
    }
}

extension MLXLanguageModel {
    /// Generate text constrained to a JSON schema by grammar — the output is valid-by-construction JSON.
    /// Drives the low-level `TokenIterator` with a `GrammarLogitProcessor` (ChatSession can only wire the
    /// penalty processor). Greedy (ArgMax) since the grammar already removes invalid branches.
    ///
    /// `grammarTokenizer` MUST be a `MiniBPE` loaded from the same `tokenizer.json` the model uses
    /// (`MiniBPE.grammarTokenizer(modelDir:)`), so its ids align with the model's logit indices.
    public func generateConstrained(
        _ prompt: String,
        schema: JSONSchemaConvertible.Type,
        grammarTokenizer: any GrammarTokenizer,
        instructions: String? = nil,
        runtimeEnums: [String: [String]]? = nil,
        maxTokens: Int = 512
    ) async throws -> String {
        try await container.perform { context in
            var messages: [Chat.Message] = []
            if let instructions { messages.append(.system(instructions)) }
            messages.append(.user(prompt))
            let input = try await context.processor.prepare(input: UserInput(chat: messages))

            let processor = GrammarLogitProcessor(schema: schema, tokenizer: grammarTokenizer,
                                                  runtimeEnums: runtimeEnums)
            var iterator = try TokenIterator(
                input: input, model: context.model, cache: nil,
                processor: processor, sampler: ArgMaxSampler(), maxTokens: maxTokens)

            var tokens: [Int] = []
            while let t = iterator.next() {
                tokens.append(t)
                if processor.isComplete {
                    // `next()` lags one token (returns the previous, advances internally), so the closing
                    // token is still pending — flush exactly one more, then stop.
                    if let last = iterator.next() { tokens.append(last) }
                    break
                }
            }
            return context.tokenizer.decode(tokenIds: tokens)
        }
    }

    /// Constrain generation to a RUNTIME `[SchemaProperty]` (e.g. a tool's args from `requiredToolArgFields`)
    /// — same loop as the schema variant, but the fields are known only at runtime.
    public func generateConstrained(
        _ prompt: String,
        fields: [SchemaProperty],
        grammarTokenizer: any GrammarTokenizer,
        instructions: String? = nil,
        runtimeEnums: [String: [String]]? = nil,
        maxTokens: Int = 512
    ) async throws -> String {
        try await container.perform { context in
            var messages: [Chat.Message] = []
            if let instructions { messages.append(.system(instructions)) }
            messages.append(.user(prompt))
            let input = try await context.processor.prepare(input: UserInput(chat: messages))

            let processor = GrammarLogitProcessor(fields: fields, tokenizer: grammarTokenizer,
                                                  runtimeEnums: runtimeEnums)
            var iterator = try TokenIterator(input: input, model: context.model, cache: nil,
                                             processor: processor, sampler: ArgMaxSampler(), maxTokens: maxTokens)
            var tokens: [Int] = []
            while let t = iterator.next() {
                tokens.append(t)
                if processor.isComplete {
                    if let last = iterator.next() { tokens.append(last) }
                    break
                }
            }
            return context.tokenizer.decode(tokenIds: tokens)
        }
    }

    /// INTERLEAVED constrained tool-call generation: `tools` go in the chat template (so the model knows to
    /// emit `<tool_call>`), and a `ConditionalGrammarProcessor` free-texts during prose but forces a valid
    /// tool call (with `name` ∈ `toolNames`, by construction) at the `<tool_call>` span. Returns the full
    /// decoded text (caller extracts/dispatches the call). The end-to-end of #31.
    public func generateToolCallConstrained(
        _ prompt: String,
        tools: [ToolSpec],
        toolNames: [String],
        schema: JSONSchemaConvertible.Type,
        grammarTokenizer: any GrammarTokenizer,
        instructions: String? = nil,
        maxTokens: Int = 256
    ) async throws -> String {
        try await container.perform { context in
            var messages: [Chat.Message] = []
            if let instructions { messages.append(.system(instructions)) }
            messages.append(.user(prompt))
            let input = try await context.processor.prepare(input: UserInput(chat: messages, tools: tools))

            let proc = ConditionalGrammarProcessor(schema: schema, tokenizer: grammarTokenizer,
                                                   runtimeEnums: ["name": toolNames])
            var iterator = try TokenIterator(input: input, model: context.model, cache: nil,
                                             processor: proc, sampler: ArgMaxSampler(), maxTokens: maxTokens)
            var tokens: [Int] = []
            let eos = grammarTokenizer.eosTokenId
            while let t = iterator.next() {
                tokens.append(t)
                if let eos, t == eos { break }
            }
            return context.tokenizer.decode(tokenIds: tokens)
        }
    }

    /// COMPLETE constrained tool call → DISPATCH (two-phase, every part valid by construction):
    ///  1. constrain the tool `name` ∈ the host's real tools (runtime enum),
    ///  2. constrain that tool's args to ITS schema (runtime fields from the tool's MCP inputSchema),
    ///  3. dispatch via the host.
    /// Constrain-from-start (no `<tool_call>` tag needed — Qwen-Coder-MLX emits raw JSON). Returns the
    /// chosen name, the args JSON, and the tool's result. This is the end-to-end of #31.
    public func callToolConstrained(
        _ prompt: String,
        host: MCPHost,
        grammarTokenizer: any GrammarTokenizer,
        nameSchema: JSONSchemaConvertible.Type,
        maxTokens: Int = 128
    ) async throws -> (name: String, argsJSON: String, result: String) {
        let toolNames = await host.toolNames
        let specs = await host.specs

        // Phase 1 — force the NAME to a real tool.
        let nameJSON = try await generateConstrained(
            "\(prompt)\n\nChoose the single most relevant tool.",
            schema: nameSchema, grammarTokenizer: grammarTokenizer,
            instructions: "Respond with ONLY {\"name\": \"<tool>\"}.",
            runtimeEnums: ["name": toolNames], maxTokens: 48)
        let name: String = {
            if let d = nameJSON.data(using: .utf8),
               let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
               let n = o["name"] as? String { return n }
            return toolNames.first ?? ""
        }()

        // Phase 2 — force the ARGS to the chosen tool's schema (skip if it has no required args).
        let argsJSON = try await constrainedToolArgs(for: name, prompt: prompt, specs: specs,
                                                     grammarTokenizer: grammarTokenizer, maxTokens: maxTokens)

        // Phase 3 — dispatch the by-construction-valid call.
        let result = try await host.dispatch(name: name, argumentsJSON: argsJSON)
        return (name, argsJSON, result)
    }

    /// A tool's `parameters` (fn-schema object) looked up from the host specs (empty if not found).
    private func toolParameters(_ name: String, specs: [ToolSpec]) -> [String: any Sendable] {
        specs.first {
            (($0["function"] as? [String: any Sendable])?["name"] as? String) == name
        }.flatMap { ($0["function"] as? [String: any Sendable])?["parameters"] as? [String: any Sendable] } ?? [:]
    }

    /// Regenerate `name`'s arguments constrained to ITS schema (valid-by-construction JSON); `"{}"` when the
    /// tool has no required args. Shared by `callToolConstrained` (Phase 2) and the live `runWithTools` loop.
    public func constrainedToolArgs(
        for name: String, prompt: String, specs: [ToolSpec],
        grammarTokenizer: any GrammarTokenizer, maxTokens: Int = 128
    ) async throws -> String {
        let fields = requiredToolArgFields(parameters: toolParameters(name, specs: specs))
        guard !fields.isEmpty else { return "{}" }
        // The grammar enforces STRUCTURE (the right field names + types); the prompt nudge reduces the
        // residual CONTENT error of a small model re-deriving a free-string value (e.g. dropping a leading
        // slash → "etc/hosts"). Constrained decoding can't make a 7B copy a path correctly — only verbatim.
        let raw = try await generateConstrained(
            "\(prompt)\n\nYou are calling the tool `\(name)`. Produce its arguments.",
            fields: fields, grammarTokenizer: grammarTokenizer,
            instructions: "Respond with ONLY the tool's arguments as a JSON object. Copy file paths, names, "
                + "and literal values from the request EXACTLY — keep leading slashes and full spelling.",
            maxTokens: maxTokens)
        return Self.trimArgStringValues(raw)
    }

    /// Strip a LEADING space from a flat JSON object's string VALUES. Qwen tokenizes many path/word tokens
    /// WITH a leading space (` /etc` is one token), so a CONSTRAINED-generated string value often picks up a
    /// spurious leading space (`" /etc/hosts"`) that breaks the call. ONLY leading spaces are removed — NOT
    /// trailing whitespace and NOT newlines, which are semantically meaningful for free-text / exact-match
    /// args (edit_file `old_string`, write_file `content`, grep `pattern`). Applied only to values this
    /// grammar generated, never to the model's verbatim routed args. Falls through if not a JSON object.
    static func trimArgStringValues(_ jsonObject: String) -> String {
        guard let data = jsonObject.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return jsonObject }
        var fixed: [String: Any] = [:]
        for (k, v) in obj {
            fixed[k] = (v as? String).map { String($0.drop(while: { $0 == " " })) } ?? v
        }
        guard let out = try? JSONSerialization.data(withJSONObject: fixed),
              let s = String(data: out, encoding: .utf8) else { return jsonObject }
        return s
    }

    /// REPAIR a tool call the unconstrained pass already ROUTED (a `name` + the model's own args) into a
    /// valid call, PRESERVING the values the model already chose and fixing only STRUCTURE:
    ///  1. keep the routed name if it's a real tool; else re-pick under a runtime-enum constraint; else keep
    ///     `routedName` so a hallucination surfaces as an honest "unknown tool" error (never silently run a
    ///     wrong real tool);
    ///  2. keep the routed args that match the tool's schema, dropping bogus fields (`{"action":…}`);
    ///  3. constrain-generate ONLY the genuinely-required fields the model OMITTED — seeded with the routed
    ///     args so distinct calls differentiate (no multi-call collapse) and the model can copy its own
    ///     values (leading-space-cleaned).
    /// Unlike `callToolConstrained` this does NOT force a call (the model's decision to call is preserved),
    /// and unlike a blind regeneration it never discards args the model got right — which both fixes the
    /// empirical wrong-args failure (`{"action":"read"}` → `{"path":…}`) AND avoids re-deriving a correct
    /// path into a wrong one. NOTE: the regeneration sees only the original `prompt` (single-tool-round
    /// assumption — the loop caps tools at round 0); multi-round / instruction context is not threaded in.
    public func constrainRoutedToolCall(
        routedName: String, routedArgsJSON: String, prompt: String, specs: [ToolSpec], toolNames: [String],
        grammarTokenizer: any GrammarTokenizer, maxTokens: Int = 128
    ) async throws -> (name: String, argsJSON: String) {
        // 1 — name: keep if real; else repair ONLY an unambiguous near-miss of one specific tool (a
        // case/separator variation or ≤2-char typo of the EMITTED name); else keep routedName so dispatch
        // returns an honest "unknown tool". We deliberately do NOT re-pick "the most relevant tool" from the
        // task prompt — that manufactured a wrong, possibly destructive call when the model emitted a name
        // that is genuinely not a tool (e.g. a hook-injected `session-learner` re-picked into `write_file`).
        var name = routedName
        if !toolNames.contains(name), let match = closestToolName(routedName, in: toolNames) {
            name = match
        }
        // else (unmatched): name stays routedName → dispatch returns "unknown tool" (never a wrong real tool).

        // 2 — keep the routed args that belong to this tool's schema (drop bogus fields the model invented).
        let params = toolParameters(name, specs: specs)
        let schemaProps: Set<String> = {
            guard let p = params["properties"] as? [String: any Sendable] else { return [] }
            return Set(p.keys)
        }()
        var merged: [String: Any] = [:]
        if let d = routedArgsJSON.data(using: .utf8),
           let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            for (k, v) in o where schemaProps.isEmpty || schemaProps.contains(k) { merged[k] = v }
        }

        // 3 — constrain-generate ONLY the required fields the model omitted (seeded with the routed args).
        let missing = requiredToolArgFields(parameters: params).filter { merged[$0.name] == nil }
        if !missing.isEmpty {
            let constrained = try await generateConstrained(
                "\(prompt)\n\nYou are calling the tool `\(name)` with draft arguments \(routedArgsJSON). "
                    + "Produce the corrected JSON arguments.",
                fields: missing, grammarTokenizer: grammarTokenizer,
                instructions: "Respond with ONLY the tool's arguments as a JSON object. Copy file paths, "
                    + "names, and literal values from the request EXACTLY — keep leading slashes and spelling.",
                maxTokens: maxTokens)
            if let d = Self.trimArgStringValues(constrained).data(using: .utf8),
               let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                for (k, v) in o where merged[k] == nil { merged[k] = v }
            }
        }

        let argsJSON = (try? JSONSerialization.data(withJSONObject: merged))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return (name, argsJSON)
    }
}

public extension MiniBPE {
    /// Load a MiniBPE grammar tokenizer from a model directory containing `tokenizer.json` (the same file
    /// the MLX model loads — so the grammar's token ids align with the model's logit indices).
    static func grammarTokenizer(modelDir: URL) throws -> MiniBPE {
        try MiniBPE(tokenizerJSON: modelDir.appending(path: "tokenizer.json"))
    }

    /// The HuggingFace hub snapshot directory for a model id
    /// (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/`) — the snapshot containing
    /// `tokenizer.json`, else any snapshot, else nil. Shared by the grammar-tokenizer loader and the
    /// model-family detector (which reads `config.json` from the same dir). No network.
    static func snapshotDir(forModelId modelId: String) -> URL? {
        let hub = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub")
        let snapshots = hub
            .appending(path: "models--" + modelId.replacingOccurrences(of: "/", with: "--"))
            .appending(path: "snapshots")
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)) ?? []
        return entries.first(where: {
            FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path)
        }) ?? entries.first
    }

    /// Load a MiniBPE grammar tokenizer for an mlx-community model id from its cached `tokenizer.json`.
    static func grammarTokenizer(forModelId modelId: String) throws -> MiniBPE {
        guard let dir = snapshotDir(forModelId: modelId) else {
            throw NSError(domain: "SwiftLM", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "tokenizer.json not found for \(modelId) in HF cache"])
        }
        return try MiniBPE(tokenizerJSON: dir.appending(path: "tokenizer.json"))
    }
}

/// Build `[SchemaProperty]` for a tool's REQUIRED args from its MCP `parameters` (fn-schema) object, using
/// primitive metatypes (`String.self`/`Int.self`/`Double.self`/`Bool.self`). Object/array-typed args are
/// skipped (v1 = flat args), in the schema's `required` order so the grammar forces them deterministically.
/// This is what lets a constrained tool call's ARGS match the chosen tool's schema, fixing the wrong-args
/// failure (the model emitted `{"action":"read"}` instead of `{"path":…}`).
///
/// ONLY genuinely-required args are forced. A tool with no `required` array (all-optional args, e.g.
/// `current_time`/`uuid`) yields `[]` ⇒ a `{}` call that uses the tool's own defaults — instead of forcing
/// the model to ramble values into optional fields (which, with no enum bound, produced 200-char garbage).
public func requiredToolArgFields(parameters: [String: any Sendable]) -> [SchemaProperty] {
    guard let props = parameters["properties"] as? [String: any Sendable] else { return [] }
    let required = (parameters["required"] as? [String]) ?? []
    func conformer(_ t: String) -> (any JSONSchemaConvertible.Type)? {
        switch t {
        case "string":  return String.self
        case "integer": return Int.self
        case "number":  return Double.self
        case "boolean": return Bool.self
        default:        return nil   // object/array — flat args only in v1
        }
    }
    return required.compactMap { name in
        guard let s = props[name] as? [String: any Sendable], let t = s["type"] as? String,
              let c = conformer(t) else { return nil }
        return SchemaProperty(name, c)
    }
}
