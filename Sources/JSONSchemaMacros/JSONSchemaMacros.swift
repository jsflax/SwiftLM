import SwiftSyntaxMacros
import SwiftCompilerPlugin
import SwiftSyntax
import Foundation

private struct MemberView {
    let name: String
    let type: String
    let isStatic: Bool
    let isOptional: Bool
    var attributeKey: String?
    var assignment: String?
    var isComputed: Bool
    var guideDescription: String?
    var guideConstraints: [String] = []  // Raw constraint expressions like ".maxLength(50)"

    /// Whether this property has .skip constraint and should be excluded from generation
    var shouldSkip: Bool {
        guideConstraints.contains(".skip")
    }
}

/// Extract the parent type path from the lexical context (excluding the current type being expanded)
/// For `Itinerary.Day`, this returns `"Itinerary"` (not `"Itinerary.Day"`)
private func getTypeQualifier(from context: some MacroExpansionContext) -> String {
    var components: [String] = []
    // lexicalContext goes from innermost to outermost
    // The first item IS the parent (not the current declaration)
    for lexicalContext in context.lexicalContext {
        if let structDecl = lexicalContext.as(StructDeclSyntax.self) {
            components.append(structDecl.name.text)
        } else if let classDecl = lexicalContext.as(ClassDeclSyntax.self) {
            components.append(classDecl.name.text)
        } else if let enumDecl = lexicalContext.as(EnumDeclSyntax.self) {
            components.append(enumDecl.name.text)
        } else if let actorDecl = lexicalContext.as(ActorDeclSyntax.self) {
            components.append(actorDecl.name.text)
        }
    }
    // Reverse because lexicalContext goes from innermost to outermost
    return components.reversed().joined(separator: ".")
}

/// Get sibling type names from the parent's lexical context
private func getSiblingTypeNames(from context: some MacroExpansionContext) -> Set<String> {
    var siblings: Set<String> = []
    // lexicalContext goes from innermost to outermost
    // The first item should be the parent that contains siblings
    for lexicalContext in context.lexicalContext {
        if let structDecl = lexicalContext.as(StructDeclSyntax.self) {
            // Found a parent struct, get its nested type declarations
            for member in structDecl.memberBlock.members {
                if let nestedStruct = member.decl.as(StructDeclSyntax.self) {
                    siblings.insert(nestedStruct.name.text)
                } else if let nestedClass = member.decl.as(ClassDeclSyntax.self) {
                    siblings.insert(nestedClass.name.text)
                } else if let nestedEnum = member.decl.as(EnumDeclSyntax.self) {
                    siblings.insert(nestedEnum.name.text)
                }
            }
            break // Only look at immediate parent
        } else if let classDecl = lexicalContext.as(ClassDeclSyntax.self) {
            for member in classDecl.memberBlock.members {
                if let nestedStruct = member.decl.as(StructDeclSyntax.self) {
                    siblings.insert(nestedStruct.name.text)
                } else if let nestedClass = member.decl.as(ClassDeclSyntax.self) {
                    siblings.insert(nestedClass.name.text)
                } else if let nestedEnum = member.decl.as(EnumDeclSyntax.self) {
                    siblings.insert(nestedEnum.name.text)
                }
            }
            break
        }
    }
    return siblings
}

/// Qualify a type name with the parent path if it's a sibling nested type
private func qualifyType(_ typeName: String, qualifier: String, siblings: Set<String>) -> String {
    // Extract the base type name (handle arrays, optionals, dictionaries)
    let baseName = extractBaseTypeName(typeName)

    // Debug: print what we're working with
    // print("DEBUG qualifyType: typeName=\(typeName), baseName=\(baseName), qualifier=\(qualifier), siblings=\(siblings)")

    // If it's a sibling type and not already qualified, add the qualifier
    if siblings.contains(baseName) && !typeName.contains(".") && !qualifier.isEmpty {
        // Replace the base name with qualified version
        return typeName.replacingOccurrences(of: baseName, with: "\(qualifier).\(baseName)")
    }
    return typeName
}

/// Extract the innermost type name from complex types like [Foo], Foo?, [String: Foo]
private func extractBaseTypeName(_ typeName: String) -> String {
    var name = typeName
    // Remove array brackets
    if name.hasPrefix("[") && name.hasSuffix("]") {
        name = String(name.dropFirst().dropLast())
        // Handle dictionary [Key: Value]
        if let colonIndex = name.firstIndex(of: ":") {
            name = String(name[name.index(after: colonIndex)...]).trimmingCharacters(in: .whitespaces)
        }
    }
    // Remove optional
    if name.hasSuffix("?") || name.hasSuffix("!") {
        name = String(name.dropLast())
    }
    return name
}

private func view(for member: MemberBlockItemListSyntax.Element) throws -> [MemberView] {
    guard let decl = member.decl.as(VariableDeclSyntax.self) else {
        return []
    }
    var members: [MemberView] = []
    for binding in decl.bindings.compactMap({
        $0.pattern.as(IdentifierPatternSyntax.self)
    }) {
        guard let type = decl.bindings.compactMap({
            $0.typeAnnotation?.type
        }).first,
        !(type.syntaxNodeType is StructDeclSyntax.Type) else { continue }

        // Get the fully qualified type string, handling nested/namespaced types
        let typeString = type.trimmedDescription

        var isComputed = false
        // Check if this property has an accessorBlock (i.e. it's computed)
        if let accessorBlock = decl.bindings.first?.accessorBlock {
            switch accessorBlock.accessors {
            case .accessors(let accessors):
                // Only treat as computed if it has a get/set/didSet/willSet with a body
                // Skip if it's just access level modifiers like private(set)
                let hasComputedAccessor = accessors.contains(where: { accessor in
                    let specifier = accessor.accessorSpecifier.text
                    return (specifier == "get" || specifier == "set" || specifier == "didSet" || specifier == "willSet")
                        && accessor.body != nil
                })
                if hasComputedAccessor {
                    isComputed = true
                }
            case .getter(let getter):
                isComputed = true
            }
        }
        
        var memberView = MemberView(name: "\(binding.identifier.trimmedDescription)",
                                    type: typeString,
                                    isStatic: decl.modifiers.contains(where: { $0.name.tokenKind == .keyword(.static)}),
                                    isOptional: type.syntaxNodeType is OptionalTypeSyntax.Type,
                                    attributeKey: nil,
                                    isComputed: isComputed)

        // Check for @SchemaGuide attribute (renamed from Guide to avoid FoundationModels conflict)
        for attr in decl.attributes {
            if let attrSyntax = attr.as(AttributeSyntax.self),
               attrSyntax.attributeName.trimmedDescription == "SchemaGuide" {
                // Parse SchemaGuide arguments
                if let args = attrSyntax.arguments?.as(LabeledExprListSyntax.self) {
                    for arg in args {
                        // Check for description (first string literal or labeled description:)
                        if let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self) {
                            memberView.guideDescription = stringLiteral.segments.trimmedDescription
                        } else {
                            // Constraint argument (like .maxLength(50), .range(1...10))
                            memberView.guideConstraints.append(arg.expression.trimmedDescription)
                        }
                    }
                }
            }
        }

        if let macroName = decl.attributes.first?.as(AttributeSyntax.self)?
            .arguments?.as(LabeledExprListSyntax.self)?.first?.expression.as(StringLiteralExprSyntax.self) {
            memberView.attributeKey = "\(macroName.segments)"
        }
        if let assignment = decl.bindings.compactMap({
            $0.initializer?.value
        }).first {
            memberView.assignment = "\(assignment)"
        }
        members.append(memberView)
    }
    return members
}

class JSONSchemaMacro: ExtensionMacro, MemberMacro, FreestandingMacro {
    struct MacroError: Error {
        let error: String
        init(_ error: String) {
            self.error = error
        }
    }
    // Add to JSONSchemaMacro class
    static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        // Parse arguments: first arg is the type, rest are (name, type) tuples
        guard let arguments = node.argumentList.as(LabeledExprListSyntax.self),
              let firstArg = arguments.first else {
            throw MacroError("Expected type argument")
        }

        // Get the target type (e.g., MKCoordinateRegion.self)
        let targetType: String
        if let memberAccess = firstArg.expression.as(MemberAccessExprSyntax.self),
           memberAccess.declName.baseName.text == "self" {
            targetType = memberAccess.base?.trimmedDescription ?? ""
        } else {
            targetType = firstArg.expression.description
        }

        // Parse property tuples: ("name", Type.self)
        var properties: [(name: String, type: String)] = []
        for arg in arguments.dropFirst() {
            if let tuple = arg.expression.as(TupleExprSyntax.self),
               tuple.elements.count == 2 {
                let elements = Array(tuple.elements)

                // First element is the property name (string literal)
                guard let nameLiteral = elements[0].expression.as(StringLiteralExprSyntax.self) else {
                    continue
                }
                let name = nameLiteral.segments.description

                // Second element is the type (e.g., Double.self)
                let typeExpr = elements[1].expression
                let type: String
                if let memberAccess = typeExpr.as(MemberAccessExprSyntax.self),
                   memberAccess.declName.baseName.text == "self" {
                    type = memberAccess.base?.trimmedDescription ?? "String"
                } else {
                    type = typeExpr.description
                }

                properties.append((name: name, type: type))
            }
        }

        let propertiesCode = properties.map { prop in
            """
            ("\(prop.name)", _GrammarJSONSchema.Property(type: \(prop.type).jsonSchema["type"] as! String, items: \(prop.type).jsonSchema["items"] as? _GrammarJSONSchema.Items, description: ""))
            """
        }.joined(separator: ",\n")

        let jsonSchemaProperties = properties.map { prop in
            """
            "\(prop.name)": \(prop.type).jsonSchema
            """
        }.joined(separator: ",\n")

        let initFromJsonCode = properties.map { prop in
            """
            let \(prop.name) = try \(prop.type)(from: json["\(prop.name)"]!)
            """
        }.joined(separator: "\n")

        // Generate the extension
        return [
            """
            extension \(raw: targetType): _GrammarJSONSchemaConvertible {
                public static var type: String { "object" }
                
                public static var _swiftLMJsonSchemaProperties: [(String, _GrammarJSONSchema.Property)]? {
                    [\(raw: propertiesCode)]
                }
                
                public static var jsonSchema: [String: Any] {
                    [
                        "type": "object",
                        "properties": [
                            \(raw: jsonSchemaProperties)
                        ]
                    ]
                }
                
                public init(from json: Any) throws {
                    guard let json = json as? [String: Any] else {
                        throw JSONDecodingError.invalidType
                    }
                    \(raw: initFromJsonCode)
                    // You'll need to customize this initializer based on the actual type
                    fatalError("Custom initializer needed for \\(Self.self)")
                }
            }
            """
        ]
    }

    
    static func expansion(of node: AttributeSyntax, providingMembersOf declaration: some DeclGroupSyntax, conformingTo protocols: [TypeSyntax], in context: some MacroExpansionContext) throws -> [DeclSyntax] {
        // Get type qualifier for nested types
        let qualifier = getTypeQualifier(from: context)
        let siblings = getSiblingTypeNames(from: context)

        let members = try declaration.memberBlock.members.flatMap(view(for:)).filter {
            !$0.isStatic && !$0.isComputed
        }

        // Qualify member types that reference sibling nested types
        let qualifiedMembers = members.map { member -> MemberView in
            MemberView(
                name: member.name,
                type: qualifyType(member.type, qualifier: qualifier, siblings: siblings),
                isStatic: member.isStatic,
                isOptional: member.isOptional,
                attributeKey: member.attributeKey,
                assignment: member.assignment,
                isComputed: member.isComputed,
                guideDescription: member.guideDescription,
                guideConstraints: member.guideConstraints
            )
        }

        // Handle enums separately
        if let enumDecl = declaration.as(EnumDeclSyntax.self) {
            // Extract case names from the enum
            let caseNames = enumDecl.memberBlock.members.compactMap { member -> [String]? in
                guard let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) else {
                    return nil
                }
                return caseDecl.elements.map { $0.name.text }
            }.flatMap { $0 }

            let casesString = caseNames.map { "\"\($0)\"" }.joined(separator: ", ")

            return [
                // FoundationModels Generable support for enums
                """
                @available(iOS 26.0, macOS 26.0, *)
                public static var generationSchema: FoundationModels.GenerationSchema {
                    FoundationModels.GenerationSchema(type: Self.self, anyOf: [\(raw: casesString)])
                }
                """,
                """
                @available(iOS 26.0, macOS 26.0, *)
                public init(_ content: FoundationModels.GeneratedContent) throws {
                    let rawValue: String = try content.value()
                    guard let value = Self(rawValue: rawValue as! RawValue) else {
                        throw JSONDecodingError.invalidValue
                    }
                    self = value
                }
                """,
                """
                @available(iOS 26.0, macOS 26.0, *)
                public var generatedContent: FoundationModels.GeneratedContent {
                    FoundationModels.GeneratedContent(rawValue as! String)
                }
                """
            ]
        }
        // Filter out skipped members for generation schema
        let generatedMembers = qualifiedMembers.filter { !$0.shouldSkip }

        // Generate GenerationSchema.Property entries for Generable (only non-skipped)
        // Include description and guides from @SchemaGuide attributes
        let generationSchemaProperties = generatedMembers.map { member -> String in
            var args = "name: \"\(member.name)\""

            // Add description if present
            if let desc = member.guideDescription {
                args += ", description: \"\(desc)\""
            }

            // Add type
            args += ", type: \(member.type).self"

            // Map GuideConstraints to FoundationModels GenerationGuide
            // Supported: .range for Int/Double, .count for arrays, .anyOf for strings
            let guides = member.guideConstraints.compactMap { constraint -> String? in
                // The constraint is stored as a raw string like ".range(0...150)"
                // We need to convert it to GenerationGuide format
                if constraint.hasPrefix(".range(") {
                    // .range works for both Int and Double in FoundationModels
                    return "FoundationModels.GenerationGuide\(constraint)"
                } else if constraint.hasPrefix(".count(") {
                    // .count for arrays
                    return "FoundationModels.GenerationGuide\(constraint)"
                } else if constraint.hasPrefix(".anyOf(") {
                    // .anyOf for strings
                    return "FoundationModels.GenerationGuide\(constraint)"
                } else if constraint.hasPrefix(".doubleRange(") {
                    // Map .doubleRange to .range for FoundationModels
                    let rangeContent = constraint.dropFirst(".doubleRange(".count).dropLast()
                    return "FoundationModels.GenerationGuide.range(\(rangeContent))"
                }
                // .maxLength and .skip are not directly supported by FoundationModels
                return nil
            }

            if !guides.isEmpty {
                args += ", guides: [\(guides.joined(separator: ", "))]"
            }

            return ".init(\(args))"
        }.joined(separator: ",\n                ")

        // Generate init from GeneratedContent (handle skipped members with defaults)
        let initFromGeneratedContent = qualifiedMembers.map { member -> String in
            if member.shouldSkip {
                // Use default value for skipped properties
                if let assignment = member.assignment {
                    return "self.\(member.name) = \(assignment)"
                } else if member.isOptional {
                    return "self.\(member.name) = nil"
                } else {
                    // No default available - use type's init()
                    return "self.\(member.name) = \(member.type)()"
                }
            } else {
                return "self.\(member.name) = try content.value(forProperty: \"\(member.name)\")"
            }
        }.joined(separator: "\n            ")

        // Generate generatedContent property (only include non-skipped)
        let generatedContentProperties = generatedMembers.map {
            """
            "\($0.name)": \($0.name)
            """
        }.joined(separator: ",\n                ")

        return [
            """
            enum CodingKeys: CodingKey {
                case \(raw: qualifiedMembers.map(\.name).joined(separator: ", "))
            }
            """,
            """
            public init(from decoder: Swift.Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                \(raw: qualifiedMembers.map { member -> String in
                    if member.shouldSkip {
                        // Skipped properties: decode if present, otherwise use default
                        // For optional types, strip the ? since decodeIfPresent already returns Optional
                        let decodeType = member.isOptional ? String(member.type.dropLast()) : member.type
                        if let assignment = member.assignment {
                            return "self.\(member.name) = try container.decodeIfPresent(\(decodeType).self, forKey: .\(member.name)) ?? \(assignment)"
                        } else if member.isOptional {
                            return "self.\(member.name) = try container.decodeIfPresent(\(decodeType).self, forKey: .\(member.name))"
                        } else {
                            return "self.\(member.name) = try container.decodeIfPresent(\(decodeType).self, forKey: .\(member.name)) ?? \(member.type)()"
                        }
                    } else {
                        return "self.\(member.name) = try \(member.type).decode(from: container, forKey: .\(member.name))"
                    }
                }.joined(separator: "\n"))
            }
            """,
            """
            public func encode(to encoder: Swift.Encoder) throws {
                var container = try encoder.container(keyedBy: CodingKeys.self)
                \(raw: qualifiedMembers.map {
                    """
                    try container.encode(\($0.name), forKey: .\($0.name))
                    """
                }.joined(separator: "\n"))
            }
            """,
            """
            public init(\(raw: qualifiedMembers.filter { $0.assignment == nil }.map {
                var str = "\($0.name): \($0.type)"
                if let assignment = $0.assignment {
                    str += " = \(assignment)"
                } else if $0.isOptional {
                    str += " = nil"
                }

                return str
            }.joined(separator: ",\n"))) {
            \(raw: qualifiedMembers.filter { $0.assignment == nil }.map {
                """
                self.\($0.name) = \($0.name)
                """
            }.joined(separator: "\n"))
            }
            """,
            """
            public init(from json: Any) throws {
                guard let json = json as? [String: Any] else {
                    throw JSONDecodingError.invalidType
                }
                \(raw: qualifiedMembers.map { member -> String in
                    if member.shouldSkip {
                        // Use default value for skipped properties
                        if let assignment = member.assignment {
                            return "self.\(member.name) = \(assignment)"
                        } else if member.isOptional {
                            return "self.\(member.name) = nil"
                        } else {
                            return "self.\(member.name) = \(member.type)()"
                        }
                    } else {
                        return """
                        guard let value = json["\(member.name)"] else {
                            throw JSONDecodingError.missingRequiredProperty("\(member.name)")
                        }
                        self.\(member.name) = try \(member.type)(from: value)
                        """
                    }
                }.joined(separator: "\n"))
            }
            """,
            // FoundationModels Generable support
            """
            @available(iOS 26.0, macOS 26.0, *)
            public static var generationSchema: FoundationModels.GenerationSchema {
                FoundationModels.GenerationSchema(type: Self.self, properties: [
                    \(raw: generationSchemaProperties)
                ])
            }
            """,
            """
            @available(iOS 26.0, macOS 26.0, *)
            public init(_ content: FoundationModels.GeneratedContent) throws {
                \(raw: initFromGeneratedContent)
            }
            """,
            """
            @available(iOS 26.0, macOS 26.0, *)
            public var generatedContent: FoundationModels.GeneratedContent {
                FoundationModels.GeneratedContent(properties: [
                    \(raw: generatedContentProperties)
                ])
            }
            """
        ]
    }
    
    static func expansion(of node: SwiftSyntax.AttributeSyntax,
                          attachedTo declaration: some SwiftSyntax.DeclGroupSyntax,
                          providingExtensionsOf type: some SwiftSyntax.TypeSyntaxProtocol,
                          conformingTo protocols: [SwiftSyntax.TypeSyntax],
                          in context: some SwiftSyntaxMacros.MacroExpansionContext) throws -> [SwiftSyntax.ExtensionDeclSyntax] {
        // Get the full type path from the `type` parameter (e.g., "Itinerary.Day")
        let fullTypePath = type.trimmedDescription
        // Extract the parent qualifier (e.g., "Itinerary" from "Itinerary.Day")
        let qualifier: String
        if let lastDotIndex = fullTypePath.lastIndex(of: ".") {
            qualifier = String(fullTypePath[..<lastDotIndex])
        } else {
            qualifier = ""
        }

        // Get sibling type names: nested types declared in the current declaration's parent
        // Try lexical context first, fall back to checking nested types in declaration
        var siblings = getSiblingTypeNames(from: context)
        // Also add nested types from the current declaration (for self-references)
        for member in declaration.memberBlock.members {
            if let nestedStruct = member.decl.as(StructDeclSyntax.self) {
                siblings.insert(nestedStruct.name.text)
            } else if let nestedClass = member.decl.as(ClassDeclSyntax.self) {
                siblings.insert(nestedClass.name.text)
            } else if let nestedEnum = member.decl.as(EnumDeclSyntax.self) {
                siblings.insert(nestedEnum.name.text)
            }
        }

        let members = try declaration.memberBlock.members.flatMap(view(for:))
            .filter {
                !$0.isStatic && !$0.isComputed
            }

        // Qualify member types that reference sibling nested types
        let qualifiedMembers = members.map { member -> MemberView in
            MemberView(
                name: member.name,
                type: qualifyType(member.type, qualifier: qualifier, siblings: siblings),
                isStatic: member.isStatic,
                isOptional: member.isOptional,
                attributeKey: member.attributeKey,
                assignment: member.assignment,
                isComputed: member.isComputed,
                guideDescription: member.guideDescription,
                guideConstraints: member.guideConstraints
            )
        }

        var inheritedTypes: [InheritedTypeSyntax] = []
        inheritedTypes.append(InheritedTypeSyntax(type: TypeSyntax("_GrammarJSONSchemaConvertible")))
        if declaration is EnumDeclSyntax {
            inheritedTypes.append(InheritedTypeSyntax(type: TypeSyntax(", CaseIterable")))
        }

        // Filter out skipped members for schema/grammar generation
        let generatableMembers = qualifiedMembers.filter { !$0.shouldSkip }

        let properties = generatableMembers.map {
            """
            "\($0.name)": \($0.type).jsonSchema
            """
        }
        if !(declaration is EnumDeclSyntax) {
            // Generate schemaProperties with metatype references, descriptions, and constraints (only non-skipped)
            let schemaPropertiesCode = generatableMembers.map { member -> String in
                var args = "\"\(member.name)\", \(member.type).self"
                if let desc = member.guideDescription {
                    args += ", description: \"\(desc)\""
                }
                if !member.guideConstraints.isEmpty {
                    let constraints = member.guideConstraints.joined(separator: ", ")
                    args += ", constraints: [\(constraints)]"
                }
                return "_GrammarSchemaProperty(\(args))"
            }.joined(separator: ",\n                        ")

            // JSONSchemaConvertible extension
            let jsonSchemaExtension = ExtensionDeclSyntax(
                extendedType: type,
                inheritanceClause: .init(inheritedTypes: .init(inheritedTypes)),
                memberBlock: """
                {
                    public static var type: String {
                        "object"
                    }
                    public static var schemaProperties: [_GrammarSchemaProperty]? {
                        [\(raw: schemaPropertiesCode)]
                    }
                    public static var jsonSchema: [String: Any] {
                        [
                            "type": "object",
                            "properties": [
                                \(raw: properties.joined(separator: ","))
                            ]
                        ]
                    }
                    public static var _swiftLMJsonSchemaProperties: [(String, _GrammarJSONSchema.Property)]? {
                        [\(raw: generatableMembers.map {
                            """
                            ("\($0.name)", _GrammarJSONSchema.Property(type: \($0.type).type, items: \($0.type).items, description: ""))
                            """
                        }.joined(separator: ","))]
                    }
                }
                """)

            // Generable conformance extension for structs (availability-gated)
            let generableExtension = try ExtensionDeclSyntax(
                """
                @available(iOS 26.0, macOS 26.0, *)
                extension \(type): FoundationModels.Generable {}
                """
            )

            return [jsonSchemaExtension, generableExtension]
        } else {
            // Enum case - add JSONSchemaConvertible extension
            let jsonSchemaExtension = ExtensionDeclSyntax(
                extendedType: type,
                inheritanceClause: .init(inheritedTypes: .init(inheritedTypes)),
                memberBlock: """
                {
                    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
                        if RawValue.self is Int.Type {
                            return Self(rawValue: Int(try container.decode(String.self, forKey: key)) as! Self.RawValue)!
                        } else {
                            return try container.decode(Self.self, forKey: key)
                        }
                    }

                    public static var _swiftLMJsonSchemaProperties: [(String, _GrammarJSONSchema.Property)]? {
                        nil
                    }
                }
                """)

            // Generable conformance extension for enums (availability-gated)
            let generableExtension = try ExtensionDeclSyntax(
                """
                @available(iOS 26.0, macOS 26.0, *)
                extension \(type): FoundationModels.Generable {}
                """
            )

            return [jsonSchemaExtension, generableExtension, try ExtensionDeclSyntax("""
            extension \(type): JSONSchemaKey {}
            """)]
        }
    }
}

/// Macro that bridges external types to JSONSchemaConvertible via a marker struct.
///
/// Usage:
/// ```swift
/// @JSONSchemaBridge(
///     for: MKCoordinateRegion.self,
///     ("latitude", Double.self),
///     ("longitude", Double.self)
/// )
/// private struct _CoordinateSchema {}
/// ```
struct JSONSchemaBridgeMacro: ExtensionMacro {
    static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        // Parse the macro arguments
        guard let arguments = node.arguments?.as(LabeledExprListSyntax.self) else {
            throw JSONSchemaMacro.MacroError("@JSONSchemaBridge requires arguments")
        }

        // First argument should be `for: SomeType.self`
        guard let firstArg = arguments.first,
              firstArg.label?.text == "for" else {
            throw JSONSchemaMacro.MacroError("@JSONSchemaBridge requires 'for:' argument specifying the target type")
        }

        // Extract the target type from `SomeType.self`
        let targetType: String
        if let memberAccess = firstArg.expression.as(MemberAccessExprSyntax.self),
           memberAccess.declName.baseName.text == "self",
           let base = memberAccess.base {
            targetType = base.trimmedDescription
        } else {
            targetType = firstArg.expression.trimmedDescription
        }

        // Parse remaining arguments as property tuples
        var properties: [(name: String, type: String)] = []

        for arg in arguments.dropFirst() {
            // Each argument should be a tuple like ("name", Type.self)
            if let tuple = arg.expression.as(TupleExprSyntax.self),
               tuple.elements.count == 2 {
                let elements = Array(tuple.elements)

                // First element: property name (string literal)
                guard let nameLiteral = elements[0].expression.as(StringLiteralExprSyntax.self) else {
                    continue
                }
                let name = nameLiteral.segments.trimmedDescription

                // Second element: type (e.g., Double.self)
                let typeExpr = elements[1].expression
                let type: String
                if let memberAccess = typeExpr.as(MemberAccessExprSyntax.self),
                   memberAccess.declName.baseName.text == "self",
                   let base = memberAccess.base {
                    type = base.trimmedDescription
                } else {
                    type = typeExpr.trimmedDescription
                }

                properties.append((name: name, type: type))
            }
        }

        guard !properties.isEmpty else {
            throw JSONSchemaMacro.MacroError("@JSONSchemaBridge requires at least one property tuple")
        }

        // Generate the properties array code
        let propertiesCode = properties.map { prop in
            """
            ("\(prop.name)", _GrammarJSONSchema.Property(type: \(prop.type).jsonSchema["type"] as! String, items: \(prop.type).jsonSchema["items"] as? _GrammarJSONSchema.Items, description: ""))
            """
        }.joined(separator: ",\n            ")

        // Generate the jsonSchema properties dict
        let jsonSchemaProperties = properties.map { prop in
            """
            "\(prop.name)": \(prop.type).jsonSchema
            """
        }.joined(separator: ",\n                ")

        // Generate init(from json:) body - extracts values but requires manual mapping
        let initFromJsonCode = properties.map { prop in
            """
            guard let \(prop.name)Value = json["\(prop.name)"] else {
                        throw JSONDecodingError.missingRequiredProperty("\(prop.name)")
                    }
                    let _\(prop.name) = try \(prop.type)(from: \(prop.name)Value)
            """
        }.joined(separator: "\n        ")

        let propertyNames = properties.map { "_\($0.name)" }.joined(separator: ", ")

        // Generate the extension on the TARGET type (not the marker struct)
        let extensionDecl = try ExtensionDeclSyntax(
            """
            extension \(raw: targetType): _GrammarJSONSchemaConvertible {
                public static var type: String { "object" }

                public static var _swiftLMJsonSchemaProperties: [(String, JSONSchema.Property)]? {
                    [\(raw: propertiesCode)]
                }

                public static var jsonSchema: [String: Any] {
                    [
                        "type": "object",
                        "properties": [
                            \(raw: jsonSchemaProperties)
                        ]
                    ]
                }

                public init(from json: Any) throws {
                    guard let json = json as? [String: Any] else {
                        throw JSONDecodingError.invalidType
                    }
                    \(raw: initFromJsonCode)
                    // TODO: Implement custom initialization using: \(raw: propertyNames)
                    fatalError("@JSONSchemaBridge: Implement custom initializer for \\(Self.self)")
                }
            }
            """
        )

        return [extensionDecl]
    }
}

/// A peer macro that does nothing at runtime.
/// It exists solely to allow @SchemaGuide annotations on properties within @JSONSchema structs.
/// The JSONSchema macro reads these annotations to extract descriptions and constraints.
struct SchemaGuideMacro: PeerMacro {
    static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        // This macro produces no code - it's just a marker that JSONSchema reads
        return []
    }
}

@main
struct JSONSchemaMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        JSONSchemaMacro.self,
        JSONSchemaBridgeMacro.self,
        SchemaGuideMacro.self
    ]
}
