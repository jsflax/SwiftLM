import Foundation
@_exported import FoundationModels

public enum JSONType: String, Codable {
    case string, integer, number, boolean, array, object
}

// MARK: - Guide Constraints (similar to Apple's @Guide)

/// Constraints that can be applied to guide generation
public enum GuideConstraint: Sendable, Equatable {
    /// String length constraint
    case maxLength(Int)
    /// Numeric range constraint
    case range(ClosedRange<Int>)
    /// Double range constraint
    case doubleRange(ClosedRange<Double>)
    /// Array count constraint
    case count(ClosedRange<Int>)
    /// Enum-like options
    case anyOf([String])
    /// Skip generation - use default value instead
    case skip
}

// Note: SchemaGuide is now defined as a peer macro at the bottom of this file.
// It allows annotating properties within @JSONSchema structs with descriptions and constraints.
// Usage: @SchemaGuide("Description", .range(0...100))

/// A special type that indicates this field should be skipped during generation
/// The LLM won't generate a value - the default will be used instead
/// Similar to Apple's GenerationID
public struct GenerationID: Codable, Sendable, Hashable, JSONSchemaConvertible {
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        .init(type: String.self, properties: [])
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
    
    public let value: String

    public init() {
        self.value = UUID().uuidString
    }

    public init(_ value: String) {
        self.value = value
    }

    // JSONSchemaConvertible conformance - but marked as skip
    public static var type: String { "string" }
    public static var shouldSkipGeneration: Bool { true }
    public static var properties: [(String, JSONSchema.Property)]? { nil }
    public static var jsonSchema: [String: Any] { ["type": "string"] }

    public init(from json: Any) throws {
        // This shouldn't be called since we skip generation
        self.value = UUID().uuidString
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self.value = str
        } else {
            self.value = UUID().uuidString
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

public struct JSONSchema : Codable {
    public struct Items : Sendable, Codable {
        public let type: String
        public let `enum`: [String]?
        public let properties: [String: Property]?
        
        public init(type: String, `enum`: [String]?, properties: [String: Property]?) {
            self.type = type
            self.enum = `enum`
            self.properties = properties
        }
    }
    public struct Property : Sendable, Codable {
        public let type: String
        public let items: Items?
        public let description: String?
        
        public init(type: String, items: Items?, description: String?) {
            self.type = type
            self.items = items
            self.description = description
        }
    }
    public let type: JSONType
    public let items: Items?
    public let properties: [String : Property]?
    
    public init(type: String, items: Items?, properties: [String : Property]?) {
        self.type = JSONType(rawValue: type)!
        self.items = items
        self.properties = properties
    }
}


public struct _JSONFunctionSchema: Sendable, Codable {
    public typealias Items = JSONSchema.Items

    public struct Property: Sendable, Codable {
        public let type: String
        public let items: Items?
        public let `enum`: [String]?
        public let description: String?
        
        enum CodingKeys: CodingKey {
            case type
            case items
            case `enum`
            case description
        }
        
        public init(type: String.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: Int.Type, description: String?) {
            self.type = "integer"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        
        public init(type: Double.Type, description: String?) {
            self.type = "number"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        public init(type: Float.Type, description: String?) {
            self.type = "number"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        public init(type: Bool.Type, description: String?) {
            self.type = "boolean"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: UUID.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init(type: Date.Type, description: String?) {
            self.type = "string"
            self.description = description
            self.items = nil
            self.enum = nil
        }
        
        public init<T: JSONSchemaConvertible>(type: Array<T>.Type, description: String?) {
            self.type = "array"
            self.description = description
            self.items = Array<T>.items
            self.enum = nil
        }
        public init<T: JSONSchemaConvertible>(type: T.Type, description: String?) {
            self.type = "object"
            self.description = description
            self.items = T.items
            self.enum = nil
        }
        public init<T: CaseIterable>(type: T.Type, description: String?) where T: RawRepresentable,
        T: StringProtocol {
            self.type = "string"
            self.enum = Array(type.allCases.map { $0.rawValue as! String })
            self.description = description
            self.items = nil
        }
    }
    
    
    public struct Parameters: Sendable, Codable {
        public let properties: [String: Property]
        public let required: [String]
        public var type = "dict"
        
        public init(properties: [String : Property], required: [String]) {
            self.properties = properties
            self.required = required
        }
    }
    
    public let name: String
    public let description: String
    public let parameters: Parameters
    
    public init(name: String, description: String, parameters: Parameters) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public enum JSONDecodingError: Error, CustomStringConvertible {
    case invalidType
    case missingRequiredProperty(String)
    case invalidValue

    public var description: String {
        switch self {
        case .invalidType:
            "Invalid type"
        case .missingRequiredProperty(let string):
            "Missing required property: \(string)"
        case .invalidValue:
            "Invalid value"
        }
    }
}

// MARK: - New Schema Types with Metatype References

/// A property in a JSON schema that holds a direct reference to the Swift type
public struct SchemaProperty: @unchecked Sendable {
    public let name: String
    public let valueType: any JSONSchemaConvertible.Type
    public let description: String?
    public let constraints: [GuideConstraint]

    public init(_ name: String, _ type: any JSONSchemaConvertible.Type, description: String? = nil, constraints: [GuideConstraint] = []) {
        self.name = name
        self.valueType = type
        self.description = description
        self.constraints = constraints
    }

    /// The JSON type name ("string", "integer", "number", "boolean", "array", "object")
    public var typeName: String { valueType.type }

    /// Whether this property is a nested object
    public var isObject: Bool { valueType.type == "object" }

    /// Whether this property is an array
    public var isArray: Bool { valueType.type == "array" }

    /// Whether this property should be skipped during generation
    public var shouldSkip: Bool {
        valueType.shouldSkipGeneration || constraints.contains(.skip)
    }

    /// For nested object types, get their properties
    public var nestedProperties: [SchemaProperty]? {
        valueType.schemaProperties
    }

    /// For array types, get the element type
    public var elementType: (any JSONSchemaConvertible.Type)? {
        valueType.elementType
    }

    /// Get max string length constraint if any
    public var maxLength: Int? {
        for constraint in constraints {
            if case .maxLength(let len) = constraint { return len }
        }
        return nil
    }

    /// Get integer range constraint if any
    public var intRange: ClosedRange<Int>? {
        for constraint in constraints {
            if case .range(let r) = constraint { return r }
        }
        return nil
    }

    /// Get array count constraint if any
    public var countRange: ClosedRange<Int>? {
        for constraint in constraints {
            if case .count(let r) = constraint { return r }
        }
        return nil
    }

    /// Get allowed values if any
    public var allowedValues: [String]? {
        for constraint in constraints {
            if case .anyOf(let values) = constraint { return values }
        }
        return nil
    }
}

public protocol JSONSchemaConvertible: Codable {
    /// The JSON type name ("string", "integer", "number", "boolean", "array", "object")
    static var type: String { get }

    /// Properties for object types, using metatype references
    static var schemaProperties: [SchemaProperty]? { get }

    /// For array types, the element type
    static var elementType: (any JSONSchemaConvertible.Type)? { get }

    /// Whether this type should be skipped during generation (like GenerationID)
    static var shouldSkipGeneration: Bool { get }

    /// Legacy JSON schema dictionary (for compatibility)
    static var jsonSchema: [String: Any] { get }

    /// Legacy properties (for compatibility, will be deprecated)
    static var properties: [(String, JSONSchema.Property)]? { get }

    /// Initialize from a JSON value
    init(from json: Any) throws

    /// Decode from a keyed container
    static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>,
                                     forKey key: K) throws -> Self
    
    @available(iOS 26.0, macOS 26.0, *)
    static var generationSchema: FoundationModels.GenerationSchema { get }
    @available(iOS 26.0, macOS 26.0, *)
    init(_ content: FoundationModels.GeneratedContent) throws
}

public typealias _GrammarJSONSchemaConvertible = JSONSchemaConvertible
public typealias _GrammarJSONSchema = JSONSchema
public typealias _GrammarSchemaProperty = SchemaProperty

extension RawRepresentable where Self : CaseIterable, RawValue : JSONSchemaConvertible, Self: Codable {
    public static var type: String {
        RawValue.type
    }
    public static var properties: [String: JSONSchema.Property]? { nil }
    public init(from json: Any) throws {
        guard let json = json as? RawValue else {
            throw JSONDecodingError.invalidType
        }
        self.init(rawValue: json)!
    }
    public static var jsonSchema: [String: Any] {
        [
            "type": RawValue.type,
            "enum": Self.allCases.map(\.rawValue)
        ]
    }
}

extension JSONSchemaConvertible {
    // New metatype-based properties - default implementations
    public static var schemaProperties: [SchemaProperty]? { nil }
    public static var elementType: (any JSONSchemaConvertible.Type)? { nil }
    public static var shouldSkipGeneration: Bool { false }

    // Legacy compatibility
    public static var items: JSONSchema.Items? { nil }
    public static var `enum`: [String]? { nil }

    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
        return try container.decode(Self.self, forKey: key)
    }

    /// Generate a description of this type's schema for prompt injection
    public static func schemaDescription(indent: String = "") -> String {
        guard let props = schemaProperties else {
            return "\(indent)\(type)"
        }

        var lines: [String] = []
        lines.append("\(indent){")
        for prop in props where !prop.shouldSkip {
            // Build constraint annotations
            var constraints: [String] = []
            if let desc = prop.description {
                constraints.append(desc)
            }
            if let maxLen = prop.maxLength {
                constraints.append("max \(maxLen) chars")
            }
            if let range = prop.intRange {
                constraints.append("range: \(range.lowerBound)-\(range.upperBound)")
            }
            if let countRange = prop.countRange {
                constraints.append("count: \(countRange.lowerBound)-\(countRange.upperBound)")
            }
            if let allowed = prop.allowedValues {
                constraints.append("one of: \(allowed.joined(separator: ", "))")
            }

            let constraintStr = constraints.isEmpty ? "" : " // \(constraints.joined(separator: "; "))"

            // Handle nested objects
            if prop.isObject, let nestedProps = prop.nestedProperties, !nestedProps.isEmpty {
                lines.append("\(indent)  \"\(prop.name)\":\(constraintStr)")
                lines.append(prop.valueType.schemaDescription(indent: indent + "  "))
            }
            // Handle arrays
            else if prop.isArray, let elementType = prop.elementType {
                if let elementProps = elementType.schemaProperties, !elementProps.isEmpty {
                    // Array of objects - show element schema
                    lines.append("\(indent)  \"\(prop.name)\": [\(constraintStr)")
                    lines.append(elementType.schemaDescription(indent: indent + "    "))
                    lines.append("\(indent)  ]")
                } else if let enumVals = elementType.jsonSchema["enum"] as? [Any], !enumVals.isEmpty {
                    // Array of enums - show allowed values
                    let enumStr = enumVals.map { "\"\($0)\"" }.joined(separator: " | ")
                    lines.append("\(indent)  \"\(prop.name)\": [\(enumStr)]\(constraintStr)")
                } else {
                    // Array of primitives
                    lines.append("\(indent)  \"\(prop.name)\": [\(elementType.type)]\(constraintStr)")
                }
            }
            // Primitive types (including enums)
            else {
                // Check if this is an enum type with known values (from jsonSchema["enum"])
                if let enumVals = prop.valueType.jsonSchema["enum"] as? [Any], !enumVals.isEmpty {
                    let enumStr = enumVals.map { "\"\($0)\"" }.joined(separator: " | ")
                    lines.append("\(indent)  \"\(prop.name)\": \(enumStr)\(constraintStr)")
                } else {
                    lines.append("\(indent)  \"\(prop.name)\": \(prop.typeName)\(constraintStr)")
                }
            }
        }
        lines.append("\(indent)}")
        return lines.joined(separator: "\n")
    }
}
extension String : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "string" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }
}
extension UUID : JSONSchemaConvertible {
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        .init(type: String.self, properties: [])
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
    
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        guard let json = UUID(uuidString: json) else {
            throw JSONDecodingError.invalidValue
        }
        self = json
    }
    
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "string" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }
}
extension URL : JSONSchemaConvertible {
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        .init(type: String.self, properties: [])
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
    
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        guard let json = URL(string: json) else {
            throw JSONDecodingError.invalidValue
        }
        self = json
    }
    
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "string" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }
}
extension Int : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Int else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }

    public static var type: String { "integer" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "integer"
        ]
    }
}
extension Double : JSONSchemaConvertible {
    public init(from json: Any) throws {
        if let json = json as? Double {
            self = json
            return
        }
        if let json = json as? Int {
            self = Double(json)
            return
        }
        throw JSONDecodingError.invalidType
        
    }
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "number" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "number"
        ]
    }
}


extension Float : JSONSchemaConvertible {
    public init(from json: Any) throws {
        if let json = json as? Float {
            self = json
            return
        }
        if let json = json as? Int {
            self = Float(json)
            return
        }
        throw JSONDecodingError.invalidType
        
    }
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "number" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "number"
        ]
    }
}


extension Bool : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Bool else {
            throw JSONDecodingError.invalidType
        }
        self = json
    }
    public static var properties: [(String, JSONSchema.Property)]? {
        nil
    }
    
    public static var type: String { "boolean" }
    public static var jsonSchema: [String: Any] {
        [
            "type": "boolean"
        ]
    }
}
extension Date : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? String else {
            throw JSONDecodingError.invalidType
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        
        guard let json = formatter.date(from: json) else {
            throw JSONDecodingError.invalidValue
        }
        self = json
    }
    public static var properties: [(String, JSONSchema.Property)]? { nil }
    public static var type: String { "string" }

    public static var jsonSchema: [String: Any] {
        [
            "type": "string"
        ]
    }

    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
        let value = try container.decode(String.self, forKey: key)
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue)
        let matches = detector?.matches(in: value, options: [], range: NSMakeRange(0, value.utf16.count))
        return matches!.first!.date!
        // return ISO8601DateFormatter().date(from: value)!
    }
}

extension DateInterval: JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? [String: Any] else {
            throw JSONDecodingError.invalidType
        }
        guard let startValue = json["start"] else {
            throw JSONDecodingError.missingRequiredProperty("start")
        }
        guard let endValue = json["end"] else {
            throw JSONDecodingError.missingRequiredProperty("end")
        }
        let start = try Date(from: startValue)
        let end = try Date(from: endValue)
        self.init(start: start, end: end)
    }

    public static var type: String { "object" }

    public static var schemaProperties: [SchemaProperty]? {
        [
            SchemaProperty("start", Date.self, description: "Start date"),
            SchemaProperty("end", Date.self, description: "End date")
        ]
    }

    public static var properties: [(String, JSONSchema.Property)]? {
        [
            ("start", JSONSchema.Property(type: "string", items: nil, description: "Start date")),
            ("end", JSONSchema.Property(type: "string", items: nil, description: "End date"))
        ]
    }

    public static var jsonSchema: [String: Any] {
        [
            "type": "object",
            "properties": [
                "start": Date.jsonSchema,
                "end": Date.jsonSchema
            ]
        ]
    }

    private enum CodingKeys: String, CodingKey {
        case start, end
    }

    public static func decode<K: CodingKey>(from container: KeyedDecodingContainer<K>, forKey key: K) throws -> Self {
        let nested = try container.nestedContainer(keyedBy: CodingKeys.self, forKey: key)
        let start = try Date.decode(from: nested, forKey: .start)
        let end = try Date.decode(from: nested, forKey: .end)
        return DateInterval(start: start, end: end)
    }
}

extension Array : JSONSchemaConvertible where Element : JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? Array<Any> else {
            throw JSONDecodingError.invalidType
        }
        self = try json.map(Element.init)
    }
    public static var type: String { "array" }

    // New: provide the element type as a metatype
    public static var elementType: (any JSONSchemaConvertible.Type)? { Element.self }

    // Legacy
    public static var items: JSONSchema.Items? {
        JSONSchema.Items(type: Element.type, enum: Element.enum, properties: Element.properties?.reduce(into: [:], {
            $0[$1.0] = $1.1
        }))
    }
    public static var properties: [(String, JSONSchema.Property)]? { nil }
    public static var jsonSchema: [String : Any] {
        [
            "type": "array",
            "items": Element.jsonSchema
        ]
    }
    
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        fatalError()
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
}

// MARK: - Dictionary Support
public protocol JSONSchemaKey: RawRepresentable, Codable where RawValue == String {}
extension String: JSONSchemaKey {
    public typealias RawValue = String
    
    public init?(rawValue: String) {
        self = rawValue
    }
    
    public var rawValue: String {
        self
    }
    
}

/// Dictionary with String keys - represented as a JSON object with dynamic keys
extension Dictionary: JSONSchemaConvertible where Key: JSONSchemaKey, Value: JSONSchemaConvertible {
    public init(from json: Any) throws {
        guard let json = json as? [String: Any] else {
            throw JSONDecodingError.invalidType
        }
        var result: [Key: Value] = [:]
        for (key, value) in json {
            guard let key = Key(rawValue: key) else {
                preconditionFailure("Trying to initialize a Dictionary with a key that cannot be converted from String raw value")
            }
            result[key] = try Value(from: value)
        }
        self = result
    }

    public static var type: String { "object" }

    /// For dictionaries, we don't have fixed properties - it's a dynamic object
    public static var schemaProperties: [SchemaProperty]? { nil }

    public static var properties: [(String, JSONSchema.Property)]? { nil }

    public static var jsonSchema: [String: Any] {
        [
            "type": "object",
            "additionalProperties": Value.jsonSchema
        ]
    }

    /// The value type for dictionary entries
    public static var valueType: (any JSONSchemaConvertible.Type)? { Value.self }
    
    
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        fatalError()
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
    
}

extension Optional : JSONSchemaConvertible where Wrapped : JSONSchemaConvertible {
    public init(from json: Any) throws {
        if json is NSNull {
            self = nil
        } else {
            self = try? Wrapped.init(from: json)
        }
    }

    public static var type: String { Wrapped.type }
    public static var properties: [(String, JSONSchema.Property)]? { Wrapped.properties }
    public static var jsonSchema: [String : Any] {
        Wrapped.jsonSchema
    }
    public static var schemaProperties: [SchemaProperty]? { Wrapped.schemaProperties }
    public static var elementType: (any JSONSchemaConvertible.Type)? { Wrapped.elementType }
    
    
    @available(macOS 26.0, *)
    public static var generationSchema: GenerationSchema {
        fatalError()
    }
    
    @available(macOS 26.0, *)
    public init(_ content: GeneratedContent) throws {
        fatalError()
    }
}

#if canImport(MapKit)
import MapKit
import CoreLocation

extension CLLocationCoordinate2D: JSONSchemaConvertible {
    public static var type: String { "object" }

    public static var schemaProperties: [SchemaProperty]? {
        [
            SchemaProperty("latitude", Double.self, description: "Latitude in degrees (-90 to 90)"),
            SchemaProperty("longitude", Double.self, description: "Longitude in degrees (-180 to 180)")
        ]
    }

    public static var properties: [(String, JSONSchema.Property)]? {
        [
            ("latitude", JSONSchema.Property(type: "number", items: nil, description: "Latitude in degrees")),
            ("longitude", JSONSchema.Property(type: "number", items: nil, description: "Longitude in degrees"))
        ]
    }

    public static var jsonSchema: [String: Any] {
        [
            "type": "object",
            "properties": [
                "latitude": Double.jsonSchema,
                "longitude": Double.jsonSchema
            ]
        ]
    }

    public init(from json: Any) throws {
        guard let json = json as? [String: Any] else {
            throw JSONDecodingError.invalidType
        }
        guard let lat = json["latitude"] else {
            throw JSONDecodingError.missingRequiredProperty("latitude")
        }
        guard let lon = json["longitude"] else {
            throw JSONDecodingError.missingRequiredProperty("longitude")
        }
        let latitude = try Double(from: lat)
        let longitude = try Double(from: lon)
        self.init(latitude: latitude, longitude: longitude)
    }

    private enum CodingKeys: String, CodingKey {
        case latitude, longitude
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let latitude = try container.decode(Double.self, forKey: .latitude)
        let longitude = try container.decode(Double.self, forKey: .longitude)
        self.init(latitude: latitude, longitude: longitude)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(latitude, forKey: .latitude)
        try container.encode(longitude, forKey: .longitude)
    }
}

extension MKCoordinateSpan: JSONSchemaConvertible {
    public static var type: String { "object" }

    public static var schemaProperties: [SchemaProperty]? {
        [
            SchemaProperty("latitudeDelta", Double.self, description: "Latitude span in degrees"),
            SchemaProperty("longitudeDelta", Double.self, description: "Longitude span in degrees")
        ]
    }

    public static var properties: [(String, JSONSchema.Property)]? {
        [
            ("latitudeDelta", JSONSchema.Property(type: "number", items: nil, description: "Latitude span in degrees")),
            ("longitudeDelta", JSONSchema.Property(type: "number", items: nil, description: "Longitude span in degrees"))
        ]
    }

    public static var jsonSchema: [String: Any] {
        [
            "type": "object",
            "properties": [
                "latitudeDelta": Double.jsonSchema,
                "longitudeDelta": Double.jsonSchema
            ]
        ]
    }

    public init(from json: Any) throws {
        guard let json = json as? [String: Any] else {
            throw JSONDecodingError.invalidType
        }
        guard let latDelta = json["latitudeDelta"] else {
            throw JSONDecodingError.missingRequiredProperty("latitudeDelta")
        }
        guard let lonDelta = json["longitudeDelta"] else {
            throw JSONDecodingError.missingRequiredProperty("longitudeDelta")
        }
        let latitudeDelta = try Double(from: latDelta)
        let longitudeDelta = try Double(from: lonDelta)
        self.init(latitudeDelta: latitudeDelta, longitudeDelta: longitudeDelta)
    }

    private enum CodingKeys: String, CodingKey {
        case latitudeDelta, longitudeDelta
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let latitudeDelta = try container.decode(Double.self, forKey: .latitudeDelta)
        let longitudeDelta = try container.decode(Double.self, forKey: .longitudeDelta)
        self.init(latitudeDelta: latitudeDelta, longitudeDelta: longitudeDelta)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(latitudeDelta, forKey: .latitudeDelta)
        try container.encode(longitudeDelta, forKey: .longitudeDelta)
    }
}

extension MKCoordinateRegion: JSONSchemaConvertible {
    public static var type: String { "object" }

    public static var schemaProperties: [SchemaProperty]? {
        [
            SchemaProperty("center", CLLocationCoordinate2D.self, description: "Center coordinate of the region"),
            SchemaProperty("span", MKCoordinateSpan.self, description: "Span of the region in degrees")
        ]
    }

    public static var properties: [(String, JSONSchema.Property)]? {
        [
            ("center", JSONSchema.Property(type: "object", items: nil, description: "Center coordinate of the region")),
            ("span", JSONSchema.Property(type: "object", items: nil, description: "Span of the region in degrees"))
        ]
    }

    public static var jsonSchema: [String: Any] {
        [
            "type": "object",
            "properties": [
                "center": CLLocationCoordinate2D.jsonSchema,
                "span": MKCoordinateSpan.jsonSchema
            ]
        ]
    }

    public init(from json: Any) throws {
        guard let json = json as? [String: Any] else {
            throw JSONDecodingError.invalidType
        }
        guard let centerJson = json["center"] else {
            throw JSONDecodingError.missingRequiredProperty("center")
        }
        guard let spanJson = json["span"] else {
            throw JSONDecodingError.missingRequiredProperty("span")
        }
        let center = try CLLocationCoordinate2D(from: centerJson)
        let span = try MKCoordinateSpan(from: spanJson)
        self.init(center: center, span: span)
    }

    private enum CodingKeys: String, CodingKey {
        case center, span
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let center = try container.decode(CLLocationCoordinate2D.self, forKey: .center)
        let span = try container.decode(MKCoordinateSpan.self, forKey: .span)
        self.init(center: center, span: span)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(center, forKey: .center)
        try container.encode(span, forKey: .span)
    }
}

// MARK: - FoundationModels Generable Support
#if canImport(FoundationModels)
import FoundationModels

@available(iOS 26.0, macOS 26.0, *)
extension CLLocationCoordinate2D: Generable {
    public static var generationSchema: GenerationSchema {
        GenerationSchema(type: Self.self, properties: [
            .init(name: "latitude", type: Double.self),
            .init(name: "longitude", type: Double.self)
        ])
    }

    public init(_ content: GeneratedContent) throws {
        let latitude: Double = try content.value(forProperty: "latitude")
        let longitude: Double = try content.value(forProperty: "longitude")
        self.init(latitude: latitude, longitude: longitude)
    }

    public var generatedContent: GeneratedContent {
        GeneratedContent(properties: [
            "latitude": latitude,
            "longitude": longitude
        ])
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension MKCoordinateSpan: Generable {
    public static var generationSchema: GenerationSchema {
        GenerationSchema(type: Self.self, properties: [
            .init(name: "latitudeDelta", type: Double.self),
            .init(name: "longitudeDelta", type: Double.self)
        ])
    }

    public init(_ content: GeneratedContent) throws {
        let latitudeDelta: Double = try content.value(forProperty: "latitudeDelta")
        let longitudeDelta: Double = try content.value(forProperty: "longitudeDelta")
        self.init(latitudeDelta: latitudeDelta, longitudeDelta: longitudeDelta)
    }

    public var generatedContent: GeneratedContent {
        GeneratedContent(properties: [
            "latitudeDelta": latitudeDelta,
            "longitudeDelta": longitudeDelta
        ])
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension MKCoordinateRegion: Generable {
    public static var generationSchema: GenerationSchema {
        GenerationSchema(type: Self.self, properties: [
            .init(name: "center", type: CLLocationCoordinate2D.self),
            .init(name: "span", type: MKCoordinateSpan.self)
        ])
    }

    public init(_ content: GeneratedContent) throws {
        let center: CLLocationCoordinate2D = try content.value(forProperty: "center")
        let span: MKCoordinateSpan = try content.value(forProperty: "span")
        self.init(center: center, span: span)
    }

    public var generatedContent: GeneratedContent {
        GeneratedContent(properties: [
            "center": center,
            "span": span
        ])
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension Date: Generable {
    public static var generationSchema: GenerationSchema {
        // Date is represented as a string in ISO format
        GenerationSchema(type: Self.self, properties: [])
    }

    public init(_ content: GeneratedContent) throws {
        let dateString: String = try content.value()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        if let date = formatter.date(from: dateString) {
            self = date
        } else {
            // Try natural language date detection as fallback
            let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue)
            let matches = detector?.matches(in: dateString, options: [], range: NSMakeRange(0, dateString.utf16.count))
            if let date = matches?.first?.date {
                self = date
            } else {
                throw JSONDecodingError.invalidValue
            }
        }
    }

    public var generatedContent: GeneratedContent {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return GeneratedContent(formatter.string(from: self))
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension DateInterval: Generable {
    public static var generationSchema: GenerationSchema {
        GenerationSchema(type: Self.self, properties: [
            .init(name: "start", type: Date.self),
            .init(name: "end", type: Date.self)
        ])
    }

    public init(_ content: GeneratedContent) throws {
        let start: Date = try content.value(forProperty: "start")
        let end: Date = try content.value(forProperty: "end")
        self.init(start: start, end: end)
    }

    public var generatedContent: GeneratedContent {
        GeneratedContent(properties: [
            "start": start,
            "end": end
        ])
    }
}
#endif
#endif


@attached(member, names: arbitrary)
@attached(extension, conformances: JSONSchemaConvertible, CaseIterable, Generable, JSONSchemaKey,
          names: arbitrary)
public macro JSONSchema() = #externalMacro(module: "JSONSchemaMacros",
                                           type: "JSONSchemaMacro")

/// Bridges an external type to JSONSchemaConvertible using a marker struct.
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
///
/// This generates an extension on `MKCoordinateRegion` conforming to `JSONSchemaConvertible`.
/// Note: The compiler warning about "extending a protocol composition" can be ignored - the macro
/// correctly generates the extension on the target type specified in the `for:` parameter.
@attached(extension, conformances: JSONSchemaConvertible, names: arbitrary)
public macro JSONSchemaBridge<T>(
    for type: T.Type,
    _ properties: (String, Any.Type)...
) = #externalMacro(
    module: "JSONSchemaMacros",
    type: "JSONSchemaBridgeMacro"
)

/// Peer macro for annotating properties with descriptions and constraints.
/// Used within @JSONSchema structs to provide metadata for generation.
/// This macro does nothing at runtime - it's only read by the JSONSchema macro during expansion.
@attached(peer)
public macro SchemaGuide(_ description: String? = nil, _ constraints: GuideConstraint...) = #externalMacro(
    module: "JSONSchemaMacros",
    type: "SchemaGuideMacro"
)

// MARK: - Dictionary Generable Support (Enum Keys with CaseIterable)

@available(iOS 26.0, macOS 26.0, *)
extension Dictionary: ConvertibleFromGeneratedContent where Key: RawRepresentable & CaseIterable, Key.RawValue == String, Value: Generable {
    public init(_ content: GeneratedContent) throws {
        var result: [Key: Value] = [:]
        for enumCase in Key.allCases {
            let value: Value = try content.value(forProperty: enumCase.rawValue)
            result[enumCase] = value
        }
        self = result
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension Dictionary: ConvertibleToGeneratedContent where Key: RawRepresentable & CaseIterable, Key.RawValue == String, Value: Generable & Encodable {
    public var generatedContent: GeneratedContent {
        // Encode the dictionary as JSON and create GeneratedContent from the raw JSON string
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        var dict: [String: Value] = [:]
        for (key, value) in self {
            dict[key.rawValue] = value
        }
        if let data = try? encoder.encode(dict),
           let jsonString = String(data: data, encoding: .utf8) {
            return GeneratedContent(jsonString)
        }
        return GeneratedContent("{}")
    }
}

@available(iOS 26.0, macOS 26.0, *)
extension Dictionary: InstructionsRepresentable where Key: RawRepresentable & CaseIterable, Key.RawValue == String, Value: Generable & Encodable {
}

@available(iOS 26.0, macOS 26.0, *)
extension Dictionary: PromptRepresentable where Key: RawRepresentable & CaseIterable, Key.RawValue == String, Value: Generable & Encodable {
}

@available(iOS 26.0, macOS 26.0, *)
extension Dictionary: Generable where Key: RawRepresentable & CaseIterable, Key.RawValue == String, Value: Generable & Encodable {
    public static var generationSchema: GenerationSchema {
        // Build properties from enum cases
        let properties: [GenerationSchema.Property] = Key.allCases.map { enumCase in
            .init(name: enumCase.rawValue, type: Value.self)
        }
        return GenerationSchema(type: Self.self, properties: properties)
    }
}

#if canImport(OpenAI)
import protocol OpenAI.JSONSchemaConvertible

extension _OpenAI {
    public typealias JSONSchemaConvertible = OpenAI.JSONSchemaConvertible
}

public typealias OpenAIJSONSchemaConvertible = _OpenAI.JSONSchemaConvertible
#else
@_marker protocol OpenAIJSONSchemaConvertible {}
#endif
