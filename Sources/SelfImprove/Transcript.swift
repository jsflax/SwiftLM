import Foundation

/// Lenient decode of one Claude Code transcript JSONL line. Every field optional —
/// records are heterogeneous (assistant/user/system/mode/attachment/snapshot/...).
struct Record: Decodable {
    let type: String?
    let uuid: String?
    let parentUuid: String?
    let isSidechain: Bool?
    let isMeta: Bool?
    let message: Message?
}

struct Message: Decodable {
    let role: String?
    let content: Content?
}

/// `message.content` is either a bare string (real human prompt) or an array of typed blocks.
enum Content: Decodable {
    case string(String)
    case blocks([Block])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .string(s)
        } else {
            self = .blocks((try? c.decode([Block].self)) ?? [])
        }
    }

    /// Concatenated visible text of `text` blocks (assistant final answer / user text).
    var texts: [String] {
        switch self {
        case .string(let s): return [s]
        case .blocks(let bs): return bs.compactMap { $0.type == "text" ? $0.text : nil }
        }
    }
    /// Raw typed blocks ([] for a bare string).
    var blocks: [Block] { if case .blocks(let b) = self { return b }; return [] }
    var isBareString: Bool { if case .string = self { return true }; return false }
}

/// A content block. We only model the fields the harvester reads.
struct Block: Decodable {
    let type: String?
    let text: String?
    let name: String?       // tool_use name
    let input: JSONValue?   // tool_use arguments
}

/// Minimal, re-serializable JSON value — captures arbitrary tool_use `input`.
enum JSONValue: Decodable {
    case null, bool(Bool), int(Int), double(Double), string(String)
    case array([JSONValue]), object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let i = try? c.decode(Int.self) { self = .int(i) }
        else if let d = try? c.decode(Double.self) { self = .double(d) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else if let o = try? c.decode([String: JSONValue].self) { self = .object(o) }
        else { self = .null }
    }

    private static func quoted(_ s: String) -> String {
        (try? JSONEncoder().encode(s)).flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
    }

    /// Compact JSON serialization (for embedding tool arguments in a training example).
    var jsonString: String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .string(let s): return Self.quoted(s)
        case .array(let a): return "[" + a.map(\.jsonString).joined(separator: ",") + "]"
        case .object(let o):
            return "{" + o.map { "\(Self.quoted($0.key)):\($0.value.jsonString)" }.joined(separator: ",") + "}"
        }
    }
}

/// Read a `.jsonl` file into decoded records, skipping malformed lines.
func loadRecords(_ url: URL) -> [Record] {
    guard let raw = try? String(contentsOf: url, encoding: .utf8) else { return [] }
    let dec = JSONDecoder()
    var out: [Record] = []
    out.reserveCapacity(1024)
    raw.enumerateLines { line, _ in
        guard !line.isEmpty, let data = line.data(using: .utf8),
              let rec = try? dec.decode(Record.self, from: data) else { return }
        out.append(rec)
    }
    return out
}
