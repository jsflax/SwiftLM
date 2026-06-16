import Foundation

/// GPT-2 / Qwen "bytes ↔ unicode" reversible map. Every one of the 256 byte values
/// is mapped to a single printable Unicode scalar so byte sequences survive as text
/// through the BPE merge process (e.g. space 0x20 → "Ġ" U+0120, newline 0x0A → "Ċ").
enum ByteLevel {
    /// byte (0...255) -> mapped Character
    static let byteToChar: [Character] = {
        // "Printable" ranges kept as-is; everything else shifted into 256+n.
        var bs: [Int] = Array(UInt8(ascii: "!")...UInt8(ascii: "~")).map(Int.init)
            + Array(0xA1...0xAC) + Array(0xAE...0xFF)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        // Build the 0...255 ordered table.
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) {
            table[b] = Character(UnicodeScalar(c)!)
        }
        return table
    }()

    /// mapped Character -> byte
    static let charToByte: [Character: UInt8] = {
        var m: [Character: UInt8] = [:]
        for b in 0..<256 { m[byteToChar[b]] = UInt8(b) }
        return m
    }()

    /// Map raw UTF-8 bytes of a string to the byte-level char string fed to BPE.
    static func encode(_ s: Substring) -> [Character] {
        Array(s.utf8).map { byteToChar[Int($0)] }
    }

    /// Inverse: a sequence of mapped chars -> original bytes -> UTF-8 string.
    static func decode(_ chars: [Character]) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(chars.count)
        for c in chars { if let b = charToByte[c] { bytes.append(b) } }
        return String(decoding: bytes, as: UTF8.self)
    }
}
