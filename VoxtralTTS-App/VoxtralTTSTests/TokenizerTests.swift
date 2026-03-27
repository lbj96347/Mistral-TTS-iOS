import XCTest
import Foundation

final class TokenizerTests: XCTestCase {

    private var tokenizer: VoxtralTokenizer!

    override func setUp() async throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }
        let (_, _, tok) = try await SharedModelLoader.shared.load()
        tokenizer = tok
    }

    // MARK: - Normal Text

    func testEncodeSimpleSentence() {
        let tokens = tokenizer.encode("Hello, how are you?")
        XCTAssertFalse(tokens.isEmpty)
        XCTAssertEqual(tokens.first, tokenizer.bosTokenId, "Should start with BOS")
        XCTAssertGreaterThan(tokens.count, 2, "Should produce multiple tokens")
    }

    func testEncodeAndDecodeRoundTrip() {
        let original = "The quick brown fox jumps over the lazy dog"
        let tokens = tokenizer.encode(original, addSpecialTokens: false)
        let decoded = tokenizer.decode(tokens)
        // BPE may normalize whitespace, so check containment of key words
        for word in ["quick", "brown", "fox", "lazy", "dog"] {
            XCTAssertTrue(
                decoded.lowercased().contains(word),
                "Decoded text should contain '\(word)', got: \(decoded)"
            )
        }
    }

    // MARK: - Edge Cases

    func testEncodeEmptyString() {
        let tokens = tokenizer.encode("")
        // Should return at least BOS, must not crash
        XCTAssertTrue(tokens.count >= 1, "Empty string should produce at least BOS token")
    }

    func testEncodeSingleCharacter() {
        for char in ["a", "!", " ", "Z", "0"] {
            let tokens = tokenizer.encode(char)
            XCTAssertFalse(tokens.isEmpty, "Single character '\(char)' should produce tokens")
        }
    }

    func testEncodeWhitespaceOnly() {
        let inputs = ["     ", "\n\n\n", "\t\t", " \n \t "]
        for input in inputs {
            let tokens = tokenizer.encode(input)
            XCTAssertFalse(tokens.isEmpty, "Whitespace input should produce at least BOS")
        }
    }

    func testEncodeVeryLongText() {
        let paragraph = "This is a test sentence for the tokenizer. "
        let longText = String(repeating: paragraph, count: 250) // ~10,000 chars
        XCTAssertGreaterThan(longText.count, 9000)

        let tokens = tokenizer.encode(longText)
        XCTAssertGreaterThan(tokens.count, 100, "Long text should produce many tokens")
    }

    func testEncodeRepeatedCharacters() {
        let repeated = String(repeating: "a", count: 1000)
        let tokens = tokenizer.encode(repeated)
        XCTAssertFalse(tokens.isEmpty, "Repeated characters should produce tokens")
    }

    // MARK: - Unicode

    func testEncodeUnicodeFrench() {
        let tokens = tokenizer.encode("Bonjour le monde")
        XCTAssertGreaterThan(tokens.count, 2)
    }

    func testEncodeUnicodeGerman() {
        let tokens = tokenizer.encode("Hallo Welt, wie geht es Ihnen?")
        XCTAssertGreaterThan(tokens.count, 2)
    }

    func testEncodeUnicodeRussian() {
        let tokens = tokenizer.encode("\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043C}\u{0438}\u{0440}")
        XCTAssertGreaterThan(tokens.count, 2)
    }

    func testEncodeUnicodeJapanese() {
        let tokens = tokenizer.encode("\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\u{4E16}\u{754C}")
        XCTAssertGreaterThan(tokens.count, 2)
    }

    func testEncodeUnicodeArabic() {
        let tokens = tokenizer.encode("\u{0645}\u{0631}\u{062D}\u{0628}\u{0627} \u{0628}\u{0627}\u{0644}\u{0639}\u{0627}\u{0644}\u{0645}")
        XCTAssertGreaterThan(tokens.count, 2)
    }

    // MARK: - Emoji

    func testEncodeEmoji() {
        let inputs = [
            "Hello \u{1F30D}\u{1F3B5}\u{1F525}",
            "\u{1F916}\u{1F4AC}\u{1F4E2}",
            "I love \u{2764}\u{FE0F} this!",
        ]
        for input in inputs {
            let tokens = tokenizer.encode(input)
            XCTAssertFalse(tokens.isEmpty, "Emoji input '\(input)' should produce tokens")
        }
    }

    // MARK: - Numbers & Special Characters

    func testEncodeNumbersOnly() {
        let inputs = ["12345", "3.14159", "1,000,000", "0", "-42"]
        for input in inputs {
            let tokens = tokenizer.encode(input)
            XCTAssertFalse(tokens.isEmpty, "Number input '\(input)' should produce tokens")
        }
    }

    func testEncodeSpecialCharacters() {
        let inputs = [
            "<html>&amp;</html>",
            "@#$%^&*()",
            "\\n\\t\\r",
            "\"quoted\" and 'single'",
            "path/to/file.txt",
            "user@example.com",
        ]
        for input in inputs {
            let tokens = tokenizer.encode(input)
            XCTAssertFalse(tokens.isEmpty, "Special chars '\(input)' should produce tokens")
        }
    }

    // MARK: - Mixed Content

    func testEncodeMixedLanguageAndEmoji() {
        let input = "Hello \u{4E16}\u{754C}! \u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{1F30D} \u{0645}\u{0631}\u{062D}\u{0628}\u{0627}"
        let tokens = tokenizer.encode(input)
        XCTAssertGreaterThan(tokens.count, 5, "Mixed content should produce several tokens")
    }

    // MARK: - Special Tokens Flag

    func testEncodeWithoutSpecialTokens() {
        let withSpecial = tokenizer.encode("test", addSpecialTokens: true)
        let withoutSpecial = tokenizer.encode("test", addSpecialTokens: false)

        XCTAssertEqual(withSpecial.first, tokenizer.bosTokenId)
        XCTAssertNotEqual(withoutSpecial.first, tokenizer.bosTokenId,
                          "Without special tokens, first token should not be BOS")
        XCTAssertEqual(withSpecial.count, withoutSpecial.count + 1,
                       "With special tokens should have one more token (BOS)")
    }
}
