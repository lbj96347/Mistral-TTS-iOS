import XCTest
import MLX
import Foundation

final class GenerationTests: XCTestCase {

    private var model: VoxtralTTSModel!
    private var config: ModelConfig!
    private var tokenizer: VoxtralTokenizer!

    override func setUp() async throws {
        let loaded = try await SharedModelLoader.shared.load()
        model = loaded.0
        config = loaded.1
        tokenizer = loaded.2
    }

    // MARK: - Helper

    private func tokenize(_ text: String) -> MLXArray {
        let ids = tokenizer.encode(text)
        return MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
    }

    private func assertValidResult(
        _ result: GenerationResult?,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard let result = result else {
            XCTFail("Generation returned nil", file: file, line: line)
            return
        }
        XCTAssertGreaterThan(result.audio.dim(0), 0, "Audio should have samples", file: file, line: line)
        XCTAssertEqual(result.sampleRate, 24000, "Sample rate should be 24kHz", file: file, line: line)
        XCTAssertGreaterThan(result.processingTimeSeconds, 0, "Processing time should be positive", file: file, line: line)
        XCTAssertGreaterThan(result.tokenCount, 0, "Token count should be positive", file: file, line: line)
    }

    // MARK: - Basic Generation

    func testGenerateSimpleEnglish() {
        let input = tokenize("Hello world")
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        assertValidResult(result)
    }

    func testGenerateSingleWord() {
        let input = tokenize("Yes")
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        // Single word may produce fewer frames but should still work
        assertValidResult(result)
    }

    func testGenerateLongSentence() {
        let text = "The quick brown fox jumps over the lazy dog, and then it ran across the field while barking loudly at the birds flying overhead."
        let input = tokenize(text)
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        assertValidResult(result)
    }

    func testGenerateNumbers() {
        let input = tokenize("One two three four five")
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        assertValidResult(result)
    }

    // MARK: - With Voice Embedding

    func testGenerateWithVoiceEmbedding() throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        let voiceDir = url.appendingPathComponent("voice_embedding")
        guard FileManager.default.fileExists(atPath: voiceDir.path) else {
            throw XCTSkip("No voice_embedding directory")
        }

        let files = try FileManager.default.contentsOfDirectory(at: voiceDir, includingPropertiesForKeys: nil)
        guard let voiceFile = files.first(where: { $0.pathExtension == "safetensors" }) else {
            throw XCTSkip("No voice embedding files found")
        }

        let voiceEmb = model.loadVoiceEmbedding(from: voiceFile)
        let input = tokenize("Hello world")
        let result = model.generate(tokenIds: input, voiceEmbedding: voiceEmb, maxAudioFrames: 50)
        assertValidResult(result)
    }

    // MARK: - Unicode & Special Inputs

    func testGenerateUnicode() {
        let input = tokenize("Bonjour le monde")
        // May not produce great audio for non-English but must not crash
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        // Result can be nil (model may not handle French), but no crash is the key assertion
        if let result = result {
            XCTAssertGreaterThan(result.audio.dim(0), 0)
        }
    }

    func testGenerateEmoji() {
        let input = tokenize("Hello \u{1F30D}")
        // Must not crash. Result may be nil.
        let _ = model.generate(tokenIds: input, maxAudioFrames: 30)
    }

    func testGenerateSpecialCharacters() {
        let input = tokenize("Hello... world!!! How are you???")
        let result = model.generate(tokenIds: input, maxAudioFrames: 50)
        if let result = result {
            XCTAssertGreaterThan(result.audio.dim(0), 0)
        }
    }

    func testGenerateVeryShortInput() {
        let input = tokenize("A")
        // Single character -- must not crash
        let _ = model.generate(tokenIds: input, maxAudioFrames: 30)
    }

    // MARK: - Random Strings

    func testGenerateRandomStrings() {
        let charPool = Array("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*(),.?\u{1F30D}\u{1F3B5}\u{1F600}\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}")
        var rng = SeededRandomNumberGenerator(seed: 42)

        for i in 0..<10 {
            let length = Int.random(in: 1...200, using: &rng)
            let randomText = String((0..<length).map { _ in charPool.randomElement(using: &rng)! })

            // Tokenization must not crash
            let tokens = tokenizer.encode(randomText)
            XCTAssertFalse(tokens.isEmpty, "Tokenization failed for random string \(i)")

            let inputArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)

            // Generation must not crash -- result can be nil
            let _ = model.generate(tokenIds: inputArray, maxAudioFrames: 20)
        }
    }

    // MARK: - Progress Callback

    func testGenerateProgressCallback() {
        let input = tokenize("Hello world")
        var callbackFrames: [Int] = []

        let _ = model.generate(
            tokenIds: input,
            maxAudioFrames: 50,
            progressCallback: { frame, total in
                callbackFrames.append(frame)
            }
        )

        XCTAssertFalse(callbackFrames.isEmpty, "Progress callback should have been called at least once")
        // Frames should be monotonically increasing
        for i in 1..<callbackFrames.count {
            XCTAssertGreaterThan(callbackFrames[i], callbackFrames[i - 1],
                                 "Frame numbers should increase monotonically")
        }
    }

    // MARK: - Audio Quality Validation

    func testGenerateAudioSampleValues() {
        let input = tokenize("Hello world, this is a test of audio generation.")
        guard let result = model.generate(tokenIds: input, maxAudioFrames: 50) else {
            XCTFail("Generation returned nil for valid input")
            return
        }

        let floats = result.audio.asType(.float32).asArray(Float.self)
        XCTAssertFalse(floats.isEmpty, "Audio should have samples")

        // All values must be finite (no NaN or Inf)
        let allFinite = floats.allSatisfy { $0.isFinite }
        XCTAssertTrue(allFinite, "All audio samples must be finite (no NaN/Inf)")

        // At least some samples should be non-silent
        let maxAmplitude = floats.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmplitude, 0.001,
                             "Audio should not be total silence (max amplitude: \(maxAmplitude))")

        // Duration should be plausible (0.1s to 30s for a short sentence with 50 frames)
        let durationSec = Float(floats.count) / 24000.0
        XCTAssertGreaterThan(durationSec, 0.1, "Audio too short: \(durationSec)s")
        XCTAssertLessThan(durationSec, 30.0, "Audio too long: \(durationSec)s")
    }

    // MARK: - Stats Consistency

    func testGenerateStatsConsistency() {
        let input = tokenize("Testing generation statistics")
        guard let result = model.generate(tokenIds: input, maxAudioFrames: 50) else {
            XCTFail("Generation returned nil")
            return
        }

        XCTAssertGreaterThan(result.processingTimeSeconds, 0)
        XCTAssertGreaterThan(result.tokenCount, 0)
        XCTAssertTrue(result.realTimeFactor.isFinite, "RTF must be finite")
        XCTAssertGreaterThan(result.realTimeFactor, 0, "RTF must be positive")
        XCTAssertGreaterThan(result.samples, 0)
        XCTAssertFalse(result.audioDuration.isEmpty, "Duration string should not be empty")
    }
}
