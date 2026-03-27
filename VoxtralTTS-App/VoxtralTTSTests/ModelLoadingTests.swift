import XCTest
import MLX
import Foundation

final class ModelLoadingTests: XCTestCase {

    // MARK: - Model Directory Validation

    func testModelDirectoryExists() throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        var isDirectory: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            "Model path does not exist: \(url.path)"
        )
        XCTAssertTrue(isDirectory.boolValue, "Model path is not a directory")

        // Check required files
        let configPath = url.appendingPathComponent("config.json").path
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: configPath),
            "config.json not found in model directory"
        )

        // Check for at least one safetensors file
        let contents = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        let safetensorsFiles = contents.filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(safetensorsFiles.isEmpty, "No .safetensors files found in model directory")
    }

    // MARK: - Config Parsing

    func testConfigJsonParsing() throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        let configURL = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(ModelConfig.self, from: configData)

        XCTAssertGreaterThan(config.dim, 0, "dim must be positive")
        XCTAssertGreaterThan(config.nLayers, 0, "nLayers must be positive")
        XCTAssertGreaterThan(config.nHeads, 0, "nHeads must be positive")
        XCTAssertGreaterThan(config.vocabSize, 0, "vocabSize must be positive")
        XCTAssertEqual(config.samplingRate, 24000, "Expected 24kHz sampling rate")
    }

    // MARK: - Model Loading

    func testModelLoadsSuccessfully() async throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        let (model, config) = try loadVoxtralModel(from: url)

        XCTAssertGreaterThan(config.dim, 0)
        XCTAssertGreaterThan(config.nLayers, 0)
        XCTAssertEqual(model.config.dim, config.dim)
        XCTAssertEqual(model.config.nLayers, config.nLayers)
        XCTAssertEqual(model.sampleRate, 24000)
    }

    // MARK: - Tokenizer Loading

    func testTokenizerLoadsSuccessfully() async throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        let tokenizer = try await VoxtralTokenizer(modelPath: url)

        XCTAssertEqual(tokenizer.bosTokenId, 1, "BOS token should be 1")
        XCTAssertEqual(tokenizer.eosTokenId, 2, "EOS token should be 2")

        // Basic sanity: encoding a simple string should produce tokens
        let tokens = tokenizer.encode("Hello")
        XCTAssertFalse(tokens.isEmpty, "Encoding 'Hello' should produce tokens")
        XCTAssertEqual(tokens.first, 1, "First token should be BOS")
    }

    // MARK: - Voice Embeddings

    func testVoiceEmbeddingsExist() async throws {
        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set")
        }

        let voiceDir = url.appendingPathComponent("voice_embedding")
        guard FileManager.default.fileExists(atPath: voiceDir.path) else {
            throw XCTSkip("No voice_embedding directory found")
        }

        let files = try FileManager.default.contentsOfDirectory(at: voiceDir, includingPropertiesForKeys: nil)
        let voiceFiles = files.filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(voiceFiles.isEmpty, "No voice embedding files found")

        // Load the first voice embedding and validate shape
        let (model, _) = try loadVoxtralModel(from: url)
        let embedding = model.loadVoiceEmbedding(from: voiceFiles[0])
        XCTAssertGreaterThanOrEqual(embedding.ndim, 2, "Voice embedding should be at least 2D")
        XCTAssertGreaterThan(embedding.dim(-1), 0, "Voice embedding last dimension should be positive")
    }
}
