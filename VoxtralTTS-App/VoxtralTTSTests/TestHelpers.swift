import XCTest
import MLX
import Foundation

// MARK: - Test Fixtures

enum TestFixtures {
    static var modelURL: URL? {
        guard let path = ProcessInfo.processInfo.environment["VOXTRAL_MODEL_PATH"] else {
            return nil
        }
        return URL(fileURLWithPath: path)
    }

    static let generationTimeout: TimeInterval = 120
}

// MARK: - Shared Model Loader

/// Loads model + tokenizer once and caches for all tests to avoid repeated multi-GB loads.
actor SharedModelLoader {
    static let shared = SharedModelLoader()

    private var cached: (VoxtralTTSModel, ModelConfig, VoxtralTokenizer)?

    func load() async throws -> (VoxtralTTSModel, ModelConfig, VoxtralTokenizer) {
        if let cached { return cached }

        guard let url = TestFixtures.modelURL else {
            throw XCTSkip("VOXTRAL_MODEL_PATH not set -- skipping integration test")
        }

        let (model, config) = try loadVoxtralModel(from: url)
        let tokenizer = try await VoxtralTokenizer(modelPath: url)
        cached = (model, config, tokenizer)
        return (model, config, tokenizer)
    }

    func reset() {
        cached = nil
    }
}

// MARK: - Seeded Random Number Generator

/// Deterministic RNG for reproducible random string tests.
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
