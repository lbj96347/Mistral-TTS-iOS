import Foundation
import Tokenizers

class VoxtralTokenizer {
    let tokenizer: Tokenizer
    let bosTokenId: Int
    let eosTokenId: Int

    init(modelPath: URL) async throws {
        print("[TTS-Tokenizer] Loading tokenizer from: \(modelPath.path)")

        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        let configURL = modelPath.appendingPathComponent("tokenizer_config.json")

        let tokenizerExists = FileManager.default.fileExists(atPath: tokenizerURL.path)
        let configExists = FileManager.default.fileExists(atPath: configURL.path)
        print("[TTS-Tokenizer] tokenizer.json exists: \(tokenizerExists)")
        print("[TTS-Tokenizer] tokenizer_config.json exists: \(configExists)")

        // Load tokenizer config and patch unsupported tokenizer_class
        var tokenizerConfigDict: [String: Any] = [:]
        if configExists {
            let configData = try Data(contentsOf: configURL)
            if var dict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] {
                let tokenizerClass = dict["tokenizer_class"] as? String ?? "unknown"
                print("[TTS-Tokenizer] tokenizer_class: \(tokenizerClass)")

                // Patch unsupported tokenizer classes
                let unsupported = ["TokenizersBackend"]
                if unsupported.contains(tokenizerClass) {
                    print("[TTS-Tokenizer] Patching unsupported tokenizer_class to PreTrainedTokenizerFast")
                    dict["tokenizer_class"] = "PreTrainedTokenizerFast"
                }
                tokenizerConfigDict = dict
            }
        }

        // Write patched config to temp file and load from there
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Copy tokenizer.json
        if tokenizerExists {
            try FileManager.default.copyItem(at: tokenizerURL, to: tempDir.appendingPathComponent("tokenizer.json"))
        }

        // Copy config.json (required by LanguageModelConfigurationFromHub)
        let modelConfigURL = modelPath.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: modelConfigURL.path) {
            try FileManager.default.copyItem(at: modelConfigURL, to: tempDir.appendingPathComponent("config.json"))
        }

        // Write patched tokenizer_config.json
        let patchedConfigData = try JSONSerialization.data(withJSONObject: tokenizerConfigDict, options: .prettyPrinted)
        try patchedConfigData.write(to: tempDir.appendingPathComponent("tokenizer_config.json"))

        print("[TTS-Tokenizer] Loading tokenizer from patched temp dir...")
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tempDir)
        print("[TTS-Tokenizer] Tokenizer loaded successfully")

        // Standard Mistral special tokens
        self.bosTokenId = 1  // <s>
        self.eosTokenId = 2  // </s>
    }

    func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        let encoded = tokenizer.encode(text: text)

        if addSpecialTokens {
            return [bosTokenId] + encoded
        }
        return encoded
    }

    func decode(_ tokenIds: [Int]) -> String {
        tokenizer.decode(tokens: tokenIds)
    }
}
