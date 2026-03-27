import Foundation
import Tokenizers

class VoxtralTokenizer {
    let tokenizer: Tokenizer
    let bosTokenId: Int
    let eosTokenId: Int

    init(modelPath: URL) async throws {
        // Load HuggingFace-compatible tokenizer.json
        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        let configURL = modelPath.appendingPathComponent("tokenizer_config.json")

        // swift-transformers can load from a directory
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelPath)

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
