import Foundation
import MLX
import MLXNN

/// Quick CLI test for model loading — invoke with `--test-load <path>`
func testLoadModel(path: String) {
    let url = URL(fileURLWithPath: path)
    print("Testing model load from: \(path)")

    do {
        // Step 1: Load config
        let configURL = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(ModelConfig.self, from: configData)
        print("[OK] Config loaded: dim=\(config.dim), bits=\(config.quantization?.bits ?? 0)")
        if let cb = config.quantization?.componentBits {
            print("     Component bits: \(cb)")
        }

        // Step 2: Create model
        let model = VoxtralTTSModel(config: config)
        print("[OK] Model instantiated")

        // Step 3: Quantize (replace Linear/Embedding with quantized versions)
        if let quantConfig = config.quantization {
            quantize(model: model) { path, module in
                let bits: Int
                if let componentBits = quantConfig.componentBits {
                    if path.hasPrefix("language_model") {
                        bits = componentBits["language_model"] ?? quantConfig.bits
                    } else if path.hasPrefix("acoustic_transformer") {
                        bits = componentBits["acoustic_transformer"] ?? quantConfig.bits
                    } else if path.hasPrefix("audio_tokenizer") {
                        bits = componentBits["audio_tokenizer"] ?? quantConfig.bits
                    } else if path.hasPrefix("audio_token_embedding") {
                        bits = componentBits["audio_token_embedding"] ?? quantConfig.bits
                    } else {
                        bits = quantConfig.bits
                    }
                } else {
                    bits = quantConfig.bits
                }

                if let linear = module as? Linear {
                    let shape = linear.weight.shape
                    let lastDim = shape[shape.count - 1]
                    let minDim = shape.min() ?? 0
                    guard lastDim % quantConfig.groupSize == 0 && minDim > quantConfig.groupSize else {
                        return nil
                    }
                    return (groupSize: quantConfig.groupSize, bits: bits, mode: .affine)
                }
                if let emb = module as? Embedding {
                    let (count, dim) = emb.shape
                    guard dim % quantConfig.groupSize == 0 && min(count, dim) > quantConfig.groupSize else {
                        return nil
                    }
                    return (groupSize: quantConfig.groupSize, bits: bits, mode: .affine)
                }
                return nil
            }
            print("[OK] Quantization applied")

            // Verify some modules were replaced
            let leaves = model.leafModules().flattened()
            let quantizedCount = leaves.filter { $0.1 is Quantized }.count
            let totalCount = leaves.count
            print("     Quantized modules: \(quantizedCount)/\(totalCount)")
        }

        // Step 4: Load weights
        let contents = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        let weightFiles = contents.filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var allWeights = [String: MLXArray]()
        for weightFile in weightFiles {
            let weights = try MLX.loadArrays(url: weightFile)
            print("     Loaded \(weightFile.lastPathComponent): \(weights.count) keys")
            for (key, value) in weights {
                allWeights[key] = value
            }
        }
        print("[OK] Weights loaded: \(allWeights.count) total keys")

        // Step 5: Apply weights to model
        let parameters = toModuleParameters(allWeights)
        model.update(parameters: parameters)
        print("[OK] Weights applied to model")

        // Step 6: Quick eval test
        eval(model.languageModel.norm.weight)
        print("[OK] Model eval works")

        print("\n=== ALL TESTS PASSED ===")

    } catch {
        print("[FAIL] Error: \(error)")
    }
}
