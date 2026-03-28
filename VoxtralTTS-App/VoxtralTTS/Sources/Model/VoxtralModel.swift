import MLX
import MLXFast
import MLXNN
import MLXRandom
import Foundation

// MARK: - Generation Result

struct GenerationResult {
    let audio: MLXArray
    let samples: Int
    let sampleRate: Int
    let tokenCount: Int
    let audioDuration: String
    let realTimeFactor: Float
    let processingTimeSeconds: Float
}

// MARK: - Constants

let EMPTY_AUDIO_ID = 0
let END_AUDIO_ID = 1
let AUDIO_CODE_OFFSET = 2

// Special token IDs from the Tekken tokenizer (absolute IDs)
let BOS_TOKEN_ID: Int32 = 1
let AUDIO_TOKEN_ID: Int32 = 24          // [AUDIO] — voice embedding placeholder
let BEGIN_AUDIO_TOKEN_ID: Int32 = 25    // [BEGIN_AUDIO] — audio section marker
let AUDIO_TO_TEXT_TOKEN_ID: Int32 = 35  // [REPEAT_AUDIO_TEXT]
let TEXT_TO_AUDIO_TOKEN_ID: Int32 = 36  // [NEXT_AUDIO_TEXT]

// MARK: - Sampling

func sampleToken(_ logits: MLXArray, temperature: Float = 0.0, topP: Float = 1.0) -> MLXArray {
    if temperature == 0.0 {
        return MLX.argMax(logits, axis: -1)
    }

    let scaled = logits / temperature

    if topP < 1.0 {
        let sortedIndices = MLX.argSort(-scaled, axis: -1)
        let sortedLogits = scaled.take(sortedIndices, axis: -1)
        let cumulativeProbs = MLX.cumsum(softmax(sortedLogits, axis: -1), axis: -1)
        let mask = cumulativeProbs - softmax(sortedLogits, axis: -1) .>= topP
        let maskedLogits = MLX.where(mask, MLXArray(-Float.infinity), sortedLogits)
        // Scatter back - simplified: just use sorted
        return MLXRandom.categorical(maskedLogits)
    }

    return MLXRandom.categorical(scaled)
}

// MARK: - Main Model

class VoxtralTTSModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: MistralTransformerDecoder
    @ModuleInfo(key: "acoustic_transformer") var acousticTransformer: FlowMatchingAcousticTransformer
    @ModuleInfo(key: "audio_tokenizer") var audioTokenizer: VoxtralCodec
    @ModuleInfo(key: "audio_token_embedding") var audioTokenEmbedding: AudioTokenEmbeddingContainer

    let config: ModelConfig

    init(config: ModelConfig) {
        self.config = config
        let audioConfig = config.audioConfig
        let codecConfig = config.codecConfig

        self._languageModel.wrappedValue = MistralTransformerDecoder(config: config)
        self._acousticTransformer.wrappedValue = FlowMatchingAcousticTransformer(
            audioConfig: audioConfig, llmDim: config.dim)
        self._audioTokenizer.wrappedValue = VoxtralCodec(config: codecConfig)

        // Audio codebook embeddings
        let codebookSizes = audioConfig.codebookSizes
        let totalAudioVocab = codebookSizes.reduce(0, +)  // 8948
        let paddedVocab = ((totalAudioVocab + 127) / 128) * 128  // 9088
        self._audioTokenEmbedding.wrappedValue = AudioTokenEmbeddingContainer(
            vocabSize: paddedVocab, dim: config.dim)
    }

    var sampleRate: Int { config.samplingRate }

    // MARK: - Voice Embedding Loading

    func loadVoiceEmbedding(from url: URL) -> MLXArray {
        // Load from .safetensors format
        let weights = try! MLX.loadArrays(url: url)
        guard let embedding = weights["embedding"] else {
            fatalError("Voice embedding file missing 'embedding' key")
        }
        return embedding
    }

    // MARK: - Audio Frame Encoding

    func encodeAudioFrame(semanticCode: MLXArray, acousticCodes: MLXArray) -> MLXArray {
        let audioConfig = config.audioConfig
        let embTable = audioTokenEmbedding.embeddings

        // Semantic embedding: first semanticCodebookSize entries
        let semEmb = embTable(semanticCode)

        // Acoustic embeddings: each codebook k starts at offset
        let baseOffset = audioConfig.semanticCodebookSize
        var acouEmb = MLXArray.zeros(like: semEmb)

        for k in 0 ..< acousticCodes.dim(1) {
            let kOffset = baseOffset + k * audioConfig.acousticCodebookSize
            let kTokens = acousticCodes[0..., k] + kOffset
            acouEmb = acouEmb + embTable(kTokens)
        }

        return semEmb + acouEmb
    }

    // MARK: - Prompt Building

    /// Build TTS prompt: [BOS][BEGIN_AUDIO][AUDIO]*N[TEXT_TO_AUDIO] text [AUDIO_TO_TEXT][BEGIN_AUDIO]
    func buildPrompt(textTokenIds: [Int32], nVoiceTokens: Int) -> MLXArray {
        var tokens: [Int32] = [BOS_TOKEN_ID, BEGIN_AUDIO_TOKEN_ID]
        tokens.append(contentsOf: [Int32](repeating: AUDIO_TOKEN_ID, count: nVoiceTokens))
        tokens.append(TEXT_TO_AUDIO_TOKEN_ID)
        tokens.append(contentsOf: textTokenIds)
        tokens.append(AUDIO_TO_TEXT_TOKEN_ID)
        tokens.append(BEGIN_AUDIO_TOKEN_ID)
        return MLXArray(tokens).reshaped(1, -1)
    }

    // MARK: - Generation

    func generate(
        textTokenIds: [Int32],
        voiceEmbedding: MLXArray? = nil,
        temperature: Float = 0.0,
        topP: Float = 1.0,
        maxAudioFrames: Int = 2048,
        repetitionPenalty: Float = 1.1,
        progressCallback: ((Int, Int) -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) -> GenerationResult? {
        let startTime = CFAbsoluteTimeGetCurrent()
        let audioConfig = config.audioConfig

        // Build prompt with correct TTS template
        let nVoice = voiceEmbedding?.dim(0) ?? 0
        let tokenIds = buildPrompt(textTokenIds: textTokenIds, nVoiceTokens: nVoice)
        let B = tokenIds.dim(0)

        // Build input embeddings: token embeddings with voice at [AUDIO] positions
        var inputEmbeds = languageModel.tokEmbeddings(tokenIds)  // [B, T, dim]

        if let voiceEmb = voiceEmbedding, nVoice > 0 {
            let voiceExpanded = voiceEmb.ndim == 2
                ? voiceEmb.expandedDimensions(axis: 0)  // [1, N, dim]
                : voiceEmb

            // [AUDIO] positions start at index 2 (after BOS, BEGIN_AUDIO)
            let firstAudioPos = 2
            let lastAudioPos = firstAudioPos + nVoice

            let before = inputEmbeds[0..., ..<firstAudioPos, 0...]
            let after = inputEmbeds[0..., lastAudioPos..., 0...]
            let voiceBlock = MLX.broadcast(voiceExpanded, to: [B, nVoice, config.dim])
            inputEmbeds = MLX.concatenated([before, voiceBlock, after], axis: 1)
        }

        // Stage 1: Single forward pass with combined embeddings
        var result = languageModel(tokens: nil, cache: nil, inputEmbeds: inputEmbeds)
        var logits = result.logits
        var hiddenStates = result.hiddenStates
        var cache = result.newCache

        // Audio token range for logit masking
        let audioStart = audioConfig.audioTokenId       // 24
        let audioEnd = audioStart + AUDIO_CODE_OFFSET + audioConfig.semanticCodebookSize  // 8218

        // Autoregressive audio generation
        var allSemanticCodes: [MLXArray] = []
        var allAcousticCodes: [MLXArray] = []
        var pastAudioTokens: [Int32] = []

        for frameIdx in 0 ..< maxAudioFrames {
            if shouldCancel?() == true { break }

            // Extract last logits and mask to audio token range
            var lastLogits = logits[0..., -1, 0...]  // [B, vocab_size]

            // Apply repetition penalty via mask
            if repetitionPenalty != 1.0 && !pastAudioTokens.isEmpty {
                let uniqueTokens = Array(Set(pastAudioTokens))
                // Build a penalty multiplier array (1.0 everywhere, modified at past token positions)
                var penaltyValues = [Float](repeating: 1.0, count: lastLogits.dim(1))
                for tokenId in uniqueTokens {
                    let idx = Int(tokenId)
                    if idx < penaltyValues.count {
                        let score: Float = lastLogits[0, idx].item()
                        penaltyValues[idx] = score > 0 ? (1.0 / repetitionPenalty) : repetitionPenalty
                    }
                }
                let penaltyMask = MLXArray(penaltyValues).reshaped(1, -1)
                lastLogits = lastLogits * penaltyMask
            }

            // Slice to audio range [24..8218)
            let audioLogits = lastLogits[0..., audioStart..<audioEnd]  // [B, 8194]
            let relativeToken = sampleToken(audioLogits, temperature: temperature, topP: topP)
            let semToken = relativeToken + Int32(audioStart)

            // Check end-of-audio: relative token 0 or 1 = special (EMPTY or BEGIN_AUDIO)
            if MLX.any(relativeToken .< Int32(AUDIO_CODE_OFFSET)).item(Bool.self) {
                break
            }

            // Map to semantic codebook index (0-8191)
            let semCode = semToken - Int32(audioConfig.audioTokenId + AUDIO_CODE_OFFSET)

            // Track for repetition penalty
            pastAudioTokens.append(semToken[0].item(Int32.self))

            // Stage 2: Flow matching for acoustic codes
            let lastHidden = hiddenStates[0..., -1, 0...]
            let acouCode = acousticTransformer.decodeOneFrame(llmHidden: lastHidden)

            allSemanticCodes.append(semCode)
            allAcousticCodes.append(acouCode)

            // Encode audio frame back to embedding
            let audioEmb = encodeAudioFrame(semanticCode: semCode, acousticCodes: acouCode)
            let audioEmbExpanded = audioEmb.expandedDimensions(axis: 1)

            // Feed back to LLM
            result = languageModel(tokens: nil, cache: cache, inputEmbeds: audioEmbExpanded)
            logits = result.logits
            hiddenStates = result.hiddenStates
            cache = result.newCache

            eval(semCode, acouCode)
            progressCallback?(frameIdx, maxAudioFrames)
        }

        guard !allSemanticCodes.isEmpty else { return nil }

        // Stack codes
        let semanticCodes = MLX.stacked(allSemanticCodes, axis: 1)
        let acousticCodes = MLX.stacked(allAcousticCodes, axis: 1)

        // Stage 3: Decode to waveform
        var audio = audioTokenizer.decode(semanticCodes: semanticCodes, acousticCodes: acousticCodes)
        audio = audio[0]
        eval(audio)

        let elapsed = Float(CFAbsoluteTimeGetCurrent() - startTime)
        let samples = audio.dim(0)
        let audioDurationSec = Float(samples) / Float(sampleRate)
        let rtf = audioDurationSec / elapsed

        let hours = Int(audioDurationSec / 3600)
        let mins = Int((audioDurationSec.truncatingRemainder(dividingBy: 3600)) / 60)
        let secs = Int(audioDurationSec.truncatingRemainder(dividingBy: 60))
        let ms = Int((audioDurationSec.truncatingRemainder(dividingBy: 1)) * 1000)

        return GenerationResult(
            audio: audio,
            samples: samples,
            sampleRate: sampleRate,
            tokenCount: tokenIds.dim(1) + allSemanticCodes.count,
            audioDuration: String(format: "%02d:%02d:%02d.%03d", hours, mins, secs, ms),
            realTimeFactor: rtf,
            processingTimeSeconds: elapsed
        )
    }
}

// MARK: - Audio Token Embedding Container

class AudioTokenEmbeddingContainer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding

    init(vocabSize: Int, dim: Int) {
        self._embeddings.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
    }
}

// MARK: - Model Loading

func loadVoxtralModel(from modelPath: URL) throws -> (VoxtralTTSModel, ModelConfig) {
    // Load config
    let configURL = modelPath.appendingPathComponent("config.json")
    print("[TTS-Model] Loading config from: \(configURL.path)")
    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(ModelConfig.self, from: configData)
    print("[TTS-Model] Config loaded — dim=\(config.dim), nLayers=\(config.nLayers), nHeads=\(config.nHeads), vocabSize=\(config.vocabSize)")
    if let q = config.quantization {
        print("[TTS-Model] Quantization: bits=\(q.bits), groupSize=\(q.groupSize)")
    } else {
        print("[TTS-Model] No quantization configured")
    }

    // Instantiate model
    print("[TTS-Model] Instantiating VoxtralTTSModel...")
    let model = VoxtralTTSModel(config: config)
    print("[TTS-Model] Model instantiated")

    // Handle quantization - replace compatible Linear layers with QuantizedLinear
    if let quantConfig = config.quantization {
        print("[TTS-Model] Applying quantization: bits=\(quantConfig.bits), groupSize=\(quantConfig.groupSize)")
        let linearFilter: (String, Module) -> Bool = { _, module in
            if let linear = module as? Linear {
                let shape = linear.weight.shape
                let lastDim = shape[shape.count - 1]
                let minDim = shape.min() ?? 0
                return lastDim % quantConfig.groupSize == 0 && minDim > quantConfig.groupSize
            }
            return false
        }

        quantize(
            model: model.languageModel,
            groupSize: quantConfig.groupSize,
            bits: quantConfig.bits,
            filter: { path, module in
                if let embedding = module as? Embedding {
                    let (_, dims) = embedding.shape
                    return dims % quantConfig.groupSize == 0 && dims > quantConfig.groupSize
                }
                return linearFilter(path, module)
            }
        )
        quantize(
            model: model.acousticTransformer,
            groupSize: quantConfig.groupSize,
            bits: quantConfig.bits,
            filter: linearFilter
        )
        quantize(
            model: model.audioTokenEmbedding,
            groupSize: quantConfig.groupSize,
            bits: quantConfig.bits,
            filter: { _, module in
                if let embedding = module as? Embedding {
                    let (_, dims) = embedding.shape
                    return dims % quantConfig.groupSize == 0 && dims > quantConfig.groupSize
                }
                return false
            }
        )
        // Quantize codec transformer blocks individually (heterogeneous decoder_blocks
        // array is incompatible with quantize's update on the whole codec)
        for block in model.audioTokenizer.decoderBlocks {
            if let tfBlock = block as? DecoderTransformerBlock {
                quantize(
                    model: tfBlock,
                    groupSize: quantConfig.groupSize,
                    bits: quantConfig.bits,
                    filter: linearFilter
                )
            }
        }
        print("[TTS-Model] Quantization applied")
    }

    // Load weights from safetensors
    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
    let weightFiles = contents.filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    print("[TTS-Model] Found \(weightFiles.count) safetensors file(s)")

    var allWeights = [String: MLXArray]()
    for weightFile in weightFiles {
        print("[TTS-Model] Loading weights from: \(weightFile.lastPathComponent)")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let weights = try MLX.loadArrays(url: weightFile)
        let loadElapsed = CFAbsoluteTimeGetCurrent() - loadStart
        print("[TTS-Model]   → \(weights.count) tensors loaded in \(String(format: "%.2f", loadElapsed))s")
        for (key, value) in weights {
            allWeights[key] = value
        }
    }

    print("[TTS-Model] Total tensors: \(allWeights.count)")

    // Log a sample of weight keys for debugging
    let sampleKeys = allWeights.keys.sorted().prefix(10)
    for key in sampleKeys {
        if let arr = allWeights[key] {
            print("[TTS-Model]   key: \(key) shape=\(arr.shape)")
        }
    }

    // Convert flat dict to nested ModuleParameters and load
    print("[TTS-Model] Applying weights to model...")
    let applyStart = CFAbsoluteTimeGetCurrent()
    let parameters = toModuleParameters(allWeights)
    try model.update(parameters: parameters, verify: .none)
    let applyElapsed = CFAbsoluteTimeGetCurrent() - applyStart
    print("[TTS-Model] Weights applied in \(String(format: "%.2f", applyElapsed))s")

    return (model, config)
}

// MARK: - Weight Dictionary Helper

func toModuleParameters(_ flat: [String: MLXArray]) -> ModuleParameters {
    var result = ModuleParameters()
    for (key, value) in flat {
        let parts = key.split(separator: ".").map(String.init)
        insertNested(&result, keys: parts, value: value)
    }
    // Convert numeric-keyed dictionaries to arrays (e.g., layers.0, layers.1 → [layer0, layer1])
    convertNumericKeysToArrays(&result)
    return result
}

private func insertNested(_ dict: inout ModuleParameters, keys: [String], value: MLXArray) {
    guard let first = keys.first else { return }

    if keys.count == 1 {
        dict[first] = .value(value)
        return
    }

    var nested: ModuleParameters
    if case .dictionary(let existing)? = dict[first] {
        nested = ModuleParameters(values: existing)
    } else {
        nested = ModuleParameters()
    }
    insertNested(&nested, keys: Array(keys.dropFirst()), value: value)
    dict[first] = nested.asItem()
}

private func convertNumericKeysToArrays(_ params: inout ModuleParameters) {
    for key in params.keys {
        guard case .dictionary(let nested) = params[key] else { continue }

        // Recursively convert children first
        var nestedParams = ModuleParameters(values: nested)
        convertNumericKeysToArrays(&nestedParams)

        // Check if all keys are consecutive integers starting from 0
        let allNumeric = nestedParams.keys.allSatisfy { Int($0) != nil }
        if allNumeric && !nestedParams.keys.isEmpty {
            let sorted = nestedParams.keys.sorted { Int($0)! < Int($1)! }
            let isConsecutive = sorted.enumerated().allSatisfy { $0.offset == Int($0.element)! }
            if isConsecutive {
                let array = sorted.map { nestedParams[$0]! }
                params[key] = .array(array)
                continue
            }
        }

        params[key] = nestedParams.asItem()
    }
}
