import Foundation

// MARK: - Acoustic Transformer Config

struct AcousticTransformerConfig: Codable {
    var inputDim: Int
    var dim: Int
    var nLayers: Int
    var headDim: Int
    var hiddenDim: Int
    var nHeads: Int
    var nKvHeads: Int
    var useBiases: Bool
    var normEps: Float
    var ropeTheta: Float
    var sigma: Float
    var sigmaMax: Float

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case hiddenDim = "hidden_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case useBiases = "use_biases"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case sigma
        case sigmaMax = "sigma_max"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? 3072
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 3072
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 3
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 9216
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 32
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        useBiases = try c.decodeIfPresent(Bool.self, forKey: .useBiases) ?? false
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        sigma = try c.decodeIfPresent(Float.self, forKey: .sigma) ?? 1e-5
        sigmaMax = try c.decodeIfPresent(Float.self, forKey: .sigmaMax) ?? 1.0
    }

    init() {
        self.inputDim = 3072; self.dim = 3072; self.nLayers = 3; self.headDim = 128
        self.hiddenDim = 9216; self.nHeads = 32; self.nKvHeads = 8; self.useBiases = false
        self.normEps = 1e-5; self.ropeTheta = 10000.0; self.sigma = 1e-5; self.sigmaMax = 1.0
    }
}

// MARK: - Audio Tokenizer (Codec) Config

struct AudioTokenizerConfig: Codable {
    var channels: Int
    var samplingRate: Int
    var pretransformPatchSize: Int
    var patchProjectionKernelSize: Int

    var semanticCodebookSize: Int
    var semanticDim: Int
    var acousticCodebookSize: Int
    var acousticDim: Int

    var dim: Int
    var hiddenDim: Int
    var headDim: Int
    var nHeads: Int
    var nKvHeads: Int
    var normEps: Float
    var qkNorm: Bool
    var qkNormEps: Float
    var causal: Bool
    var attnSlidingWindowSize: Int
    var halfAttnWindowUponDownsampling: Bool

    var layerScale: Bool
    var layerScaleInit: Float
    var convWeightNorm: Bool

    var decoderTransformerLengths: [Int]
    var decoderConvKernels: [Int]
    var decoderConvStrides: [Int]

    enum CodingKeys: String, CodingKey {
        case channels
        case samplingRate = "sampling_rate"
        case pretransformPatchSize = "pretransform_patch_size"
        case patchProjectionKernelSize = "patch_projection_kernel_size"
        case semanticCodebookSize = "semantic_codebook_size"
        case semanticDim = "semantic_dim"
        case acousticCodebookSize = "acoustic_codebook_size"
        case acousticDim = "acoustic_dim"
        case dim
        case hiddenDim = "hidden_dim"
        case headDim = "head_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case normEps = "norm_eps"
        case qkNorm = "qk_norm"
        case qkNormEps = "qk_norm_eps"
        case causal
        case attnSlidingWindowSize = "attn_sliding_window_size"
        case halfAttnWindowUponDownsampling = "half_attn_window_upon_downsampling"
        case layerScale = "layer_scale"
        case layerScaleInit = "layer_scale_init"
        case convWeightNorm = "conv_weight_norm"
        case decoderTransformerLengths = "decoder_transformer_lengths"
        case decoderConvKernels = "decoder_conv_kernels"
        case decoderConvStrides = "decoder_conv_strides"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        pretransformPatchSize = try c.decodeIfPresent(Int.self, forKey: .pretransformPatchSize) ?? 240
        patchProjectionKernelSize = try c.decodeIfPresent(Int.self, forKey: .patchProjectionKernelSize) ?? 7
        semanticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 8192
        semanticDim = try c.decodeIfPresent(Int.self, forKey: .semanticDim) ?? 256
        acousticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .acousticCodebookSize) ?? 21
        acousticDim = try c.decodeIfPresent(Int.self, forKey: .acousticDim) ?? 36
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 1024
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 4096
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 0.01
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? true
        qkNormEps = try c.decodeIfPresent(Float.self, forKey: .qkNormEps) ?? 1e-6
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        attnSlidingWindowSize = try c.decodeIfPresent(Int.self, forKey: .attnSlidingWindowSize) ?? 16
        halfAttnWindowUponDownsampling = try c.decodeIfPresent(Bool.self, forKey: .halfAttnWindowUponDownsampling) ?? true
        layerScale = try c.decodeIfPresent(Bool.self, forKey: .layerScale) ?? true
        layerScaleInit = try c.decodeIfPresent(Float.self, forKey: .layerScaleInit) ?? 0.01
        convWeightNorm = try c.decodeIfPresent(Bool.self, forKey: .convWeightNorm) ?? true
        decoderTransformerLengths = try c.decodeIfPresent([Int].self, forKey: .decoderTransformerLengths) ?? [2, 2, 2, 2]
        decoderConvKernels = try c.decodeIfPresent([Int].self, forKey: .decoderConvKernels) ?? [3, 4, 4, 4]
        decoderConvStrides = try c.decodeIfPresent([Int].self, forKey: .decoderConvStrides) ?? [1, 2, 2, 2]
    }

    init() {
        self.channels = 1; self.samplingRate = 24000; self.pretransformPatchSize = 240
        self.patchProjectionKernelSize = 7; self.semanticCodebookSize = 8192; self.semanticDim = 256
        self.acousticCodebookSize = 21; self.acousticDim = 36; self.dim = 1024; self.hiddenDim = 4096
        self.headDim = 128; self.nHeads = 8; self.nKvHeads = 8; self.normEps = 0.01
        self.qkNorm = true; self.qkNormEps = 1e-6; self.causal = true; self.attnSlidingWindowSize = 16
        self.halfAttnWindowUponDownsampling = true; self.layerScale = true; self.layerScaleInit = 0.01
        self.convWeightNorm = true; self.decoderTransformerLengths = [2, 2, 2, 2]
        self.decoderConvKernels = [3, 4, 4, 4]; self.decoderConvStrides = [1, 2, 2, 2]
    }
}

// MARK: - Multimodal Audio Model Config

struct MultimodalAudioModelConfig: Codable {
    var semanticCodebookSize: Int
    var acousticCodebookSize: Int
    var nAcousticCodebook: Int
    var audioTokenId: Int
    var beginAudioTokenId: Int
    var bosTokenId: Int
    var samplingRate: Int
    var frameRate: Float
    var nCodebook: Int
    var inputEmbeddingConcatType: String
    var acousticTransformerArgs: AcousticTransformerConfig?

    enum CodingKeys: String, CodingKey {
        case semanticCodebookSize = "semantic_codebook_size"
        case acousticCodebookSize = "acoustic_codebook_size"
        case nAcousticCodebook = "n_acoustic_codebook"
        case audioTokenId = "audio_token_id"
        case beginAudioTokenId = "begin_audio_token_id"
        case bosTokenId = "bos_token_id"
        case samplingRate = "sampling_rate"
        case frameRate = "frame_rate"
        case nCodebook = "n_codebook"
        case inputEmbeddingConcatType = "input_embedding_concat_type"
        case acousticTransformerArgs = "acoustic_transformer_args"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        semanticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 8192
        acousticCodebookSize = try c.decodeIfPresent(Int.self, forKey: .acousticCodebookSize) ?? 21
        nAcousticCodebook = try c.decodeIfPresent(Int.self, forKey: .nAcousticCodebook) ?? 36
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 24
        beginAudioTokenId = try c.decodeIfPresent(Int.self, forKey: .beginAudioTokenId) ?? 25
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        frameRate = try c.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        nCodebook = try c.decodeIfPresent(Int.self, forKey: .nCodebook) ?? 37
        inputEmbeddingConcatType = try c.decodeIfPresent(String.self, forKey: .inputEmbeddingConcatType) ?? "sum"
        acousticTransformerArgs = try c.decodeIfPresent(AcousticTransformerConfig.self, forKey: .acousticTransformerArgs)
    }

    init() {
        self.semanticCodebookSize = 8192; self.acousticCodebookSize = 21; self.nAcousticCodebook = 36
        self.audioTokenId = 24; self.beginAudioTokenId = 25; self.bosTokenId = 1
        self.samplingRate = 24000; self.frameRate = 12.5; self.nCodebook = 37
        self.inputEmbeddingConcatType = "sum"; self.acousticTransformerArgs = nil
    }

    var codebookSizes: [Int] {
        [semanticCodebookSize] + Array(repeating: acousticCodebookSize, count: nAcousticCodebook)
    }

    var acousticConfig: AcousticTransformerConfig {
        acousticTransformerArgs ?? AcousticTransformerConfig()
    }
}

// MARK: - Top-Level Model Config

struct ModelConfig: Codable {
    var modelType: String
    var dim: Int
    var nLayers: Int
    var headDim: Int
    var nHeads: Int
    var nKvHeads: Int
    var hiddenDim: Int
    var vocabSize: Int
    var ropeTheta: Float
    var normEps: Float
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool
    var audioModelArgs: MultimodalAudioModelConfig?
    var codecArgs: AudioTokenizerConfig?
    var samplingRate: Int
    var quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case dim
        case nLayers = "n_layers"
        case headDim = "head_dim"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case hiddenDim = "hidden_dim"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case normEps = "norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case audioModelArgs = "audio_model_args"
        case codecArgs = "codec_args"
        case samplingRate = "sampling_rate"
        case quantization
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral_tts"
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? 3072
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 26
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 32
        nKvHeads = try c.decodeIfPresent(Int.self, forKey: .nKvHeads) ?? 8
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 9216
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131072
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 128000
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        audioModelArgs = try c.decodeIfPresent(MultimodalAudioModelConfig.self, forKey: .audioModelArgs)
        codecArgs = try c.decodeIfPresent(AudioTokenizerConfig.self, forKey: .codecArgs)
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }

    var audioConfig: MultimodalAudioModelConfig {
        audioModelArgs ?? MultimodalAudioModelConfig()
    }

    var codecConfig: AudioTokenizerConfig {
        codecArgs ?? AudioTokenizerConfig()
    }
}

struct QuantizationConfig: Codable {
    var groupSize: Int
    var bits: Int
    var componentBits: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case componentBits = "component_bits"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
        componentBits = try c.decodeIfPresent([String: Int].self, forKey: .componentBits)
    }
}

