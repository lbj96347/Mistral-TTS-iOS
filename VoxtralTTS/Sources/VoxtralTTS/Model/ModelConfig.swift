import Foundation

// MARK: - Acoustic Transformer Config

struct AcousticTransformerConfig: Codable {
    var inputDim: Int = 3072
    var dim: Int = 3072
    var nLayers: Int = 3
    var headDim: Int = 128
    var hiddenDim: Int = 9216
    var nHeads: Int = 32
    var nKvHeads: Int = 8
    var useBiases: Bool = false
    var normEps: Float = 1e-5
    var ropeTheta: Float = 10000.0
    var sigma: Float = 1e-5
    var sigmaMax: Float = 1.0

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
}

// MARK: - Audio Tokenizer (Codec) Config

struct AudioTokenizerConfig: Codable {
    var channels: Int = 1
    var samplingRate: Int = 24000
    var pretransformPatchSize: Int = 240
    var patchProjectionKernelSize: Int = 7

    var semanticCodebookSize: Int = 8192
    var semanticDim: Int = 256
    var acousticCodebookSize: Int = 21
    var acousticDim: Int = 36

    var dim: Int = 1024
    var hiddenDim: Int = 4096
    var headDim: Int = 128
    var nHeads: Int = 8
    var nKvHeads: Int = 8
    var normEps: Float = 0.01
    var qkNorm: Bool = true
    var qkNormEps: Float = 1e-6
    var causal: Bool = true
    var attnSlidingWindowSize: Int = 16
    var halfAttnWindowUponDownsampling: Bool = true

    var layerScale: Bool = true
    var layerScaleInit: Float = 0.01
    var convWeightNorm: Bool = true

    var decoderTransformerLengths: [Int] = [2, 2, 2, 2]
    var decoderConvKernels: [Int] = [3, 4, 4, 4]
    var decoderConvStrides: [Int] = [1, 2, 2, 2]

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
}

// MARK: - Multimodal Audio Model Config

struct MultimodalAudioModelConfig: Codable {
    var semanticCodebookSize: Int = 8192
    var acousticCodebookSize: Int = 21
    var nAcousticCodebook: Int = 36
    var audioTokenId: Int = 24
    var beginAudioTokenId: Int = 25
    var bosTokenId: Int = 1
    var samplingRate: Int = 24000
    var frameRate: Float = 12.5
    var nCodebook: Int = 37
    var inputEmbeddingConcatType: String = "sum"
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

    var codebookSizes: [Int] {
        [semanticCodebookSize] + Array(repeating: acousticCodebookSize, count: nAcousticCodebook)
    }

    var acousticConfig: AcousticTransformerConfig {
        acousticTransformerArgs ?? AcousticTransformerConfig()
    }
}

// MARK: - Top-Level Model Config

struct ModelConfig: Codable {
    var modelType: String = "voxtral_tts"

    // LLM config
    var dim: Int = 3072
    var nLayers: Int = 26
    var headDim: Int = 128
    var nHeads: Int = 32
    var nKvHeads: Int = 8
    var hiddenDim: Int = 9216
    var vocabSize: Int = 131072
    var ropeTheta: Float = 1000000.0
    var normEps: Float = 1e-5
    var maxPositionEmbeddings: Int = 128000
    var tieWordEmbeddings: Bool = true

    // Audio config
    var audioModelArgs: MultimodalAudioModelConfig?
    var codecArgs: AudioTokenizerConfig?
    var samplingRate: Int = 24000

    // Quantization
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

    var audioConfig: MultimodalAudioModelConfig {
        audioModelArgs ?? MultimodalAudioModelConfig()
    }

    var codecConfig: AudioTokenizerConfig {
        codecArgs ?? AudioTokenizerConfig()
    }
}

struct QuantizationConfig: Codable {
    var groupSize: Int = 64
    var bits: Int = 4

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}
