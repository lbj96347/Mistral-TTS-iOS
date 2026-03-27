import MLX
import MLXFast
import MLXNN
import Foundation

// MARK: - Codebook Components

class SemanticCodebook: Module {
    var embeddingSum: MLXArray
    var clusterUsage: MLXArray
    let codebookSize: Int
    let dim: Int

    init(codebookSize: Int, dim: Int) {
        self.codebookSize = codebookSize
        self.dim = dim
        self.embeddingSum = MLXArray.zeros([codebookSize, dim])
        self.clusterUsage = MLXArray.ones([codebookSize])
    }

    var embeddings: MLXArray {
        embeddingSum / MLX.maximum(clusterUsage.expandedDimensions(axis: 1), MLXArray(1e-7))
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        embeddings[codes]
    }
}

class AcousticCodebook: Module {
    let codebookSize: Int
    let dim: Int
    let halfLevels: Float

    init(codebookSize: Int, dim: Int) {
        self.codebookSize = codebookSize
        self.dim = dim
        self.halfLevels = Float(codebookSize - 1) / 2.0
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        codes.asType(.float32) / halfLevels - 1.0
    }
}

class MistralAudioCodebook: Module {
    @ModuleInfo(key: "semantic_codebook") var semanticCodebook: SemanticCodebook
    @ModuleInfo(key: "acoustic_codebook") var acousticCodebook: AcousticCodebook

    init(config: AudioTokenizerConfig) {
        self._semanticCodebook.wrappedValue = SemanticCodebook(
            codebookSize: config.semanticCodebookSize, dim: config.semanticDim)
        self._acousticCodebook.wrappedValue = AcousticCodebook(
            codebookSize: config.acousticCodebookSize, dim: config.acousticDim)
    }

    func decode(semanticCodes: MLXArray, acousticCodes: MLXArray) -> MLXArray {
        let semanticEmb = semanticCodebook.decode(semanticCodes)
        let acousticEmb = acousticCodebook.decode(acousticCodes)
        return MLX.concatenated([semanticEmb, acousticEmb], axis: -1)
    }
}

// MARK: - Weight-Normed Convolution Parametrizations

class ParametrizationsWeight: Module {
    var original0: MLXArray  // [out, 1, 1] magnitude
    var original1: MLXArray  // [out, kernel, in] direction

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self.original0 = MLXArray.ones([outChannels, 1, 1])
        self.original1 = MLXArray.zeros([outChannels, kernelSize, inChannels])
    }
}

class ParametrizationsContainer: Module {
    @ModuleInfo(key: "weight") var weight: ParametrizationsWeight

    init(outChannels: Int, inChannels: Int, kernelSize: Int) {
        self._weight.wrappedValue = ParametrizationsWeight(
            outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
    }
}

// MARK: - Weight-Normed ConvTranspose1d

class WeightNormedConvTranspose1d: Module {
    @ModuleInfo(key: "parametrizations") var parametrizations: ParametrizationsContainer

    let stride: Int
    let kernelSize: Int
    let trim: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.stride = stride
        self.kernelSize = kernelSize
        self.trim = kernelSize - stride

        self._parametrizations.wrappedValue = ParametrizationsContainer(
            outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
    }

    func getWeight() -> MLXArray {
        let g = parametrizations.weight.original0
        let v = parametrizations.weight.original1
        let vNorm = MLX.sqrt(MLX.sum(v * v, axes: [1, 2], keepDims: true) + 1e-12)
        return g * (v / vNorm)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = getWeight()
        var y = MLX.convTransposed1d(x, weight, stride: stride, padding: 0)
        if trim > 0 {
            y = y[0..., 0 ..< (y.dim(1) - trim), 0...]
        }
        return y
    }
}

// MARK: - Weight-Normed Conv1d

class WeightNormedConv1d: Module {
    @ModuleInfo(key: "parametrizations") var parametrizations: ParametrizationsContainer

    let stride: Int
    let kernelSize: Int
    let padding: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.stride = stride
        self.kernelSize = kernelSize
        self.padding = kernelSize - 1

        self._parametrizations.wrappedValue = ParametrizationsContainer(
            outChannels: outChannels, inChannels: inChannels, kernelSize: kernelSize)
    }

    func getWeight() -> MLXArray {
        let g = parametrizations.weight.original0
        let v = parametrizations.weight.original1
        let vNorm = MLX.sqrt(MLX.sum(v * v, axes: [1, 2], keepDims: true) + 1e-12)
        return g * (v / vNorm)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        if padding > 0 {
            let widths: [IntOrPair] = [
                IntOrPair((0, 0)),
                IntOrPair((padding, 0)),
                IntOrPair((0, 0)),
            ]
            input = MLX.padded(input, widths: widths)
        }
        let weight = getWeight()
        return MLX.conv1d(input, weight, stride: stride, padding: 0)
    }
}

// MARK: - Codec Attention

class CodecAttention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float
    let causal: Bool
    let slidingWindow: Int
    let useQkNorm: Bool

    init(dim: Int, nHeads: Int, nKvHeads: Int, headDim: Int,
         causal: Bool, slidingWindow: Int,
         qkNorm: Bool, qkNormEps: Float) {
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        self.nRep = nHeads / nKvHeads
        self.scale = pow(Float(headDim), -0.5)
        self.causal = causal
        self.slidingWindow = slidingWindow
        self.useQkNorm = qkNorm

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        if qkNorm {
            self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
            self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)

        var q = wq(x).reshaped(B, T, nHeads, headDim)
        var k = wk(x).reshaped(B, T, nKvHeads, headDim)
        var v = wv(x).reshaped(B, T, nKvHeads, headDim)

        if useQkNorm, let qn = qNorm, let kn = kNorm {
            q = qn(q)
            k = kn(k)
        }

        if nRep > 1 {
            k = repeatKv(k, nRep: nRep)
            v = repeatKv(v, nRep: nRep)
        }

        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        var scores = qT.matmul(kT.transposed(0, 1, 3, 2)) * scale

        if causal || slidingWindow > 0 {
            let mask = buildMask(T: T)
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1)
        let output = weights.matmul(vT)
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, T, -1))
    }

    func buildMask(T: Int) -> MLXArray {
        let rows = MLXArray(0 ..< T).reshaped(T, 1)
        let cols = MLXArray(0 ..< T).reshaped(1, T)
        var mask = MLXArray.zeros([T, T])
        if causal {
            mask = MLX.where(cols .> rows, MLXArray(-1e9), mask)
        }
        if slidingWindow > 0 {
            mask = MLX.where((rows - cols) .>= slidingWindow, MLXArray(-1e9), mask)
        }
        return mask
    }
}

// MARK: - Codec FeedForward

class CodecFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Codec Transformer Block

class CodecTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: CodecAttention
    @ModuleInfo(key: "feed_forward") var feedForward: CodecFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    var attentionScale: MLXArray?
    var ffnScale: MLXArray?
    let useLayerScale: Bool

    init(config: AudioTokenizerConfig, slidingWindow: Int) {
        self.useLayerScale = config.layerScale

        self._attention.wrappedValue = CodecAttention(
            dim: config.dim, nHeads: config.nHeads, nKvHeads: config.nKvHeads,
            headDim: config.headDim, causal: config.causal,
            slidingWindow: slidingWindow,
            qkNorm: config.qkNorm, qkNormEps: config.qkNormEps)
        self._feedForward.wrappedValue = CodecFeedForward(dim: config.dim, hiddenDim: config.hiddenDim)
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        if config.layerScale {
            self.attentionScale = MLXArray.ones([config.dim])
            self.ffnScale = MLXArray.ones([config.dim])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = attention(attentionNorm(x))
        if useLayerScale, let scale = attentionScale {
            h = h * scale
        }
        var out = x + h

        h = feedForward(ffnNorm(out))
        if useLayerScale, let scale = ffnScale {
            h = h * scale
        }
        out = out + h
        return out
    }
}

// MARK: - Decoder Blocks

class DecoderConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: WeightNormedConvTranspose1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self._conv.wrappedValue = WeightNormedConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

class DecoderTransformerBlock: Module {
    @ModuleInfo(key: "layers") var layers: [CodecTransformerBlock]

    init(config: AudioTokenizerConfig, nLayers: Int, slidingWindow: Int) {
        self._layers.wrappedValue = (0 ..< nLayers).map { _ in
            CodecTransformerBlock(config: config, slidingWindow: slidingWindow)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

class OutputProjection: Module {
    @ModuleInfo(key: "conv") var conv: WeightNormedConv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int) {
        self._conv.wrappedValue = WeightNormedConv1d(
            inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

// MARK: - Main Codec

class VoxtralCodec: Module {
    @ModuleInfo(key: "quantizer") var quantizer: MistralAudioCodebook
    @ModuleInfo(key: "decoder_blocks") var decoderBlocks: [Module]
    @ModuleInfo(key: "output_proj") var outputProj: OutputProjection

    let config: AudioTokenizerConfig

    init(config: AudioTokenizerConfig) {
        self.config = config

        self._quantizer.wrappedValue = MistralAudioCodebook(config: config)

        let strides = config.decoderConvStrides
        let kernels = config.decoderConvKernels
        let nBlocksPerStage = config.decoderTransformerLengths
        let totalCodebookDim = config.semanticDim + config.acousticDim  // 292

        var blocks: [Module] = []
        var slidingWindow = config.attnSlidingWindowSize

        for stageIdx in 0 ..< strides.count {
            let inCh = stageIdx == 0 ? totalCodebookDim : config.dim
            blocks.append(DecoderConvBlock(
                inChannels: inCh, outChannels: config.dim,
                kernelSize: kernels[stageIdx], stride: strides[stageIdx]))
            blocks.append(DecoderTransformerBlock(
                config: config, nLayers: nBlocksPerStage[stageIdx],
                slidingWindow: slidingWindow))

            if config.halfAttnWindowUponDownsampling && strides[stageIdx] > 1 {
                slidingWindow = max(1, slidingWindow / 2)
            }
        }

        self._decoderBlocks.wrappedValue = blocks
        self._outputProj.wrappedValue = OutputProjection(
            inChannels: config.dim, outChannels: config.pretransformPatchSize,
            kernelSize: config.patchProjectionKernelSize)
    }

    func decode(semanticCodes: MLXArray, acousticCodes: MLXArray) -> MLXArray {
        var h = quantizer.decode(semanticCodes: semanticCodes, acousticCodes: acousticCodes)

        for block in decoderBlocks {
            if let convBlock = block as? DecoderConvBlock {
                h = convBlock(h)
            } else if let tfBlock = block as? DecoderTransformerBlock {
                h = tfBlock(h)
            }
        }

        h = outputProj(h)

        let B = h.dim(0)
        let tUp = h.dim(1)
        let P = h.dim(2)
        return h.reshaped(B, tUp * P)
    }
}
