import MLX
import MLXFast
import MLXNN
import MLXRandom
import Foundation

// MARK: - Time Embedding

class TimeEmbedding: Module {
    let dim: Int
    let embScale: Float

    init(dim: Int) {
        self.dim = dim
        let halfDim = dim / 2
        self.embScale = log(10000.0) / Float(halfDim - 1)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let halfDim = dim / 2
        let emb = MLX.exp(MLXArray(Array(0 ..< halfDim).map { Float($0) }) * (-embScale))
        var tReshaped = t
        if t.ndim == 0 {
            tReshaped = t.reshaped(1)
        }
        let scaled = tReshaped.expandedDimensions(axis: 1) * emb.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.cos(scaled), MLX.sin(scaled)], axis: -1)
    }
}

// MARK: - Bidirectional Attention

class BidirectionalAttention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float

    init(dim: Int, nHeads: Int, nKvHeads: Int, headDim: Int, bias: Bool = false) {
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        self.nRep = nHeads / nKvHeads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: bias)
        self._wk.wrappedValue = Linear(dim, nKvHeads * headDim, bias: bias)
        self._wv.wrappedValue = Linear(dim, nKvHeads * headDim, bias: bias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)

        var q = wq(x).reshaped(B, T, nHeads, headDim)
        var k = wk(x).reshaped(B, T, nKvHeads, headDim)
        var v = wv(x).reshaped(B, T, nKvHeads, headDim)

        if nRep > 1 {
            k = repeatKv(k, nRep: nRep)
            v = repeatKv(v, nRep: nRep)
        }

        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let scores = qT.matmul(kT.transposed(0, 1, 3, 2)) * scale
        let weights = softmax(scores, axis: -1)
        let output = weights.matmul(vT)

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, T, -1))
    }
}

// MARK: - Acoustic FeedForward

class AcousticFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int, bias: Bool = false) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: bias)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Acoustic Transformer Block

class AcousticTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: BidirectionalAttention
    @ModuleInfo(key: "feed_forward") var feedForward: AcousticFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(config: AcousticTransformerConfig) {
        self._attention.wrappedValue = BidirectionalAttention(
            dim: config.dim, nHeads: config.nHeads,
            nKvHeads: config.nKvHeads, headDim: config.headDim,
            bias: config.useBiases
        )
        self._feedForward.wrappedValue = AcousticFeedForward(
            dim: config.dim, hiddenDim: config.hiddenDim,
            bias: config.useBiases
        )
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(attentionNorm(x))
        h = h + feedForward(ffnNorm(h))
        return h
    }
}

// MARK: - Flow Matching Acoustic Transformer

class FlowMatchingAcousticTransformer: Module {
    @ModuleInfo(key: "input_projection") var inputProjection: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "llm_projection") var llmProjection: Linear
    @ModuleInfo(key: "layers") var layers: [AcousticTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "semantic_codebook_output") var semanticCodebookOutput: Linear
    @ModuleInfo(key: "acoustic_codebook_output") var acousticCodebookOutput: Linear

    let timeEmbedding: TimeEmbedding
    let audioConfig: MultimodalAudioModelConfig
    let acousticConfig: AcousticTransformerConfig

    let acousticDecodeIters = 8
    let cfgAlpha: Float = 1.2
    let noiseScale: Float = 1.0

    init(audioConfig: MultimodalAudioModelConfig, llmDim: Int) {
        let acConfig = audioConfig.acousticConfig
        let dim = acConfig.dim
        let nAcoustic = audioConfig.nAcousticCodebook

        self.audioConfig = audioConfig
        self.acousticConfig = acConfig
        self.timeEmbedding = TimeEmbedding(dim: dim)

        self._inputProjection.wrappedValue = Linear(nAcoustic, dim, bias: false)
        self._timeProjection.wrappedValue = Linear(dim, dim, bias: false)
        self._llmProjection.wrappedValue = Linear(llmDim, dim, bias: false)

        self._layers.wrappedValue = (0 ..< acConfig.nLayers).map { _ in
            AcousticTransformerBlock(config: acConfig)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: dim, eps: acConfig.normEps)

        let paddedSemanticSize = ((audioConfig.semanticCodebookSize + 2 + 127) / 128) * 128
        self._semanticCodebookOutput.wrappedValue = Linear(dim, paddedSemanticSize, bias: false)
        self._acousticCodebookOutput.wrappedValue = Linear(dim, nAcoustic, bias: false)
    }

    func predictVelocity(_ xT: MLXArray, t: MLXArray, llmHidden: MLXArray) -> MLXArray {
        let acousticProj = inputProjection(xT)
        let timeEmb = timeEmbedding(t)
        let timeProj = timeProjection(timeEmb)
        let llmProj = llmProjection(llmHidden)

        // Concatenate as 3-token sequence: [noise_state, time, llm_hidden]
        let acousticTok = acousticProj.reshaped(-1, 1, acousticConfig.dim)
        let timeTok = timeProj.reshaped(-1, 1, acousticConfig.dim)
        let llmTok = llmProj.reshaped(-1, 1, acousticConfig.dim)

        var h = MLX.concatenated([acousticTok, timeTok, llmTok], axis: 1)  // [B, 3, dim]

        for layer in layers {
            h = layer(h)
        }

        h = norm(h)

        // Extract first token output (noise state position) for velocity
        return acousticCodebookOutput(h[0..., 0, 0...])  // [B, n_acoustic]
    }

    /// Predict semantic token from LLM hidden state
    func predictSemantic(llmHidden: MLXArray) -> MLXArray {
        return semanticCodebookOutput(llmHidden)  // [B, padded_semantic_size]
    }

    func decodeOneFrame(llmHidden: MLXArray) -> MLXArray {
        let B = llmHidden.dim(0)
        let nAcoustic = audioConfig.nAcousticCodebook

        var xT = MLXRandom.normal([B, nAcoustic]) * noiseScale

        // Euler integration: linspace(0, 1, num_iters) gives num_iters-1 steps
        let numIters = acousticDecodeIters  // 8
        let dt: Float = 1.0 / Float(numIters - 1)  // 1/7

        for i in 0 ..< (numIters - 1) {  // 7 steps
            let t = MLXArray.full([B], values: MLXArray(Float(i) * dt))

            let vCond = predictVelocity(xT, t: t, llmHidden: llmHidden)
            let zeroHidden = MLXArray.zeros(like: llmHidden)
            let vUncond = predictVelocity(xT, t: t, llmHidden: zeroHidden)

            let vT = cfgAlpha * vCond + (1.0 - cfgAlpha) * vUncond
            xT = xT + dt * vT
        }

        // Quantize: clamp to [-1, 1], scale to [0, codebookSize-1]
        let clamped = MLX.clip(xT, min: -1.0, max: 1.0)
        let scaled = (clamped + 1.0) * 0.5 * Float(audioConfig.acousticCodebookSize - 1)
        var codes = MLX.round(scaled).asType(.int32)
        codes = MLX.clip(codes, min: 0, max: audioConfig.acousticCodebookSize - 1)

        return codes
    }
}
