import MLX
import MLXFast
import MLXNN
import Foundation

// MARK: - RoPE Helpers

func precomputeFreqsCis(dim: Int, end: Int, theta: Float = 1_000_000.0) -> MLXArray {
    let freqs = 1.0 / MLX.pow(theta, MLXArray(stride(from: 0, to: Float(dim), by: 2)) / Float(dim))
    let t = MLXArray(Array(0 ..< end).map { Float($0) })
    let freqsGrid = MLX.outer(t, freqs)
    return MLX.stacked([MLX.cos(freqsGrid), MLX.sin(freqsGrid)], axis: -1)
}

func applyRotaryEmb(_ xq: MLXArray, _ xk: MLXArray, freqsCis: MLXArray) -> (MLXArray, MLXArray) {
    let shape = xq.shape
    let xqR = xq.reshaped(shape[0], shape[1], shape[2], -1, 2)
    let xkR = xk.reshaped(xk.shape[0], xk.shape[1], xk.shape[2], -1, 2)

    // freqs_cis shape is [T, head_dim/2, 2]
    // cos/sin: [T, head_dim/2] -> broadcast to [1, T, 1, head_dim/2]
    let cosB = freqsCis[0..., 0..., 0].reshaped(1, -1, 1, freqsCis.dim(1))
    let sinB = freqsCis[0..., 0..., 1].reshaped(1, -1, 1, freqsCis.dim(1))

    let xqOutR = xqR[0..., 0..., 0..., 0..., 0] * cosB - xqR[0..., 0..., 0..., 0..., 1] * sinB
    let xqOutI = xqR[0..., 0..., 0..., 0..., 0] * sinB + xqR[0..., 0..., 0..., 0..., 1] * cosB
    let xqOut = MLX.stacked([xqOutR, xqOutI], axis: -1).reshaped(shape)

    let xkShape = xk.shape
    let xkOutR = xkR[0..., 0..., 0..., 0..., 0] * cosB - xkR[0..., 0..., 0..., 0..., 1] * sinB
    let xkOutI = xkR[0..., 0..., 0..., 0..., 0] * sinB + xkR[0..., 0..., 0..., 0..., 1] * cosB
    let xkOut = MLX.stacked([xkOutR, xkOutI], axis: -1).reshaped(xkShape)

    return (xqOut, xkOut)
}

func repeatKv(_ x: MLXArray, nRep: Int) -> MLXArray {
    if nRep == 1 { return x }
    let (B, T, nKvHeads, headDim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let expanded = MLX.expandedDimensions(x, axis: 3)
    let broadcasted = MLX.broadcast(expanded, to: [B, T, nKvHeads, nRep, headDim])
    return broadcasted.reshaped(B, T, nKvHeads * nRep, headDim)
}

// MARK: - Attention

class Attention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float

    init(dim: Int, nHeads: Int, nKvHeads: Int, headDim: Int) {
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        self.nRep = nHeads / nKvHeads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        freqsCis: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let B = x.dim(0)
        let T = x.dim(1)

        var q = wq(x).reshaped(B, T, nHeads, headDim)
        var k = wk(x).reshaped(B, T, nKvHeads, headDim)
        var v = wv(x).reshaped(B, T, nKvHeads, headDim)

        (q, k) = applyRotaryEmb(q, k, freqsCis: freqsCis)

        if let (kCache, vCache) = cache {
            k = MLX.concatenated([kCache, k], axis: 1)
            v = MLX.concatenated([vCache, v], axis: 1)
        }

        let newCache = (k, v)

        let kExpanded = repeatKv(k, nRep: nRep)
        let vExpanded = repeatKv(v, nRep: nRep)

        // [B, T, H, D] -> [B, H, T, D]
        let qT = q.transposed(0, 2, 1, 3)
        let kT = kExpanded.transposed(0, 2, 1, 3)
        let vT = vExpanded.transposed(0, 2, 1, 3)

        var scores = (qT.matmul(kT.transposed(0, 1, 3, 2))) * scale
        if let mask = mask {
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1)
        let output = weights.matmul(vT)
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped(B, T, -1)

        return (wo(outputReshaped), newCache)
    }
}

// MARK: - FeedForward (SwiGLU)

class FeedForward: Module {
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

// MARK: - Transformer Block

class TransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: Attention
    @ModuleInfo(key: "feed_forward") var feedForward: FeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(dim: Int, nHeads: Int, nKvHeads: Int, headDim: Int, hiddenDim: Int, normEps: Float) {
        self._attention.wrappedValue = Attention(dim: dim, nHeads: nHeads, nKvHeads: nKvHeads, headDim: headDim)
        self._feedForward.wrappedValue = FeedForward(dim: dim, hiddenDim: hiddenDim)
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        freqsCis: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (h, newCache) = attention(attentionNorm(x), freqsCis: freqsCis, mask: mask, cache: cache)
        let x2 = x + h
        let x3 = x2 + feedForward(ffnNorm(x2))
        return (x3, newCache)
    }
}

// MARK: - Mistral Transformer Decoder

class MistralTransformerDecoder: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Optional output projection (only when not tying embeddings)
    @ModuleInfo(key: "output") var output: Linear?

    let dim: Int
    let vocabSize: Int
    let ropeTheta: Float
    let tieWordEmbeddings: Bool

    init(config: ModelConfig) {
        self.dim = config.dim
        self.vocabSize = config.vocabSize
        self.ropeTheta = config.ropeTheta
        self.tieWordEmbeddings = config.tieWordEmbeddings

        self._tokEmbeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dim)
        self._layers.wrappedValue = (0 ..< config.nLayers).map { _ in
            TransformerBlock(
                dim: config.dim,
                nHeads: config.nHeads,
                nKvHeads: config.nKvHeads,
                headDim: config.headDim,
                hiddenDim: config.hiddenDim,
                normEps: config.normEps
            )
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        if !config.tieWordEmbeddings {
            self._output.wrappedValue = Linear(config.dim, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        tokens: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)?]? = nil,
        inputEmbeds: MLXArray? = nil
    ) -> (logits: MLXArray, hiddenStates: MLXArray, newCache: [(MLXArray, MLXArray)]) {
        var h: MLXArray
        let T: Int

        if let embeds = inputEmbeds {
            h = embeds
            T = embeds.dim(1)
        } else {
            guard let tokens = tokens else {
                fatalError("Either tokens or inputEmbeds must be provided")
            }
            T = tokens.dim(1)
            h = tokEmbeddings(tokens)
        }

        // Position offset from cache
        var offset = 0
        if let c = cache, let first = c[0] {
            offset = first.0.dim(1)
        }

        // Precompute RoPE
        let headDim = layers[0].attention.headDim
        let allFreqs = precomputeFreqsCis(dim: headDim, end: offset + T, theta: ropeTheta)
        let freqsCis = allFreqs[offset ..< (offset + T)]

        // Causal mask
        var mask: MLXArray? = nil
        if T > 1 {
            mask = MultiHeadAttention.createAdditiveCausalMask(T).asType(h.dtype)
        }

        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (hNew, c) = layer(h, freqsCis: freqsCis, mask: mask, cache: layerCache)
            h = hNew
            newCache.append(c)
        }

        h = norm(h)
        let hiddenStates = h  // Post-norm hidden states for downstream use

        let logits: MLXArray
        if tieWordEmbeddings {
            logits = h.matmul(tokEmbeddings.weight.T)
        } else {
            logits = output!(h)
        }

        return (logits, hiddenStates, newCache)
    }
}
