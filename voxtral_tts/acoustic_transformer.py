"""Flow-Matching Acoustic Transformer for Voxtral TTS.

Takes hidden states from the LLM decoder and produces semantic + acoustic
codes via flow matching with Euler ODE integration and classifier-free guidance.

Architecture: 3 bidirectional transformer layers, dim=3072, 32 heads.
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import AcousticTransformerArgs, MultimodalAudioModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        # Precompute log-spaced frequencies
        self.emb_scale = math.log(10000) / (half_dim - 1)

    def __call__(self, t: mx.array) -> mx.array:
        """Encode timestep t into sinusoidal embedding.

        Args:
            t: Scalar or [B] timestep values in [0, 1]
        """
        half_dim = self.dim // 2
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -self.emb_scale)

        if t.ndim == 0:
            t = t.reshape(1)

        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # [B, dim]
        return emb


class BidirectionalAttention(nn.Module):
    """Multi-head attention WITHOUT causal masking.

    Used in the acoustic transformer where all positions
    can attend to all other positions.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # No causal mask — bidirectional attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class AcousticTransformerBlock(nn.Module):
    """Single transformer block for the acoustic transformer."""

    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.attention = BidirectionalAttention(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            bias=args.use_biases,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            bias=args.use_biases,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FlowMatchingAcousticTransformer(nn.Module):
    """Flow-matching acoustic transformer.

    Takes LLM hidden states and produces audio codes (1 semantic + 36 acoustic)
    per frame using Euler ODE integration with classifier-free guidance.

    Flow matching process:
    1. Sample noise x_0 ~ N(0, sigma^2)
    2. For each timestep t in [0, 1]:
       - Predict velocity v_t = f(x_t, t, hidden)
       - Apply CFG: v = alpha * v_cond + (1-alpha) * v_uncond
       - Euler step: x_{t+dt} = x_t + dt * v_t
    3. Quantize final x_1 to discrete acoustic codes
    """

    def __init__(self, audio_args: MultimodalAudioModelArgs, llm_dim: int):
        super().__init__()
        self.audio_args = audio_args
        self.acoustic_args = audio_args.get_acoustic_args()

        dim = self.acoustic_args.dim
        n_acoustic = audio_args.n_acoustic_codebook

        # Flow matching hyperparameters
        self._acoustic_decode_iters = 8
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0

        # Input projections — names match consolidated.safetensors keys:
        #   acoustic_transformer.input_projection.weight: [dim, n_acoustic]
        #   acoustic_transformer.time_projection.weight: [dim, dim]
        #   acoustic_transformer.llm_projection.weight: [dim, llm_dim]
        self.input_projection = nn.Linear(n_acoustic, dim, bias=False)
        self.time_projection = nn.Linear(dim, dim, bias=False)
        self.llm_projection = nn.Linear(llm_dim, dim, bias=False)

        # Time embedding
        self.time_embedding = TimeEmbedding(dim)

        # Transformer layers (bidirectional)
        self.layers = [
            AcousticTransformerBlock(self.acoustic_args)
            for _ in range(self.acoustic_args.n_layers)
        ]

        # Output norm
        self.norm = RMSNorm(dim, eps=self.acoustic_args.norm_eps)

        # Output heads — names match consolidated.safetensors:
        #   acoustic_transformer.semantic_codebook_output.weight: [8320, dim]
        #   acoustic_transformer.acoustic_codebook_output.weight: [n_acoustic, dim]
        padded_semantic_size = self._pad_to_multiple(
            audio_args.semantic_codebook_size + 2, 128  # +2 for special tokens = 8194 -> 8320
        )
        self.semantic_codebook_output = nn.Linear(dim, padded_semantic_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(dim, n_acoustic, bias=False)

    @staticmethod
    def _pad_to_multiple(size: int, multiple: int) -> int:
        return ((size + multiple - 1) // multiple) * multiple

    def _predict_velocity(
        self,
        x_t: mx.array,
        t: mx.array,
        llm_hidden: mx.array,
    ) -> mx.array:
        """Predict velocity for flow matching at timestep t.

        Args:
            x_t: Current acoustic state [B, n_acoustic]
            t: Current timestep [B] in [0, 1]
            llm_hidden: LLM hidden state [B, dim]

        Returns:
            Velocity prediction [B, n_acoustic]
        """
        # Project inputs to transformer dimension
        acoustic_proj = self.input_projection(x_t)  # [B, dim]
        time_emb = self.time_embedding(t)  # [B, dim]
        time_proj = self.time_projection(time_emb)
        llm_proj = self.llm_projection(llm_hidden)

        # Concatenate as 3-token sequence: [noise_state, time, llm_hidden]
        acoustic_tok = acoustic_proj.reshape(-1, 1, self.acoustic_args.dim)  # [B, 1, dim]
        time_tok = time_proj.reshape(-1, 1, self.acoustic_args.dim)          # [B, 1, dim]
        llm_tok = llm_proj.reshape(-1, 1, self.acoustic_args.dim)            # [B, 1, dim]

        h = mx.concatenate([acoustic_tok, time_tok, llm_tok], axis=1)  # [B, 3, dim]

        # Forward through transformer layers (bidirectional attention across 3 tokens)
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        # Extract first token output (noise state position) for velocity prediction
        velocity = self.acoustic_codebook_output(h[:, 0, :])  # [B, n_acoustic]
        return velocity

    def predict_semantic(self, llm_hidden: mx.array) -> mx.array:
        """Predict semantic token from LLM hidden state.

        Args:
            llm_hidden: [B, dim] post-norm hidden states from the LLM

        Returns:
            semantic_logits: [B, padded_semantic_size] logits over semantic vocab
        """
        return self.semantic_codebook_output(llm_hidden)

    def decode_one_frame(
        self,
        llm_hidden: mx.array,
    ) -> mx.array:
        """Decode one frame of acoustic codes from LLM hidden states.

        Uses Euler ODE integration with classifier-free guidance.

        Args:
            llm_hidden: [B, dim] hidden states from the LLM

        Returns:
            acoustic_codes: [B, n_acoustic] quantized acoustic codes
        """
        B = llm_hidden.shape[0]
        n_acoustic = self.audio_args.n_acoustic_codebook

        # Flow matching for acoustic codes
        # Initialize from noise
        x_t = mx.random.normal((B, n_acoustic)) * self._noise_scale

        # Euler integration: linspace(0, 1, num_iters) gives num_iters-1 steps
        num_iters = self._acoustic_decode_iters  # 8
        dt = 1.0 / (num_iters - 1)  # 1/7

        for i in range(num_iters - 1):  # 7 steps
            t = mx.full((B,), i * dt)

            # Conditional velocity (with LLM hidden)
            v_cond = self._predict_velocity(x_t, t, llm_hidden)

            # Unconditional velocity (zero hidden for CFG)
            zero_hidden = mx.zeros_like(llm_hidden)
            v_uncond = self._predict_velocity(x_t, t, zero_hidden)

            # Classifier-free guidance
            v_t = self._cfg_alpha * v_cond + (1.0 - self._cfg_alpha) * v_uncond

            # Euler step
            x_t = x_t + dt * v_t

        # Quantize acoustic codes
        # Clamp to [-1, 1] then scale to codebook range (FSQ)
        x_t = mx.clip(x_t, -1.0, 1.0)
        acoustic_codes = mx.round(
            (x_t + 1.0) * 0.5 * (self.audio_args.acoustic_codebook_size - 1)
        ).astype(mx.int32)
        acoustic_codes = mx.clip(acoustic_codes, 0, self.audio_args.acoustic_codebook_size - 1)

        return acoustic_codes

    def __call__(
        self,
        llm_hidden: mx.array,
    ) -> mx.array:
        """Generate acoustic codes for a sequence of hidden states.

        Semantic tokens are generated by the LLM, not by this module.

        Args:
            llm_hidden: [B, T, dim] sequence of hidden states

        Returns:
            acoustic_codes: [B, T, n_acoustic] acoustic codebook indices
        """
        B, T, D = llm_hidden.shape
        all_acoustic = []

        for t in range(T):
            h = llm_hidden[:, t, :]  # [B, dim]
            acou_codes = self.decode_one_frame(h)
            all_acoustic.append(acou_codes)

        acoustic_codes = mx.stack(all_acoustic, axis=1)  # [B, T, n_acoustic]
        return acoustic_codes
