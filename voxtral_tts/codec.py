"""Voxtral Codec: Audio tokenizer decoder.

Converts discrete tokens (1 semantic + 36 acoustic) to 24kHz waveform.

Architecture from consolidated.safetensors:
  audio_tokenizer.decoder_blocks.{0,2,4,6}.conv.*  — ConvTranspose1d (weight-normed)
  audio_tokenizer.decoder_blocks.{1,3,5,7}.layers.{0,1}.*  — Transformer blocks
  audio_tokenizer.output_proj.conv.*  — Final output Conv1d (weight-normed)
  audio_tokenizer.quantizer.semantic_codebook.*  — VQ codebook

Decoder stages (4 stages, each = conv + 2 transformer blocks):
  Stage 0: decoder_blocks.0 (conv, stride=1, k=3) + decoder_blocks.1 (2 transformer blocks)
  Stage 1: decoder_blocks.2 (conv, stride=2, k=4) + decoder_blocks.3 (2 transformer blocks)
  Stage 2: decoder_blocks.4 (conv, stride=2, k=4) + decoder_blocks.5 (2 transformer blocks)
  Stage 3: decoder_blocks.6 (conv, stride=2, k=4) + decoder_blocks.7 (2 transformer blocks)
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import AudioTokenizerArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 0.01):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


# --- Codebook Components ---

class SemanticCodebook(nn.Module):
    """Vector quantization codebook for semantic tokens.

    Weights from safetensors:
      quantizer.semantic_codebook.embedding_sum: [8192, 256]
      quantizer.semantic_codebook.cluster_usage: [8192]
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        # The actual weights use embedding_sum / cluster_usage for the codebook
        self.embedding_sum = mx.zeros((codebook_size, dim))
        self.cluster_usage = mx.ones((codebook_size,))

    @property
    def embeddings(self) -> mx.array:
        """Compute normalized embeddings from EMA statistics."""
        return self.embedding_sum / mx.maximum(self.cluster_usage[:, None], mx.array(1e-7))

    def decode(self, codes: mx.array) -> mx.array:
        """Look up embeddings for code indices.

        Args:
            codes: [B, T] codebook indices

        Returns:
            embeddings: [B, T, dim]
        """
        emb = self.embeddings  # [codebook_size, dim]
        return emb[codes]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization (FSQ) for acoustic tokens.

    Maps discrete levels back to continuous values via simple scaling.
    No learnable parameters.
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.half_levels = (codebook_size - 1) / 2.0

    def decode(self, codes: mx.array) -> mx.array:
        """Convert FSQ codes back to continuous values.

        Args:
            codes: [B, T, dim] discrete codes in [0, codebook_size-1]

        Returns:
            values: [B, T, dim] continuous values in [-1, 1]
        """
        return codes.astype(mx.float32) / self.half_levels - 1.0


class MistralAudioCodebook(nn.Module):
    """Combined semantic + acoustic codebook (quantizer)."""

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(args.semantic_codebook_size, args.semantic_dim)
        self.acoustic_codebook = AcousticCodebook(args.acoustic_codebook_size, args.acoustic_dim)
        self.semantic_dim = args.semantic_dim
        self.acoustic_dim = args.acoustic_dim

    def decode(self, semantic_codes: mx.array, acoustic_codes: mx.array) -> mx.array:
        """Decode codes back to continuous representation.

        Args:
            semantic_codes: [B, T]
            acoustic_codes: [B, T, acoustic_dim]

        Returns:
            x: [B, T, semantic_dim + acoustic_dim]  (292)
        """
        semantic_emb = self.semantic_codebook.decode(semantic_codes)  # [B, T, 256]
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes)  # [B, T, 36]
        return mx.concatenate([semantic_emb, acoustic_emb], axis=-1)


# --- Weight-Normed Convolution ---

class WeightNormedConvTranspose1d(nn.Module):
    """ConvTranspose1d with weight normalization (parametrized).

    Weights from safetensors are stored as:
      conv.parametrizations.weight.original0: [out_ch, 1, 1]  (magnitude/scale)
      conv.parametrizations.weight.original1: [out_ch, in_ch, kernel]  (direction)

    The effective weight = original0 * (original1 / ||original1||)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.trim = kernel_size - stride

        # Weight norm parametrization
        self.parametrizations = ParametrizationsContainer(out_channels, in_channels, kernel_size)

    def _get_weight(self) -> mx.array:
        """Compute effective weight from parametrizations."""
        g = self.parametrizations.weight.original0  # [out, 1, 1]
        v = self.parametrizations.weight.original1  # [out, in, kernel] (PyTorch) or [out, kernel, in] (MLX)
        # Normalize v along non-output dims
        v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
        return g * (v / v_norm)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: [B, T, C] input (channels-last in MLX)

        Returns:
            [B, T', C'] upsampled output
        """
        weight = self._get_weight()  # [out, kernel, in] in MLX convention
        # MLX ConvTranspose1d expects weight shape [out, kernel, in]
        # Use functional conv_transpose1d
        y = mx.conv_transpose1d(x, weight, stride=self.stride, padding=0)
        # Trim right side to maintain causal property
        if self.trim > 0:
            y = y[:, :-self.trim, :]
        return y


class WeightNormedConv1d(nn.Module):
    """Conv1d with weight normalization, used for output projection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

        self.parametrizations = ParametrizationsContainer(out_channels, in_channels, kernel_size)

    def _get_weight(self) -> mx.array:
        g = self.parametrizations.weight.original0
        v = self.parametrizations.weight.original1
        v_norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
        return g * (v / v_norm)

    def __call__(self, x: mx.array) -> mx.array:
        # Causal left-padding
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        weight = self._get_weight()
        return mx.conv1d(x, weight, stride=self.stride, padding=0)


class ParametrizationsWeight(nn.Module):
    """Container for weight normalization parametrizations."""
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int):
        super().__init__()
        self.original0 = mx.ones((out_channels, 1, 1))
        self.original1 = mx.zeros((out_channels, kernel_size, in_channels))


class ParametrizationsContainer(nn.Module):
    """Matches the nested structure: parametrizations.weight.original0/1"""
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int):
        super().__init__()
        self.weight = ParametrizationsWeight(out_channels, in_channels, kernel_size)


# --- Attention for Codec ---

class CodecAttention(nn.Module):
    """Multi-head attention for codec transformer blocks.

    Matches weight keys: attention.{wq,wk,wv,wo,q_norm,k_norm}.weight
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, causal: bool = True,
                 sliding_window: int = 16,
                 qk_norm: bool = True, qk_norm_eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim ** -0.5
        self.causal = causal
        self.sliding_window = sliding_window
        self.qk_norm = qk_norm

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=qk_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if self.causal or self.sliding_window > 0:
            mask = self._build_mask(T)
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v
        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(output)

    def _build_mask(self, T: int) -> mx.array:
        rows = mx.arange(T)[:, None]
        cols = mx.arange(T)[None, :]
        mask = mx.zeros((T, T))
        if self.causal:
            causal_mask = cols > rows
            mask = mx.where(causal_mask, mx.array(-1e9), mask)
        if self.sliding_window > 0:
            window_mask = (rows - cols) >= self.sliding_window
            mask = mx.where(window_mask, mx.array(-1e9), mask)
        return mask


class CodecFeedForward(nn.Module):
    """SwiGLU feed-forward. Matches: feed_forward.{w1,w2,w3}.weight"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class CodecTransformerBlock(nn.Module):
    """Transformer block matching weight keys:
      layers.N.attention.{wq,wk,wv,wo,q_norm,k_norm}.weight
      layers.N.attention_norm.weight
      layers.N.attention_scale  (1D, per-dim layer scale)
      layers.N.feed_forward.{w1,w2,w3}.weight
      layers.N.ffn_norm.weight
      layers.N.ffn_scale  (1D, per-dim layer scale)
    """

    def __init__(self, dim: int, hidden_dim: int, n_heads: int,
                 n_kv_heads: int, head_dim: int,
                 norm_eps: float = 0.01,
                 causal: bool = True, sliding_window: int = 16,
                 qk_norm: bool = True, qk_norm_eps: float = 1e-6,
                 layer_scale: bool = True):
        super().__init__()
        self.attention = CodecAttention(
            dim, n_heads, n_kv_heads, head_dim,
            causal, sliding_window, qk_norm, qk_norm_eps,
        )
        self.feed_forward = CodecFeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.use_layer_scale = layer_scale
        if layer_scale:
            # These are loaded from safetensors as 1D [dim] vectors
            self.attention_scale = mx.ones((dim,))
            self.ffn_scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.attention(self.attention_norm(x))
        if self.use_layer_scale:
            h = h * self.attention_scale
        x = x + h

        h = self.feed_forward(self.ffn_norm(x))
        if self.use_layer_scale:
            h = h * self.ffn_scale
        x = x + h

        return x


# --- Main Codec ---

class DecoderConvBlock(nn.Module):
    """A conv block in the decoder: ConvTranspose1d with weight normalization.

    Matches: decoder_blocks.{0,2,4,6}.conv.parametrizations.weight.original{0,1}
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = WeightNormedConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class DecoderTransformerBlock(nn.Module):
    """A transformer block group in the decoder.

    Matches: decoder_blocks.{1,3,5,7}.layers.{0,1}.*
    """
    def __init__(self, args: AudioTokenizerArgs, n_layers: int = 2,
                 sliding_window: int = 16):
        super().__init__()
        self.layers = [
            CodecTransformerBlock(
                dim=args.dim, hidden_dim=args.hidden_dim,
                n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim, norm_eps=args.norm_eps,
                causal=args.causal, sliding_window=sliding_window,
                qk_norm=args.qk_norm, qk_norm_eps=args.qk_norm_eps,
                layer_scale=args.layer_scale,
            )
            for _ in range(n_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class OutputProjection(nn.Module):
    """Final output conv projection.

    Matches: output_proj.conv.parametrizations.weight.original{0,1}
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = WeightNormedConv1d(in_channels, out_channels, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class VoxtralCodec(nn.Module):
    """Voxtral audio codec decoder.

    Converts discrete tokens to 24kHz waveform. Only the decoder is present
    in consolidated.safetensors (no encoder weights).

    Structure: 4 stages of (ConvTranspose1d + 2 TransformerBlocks) + output conv.
    """

    def __init__(self, args: AudioTokenizerArgs):
        super().__init__()
        self.args = args
        dim = args.dim

        # Quantizer (for decoding codes -> continuous representation)
        self.quantizer = MistralAudioCodebook(args)

        # Decoder blocks (alternating conv + transformer)
        # Strides/kernels from config: [1,2,2,2] / [3,4,4,4]
        strides = args.decoder_conv_strides
        kernels = args.decoder_conv_kernels
        n_blocks_per_stage = args.decoder_transformer_lengths

        # First conv takes codebook dim (292) as input
        total_codebook_dim = args.semantic_dim + args.acoustic_dim  # 292

        # Build decoder_blocks as a flat list matching safetensors indexing
        self.decoder_blocks = []
        sliding_window = args.attn_sliding_window_size

        for stage_idx in range(len(strides)):
            # Conv block (even indices: 0, 2, 4, 6)
            in_ch = total_codebook_dim if stage_idx == 0 else dim
            self.decoder_blocks.append(
                DecoderConvBlock(in_ch, dim, kernels[stage_idx], strides[stage_idx])
            )

            # Transformer block (odd indices: 1, 3, 5, 7)
            self.decoder_blocks.append(
                DecoderTransformerBlock(args, n_layers=n_blocks_per_stage[stage_idx],
                                       sliding_window=sliding_window)
            )

            # Halve sliding window after upsampling
            if args.half_attn_window_upon_downsampling and strides[stage_idx] > 1:
                sliding_window = max(1, sliding_window // 2)

        # Output projection
        self.output_proj = OutputProjection(dim, args.pretransform_patch_size,
                                            args.patch_projection_kernel_size)

    def decode(self, semantic_codes: mx.array, acoustic_codes: mx.array) -> mx.array:
        """Decode audio codes to waveform.

        Args:
            semantic_codes: [B, T] semantic codebook indices
            acoustic_codes: [B, T, n_acoustic] acoustic codebook indices

        Returns:
            audio: [B, num_samples] reconstructed waveform at 24kHz
        """
        # 1. Dequantize codes to continuous representation
        continuous = self.quantizer.decode(semantic_codes, acoustic_codes)  # [B, T, 292]

        # 2. Forward through decoder blocks
        h = continuous
        for block in self.decoder_blocks:
            h = block(h)

        # 3. Output projection -> patches
        h = self.output_proj(h)  # [B, T_upsampled, patch_size]

        # 4. Reshape patches to waveform
        B, T_up, P = h.shape
        audio = h.reshape(B, T_up * P)

        return audio
