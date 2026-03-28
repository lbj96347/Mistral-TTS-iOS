"""Mistral-based Transformer Decoder for Voxtral TTS.

This is the text LLM component (3.4B params) that generates hidden states
from text tokens. Based on Ministral-3B architecture.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(end, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def apply_rotary_emb(xq: mx.array, xk: mx.array, freqs_cis: mx.array) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors."""
    # xq, xk: [B, T, n_heads, head_dim]
    # freqs_cis: [T, head_dim/2, 2]
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)  # [B, T, H, head_dim/2, 2]
    xk_r = xk.reshape(*xk.shape[:-1], -1, 2)

    # Extract cos, sin from freqs_cis
    cos = freqs_cis[:, :, 0]  # [T, head_dim/2]
    sin = freqs_cis[:, :, 1]

    # Broadcast: [1, T, 1, head_dim/2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Apply rotation
    xq_out_r = xq_r[..., 0] * cos - xq_r[..., 1] * sin
    xq_out_i = xq_r[..., 0] * sin + xq_r[..., 1] * cos
    xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)

    xk_out_r = xk_r[..., 0] * cos - xk_r[..., 1] * sin
    xk_out_i = xk_r[..., 0] * sin + xk_r[..., 1] * cos
    xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

    return xq_out, xk_out


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads to match number of query heads."""
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    x = mx.expand_dims(x, axis=3)  # [B, T, n_kv_heads, 1, head_dim]
    x = mx.broadcast_to(x, (B, T, n_kv_heads, n_rep, head_dim))
    return x.reshape(B, T, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # GQA: repeat KV heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v  # [B, H, T, D]
        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.wo(output), new_cache


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, hidden_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        h, new_cache = self.attention(self.attention_norm(x), freqs_cis, mask, cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


class MistralTransformerDecoder(nn.Module):
    """Mistral-based transformer decoder for text-to-hidden-state generation.

    Architecture: Ministral-3B variant with 26 layers, dim=3072.
    """

    def __init__(
        self,
        dim: int = 3072,
        n_layers: int = 26,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_dim: int = 9216,
        vocab_size: int = 131072,
        norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 128000,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = [
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim, eps=norm_eps)

        if not tie_word_embeddings:
            self.output = nn.Linear(dim, vocab_size, bias=False)

    def __call__(
        self,
        tokens: Optional[mx.array] = None,
        cache: Optional[list] = None,
        input_embeds: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Optional[list]]:
        """Forward pass through the decoder.

        Args:
            tokens: Input token IDs [B, T] (mutually exclusive with input_embeds)
            cache: Optional KV cache list
            input_embeds: Pre-computed embeddings [B, T, dim] to use instead of
                token lookup. Used for feeding audio embeddings back into the LLM.

        Returns:
            logits: [B, T, vocab_size]
            hidden_states: [B, T, dim] (last layer hidden states)
            new_cache: Updated KV cache
        """
        if input_embeds is not None:
            h = input_embeds
            B, T = h.shape[:2]
        else:
            B, T = tokens.shape
            h = self.tok_embeddings(tokens)

        # Compute position offset from cache
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            self.layers[0].attention.head_dim,
            offset + T,
            self.rope_theta,
        )[offset:offset + T]

        # Causal mask
        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, freqs_cis, mask, layer_cache)
            new_cache.append(c)

        h = self.norm(h)
        hidden_states = h  # Post-norm hidden states for downstream use

        if self.tie_word_embeddings:
            logits = self.tok_embeddings.as_linear(h)
        else:
            logits = self.output(h)

        return logits, hidden_states, new_cache
