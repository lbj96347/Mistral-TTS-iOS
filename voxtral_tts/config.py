"""Configuration dataclasses for Voxtral TTS model components."""

import inspect
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class AcousticTransformerArgs:
    """Flow-matching acoustic transformer configuration."""
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sigma: float = 1e-5
    sigma_max: float = 1.0


@dataclass
class AudioTokenizerArgs:
    """Voxtral codec encoder/decoder configuration."""
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_projection_kernel_size: int = 7

    # Codebook settings
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36

    # Transformer architecture
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    norm_eps: float = 0.01
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True

    # Layer scale
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    conv_weight_norm: bool = True

    # Encoder architecture
    encoder_transformer_lengths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_conv_kernels: List[int] = field(default_factory=lambda: [4, 4, 4, 3])
    encoder_conv_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 1])

    # Decoder architecture (symmetric inverse)
    decoder_transformer_lengths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_conv_kernels: List[int] = field(default_factory=lambda: [3, 4, 4, 4])
    decoder_conv_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])


@dataclass
class MultimodalAudioModelArgs:
    """Audio model arguments for the LLM decoder stage."""
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    bos_token_id: int = 1
    sampling_rate: int = 24000
    frame_rate: float = 12.5
    codebook_pattern: str = "parallel"
    interleave_audio_tokens_per_segment: int = 8192
    interleave_text_tokens_per_segment: int = 8192
    single_trailing_segment: bool = False
    n_codebook: int = 37
    input_embedding_concat_type: str = "sum"
    acoustic_transformer_args: Optional[dict] = None

    @property
    def codebook_sizes(self) -> List[int]:
        return [self.semantic_codebook_size] + [self.acoustic_codebook_size] * self.n_acoustic_codebook

    def get_acoustic_args(self) -> AcousticTransformerArgs:
        if self.acoustic_transformer_args:
            return AcousticTransformerArgs(**{
                k: v for k, v in self.acoustic_transformer_args.items()
                if k in inspect.signature(AcousticTransformerArgs).parameters
            })
        return AcousticTransformerArgs()


@dataclass
class ModelConfig(BaseModelArgs):
    """Top-level Voxtral TTS model configuration."""
    model_type: str = "voxtral_tts"

    # Text model (Mistral) config
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    n_heads: int = 32
    n_kv_heads: int = 8
    hidden_dim: int = 9216
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    max_position_embeddings: int = 128000
    tie_word_embeddings: bool = True

    # Audio config
    audio_model_args: Optional[dict] = None
    codec_args: Optional[dict] = None
    sampling_rate: int = 24000
    sample_rate: int = 24000  # alias for mlx-audio compatibility

    def get_audio_model_args(self) -> MultimodalAudioModelArgs:
        if self.audio_model_args:
            return MultimodalAudioModelArgs(**{
                k: v for k, v in self.audio_model_args.items()
                if k in inspect.signature(MultimodalAudioModelArgs).parameters
            })
        return MultimodalAudioModelArgs()

    def get_codec_args(self) -> AudioTokenizerArgs:
        if self.codec_args:
            return AudioTokenizerArgs(**{
                k: v for k, v in self.codec_args.items()
                if k in inspect.signature(AudioTokenizerArgs).parameters
            })
        return AudioTokenizerArgs()
