"""Voxtral TTS: Main model class for text-to-speech synthesis.

Implements the full inference pipeline:
Text → Tokenize (Tekken) → LLM Decoder → Flow Transformer → Codec → WAV

Three-stage architecture:
1. Transformer Decoder (3.4B): Text tokens → hidden states, generates semantic tokens
2. Acoustic Transformer (390M): Hidden states → acoustic codes via flow matching
3. Voxtral Codec (300M): Semantic + acoustic codes → 24kHz waveform
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .acoustic_transformer import FlowMatchingAcousticTransformer
from .codec import VoxtralCodec
from .config import ModelConfig, MultimodalAudioModelArgs
from .transformer_decoder import MistralTransformerDecoder


@dataclass
class GenerationResult:
    """Standard result object compatible with mlx-audio."""
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
    is_streaming_chunk: bool = False
    is_final_chunk: bool = False


# Audio special token IDs (relative to audio vocab)
EMPTY_AUDIO_ID = 0
END_AUDIO_ID = 1
AUDIO_CODE_OFFSET = 2  # Actual codes start after special tokens


def load_tokenizer(model_path: Path):
    """Load the Tekken tokenizer from a model directory.

    Falls back to HuggingFace AutoTokenizer if mistral_common is not available.
    """
    tekken_path = model_path / "tekken.json"
    if tekken_path.exists():
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekken
            return Tekken.from_file(str(tekken_path))
        except ImportError:
            pass

    # Fallback to transformers
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path))


def load(model_path: str) -> Tuple["Model", Any]:
    """Load a converted MLX Voxtral TTS model from directory.

    Args:
        model_path: Path to directory containing config.json and *.safetensors

    Returns:
        (model, tokenizer) tuple
    """
    path = Path(model_path)

    # Load config
    with open(path / "config.json") as f:
        config_data = json.load(f)
    config = ModelConfig.from_dict(config_data)

    # Instantiate model
    model = Model(config)

    # Handle quantization if config specifies it
    quant_config = config_data.get("quantization")
    if quant_config:
        nn.quantize(
            model,
            bits=quant_config.get("bits", 4),
            group_size=quant_config.get("group_size", 64),
        )

    # Load weights (merge shards if multiple files)
    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    all_weights = []
    for wf in weight_files:
        weights = list(mx.load(str(wf)).items())
        all_weights.extend(weights)
    model.load_weights(all_weights, strict=False)

    # Load tokenizer
    tokenizer = load_tokenizer(path)

    return model, tokenizer


def _sample_token(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> mx.array:
    """Sample a token from logits.

    Args:
        logits: [B, vocab_size] logits
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        token: [B] sampled token IDs
    """
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_p < 1.0:
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        mask = cumulative_probs - mx.softmax(sorted_logits, axis=-1) >= top_p
        sorted_logits = mx.where(mask, mx.array(-float("inf")), sorted_logits)
        logits = mx.zeros_like(logits)
        logits = logits.at[mx.arange(logits.shape[0])[:, None], sorted_indices].set(sorted_logits)

    return mx.random.categorical(logits, axis=-1)


class Model(nn.Module):
    """Voxtral TTS Model.

    End-to-end text-to-speech model combining:
    - Mistral-based LLM for text understanding and semantic token generation
    - Flow-matching acoustic transformer for acoustic code generation
    - Voxtral codec for waveform synthesis
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config

        audio_args = config.get_audio_model_args()
        codec_args = config.get_codec_args()

        # Stage 1: LLM Transformer Decoder (generates semantic tokens)
        self.language_model = MistralTransformerDecoder(
            dim=config.dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
        )

        # Stage 2: Flow-Matching Acoustic Transformer
        self.acoustic_transformer = FlowMatchingAcousticTransformer(
            audio_args=audio_args,
            llm_dim=config.dim,
        )

        # Stage 3: Voxtral Codec Decoder
        self.audio_tokenizer = VoxtralCodec(codec_args)

        # Audio codebook embeddings — matches safetensors key:
        #   mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight: [9088, 3072]
        # Layout: [semantic(8192) + acoustic(36*21=756) + padding -> 9088 total]
        # This is remapped to: audio_token_embedding.embeddings.weight
        n_codebooks = audio_args.n_codebook  # 37
        codebook_sizes = audio_args.codebook_sizes
        total_audio_vocab = sum(codebook_sizes)  # 8192 + 36*21 = 8948
        padded_vocab = ((total_audio_vocab + 127) // 128) * 128  # pad to 128 -> 9088
        self.audio_token_embedding = nn.Module()
        self.audio_token_embedding.embeddings = nn.Embedding(padded_vocab, config.dim)

        # Voice embeddings storage
        self._voice_embeddings = {}

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate

    def load_voice_embedding(self, voice_path: Union[str, Path]) -> mx.array:
        """Load a voice embedding from a .pt file.

        Args:
            voice_path: Path to voice embedding file

        Returns:
            Voice embedding array [T, dim] — a sequence of embeddings
            representing the reference voice
        """
        voice_path = str(voice_path)
        if voice_path not in self._voice_embeddings:
            import torch
            data = torch.load(voice_path, map_location="cpu", weights_only=True)
            self._voice_embeddings[voice_path] = mx.array(data.numpy())
        return self._voice_embeddings[voice_path]

    def _make_generation_result(
        self, audio: mx.array, start_time: float,
        token_count: int, segment_idx: int,
    ) -> GenerationResult:
        """Create a GenerationResult from generated audio."""
        samples = audio.shape[0] if audio.ndim == 1 else audio.shape[-1]
        audio_duration_seconds = samples / self.sample_rate
        elapsed = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed if elapsed > 0 else 0

        hours = int(audio_duration_seconds // 3600)
        mins = int((audio_duration_seconds % 3600) // 60)
        secs = int(audio_duration_seconds % 60)
        ms = int((audio_duration_seconds % 1) * 1000)

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}",
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": round(token_count / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _tokenize_text(self, text: str, tokenizer) -> mx.array:
        """Tokenize text for the LLM.

        The Tekken tokenizer handles the text encoding. We need to wrap
        the text with appropriate control tokens for TTS.
        """
        # Format: [BOS] text [AUDIO] ... generate audio tokens ...
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return mx.array([tokens])

    def generate(
        self,
        text: str,
        tokenizer=None,
        voice: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_audio_frames: int = 2048,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        The correct generation flow:
        1. LLM processes text tokens, then autoregressively generates semantic tokens
        2. For each semantic token, flow-matching generates 36 acoustic tokens
        3. All codes are decoded to waveform by the codec

        Args:
            text: Input text to synthesize
            tokenizer: Tokenizer instance (from load())
            voice: Path to voice embedding file, or voice name
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Top-p sampling threshold
            max_audio_frames: Maximum number of audio frames to generate
            repetition_penalty: Repetition penalty factor
            verbose: Print progress information

        Yields:
            GenerationResult objects containing audio chunks
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required — use load() to get model and tokenizer")

        start_time = time.perf_counter()
        audio_args = self.config.get_audio_model_args()

        # Prepare input tokens
        input_ids = self._tokenize_text(text, tokenizer)
        B = input_ids.shape[0]

        # Prepend voice embedding if specified
        voice_emb = None
        if voice is not None:
            voice_emb = self.load_voice_embedding(voice)
            # voice_emb is [T_voice, dim], expand to [B, T_voice, dim]
            if voice_emb.ndim == 2:
                voice_emb = voice_emb[None, :, :].broadcast_to((B,) + voice_emb.shape)

        # Stage 1: Initial forward pass with text tokens
        cache = None
        if voice_emb is not None:
            # First process voice conditioning through the LLM
            _, _, cache = self.language_model(input_embeds=voice_emb, cache=cache)

        # Process text tokens
        logits, hidden_states, cache = self.language_model(tokens=input_ids, cache=cache)

        # Now generate audio frames autoregressively
        # The LLM generates semantic tokens; the acoustic transformer generates acoustic codes
        all_semantic_codes = []
        all_acoustic_codes = []

        for frame_idx in range(max_audio_frames):
            # Sample semantic token from LLM logits
            # The semantic token vocabulary is a subset of the LLM vocab
            # (audio_token_id marks where audio tokens start in the LLM vocab)
            last_logits = logits[:, -1, :]  # [B, vocab_size]
            sem_token = _sample_token(last_logits, temperature, top_p)  # [B]

            # Check for end-of-audio token
            if mx.any(sem_token == audio_args.audio_token_id + END_AUDIO_ID).item():
                break

            # Map from LLM vocab token to semantic codebook index
            sem_code = sem_token - audio_args.audio_token_id - AUDIO_CODE_OFFSET

            # Get the last hidden state for the acoustic transformer
            last_hidden = hidden_states[:, -1, :]  # [B, dim]

            # Stage 2: Flow matching to get acoustic codes
            acou_code = self.acoustic_transformer.decode_one_frame(last_hidden)  # [B, 36]

            all_semantic_codes.append(sem_code)
            all_acoustic_codes.append(acou_code)

            # Encode audio codes back into embedding for next LLM step
            audio_emb = self._encode_audio_frame(sem_code, acou_code)  # [B, dim]
            audio_emb = audio_emb[:, None, :]  # [B, 1, dim]

            # Feed audio embedding directly into LLM (bypass token lookup)
            logits, hidden_states, cache = self.language_model(
                input_embeds=audio_emb, cache=cache,
            )

            if verbose and frame_idx % 50 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Frame {frame_idx}/{max_audio_frames}, "
                      f"elapsed: {elapsed:.1f}s")

            mx.eval(sem_code, acou_code)

        if not all_semantic_codes:
            return

        # Stack all codes
        semantic_codes = mx.stack(all_semantic_codes, axis=1)  # [B, T]
        acoustic_codes = mx.stack(all_acoustic_codes, axis=1)  # [B, T, 36]

        if verbose:
            print(f"  Generated {semantic_codes.shape[1]} audio frames")

        # Stage 3: Decode audio codes to waveform
        audio = self.audio_tokenizer.decode(semantic_codes, acoustic_codes)  # [B, samples]
        audio = audio[0]  # Take first batch item

        mx.eval(audio)

        token_count = input_ids.shape[1] + len(all_semantic_codes)

        yield self._make_generation_result(
            audio=audio,
            start_time=start_time,
            token_count=token_count,
            segment_idx=0,
        )

    def _encode_audio_frame(self, semantic_code: mx.array, acoustic_codes: mx.array) -> mx.array:
        """Encode one frame of audio codes into an embedding for the LLM.

        The combined embedding table (audio_token_embedding.embeddings) is laid out as:
        [sem_0..sem_8191, acou_0_0..acou_0_20, acou_1_0..acou_1_20, ..., acou_35_0..acou_35_20]
        Total: 8192 + 36*21 = 8948 entries (padded to 9088)

        Args:
            semantic_code: [B] semantic codebook index (0-8191)
            acoustic_codes: [B, 36] acoustic codebook indices (0-20 each)

        Returns:
            embedding: [B, dim] combined audio embedding (sum of all codebook embeddings)
        """
        audio_args = self.config.get_audio_model_args()
        emb_table = self.audio_token_embedding.embeddings

        # Semantic embedding: first 8192 entries
        sem_emb = emb_table(semantic_code)  # [B, dim]

        # Acoustic embeddings: each codebook k starts at offset 8192 + k*21
        base_offset = audio_args.semantic_codebook_size
        acou_emb = mx.zeros_like(sem_emb)
        for k in range(acoustic_codes.shape[1]):
            k_offset = base_offset + k * audio_args.acoustic_codebook_size
            k_tokens = acoustic_codes[:, k] + k_offset
            acou_emb = acou_emb + emb_table(k_tokens)

        return sem_emb + acou_emb
