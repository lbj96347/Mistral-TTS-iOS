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

# Special token IDs from the Tekken tokenizer (absolute IDs)
BOS_TOKEN_ID = 1
AUDIO_TOKEN_ID = 24           # [AUDIO] — voice embedding placeholder
BEGIN_AUDIO_TOKEN_ID = 25     # [BEGIN_AUDIO] — audio section marker
AUDIO_TO_TEXT_TOKEN_ID = 35   # [REPEAT_AUDIO_TEXT] — transition from text to audio generation
TEXT_TO_AUDIO_TOKEN_ID = 36   # [NEXT_AUDIO_TEXT] — transition from voice to text


def load_tokenizer(model_path: Path):
    """Load the MistralTokenizer from a model directory.

    Returns a MistralTokenizer which supports encode_speech_request() for
    proper TTS prompt formatting. Falls back to raw Tekkenizer if
    MistralTokenizer is not available.
    """
    tekken_path = model_path / "tekken.json"
    if tekken_path.exists():
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            return MistralTokenizer.from_file(str(tekken_path))
        except ImportError:
            pass
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
        q_group_size = quant_config.get("group_size", 64)

        def _can_quantize(path: str, module: nn.Module) -> bool:
            if not hasattr(module, "to_quantized"):
                return False
            if isinstance(module, nn.Linear):
                # Skip layers whose input dim isn't divisible by group_size
                if module.weight.shape[-1] % q_group_size != 0:
                    return False
                if min(module.weight.shape) <= q_group_size:
                    return False
            return True

        nn.quantize(
            model,
            bits=quant_config.get("bits", 4),
            group_size=q_group_size,
            class_predicate=_can_quantize,
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
        # Sort descending
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        # Mask tokens beyond top-p threshold (keep first token always)
        mask = (cumulative_probs - sorted_probs) >= top_p
        sorted_logits = mx.where(mask, mx.array(-float("inf")), sorted_logits)
        # Sample from sorted space, then map back to original indices
        sampled_sorted = mx.random.categorical(sorted_logits, axis=-1)  # [B]
        return mx.take_along_axis(
            sorted_indices, sampled_sorted[:, None], axis=-1,
        ).squeeze(-1)

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
            # Voice embeddings may be BFloat16 — convert to float32 for numpy
            self._voice_embeddings[voice_path] = mx.array(data.float().numpy())
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

    def _build_prompt(
        self, text: str, tokenizer, voice_emb: Optional[mx.array] = None,
    ) -> mx.array:
        """Build the TTS prompt token sequence.

        Correct format (from mistral_common encode_speech_request):
          [BOS] [BEGIN_AUDIO] [AUDIO]*N [TEXT_TO_AUDIO] text_tokens [AUDIO_TO_TEXT] [BEGIN_AUDIO]

        Where [AUDIO]*N are placeholder positions replaced by voice embeddings.
        """
        # Try MistralTokenizer.encode_speech_request() first
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.protocol.speech.request import SpeechRequest
            if isinstance(tokenizer, MistralTokenizer):
                # Determine voice name from embedding or use a default
                voice_name = getattr(self, '_current_voice_name', None)
                if voice_name:
                    req = SpeechRequest(input=text, voice=voice_name)
                    result = tokenizer.encode_speech_request(req)
                    return mx.array([result.tokens])
        except (ImportError, Exception):
            pass

        # Manual prompt construction (also used by Swift side)
        # Get text tokens without BOS/EOS
        if hasattr(tokenizer, 'instruct_tokenizer'):
            # MistralTokenizer wrapper — access underlying Tekkenizer
            raw_tok = tokenizer.instruct_tokenizer.tokenizer
            text_tokens = raw_tok.encode(text, bos=False, eos=False)
        elif hasattr(tokenizer, 'encode') and 'bos' in str(type(tokenizer)):
            text_tokens = tokenizer.encode(text, bos=False, eos=False)
        else:
            # HuggingFace fallback
            text_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Number of voice embedding positions
        n_voice = voice_emb.shape[0] if voice_emb is not None and voice_emb.ndim >= 1 else 0

        # Build: [BOS, BEGIN_AUDIO, AUDIO*N, TEXT_TO_AUDIO, *text, AUDIO_TO_TEXT, BEGIN_AUDIO]
        tokens = [BOS_TOKEN_ID, BEGIN_AUDIO_TOKEN_ID]
        tokens.extend([AUDIO_TOKEN_ID] * n_voice)
        tokens.append(TEXT_TO_AUDIO_TOKEN_ID)
        tokens.extend(text_tokens)
        tokens.append(AUDIO_TO_TEXT_TOKEN_ID)
        tokens.append(BEGIN_AUDIO_TOKEN_ID)

        return mx.array([tokens])

    @staticmethod
    def _apply_repetition_penalty(
        logits: mx.array, past_tokens: List[int], penalty: float,
    ) -> mx.array:
        """Apply repetition penalty to logits for previously generated tokens."""
        if penalty == 1.0 or not past_tokens:
            return logits
        unique_tokens = list(set(past_tokens))
        indices = mx.array(unique_tokens)
        scores = logits[:, indices]
        # If score > 0 divide by penalty, if < 0 multiply by penalty
        penalized = mx.where(scores > 0, scores / penalty, scores * penalty)
        logits[:, indices] = penalized
        return logits

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
        1. Build prompt: [BOS][BEGIN_AUDIO][AUDIO]*N[TEXT_TO_AUDIO] text [AUDIO_TO_TEXT][BEGIN_AUDIO]
        2. Replace [AUDIO] positions with voice embeddings, forward through LLM
        3. Autoregressively generate semantic tokens (masked to audio vocab)
        4. For each semantic token, flow-matching generates 36 acoustic tokens
        5. All codes are decoded to waveform by the codec

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

        # Load voice embedding if specified
        voice_emb = None
        if voice is not None:
            voice_path = Path(voice)
            if voice_path.exists():
                voice_emb = self.load_voice_embedding(voice)
                # Store voice name for MistralTokenizer path
                self._current_voice_name = voice_path.stem
            else:
                # Treat as a voice name (for MistralTokenizer)
                self._current_voice_name = voice

        # Build prompt with correct TTS template
        input_ids = self._build_prompt(text, tokenizer, voice_emb)
        B = input_ids.shape[0]

        if verbose:
            print(f"  Prompt tokens: {input_ids.shape[1]}, "
                  f"voice positions: {voice_emb.shape[0] if voice_emb is not None else 0}")

        # Build input embeddings: token embeddings with voice embeddings at [AUDIO] positions
        token_embs = self.language_model.tok_embeddings(input_ids)  # [B, T, dim]

        if voice_emb is not None:
            # voice_emb is [N, dim] — expand to [B, N, dim]
            if voice_emb.ndim == 2:
                voice_emb_expanded = voice_emb[None, :, :]  # [1, N, dim]
            else:
                voice_emb_expanded = voice_emb

            n_voice = voice_emb.shape[0] if voice_emb.ndim == 2 else voice_emb.shape[1]

            # [AUDIO] positions are contiguous starting at index 2
            # (after BOS=1 and BEGIN_AUDIO=25)
            first_audio_pos = 2
            last_audio_pos = first_audio_pos + n_voice

            # Slice: [before_audio | voice_emb | after_audio]
            before = token_embs[:, :first_audio_pos, :]
            after = token_embs[:, last_audio_pos:, :]
            voice_block = mx.broadcast_to(
                voice_emb_expanded, (B, n_voice, token_embs.shape[-1])
            )
            input_embeds = mx.concatenate([before, voice_block, after], axis=1)
        else:
            input_embeds = token_embs

        # Stage 1: Forward pass with combined embeddings (single pass)
        logits, hidden_states, cache = self.language_model(
            input_embeds=input_embeds, cache=None,
        )

        # Audio token range for logit masking
        audio_start = audio_args.audio_token_id  # 24
        audio_end = audio_start + AUDIO_CODE_OFFSET + audio_args.semantic_codebook_size  # 8218

        # Autoregressive audio frame generation
        all_semantic_codes = []
        all_acoustic_codes = []
        past_audio_tokens = []

        for frame_idx in range(max_audio_frames):
            # Extract last logits and mask to audio token range only
            last_logits = logits[:, -1, :]  # [B, vocab_size=131072]

            # Apply repetition penalty before masking
            if repetition_penalty != 1.0 and past_audio_tokens:
                last_logits = self._apply_repetition_penalty(
                    last_logits, past_audio_tokens, repetition_penalty,
                )

            # Slice to audio range [24..8218) — 8194 tokens
            audio_logits = last_logits[:, audio_start:audio_end]  # [B, 8194]
            relative_token = _sample_token(audio_logits, temperature, top_p)  # [B]
            sem_token = relative_token + audio_start  # back to absolute vocab ID

            # Check for end-of-audio: tokens 24 (EMPTY) or 25 (BEGIN_AUDIO) = stop
            sem_code_raw = sem_token - audio_args.audio_token_id  # relative: 0 or 1 = special
            if mx.any(sem_code_raw < AUDIO_CODE_OFFSET).item():
                if verbose:
                    print(f"  End-of-audio at frame {frame_idx}")
                break

            # Map from absolute LLM vocab token to semantic codebook index (0-8191)
            sem_code = sem_token - audio_args.audio_token_id - AUDIO_CODE_OFFSET

            # Track for repetition penalty
            past_audio_tokens.append(sem_token.item())

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
