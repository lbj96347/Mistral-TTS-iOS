# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX port of Mistral's Voxtral-4B-TTS-2603 text-to-speech model for on-device inference on Apple Silicon. Converts the HuggingFace model (single `consolidated.safetensors`, 8GB) into MLX format with optional quantization (Q2-Q8).

**Source model**: `mistralai/Voxtral-4B-TTS-2603`
**Reference implementation**: vLLM-Omni (`vllm/model_executor/models/voxtral.py`)

## Commands

```bash
# Setup venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run construction tests (no weights needed)
python3 -m voxtral_tts.test_model --test construction

# Inspect actual HF model weight keys (run before converting!)
python3 -m voxtral_tts.convert --inspect

# Convert model from HuggingFace to MLX
python3 -m voxtral_tts.convert --output-dir mlx_model
python3 -m voxtral_tts.convert --output-dir mlx_model --quantize q4

# Convert with mixed quantization for iOS (q2 LLM + q4 audio)
python3 -m voxtral_tts.convert --output-dir mlx_model_ios \
    --quantize-llm q2 --quantize-acoustic q4 --quantize-codec q4

# Test weight loading after conversion
python3 -m voxtral_tts.test_model --model-path mlx_model --test loading

# Analyze converted model weights
python3 -m voxtral_tts.test_model --model-path mlx_model --test weights
```

## Architecture

Three-stage pipeline: `Text → LLM Decoder → Flow-Matching Transformer → Codec → 24kHz WAV`

| Component | File | Params | Purpose |
|---|---|---|---|
| Transformer Decoder | `transformer_decoder.py` | 3.4B | Mistral/Ministral-3B based LLM, text tokens → hidden states |
| Acoustic Transformer | `acoustic_transformer.py` | 390M | Flow-matching with Euler ODE + CFG, hidden states → semantic + acoustic codes |
| Voxtral Codec | `codec.py` | 300M | Conv-transformer autoencoder, codes → 24kHz waveform |
| Main Model | `voxtral_tts.py` | — | Orchestrates all stages, `Model` class with `generate()`, `load()` function |
| Config | `config.py` | — | Dataclasses: `ModelConfig`, `MultimodalAudioModelArgs`, `AudioTokenizerArgs`, `AcousticTransformerArgs` |
| Converter | `convert.py` | — | HF safetensors → MLX format with weight remapping and quantization |

### Audio Token Structure

Each audio frame = 37 discrete tokens: 1 semantic (8192-vocab VQ) + 36 acoustic (21-level FSQ). Frame rate: 12.5 Hz. All codebook embeddings are summed into a single embedding vector fed back to the LLM.

### Weight Key Remapping (convert.py / Model.sanitize)

HF `consolidated.safetensors` keys are remapped to MLX model structure:
- `layers.*` → `language_model.layers.*`
- `tok_embeddings.*` → `language_model.tok_embeddings.*`
- `norm.*` / `output.*` → `language_model.norm.*` / `language_model.output.*`
- `mm_audio_embeddings.*` → `audio_token_embedding.*`
- `acoustic_transformer.*` and `audio_tokenizer.*` kept as-is
- Conv1d 3D weights transposed: PyTorch `[out, in, kernel]` → MLX `[out, kernel, in]`

### Current Status

**Fixed**: Model loading (`load()` function), tokenizer loading, audio embedding injection
(uses `input_embeds` instead of dummy tokens), semantic token generation (LLM generates
semantic tokens, acoustic transformer only generates acoustic codes), quantization
(properly stores scales/biases), Conv1d detection (explicit key matching instead of substring).

**Remaining**: Weight key remapping has NOT been verified against the actual
`consolidated.safetensors` — run `python3 -m voxtral_tts.convert --inspect` first
and adjust `remap_weights()` based on the real keys. The `params.json` structure
also needs verification.

## Key Dependencies

- `mlx` / `mlx-lm`: Apple MLX framework for on-device ML
- `mistral-common[audio]`: Tekken tokenizer + voice embeddings
- `safetensors`: Weight file format
- `torch` (dev only): For voice embedding `.pt` file conversion
