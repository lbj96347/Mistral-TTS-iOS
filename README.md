# Voxtral TTS MLX

MLX port of Mistral's [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech model for on-device inference on Apple Silicon.

Converts the HuggingFace model (~8GB) into MLX format with optional quantization (Q2–Q8) for efficient local generation.

## Architecture

Three-stage pipeline:

```
Text → LLM Decoder → Flow-Matching Transformer → Codec → 24kHz WAV
```

| Component | Params | Description |
|---|---|---|
| Transformer Decoder | 3.4B | Mistral/Ministral-3B based LLM; text tokens → hidden states |
| Acoustic Transformer | 390M | Flow-matching with Euler ODE + CFG; hidden states → audio codes |
| Voxtral Codec | 300M | Conv-transformer autoencoder; codes → 24kHz waveform |

Each audio frame consists of 37 discrete tokens (1 semantic + 36 acoustic) at 12.5 Hz frame rate.

## Setup

Requires Python 3.10+ and Apple Silicon (M1/M2/M3/M4).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### 1. Inspect HuggingFace model weights

```bash
python3 -m voxtral_tts.convert --inspect
```

### 2. Convert model to MLX format

```bash
# Full precision
python3 -m voxtral_tts.convert --output-dir mlx_model

# With uniform quantization (q2, q4, q6, q8)
python3 -m voxtral_tts.convert --output-dir mlx_model --quantize q4

# Mixed quantization for iOS (Q4 LLM/acoustic, Q2 codec for size savings)
python3 -m voxtral_tts.convert --output-dir mlx_model_ios \
    --quantize-llm q4 --quantize-acoustic q4 --quantize-codec q2
```

#### Quantization Guide

| Target Device | RAM | Recommended Command | Estimated Size |
|---|---|---|---|
| Mac (M1/M2/M3/M4) | 16GB+ | `--quantize q4` | ~2.1 GB |
| Mac (8GB) | 8GB | `--quantize q4` | ~2.1 GB |
| iPhone 15 Pro / iPad Pro | 8GB | `--quantize-llm q4 --quantize-acoustic q4 --quantize-codec q2` | ~2.1 GB |
| iPhone 16 Pro | 8GB | `--quantize-llm q4 --quantize-acoustic q4 --quantize-codec q2` | ~2.1 GB |

Mixed quantization applies different bit widths per component:
- `--quantize-llm` — Language model (3.4B params, minimum Q4 enforced)
- `--quantize-acoustic` — Acoustic transformer (390M params, minimum Q4 enforced)
- `--quantize-codec` — Codec (300M params, tolerates Q2 since it's not in autoregressive loop)

**Note:** The LLM and acoustic transformer require Q4 minimum for intelligible speech. Values below Q4 are automatically clamped with a warning.

Per-component flags override `--quantize` when both are specified.

### 3. Test the model

```bash
# Construction test (no weights needed)
python3 -m voxtral_tts.test_model --test construction

# Weight loading test
python3 -m voxtral_tts.test_model --model-path mlx_model --test loading

# Weight analysis
python3 -m voxtral_tts.test_model --model-path mlx_model --test weights
```

## iOS App

The `VoxtralTTS/` directory contains a SwiftUI iOS app that runs the model on-device using MLX-Swift.

### iOS Memory Optimizations

The iOS app includes several optimizations to fit within iOS jetsam memory limits:

- **Quantized embeddings** — Both `Linear` and `Embedding` layers (including the 131K-vocab token embedding table) are properly loaded as quantized modules, saving ~800MB compared to unquantized embedding loading.
- **Quantized output projection** — Tied embedding output uses `quantizedMM` instead of dequantizing the full weight matrix.
- **GPU cache limit** — MLX buffer cache is capped at 20MB on iOS (per MLX recommendations) to prevent unbounded memory growth during autoregressive generation.
- **Periodic cache clearing** — `Memory.clearCache()` is called every 50 frames during generation.
- **Mixed quantization** — The codec (not in autoregressive loop) can use Q2 while LLM and acoustic transformer stay at Q4 minimum.

### Running on iOS

1. Convert the model with iOS-optimized quantization (see Quantization Guide above)
2. Copy the output directory (e.g., `mlx_model_ios/`) to your device
3. Open the Xcode project and build for your device (simulator is not supported — MLX requires Metal GPU)
4. Select the model directory in the app and generate

## Project Structure

```
voxtral_tts/
├── voxtral_tts.py            # Main model: generate() + load()
├── transformer_decoder.py    # LLM decoder (Mistral-based)
├── acoustic_transformer.py   # Flow-matching acoustic model
├── codec.py                  # Audio codec (codes → waveform)
├── config.py                 # Model configuration dataclasses
├── convert.py                # HF → MLX weight converter
└── test_model.py             # Construction/loading/weight tests
```

## Dependencies

- [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-examples) — Apple ML framework
- [mistral-common](https://github.com/mistralai/mistral-common) — Tekken tokenizer + voice embeddings
- [safetensors](https://github.com/huggingface/safetensors) — Weight file format
- `torch` (dev only) — For `.pt` voice embedding conversion

## License

Apache-2.0
