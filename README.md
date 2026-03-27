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

# With quantization (q2, q4, q8)
python3 -m voxtral_tts.convert --output-dir mlx_model --quantize q4
```

### 3. Test the model

```bash
# Construction test (no weights needed)
python3 -m voxtral_tts.test_model --test construction

# Weight loading test
python3 -m voxtral_tts.test_model --model-path mlx_model --test loading

# Weight analysis
python3 -m voxtral_tts.test_model --model-path mlx_model --test weights
```

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
