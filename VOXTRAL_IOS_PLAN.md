# Voxtral-4B-TTS on iOS: Analysis and Options

## Executive Summary

**Honest assessment**: There is currently **no existing solution** to run Voxtral-4B-TTS-2603 offline on iOS using existing tools. The model was released March 26, 2026 (yesterday), and the community hasn't had time to create iOS-compatible conversions.

**The constraint conflict**: You specified must be offline + must be Voxtral + must use existing tools. Unfortunately, these three requirements cannot all be satisfied today.

## Model Architecture

Voxtral-4B-TTS consists of three components ([source](https://mistral.ai/news/voxtral-tts)):

| Component | Size | Purpose |
|-----------|------|---------|
| Transformer Decoder | 3.4B params | Text → audio token generation |
| Flow-Matching Transformer | 390M params | Acoustic token refinement |
| Voxtral Codec | 300M params | Audio tokens → 24kHz waveform |

**Total**: ~4.1B parameters, 8GB in BF16 format

## Current Conversion Status

| Model | Direction | GGUF Support | Notes |
|-------|-----------|--------------|-------|
| Voxtral-4B-TTS-2603 | Text→Speech | **None** | The model you want |
| Voxtral-Mini-3B-2507 | Speech→Text | Yes | ASR model (opposite direction) |
| OuteTTS-0.2-500M | Text→Speech | Yes | Alternative TTS in llama.cpp |

**Key Issue**: llama.cpp currently only supports **decoder-only** models for GGUF. The Voxtral TTS model has non-standard components (flow-matching transformer, custom codec) that aren't supported yet.

## Options

### Option A: Use Alternative TTS Model (OuteTTS)
- **Pros**: Already works with llama.cpp on iOS, ~500M params (much smaller)
- **Cons**: Different voice quality, fewer languages, no voice cloning
- **Effort**: Low - can use existing tools

### Option B: Server-Based Architecture
- **Pros**: Full Voxtral quality, no conversion needed
- **Cons**: Requires network, adds latency, server costs
- **Effort**: Medium - iOS app + vLLM server

### Option C: Convert to GGUF (Pioneer Path)
Requires implementing:
1. Voxtral architecture support in llama.cpp
2. Flow-matching transformer in ggml
3. Voxtral Codec decoder in ggml
4. Conversion scripts

- **Pros**: Native on-device, full quality
- **Cons**: Significant engineering (weeks/months), may not fit in iOS memory even quantized
- **Effort**: Very High

### Option D: CoreML Conversion
Convert each component to CoreML format:
1. Export transformer decoder via torch → coremltools
2. Export flow-matching transformer
3. Export Voxtral Codec decoder

- **Pros**: Apple-optimized, uses Neural Engine
- **Cons**: Complex multi-model pipeline, memory constraints
- **Effort**: High

### Option E: Rust/Burn Implementation (Hybrid)
Port the [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) approach:
- Use Burn ML framework
- Q4 quantization (~2.5GB for the similar STT model)
- Metal backend for GPU acceleration

- **Pros**: Proven approach for similar model (STT version runs in browser)
- **Cons**: Need to implement TTS components, still complex
- **Effort**: High

## Memory Constraints

| Quantization | Estimated Size | iPhone Feasibility |
|--------------|---------------|-------------------|
| BF16 | ~8GB | Not viable |
| Q8 | ~4GB | Borderline (iPhone 15 Pro Max: 8GB) |
| Q4 | ~2-2.5GB | Feasible on newer iPhones |
| Q2 | ~1.5GB | Feasible, quality degradation |

## Recommended Path

**For production use**: Option B (Server-based) - Get Voxtral quality without mobile constraints

**For on-device TTS**: Option A (OuteTTS) - Working solution today

**For research/contribution**: Option C or D - Contribute to the ecosystem

## Current Ecosystem Status (March 2026)

| Project | Voxtral STT | Voxtral TTS | iOS Support |
|---------|-------------|-------------|-------------|
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Yes | **No** | Yes (Swift pkg) |
| [voxmlx](https://github.com/awni/voxmlx) | Yes | **No** | No (Python only) |
| [speech-swift](https://github.com/ivan-digital/qwen3-asr-swift) | Yes | **No** (has Kokoro, Qwen3-TTS) | Yes |
| llama.cpp | Yes (GGUF) | **No** (OuteTTS only) | Yes (Metal) |
| vLLM-Omni | N/A | **Yes** | No (server only) |

**Key insight**: All iOS-compatible projects support Voxtral for **STT** (speech-to-text), but none support Voxtral **TTS** yet.

## Realistic Paths Forward

### Path 1: Wait for Community Support
- **mlx-audio** is actively maintained and adding models
- Voxtral TTS support could be added within weeks/months
- Watch: https://github.com/Blaizzy/mlx-audio/issues

### Path 2: Contribute the Implementation
Convert Voxtral TTS to MLX format yourself:
1. Port the 3 components to MLX
2. Implement Voxtral Codec decoder
3. Add Swift bindings for iOS
4. Contribute back to mlx-audio

### Path 3: Use Alternative TTS (Compromise)
If you can accept a different model, these work on iOS today:
- **Kokoro TTS** - 82M params, 50 voices, CoreML, ~325MB
- **Qwen3-TTS** - Voice cloning, emotion control, MLX

### Path 4: Hybrid Architecture
- Use Voxtral TTS via API when online
- Fall back to Kokoro TTS when offline

## Implementation Plan: Build Voxtral TTS for iOS

Since you're willing to build the conversion, here's the technical roadmap:

### Phase 1: Environment Setup
1. Clone mlx-audio repo (best foundation for iOS TTS)
2. Download Voxtral-4B-TTS-2603 weights from HuggingFace
3. Set up development environment (Python 3.10+, MLX, PyTorch)

### Phase 2: Understand the Architecture
Study the three components from the safetensors file:
```
consolidated.safetensors (8GB)
├── Transformer Decoder (3.4B) - Ministral-3B based
├── Flow-Matching Transformer (390M) - Acoustic refinement
├── Voxtral Codec Decoder (300M) - Tokens → waveform
└── Voice Embeddings (in voice_embedding/ dir)
```

### Phase 3: Convert Each Component to MLX

**Step 3.1: Transformer Decoder**
- Based on Ministral-3B (Mistral architecture)
- MLX already supports Mistral models
- Use `mlx_lm.convert` as starting point

**Step 3.2: Flow-Matching Transformer**
- Novel architecture for acoustic token generation
- Need to implement flow-matching layers in MLX
- Reference: vLLM-Omni implementation

**Step 3.3: Voxtral Codec Decoder**
- Convolutional-transformer autoencoder
- Converts 37 tokens (1 semantic + 36 acoustic) → 24kHz audio
- Most complex part - custom architecture

### Phase 4: Implement Inference Pipeline
```
Text → Tokenize (Tekken) → Decoder → Flow Transformer → Codec → WAV
```

### Phase 5: Quantization
- Target Q4 quantization (~2GB total)
- Test quality vs size tradeoffs
- Ensure iOS memory compatibility

### Phase 6: Swift/iOS Integration
- Create MLX Swift bindings
- Package as Swift Package Manager module
- Implement audio playback with AVAudioEngine

### Key Files to Study

**From HuggingFace model:**
- `params.json` - Model configuration
- `tekken.json` - Tokenizer (14.9MB)
- `voice_embedding/` - Reference voices

**From vLLM-Omni (reference implementation):**
```bash
git clone https://github.com/vllm-project/vllm-omni.git
```
- `vllm_omni/model_executor/stage_configs/voxtral_tts.yaml` - Stage config
- `vllm_omni/examples/offline_inference/voxtral_tts/end2end.py` - Inference example
- PRs #1803, #2026, #2056 - Implementation commits

### First Concrete Steps

```bash
# 1. Clone reference implementation
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

# 2. Study the Voxtral TTS inference code
cat vllm_omni/model_executor/stage_configs/voxtral_tts.yaml
python examples/offline_inference/voxtral_tts/end2end.py

# 3. Clone mlx-audio (target framework for iOS)
git clone https://github.com/Blaizzy/mlx-audio.git

# 4. Download model weights
huggingface-cli download mistralai/Voxtral-4B-TTS-2603
```

### Estimated Effort
- Phase 1-2: 1-2 days
- Phase 3: 1-2 weeks (main engineering work)
- Phase 4-5: 3-5 days
- Phase 6: 3-5 days
- **Total**: 3-4 weeks for experienced ML engineer

## Key Resources

- Voxtral TTS weights: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
- mlx-audio (closest to adding support): https://github.com/Blaizzy/mlx-audio
- Research paper: https://mistral.ai/static/research/voxtral-tts.pdf
- vLLM-Omni (server reference): https://github.com/vllm-project/vllm-omni
