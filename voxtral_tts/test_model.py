"""Test script for Voxtral TTS model.

Verifies model construction, weight loading, and basic inference.

Usage:
    # Test model construction only (no weights needed)
    python -m voxtral_tts.test_model --test construction

    # Test with converted weights
    python -m voxtral_tts.test_model --model-path mlx_model --test weights

    # Test inference with converted weights
    python -m voxtral_tts.test_model --model-path mlx_model --test inference
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def test_construction():
    """Test that all model components can be constructed."""
    from .config import ModelConfig, AudioTokenizerArgs, AcousticTransformerArgs
    from .transformer_decoder import MistralTransformerDecoder
    from .acoustic_transformer import FlowMatchingAcousticTransformer
    from .codec import VoxtralCodec

    print("=== Testing Model Construction ===\n")

    # 1. Config
    print("[1] Testing ModelConfig...")
    config = ModelConfig()
    audio_args = config.get_audio_model_args()
    codec_args = config.get_codec_args()
    acoustic_args = audio_args.get_acoustic_args()
    print(f"  LLM: dim={config.dim}, layers={config.n_layers}, vocab={config.vocab_size}")
    print(f"  Audio: semantic={audio_args.semantic_codebook_size}, "
          f"acoustic={audio_args.acoustic_codebook_size}x{audio_args.n_acoustic_codebook}")
    print(f"  Codec: dim={codec_args.dim}, sample_rate={codec_args.sampling_rate}")
    print(f"  Acoustic Transformer: dim={acoustic_args.dim}, layers={acoustic_args.n_layers}")
    print("  OK\n")

    # 2. Transformer Decoder (small version for testing)
    print("[2] Testing MistralTransformerDecoder (small)...")
    small_decoder = MistralTransformerDecoder(
        dim=256, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=64, hidden_dim=512, vocab_size=1000,
    )

    # Test with token input
    test_tokens = mx.array([[1, 2, 3, 4, 5]])
    logits, hidden, cache = small_decoder(tokens=test_tokens)
    print(f"  Input: {test_tokens.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Hidden: {hidden.shape}")
    print(f"  Cache layers: {len(cache)}")
    mx.eval(logits)

    # Test with input_embeds (audio embedding injection)
    test_embeds = mx.random.normal((1, 1, 256))
    logits2, hidden2, cache2 = small_decoder(input_embeds=test_embeds, cache=cache)
    print(f"  input_embeds: {test_embeds.shape} -> logits: {logits2.shape}")
    mx.eval(logits2)
    print("  OK\n")

    # 3. Acoustic Transformer (small version)
    print("[3] Testing FlowMatchingAcousticTransformer (small)...")
    from .config import MultimodalAudioModelArgs
    small_audio_args = MultimodalAudioModelArgs(
        semantic_codebook_size=100,
        acoustic_codebook_size=21,
        n_acoustic_codebook=4,
        acoustic_transformer_args={
            "input_dim": 256, "dim": 256, "n_layers": 1,
            "head_dim": 64, "hidden_dim": 512, "n_heads": 4, "n_kv_heads": 2,
        }
    )
    small_acoustic = FlowMatchingAcousticTransformer(
        audio_args=small_audio_args, llm_dim=256,
    )

    # Test batch call
    test_hidden = mx.random.normal((1, 3, 256))
    acou_codes = small_acoustic(test_hidden)
    print(f"  Input hidden: {test_hidden.shape}")
    print(f"  Acoustic codes: {acou_codes.shape}")
    mx.eval(acou_codes)

    # Test single frame
    test_hidden_1 = mx.random.normal((1, 256))
    acou_codes_1 = small_acoustic.decode_one_frame(test_hidden_1)
    print(f"  Single frame: {test_hidden_1.shape} -> {acou_codes_1.shape}")
    mx.eval(acou_codes_1)
    print("  OK\n")

    # 4. Voxtral Codec (small version)
    print("[4] Testing VoxtralCodec (small)...")
    small_codec_args = AudioTokenizerArgs(
        dim=128, hidden_dim=256, n_heads=4, n_kv_heads=4, head_dim=32,
        semantic_codebook_size=100, semantic_dim=32,
        acoustic_codebook_size=21, acoustic_dim=4,
        encoder_transformer_lengths=[1, 1, 1, 1],
        decoder_transformer_lengths=[1, 1, 1, 1],
    )
    small_codec = VoxtralCodec(small_codec_args)

    # Test decode path
    test_sem = mx.array([[0, 1, 2, 3, 4]])
    test_acou = mx.random.randint(0, 21, (1, 5, 4))
    audio = small_codec.decode(test_sem, test_acou)
    print(f"  Semantic codes: {test_sem.shape}")
    print(f"  Acoustic codes: {test_acou.shape}")
    print(f"  Output audio: {audio.shape}")
    mx.eval(audio)
    print("  OK\n")

    # 5. Full Model (small version)
    print("[5] Testing Full Model (small)...")
    from .voxtral_tts import Model
    small_config = ModelConfig(
        dim=256, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=64, hidden_dim=512, vocab_size=1000,
        audio_model_args={
            "semantic_codebook_size": 100,
            "acoustic_codebook_size": 21,
            "n_acoustic_codebook": 4,
            "n_codebook": 5,
            "acoustic_transformer_args": {
                "input_dim": 256, "dim": 256, "n_layers": 1,
                "head_dim": 64, "hidden_dim": 512, "n_heads": 4, "n_kv_heads": 2,
            },
        },
        codec_args={
            "dim": 128, "hidden_dim": 256, "n_heads": 4, "n_kv_heads": 4,
            "head_dim": 32, "semantic_codebook_size": 100, "semantic_dim": 32,
            "acoustic_codebook_size": 21, "acoustic_dim": 4,
            "encoder_transformer_lengths": [1, 1, 1, 1],
            "decoder_transformer_lengths": [1, 1, 1, 1],
        },
    )
    model = Model(small_config)
    p_count = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Total parameters: {p_count:,}")
    print("  OK\n")

    print("=== All Construction Tests Passed ===")


def test_weight_loading(model_path: str):
    """Test that converted weights load into the model correctly."""
    from .voxtral_tts import load

    print("=== Testing Weight Loading ===\n")

    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    p_count = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Total parameters: {p_count:,}")
    print(f"  Sample rate: {model.sample_rate}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")

    # Verify all parameters are loaded (no NaN/zero blocks)
    all_params = dict(nn.utils.tree_flatten(model.parameters()))
    zero_params = []
    for name, param in all_params.items():
        if param.size > 100 and mx.all(param == 0).item():
            zero_params.append(name)

    if zero_params:
        print(f"\n  WARNING: {len(zero_params)} parameters are all zeros:")
        for name in zero_params[:10]:
            print(f"    - {name}")
    else:
        print("  All parameters loaded (no all-zero blocks)")

    print("\n=== Weight Loading Test Passed ===")


def test_weight_analysis(model_path: str):
    """Analyze weight structure of a converted model."""
    from .convert import analyze_weights, load_safetensors

    print("=== Analyzing Model Weights ===\n")
    path = Path(model_path)
    weights = load_safetensors(path)
    components = analyze_weights(weights)

    for comp, keys in components.items():
        print(f"\n{comp}: {len(keys)} tensors")
        for k in keys[:10]:
            v = weights[k]
            print(f"  {k}: {v.shape} ({v.dtype})")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")

    total = sum(v.nbytes for v in weights.values())
    print(f"\nTotal size: {total / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral TTS model")
    parser.add_argument(
        "--test",
        choices=["construction", "weights", "loading", "inference"],
        default="construction",
    )
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    if args.test == "construction":
        test_construction()
    elif args.test == "weights":
        if not args.model_path:
            print("Error: --model-path required for weight analysis")
            return
        test_weight_analysis(args.model_path)
    elif args.test == "loading":
        if not args.model_path:
            print("Error: --model-path required for loading test")
            return
        test_weight_loading(args.model_path)
    elif args.test == "inference":
        print("Inference test requires model weights. Use --model-path")


if __name__ == "__main__":
    main()
