"""Weight conversion script for Voxtral TTS.

Converts weights from HuggingFace safetensors format to MLX format.

Usage:
    # Inspect weight keys (run this first!)
    python -m voxtral_tts.convert --inspect

    # Convert to MLX format
    python -m voxtral_tts.convert --output-dir mlx_model

    # Convert with quantization
    python -m voxtral_tts.convert --output-dir mlx_model --quantize q4
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import mlx.nn as nn


def get_model_path(hf_repo: str, revision: Optional[str] = None) -> Path:
    """Download model from HuggingFace Hub and return local path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tekken.json",
            "voice_embedding/*.pt",
            "params.json",
        ],
    )
    return Path(path)


def load_safetensors(model_path: Path) -> dict:
    """Load weights from safetensors file(s)."""
    import torch
    import safetensors.torch

    weights = {}
    for sf_file in sorted(model_path.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}...")
        w = safetensors.torch.load_file(str(sf_file))
        for k, v in w.items():
            # BFloat16 -> Float32 for numpy compatibility, then to MLX bfloat16
            if v.dtype == torch.bfloat16:
                weights[k] = mx.array(v.float().numpy()).astype(mx.bfloat16)
            else:
                weights[k] = mx.array(v.numpy())

    return weights


def load_params(model_path: Path) -> dict:
    """Load model params.json and convert to our config format.

    The HF params.json nests audio config under "multimodal":
      multimodal.audio_model_args -> our audio_model_args
      multimodal.audio_tokenizer_args -> our codec_args
    """
    params_path = model_path / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    # Top-level LLM config
    config = {
        "model_type": "voxtral_tts",
        "dim": params.get("dim", 3072),
        "n_layers": params.get("n_layers", 26),
        "head_dim": params.get("head_dim", 128),
        "n_heads": params.get("n_heads", 32),
        "n_kv_heads": params.get("n_kv_heads", 8),
        "hidden_dim": params.get("hidden_dim", 9216),
        "vocab_size": params.get("vocab_size", 131072),
        "rope_theta": params.get("rope_theta", 1000000.0),
        "norm_eps": params.get("norm_eps", 1e-5),
        "max_position_embeddings": params.get("max_position_embeddings", 128000),
        "tie_word_embeddings": params.get("tied_embeddings", True),
        "sampling_rate": 24000,
        "sample_rate": 24000,
    }

    # Extract multimodal config (audio_model_args + audio_tokenizer_args)
    multimodal = params.get("multimodal", {})

    if "audio_model_args" in multimodal:
        audio_args = dict(multimodal["audio_model_args"])
        # Flatten audio_encoding_args into audio_model_args for our config
        encoding = audio_args.pop("audio_encoding_args", {})
        audio_args.setdefault("n_codebook", encoding.get("num_codebooks", 37))
        audio_args.setdefault("sampling_rate", encoding.get("sampling_rate", 24000))
        audio_args.setdefault("frame_rate", encoding.get("frame_rate", 12.5))
        audio_args.setdefault("codebook_pattern", encoding.get("codebook_pattern", "parallel"))
        audio_args.setdefault("interleave_audio_tokens_per_segment",
                              encoding.get("interleave_audio_tokens_per_segment", 8192))
        audio_args.setdefault("interleave_text_tokens_per_segment",
                              encoding.get("interleave_text_tokens_per_segment", 8192))
        audio_args.setdefault("single_trailing_segment",
                              encoding.get("single_trailing_segment", False))
        # bos_token_id comes from multimodal level
        audio_args.setdefault("bos_token_id", multimodal.get("bos_token_id", 1))
        config["audio_model_args"] = audio_args

    if "audio_tokenizer_args" in multimodal:
        codec_args = dict(multimodal["audio_tokenizer_args"])
        # Convert _str fields to lists
        for str_field, list_field in [
            ("decoder_transformer_lengths_str", "decoder_transformer_lengths"),
            ("decoder_convs_kernels_str", "decoder_conv_kernels"),
            ("decoder_convs_strides_str", "decoder_conv_strides"),
        ]:
            if str_field in codec_args:
                codec_args[list_field] = [int(x) for x in codec_args.pop(str_field).split(",")]
        # Rename patch_proj_kernel_size -> patch_projection_kernel_size
        if "patch_proj_kernel_size" in codec_args:
            codec_args["patch_projection_kernel_size"] = codec_args.pop("patch_proj_kernel_size")
        # Remove voice map (not needed in config)
        codec_args.pop("voice", None)
        config["codec_args"] = codec_args

    return config


def analyze_weights(weights: dict) -> dict:
    """Analyze weight keys and map them to model components."""
    components = {
        "language_model": [],
        "acoustic_transformer": [],
        "audio_tokenizer": [],
        "audio_embeddings": [],
        "unknown": [],
    }

    for key in sorted(weights.keys()):
        if key.startswith("acoustic_transformer."):
            components["acoustic_transformer"].append(key)
        elif key.startswith("audio_tokenizer."):
            components["audio_tokenizer"].append(key)
        elif key.startswith("mm_audio_embeddings"):
            components["audio_embeddings"].append(key)
        elif any(key.startswith(p) for p in ["layers.", "tok_embeddings.", "norm.", "output."]):
            components["language_model"].append(key)
        else:
            components["unknown"].append(key)

    return components


def _get_conv_weight_keys(weights: dict) -> Set[str]:
    """Identify Conv1d weight keys that need transposition.

    The codec uses weight-normed convolutions stored as:
      decoder_blocks.N.conv.parametrizations.weight.original1: [out, in, kernel]
    These need transposing to MLX convention: [out, kernel, in]
    """
    conv_keys = set()
    for key in weights.keys():
        if weights[key].ndim == 3:
            # Weight-normed conv weights in audio_tokenizer
            if "parametrizations.weight.original1" in key:
                conv_keys.add(key)
    return conv_keys


def remap_weights(weights: dict) -> dict:
    """Remap weight keys from HF consolidated format to our MLX model format.

    Actual consolidated.safetensors structure:
      layers.N.*                                        -> language_model.layers.N.*
      norm.weight                                       -> language_model.norm.weight
      mm_audio_embeddings.tok_embeddings.weight         -> language_model.tok_embeddings.weight
      mm_audio_embeddings.audio_codebook_embeddings.*   -> audio_token_embedding.*
      acoustic_transformer.*                            -> acoustic_transformer.* (kept)
      audio_tokenizer.*                                 -> audio_tokenizer.* (kept)

    Note: No separate output.weight — model uses tied embeddings.
    """
    conv_keys = _get_conv_weight_keys(weights)
    remapped = {}

    for key, value in weights.items():
        new_key = key

        if key.startswith("acoustic_transformer."):
            new_key = key  # Keep as-is

        elif key.startswith("audio_tokenizer."):
            new_key = key  # Keep as-is

        elif key == "mm_audio_embeddings.tok_embeddings.weight":
            # This is the LLM's token embedding — shared with output (tied)
            new_key = "language_model.tok_embeddings.weight"

        elif key.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
            # Audio codebook embeddings for feeding audio codes back to LLM
            suffix = key.replace("mm_audio_embeddings.audio_codebook_embeddings.", "")
            new_key = f"audio_token_embedding.{suffix}"

        elif key == "norm.weight":
            new_key = "language_model.norm.weight"

        elif key.startswith("layers."):
            new_key = f"language_model.{key}"

        else:
            new_key = f"language_model.{key}"
            print(f"  WARNING: Unmapped key '{key}' -> '{new_key}'")

        # Conv1d weight transposition (PyTorch [out, in, kernel] -> MLX [out, kernel, in])
        if key in conv_keys:
            value = value.transpose(0, 2, 1)

        remapped[new_key] = value

    return remapped


def inspect(hf_repo: str, revision: Optional[str] = None):
    """Download model and print all weight keys, shapes, and config.

    This is the critical first step before conversion — it reveals
    the actual structure of the safetensors file so we can verify
    our remapping assumptions.
    """
    print(f"Downloading model from {hf_repo}...")
    model_path = get_model_path(hf_repo, revision)

    # Print params.json
    params_path = model_path / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print("\n=== params.json ===")
        print(json.dumps(params, indent=2))
    else:
        print("\nWARNING: No params.json found!")

    # List all files
    print("\n=== Files ===")
    for f in sorted(model_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
        elif f.is_dir():
            count = sum(1 for _ in f.iterdir())
            print(f"  {f.name}/ ({count} files)")

    # Load and inspect weights
    print("\n=== Loading weights ===")
    weights = load_safetensors(model_path)

    # Analyze by component
    components = analyze_weights(weights)

    total_params = 0
    total_bytes = 0

    for comp_name, keys in components.items():
        if not keys:
            continue
        comp_params = sum(weights[k].size for k in keys)
        comp_bytes = sum(weights[k].nbytes for k in keys)
        total_params += comp_params
        total_bytes += comp_bytes

        print(f"\n=== {comp_name} ({len(keys)} tensors, {comp_params:,} params, {comp_bytes / 1e9:.2f} GB) ===")
        for k in keys:
            v = weights[k]
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")

    print(f"\n=== TOTAL: {len(weights)} tensors, {total_params:,} params, {total_bytes / 1e9:.2f} GB ===")

    # Check for Conv1d candidates
    conv_candidates = _get_conv_weight_keys(weights)
    if conv_candidates:
        print(f"\n=== Conv1d weights to transpose ({len(conv_candidates)}) ===")
        for k in sorted(conv_candidates):
            print(f"  {k}: {list(weights[k].shape)}")

    # Show what remapping would produce
    print("\n=== Remapping preview ===")
    remapped = remap_weights(weights)
    for old_key in sorted(weights.keys()):
        # Find corresponding new key
        for new_key in remapped:
            if new_key.endswith(old_key.split(".")[-1]) or old_key in new_key:
                if old_key != new_key:
                    print(f"  {old_key} -> {new_key}")
                break


def convert(
    hf_repo: str,
    output_dir: str = "mlx_model",
    quantize: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
):
    """Convert Voxtral TTS from HuggingFace to MLX format.

    Args:
        hf_repo: HuggingFace repository ID
        output_dir: Output directory for MLX model
        quantize: Quantization level (q4, q8, None)
        revision: HuggingFace revision
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Get model path
    if local_dir:
        print(f"[1/5] Using local model from {local_dir}...")
        model_path = Path(local_dir)
    else:
        print(f"[1/5] Downloading model from {hf_repo}...")
        model_path = get_model_path(hf_repo, revision)

    # Step 2: Load and parse config
    print("[2/5] Loading configuration...")
    config = load_params(model_path)

    # Step 3: Load weights
    print("[3/5] Loading weights...")
    weights = load_safetensors(model_path)

    # Analyze weight structure
    components = analyze_weights(weights)
    print(f"  Language model: {len(components['language_model'])} tensors")
    print(f"  Acoustic transformer: {len(components['acoustic_transformer'])} tensors")
    print(f"  Audio tokenizer: {len(components['audio_tokenizer'])} tensors")
    print(f"  Audio embeddings: {len(components['audio_embeddings'])} tensors")
    if components["unknown"]:
        print(f"  Unknown: {len(components['unknown'])} tensors")
        for k in components["unknown"]:
            print(f"    - {k}")

    # Step 4: Remap weights
    print("[4/5] Remapping weights...")
    weights = remap_weights(weights)

    # Apply quantization if requested
    if quantize:
        print(f"  Quantizing to {quantize}...")
        q_bits = {"q2": 2, "q4": 4, "q6": 6, "q8": 8}.get(quantize, 4)
        group_size = 64
        quantized = {}
        for key, value in weights.items():
            if (key.endswith(".weight") and
                    value.ndim == 2 and
                    value.shape[-1] % group_size == 0 and
                    min(value.shape) > group_size):
                # mx.quantize returns (quantized_weight, scales, biases)
                q_weight, scales, biases = mx.quantize(value, bits=q_bits, group_size=group_size)
                quantized[key] = q_weight
                quantized[key.replace(".weight", ".scales")] = scales
                quantized[key.replace(".weight", ".biases")] = biases
            else:
                quantized[key] = value
        weights = quantized
        config["quantization"] = {"group_size": group_size, "bits": q_bits}

    # Step 5: Save
    print("[5/5] Saving MLX model...")

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save weights (handle both plain arrays and quantized tuples)
    save_safetensors = mx.save_safetensors

    total_size = sum(v.nbytes for v in weights.values())
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB

    if total_size <= max_shard_size:
        save_safetensors(str(output_path / "model.safetensors"), weights)
    else:
        # Shard weights
        shard_idx = 0
        current_shard = {}
        current_size = 0

        for key, value in weights.items():
            if current_size + value.nbytes > max_shard_size and current_shard:
                save_safetensors(
                    str(output_path / f"model-{shard_idx:05d}.safetensors"),
                    current_shard,
                )
                shard_idx += 1
                current_shard = {}
                current_size = 0

            current_shard[key] = value
            current_size += value.nbytes

        if current_shard:
            save_safetensors(
                str(output_path / f"model-{shard_idx:05d}.safetensors"),
                current_shard,
            )

    # Copy tokenizer and voice embeddings
    tekken_path = model_path / "tekken.json"
    if tekken_path.exists():
        shutil.copy(tekken_path, output_path / "tekken.json")

    voice_dir = model_path / "voice_embedding"
    if voice_dir.exists():
        shutil.copytree(voice_dir, output_path / "voice_embedding", dirs_exist_ok=True)

    total_mb = total_size / (1024 * 1024)
    print(f"\nDone! Model saved to {output_path}")
    print(f"  Total size: {total_mb:.1f} MB")
    print(f"  Config: {output_path / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(description="Convert Voxtral TTS to MLX format")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="mistralai/Voxtral-4B-TTS-2603",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mlx_model",
        help="Output directory for MLX model",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["q2", "q4", "q6", "q8"],
        default=None,
        help="Quantization level",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace revision",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Download and inspect weight keys without converting",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Use local model directory instead of downloading from HF",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect(hf_repo=args.hf_repo, revision=args.revision)
    else:
        convert(
            hf_repo=args.hf_repo,
            output_dir=args.output_dir,
            quantize=args.quantize,
            revision=args.revision,
            local_dir=args.local_dir,
        )


if __name__ == "__main__":
    main()
