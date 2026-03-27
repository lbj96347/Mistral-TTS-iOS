#!/usr/bin/env python3
"""
Validate a model directory for compatibility with the VoxtralTTS Swift app.

Mirrors the Swift app's loadVoxtralModel() pipeline stage-by-stage to catch
issues before they become runtime errors. Reference:
  VoxtralTTS-App/VoxtralTTS/Sources/Model/ModelConfig.swift
  VoxtralTTS-App/VoxtralTTS/Sources/Model/VoxtralModel.swift
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Swift CodingKeys definitions (mirrored from ModelConfig.swift)
# Each dict maps JSON key -> expected Python type
# ---------------------------------------------------------------------------

MODEL_CONFIG_KEYS = {
    "model_type": str,
    "dim": (int, float),
    "n_layers": (int, float),
    "head_dim": (int, float),
    "n_heads": (int, float),
    "n_kv_heads": (int, float),
    "hidden_dim": (int, float),
    "vocab_size": (int, float),
    "rope_theta": (int, float),
    "norm_eps": (int, float),
    "max_position_embeddings": (int, float),
    "tie_word_embeddings": bool,
    "audio_model_args": dict,
    "codec_args": dict,
    "sampling_rate": (int, float),
    "quantization": dict,
}

MULTIMODAL_AUDIO_CONFIG_KEYS = {
    "semantic_codebook_size": (int, float),
    "acoustic_codebook_size": (int, float),
    "n_acoustic_codebook": (int, float),
    "audio_token_id": (int, float),
    "begin_audio_token_id": (int, float),
    "bos_token_id": (int, float),
    "sampling_rate": (int, float),
    "frame_rate": (int, float),
    "n_codebook": (int, float),
    "input_embedding_concat_type": str,
    "acoustic_transformer_args": dict,
}

ACOUSTIC_TRANSFORMER_CONFIG_KEYS = {
    "input_dim": (int, float),
    "dim": (int, float),
    "n_layers": (int, float),
    "head_dim": (int, float),
    "hidden_dim": (int, float),
    "n_heads": (int, float),
    "n_kv_heads": (int, float),
    "use_biases": bool,
    "norm_eps": (int, float),
    "rope_theta": (int, float),
    "sigma": (int, float),
    "sigma_max": (int, float),
}

AUDIO_TOKENIZER_CONFIG_KEYS = {
    "channels": (int, float),
    "sampling_rate": (int, float),
    "pretransform_patch_size": (int, float),
    "patch_projection_kernel_size": (int, float),
    "semantic_codebook_size": (int, float),
    "semantic_dim": (int, float),
    "acoustic_codebook_size": (int, float),
    "acoustic_dim": (int, float),
    "dim": (int, float),
    "hidden_dim": (int, float),
    "head_dim": (int, float),
    "n_heads": (int, float),
    "n_kv_heads": (int, float),
    "norm_eps": (int, float),
    "qk_norm": bool,
    "qk_norm_eps": (int, float),
    "causal": bool,
    "attn_sliding_window_size": (int, float),
    "half_attn_window_upon_downsampling": bool,
    "layer_scale": bool,
    "layer_scale_init": (int, float),
    "conv_weight_norm": bool,
    "decoder_transformer_lengths": list,
    "decoder_conv_kernels": list,
    "decoder_conv_strides": list,
}

QUANTIZATION_CONFIG_KEYS = {
    "group_size": (int, float),
    "bits": (int, float),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
    PASS = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    INFO = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def status(label, ok, detail=""):
    if ok:
        print(f"  {label:.<45} {Colors.PASS}OK{Colors.RESET} {detail}")
    else:
        print(f"  {label:.<45} {Colors.FAIL}FAIL{Colors.RESET} {detail}")


def warn(label, detail=""):
    print(f"  {label:.<45} {Colors.WARN}WARN{Colors.RESET} {detail}")


def info(label, detail=""):
    print(f"  {label:.<45} {Colors.INFO}INFO{Colors.RESET} {detail}")


def parse_safetensors_header(path):
    """Parse safetensors file header without loading tensor data."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return None, None
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 100_000_000:  # sanity check: >100MB header is suspicious
            return None, None
        header_json = f.read(header_size)
    header = json.loads(header_json)
    metadata = header.pop("__metadata__", {})
    return header, metadata


def check_keys(section_name, data, expected_keys):
    """Check a config section against expected CodingKeys. Returns (errors, warnings)."""
    errors = []
    warnings = []

    # Check for type mismatches on known keys
    for key, expected_type in expected_keys.items():
        if key in data:
            if not isinstance(data[key], expected_type):
                # Special case: int/float interchangeable in JSON
                if isinstance(expected_type, tuple):
                    if not isinstance(data[key], expected_type):
                        errors.append(f"{key}: expected {expected_type}, got {type(data[key]).__name__}")
                else:
                    errors.append(f"{key}: expected {expected_type.__name__}, got {type(data[key]).__name__}")

    # Find extra keys not in Swift CodingKeys (will be silently ignored)
    extra = set(data.keys()) - set(expected_keys.keys())
    if extra:
        warnings.append(f"Extra keys (ignored by Swift): {sorted(extra)}")

    # Find missing keys (will use Swift defaults)
    missing = set(expected_keys.keys()) - set(data.keys())
    if missing:
        warnings.append(f"Missing keys (Swift defaults used): {sorted(missing)}")

    return errors, warnings


# ---------------------------------------------------------------------------
# Test Stages
# ---------------------------------------------------------------------------

def stage1_file_existence(model_dir):
    """Check required files exist."""
    print(f"\n{Colors.BOLD}[1/7] File Existence{Colors.RESET}")
    errors = 0

    # Required files
    for fname in ["config.json"]:
        exists = (model_dir / fname).is_file()
        status(fname, exists)
        if not exists:
            errors += 1

    # Safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    has_weights = len(st_files) > 0
    status(f"*.safetensors ({len(st_files)} file(s))", has_weights)
    if not has_weights:
        errors += 1

    # Tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        exists = (model_dir / fname).is_file()
        status(fname, exists)
        if not exists:
            errors += 1

    # Voice embeddings (optional but warned)
    voice_dir = model_dir / "voice_embedding"
    if voice_dir.is_dir():
        voice_files = sorted(voice_dir.glob("*.safetensors"))
        pt_files = sorted(voice_dir.glob("*.pt"))
        if voice_files:
            status(f"voice_embedding/ ({len(voice_files)} .safetensors)", True)
        elif pt_files:
            warn(f"voice_embedding/ ({len(pt_files)} .pt only)",
                 "Swift app needs .safetensors, not .pt")
            errors += 1
        else:
            warn("voice_embedding/", "directory exists but empty")
    else:
        warn("voice_embedding/", "directory not found")

    return errors


def stage2_config_parsing(model_dir):
    """Parse config.json and validate against Swift CodingKeys."""
    print(f"\n{Colors.BOLD}[2/7] Config JSON Parsing{Colors.RESET}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        status("config.json", False, "file missing, skipping")
        return 1, None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        status("JSON parse", False, str(e))
        return 1, None

    status("JSON parse", True)
    total_errors = 0

    # Top-level ModelConfig
    errs, warns = check_keys("ModelConfig", config, MODEL_CONFIG_KEYS)
    status("ModelConfig keys", len(errs) == 0, "; ".join(errs) if errs else "")
    total_errors += len(errs)
    for w in warns:
        info("ModelConfig", w)

    # audio_model_args
    audio_args = config.get("audio_model_args", {})
    if audio_args:
        errs, warns = check_keys("MultimodalAudioModelConfig", audio_args, MULTIMODAL_AUDIO_CONFIG_KEYS)
        status("MultimodalAudioModelConfig keys", len(errs) == 0, "; ".join(errs) if errs else "")
        total_errors += len(errs)
        for w in warns:
            info("MultimodalAudioModelConfig", w)

        # acoustic_transformer_args (nested)
        at_args = audio_args.get("acoustic_transformer_args", {})
        if at_args:
            errs, warns = check_keys("AcousticTransformerConfig", at_args, ACOUSTIC_TRANSFORMER_CONFIG_KEYS)
            status("AcousticTransformerConfig keys", len(errs) == 0, "; ".join(errs) if errs else "")
            total_errors += len(errs)
            for w in warns:
                info("AcousticTransformerConfig", w)

    # codec_args
    codec_args = config.get("codec_args", {})
    if codec_args:
        errs, warns = check_keys("AudioTokenizerConfig", codec_args, AUDIO_TOKENIZER_CONFIG_KEYS)
        status("AudioTokenizerConfig keys", len(errs) == 0, "; ".join(errs) if errs else "")
        total_errors += len(errs)
        for w in warns:
            info("AudioTokenizerConfig", w)

    # quantization
    quant = config.get("quantization")
    if quant:
        errs, warns = check_keys("QuantizationConfig", quant, QUANTIZATION_CONFIG_KEYS)
        status("QuantizationConfig keys", len(errs) == 0, "; ".join(errs) if errs else "")
        total_errors += len(errs)
    else:
        info("QuantizationConfig", "not present (non-quantized model)")

    return total_errors, config


def stage3_config_values(config):
    """Validate config values are internally consistent."""
    print(f"\n{Colors.BOLD}[3/7] Config Value Validation{Colors.RESET}")
    if config is None:
        status("config", False, "no config loaded")
        return 1

    errors = 0

    # Basic sanity
    dim = config.get("dim", 3072)
    n_layers = config.get("n_layers", 26)
    n_heads = config.get("n_heads", 32)
    n_kv_heads = config.get("n_kv_heads", 8)
    vocab_size = config.get("vocab_size", 131072)

    ok = dim > 0
    status("dim > 0", ok, f"dim={dim}")
    errors += 0 if ok else 1

    ok = n_layers > 0
    status("n_layers > 0", ok, f"n_layers={n_layers}")
    errors += 0 if ok else 1

    ok = vocab_size > 0
    status("vocab_size > 0", ok, f"vocab_size={vocab_size}")
    errors += 0 if ok else 1

    ok = n_heads % n_kv_heads == 0
    status("n_heads % n_kv_heads == 0 (GQA)", ok, f"{n_heads} % {n_kv_heads} = {n_heads % n_kv_heads}")
    errors += 0 if ok else 1

    # Audio config consistency
    audio_args = config.get("audio_model_args", {})
    n_codebook = audio_args.get("n_codebook", 37)
    n_acoustic = audio_args.get("n_acoustic_codebook", 36)
    ok = n_codebook == 1 + n_acoustic
    status("n_codebook == 1 + n_acoustic_codebook", ok, f"{n_codebook} == 1 + {n_acoustic}")
    errors += 0 if ok else 1

    # Padded vocab calculation (mirrors VoxtralModel.swift lines 68-70)
    semantic_cb = audio_args.get("semantic_codebook_size", 8192)
    acoustic_cb = audio_args.get("acoustic_codebook_size", 21)
    total_audio_vocab = semantic_cb + acoustic_cb * n_acoustic
    padded_vocab = ((total_audio_vocab + 127) // 128) * 128
    info("Padded audio vocab", f"total={total_audio_vocab}, padded={padded_vocab}")

    return errors


def stage4_weight_structure(model_dir, config):
    """Parse safetensors headers and check weight key structure."""
    print(f"\n{Colors.BOLD}[4/7] Weight Key Structure{Colors.RESET}")
    if config is None:
        status("config", False, "no config loaded")
        return 1

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        status("safetensors files", False, "none found")
        return 1

    # Merge all headers
    all_keys = {}
    for sf in st_files:
        header, _ = parse_safetensors_header(sf)
        if header is None:
            status(sf.name, False, "failed to parse header")
            return 1
        all_keys.update(header)

    status(f"Parsed {len(st_files)} file(s)", True, f"{len(all_keys)} weight keys total")
    errors = 0

    # Check top-level prefixes
    prefixes = {"language_model", "acoustic_transformer", "audio_tokenizer", "audio_token_embedding"}
    found_prefixes = set()
    for key in all_keys:
        prefix = key.split(".")[0]
        if prefix in prefixes:
            found_prefixes.add(prefix)

    for p in sorted(prefixes):
        ok = p in found_prefixes
        status(f"prefix: {p}.*", ok)
        if not ok:
            errors += 1

    # Check layer count
    n_layers = config.get("n_layers", 26)
    layer_indices = set()
    for key in all_keys:
        if key.startswith("language_model.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))

    expected = set(range(n_layers))
    ok = layer_indices == expected
    status(f"language_model layers (expect {n_layers})", ok,
           f"found {len(layer_indices)}" + ("" if ok else f", missing: {sorted(expected - layer_indices)[:5]}"))
    if not ok:
        errors += 1

    # Check acoustic transformer layers
    at_args = config.get("audio_model_args", {}).get("acoustic_transformer_args", {})
    at_n_layers = at_args.get("n_layers", 3)
    at_layer_indices = set()
    for key in all_keys:
        if key.startswith("acoustic_transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                at_layer_indices.add(int(parts[2]))

    expected_at = set(range(at_n_layers))
    ok = at_layer_indices == expected_at
    status(f"acoustic_transformer layers (expect {at_n_layers})", ok, f"found {len(at_layer_indices)}")
    if not ok:
        errors += 1

    # Quantization check
    has_quant = config.get("quantization") is not None
    if has_quant:
        # Check a sample quantized weight has scales and biases
        sample_weight = None
        for key in all_keys:
            if "language_model.layers.0.attention.wq.weight" == key:
                sample_weight = key
                break
        if sample_weight:
            base = sample_weight.rsplit(".weight", 1)[0]
            has_scales = f"{base}.scales" in all_keys
            has_biases = f"{base}.biases" in all_keys
            ok = has_scales and has_biases
            status("Quantized weight (scales+biases)", ok,
                   f"scales={'yes' if has_scales else 'NO'}, biases={'yes' if has_biases else 'NO'}")
            if not ok:
                errors += 1
        else:
            warn("Quantization check", "could not find sample weight key")

    return errors


def stage5_weight_shapes(model_dir, config):
    """Validate key tensor shapes against config."""
    print(f"\n{Colors.BOLD}[5/7] Weight Shape Validation{Colors.RESET}")
    if config is None:
        status("config", False, "no config loaded")
        return 1

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        status("safetensors files", False, "none found")
        return 1

    all_keys = {}
    for sf in st_files:
        header, _ = parse_safetensors_header(sf)
        if header:
            all_keys.update(header)

    errors = 0
    dim = config.get("dim", 3072)
    vocab_size = config.get("vocab_size", 131072)
    has_quant = config.get("quantization") is not None

    # audio_token_embedding.embeddings.weight
    emb_key = "audio_token_embedding.embeddings.weight"
    if emb_key in all_keys:
        shape = all_keys[emb_key]["shape"]
        audio_args = config.get("audio_model_args", {})
        semantic_cb = audio_args.get("semantic_codebook_size", 8192)
        acoustic_cb = audio_args.get("acoustic_codebook_size", 21)
        n_acoustic = audio_args.get("n_acoustic_codebook", 36)
        total = semantic_cb + acoustic_cb * n_acoustic
        swift_padded = ((total + 127) // 128) * 128
        actual_vocab = shape[0]
        if actual_vocab == swift_padded:
            status(emb_key, True, f"shape={shape}, matches Swift padded={swift_padded}")
        elif actual_vocab >= total:
            # Weight is larger than Swift expects but >= raw total — model.update() will override
            warn(emb_key,
                 f"shape={shape}, Swift init creates {swift_padded} but weight has {actual_vocab}. "
                 f"model.update() will override — check indexing code uses weight shape, not computed padding")
        else:
            status(emb_key, False, f"shape={shape}, too small (need >= {total})")
            errors += 1
    else:
        status(emb_key, False, "key not found")
        errors += 1

    # language_model.tok_embeddings.weight
    tok_key = "language_model.tok_embeddings.weight"
    if tok_key in all_keys:
        shape = all_keys[tok_key]["shape"]
        if has_quant:
            # Quantized: shape may be different (packed)
            info(tok_key, f"shape={shape} (quantized)")
        else:
            ok = shape == [vocab_size, dim]
            status(tok_key, ok, f"shape={shape}, expected=[{vocab_size}, {dim}]")
            if not ok:
                errors += 1
    else:
        status(tok_key, False, "key not found")
        errors += 1

    # Check a sample attention weight shape
    head_dim = config.get("head_dim", 128)
    n_heads = config.get("n_heads", 32)
    n_kv_heads = config.get("n_kv_heads", 8)

    wq_key = "language_model.layers.0.attention.wq.weight"
    if wq_key in all_keys:
        shape = all_keys[wq_key]["shape"]
        if has_quant:
            info(wq_key, f"shape={shape} (quantized)")
        else:
            expected_wq = [n_heads * head_dim, dim]
            ok = shape == expected_wq
            status(wq_key, ok, f"shape={shape}, expected={expected_wq}")
            if not ok:
                errors += 1
    else:
        status(wq_key, False, "key not found")
        errors += 1

    wk_key = "language_model.layers.0.attention.wk.weight"
    if wk_key in all_keys:
        shape = all_keys[wk_key]["shape"]
        if has_quant:
            info(wk_key, f"shape={shape} (quantized)")
        else:
            expected_wk = [n_kv_heads * head_dim, dim]
            ok = shape == expected_wk
            status(wk_key, ok, f"shape={shape}, expected={expected_wk}")
            if not ok:
                errors += 1

    return errors


def stage6_tokenizer(model_dir):
    """Validate tokenizer files."""
    print(f"\n{Colors.BOLD}[6/7] Tokenizer Validation{Colors.RESET}")
    errors = 0

    tok_path = model_dir / "tokenizer.json"
    if tok_path.is_file():
        try:
            with open(tok_path) as f:
                tok = json.load(f)
            has_model = "model" in tok
            has_added = "added_tokens" in tok
            ok = has_model and has_added
            status("tokenizer.json structure", ok,
                   f"model={'yes' if has_model else 'NO'}, added_tokens={'yes' if has_added else 'NO'}")
            if not ok:
                errors += 1
        except json.JSONDecodeError as e:
            status("tokenizer.json parse", False, str(e))
            errors += 1
    else:
        status("tokenizer.json", False, "file missing")
        errors += 1

    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        try:
            with open(tok_cfg_path) as f:
                json.load(f)
            status("tokenizer_config.json", True, "valid JSON")
        except json.JSONDecodeError as e:
            status("tokenizer_config.json parse", False, str(e))
            errors += 1
    else:
        status("tokenizer_config.json", False, "file missing")
        errors += 1

    # Check tekken.json (used by mistral-common tokenizer)
    tekken_path = model_dir / "tekken.json"
    if tekken_path.is_file():
        info("tekken.json", "present (Mistral tokenizer)")
    else:
        info("tekken.json", "not found (may not be needed for Swift)")

    return errors


def stage7_voice_embeddings(model_dir, config):
    """Validate voice embedding files."""
    print(f"\n{Colors.BOLD}[7/7] Voice Embedding Validation{Colors.RESET}")
    voice_dir = model_dir / "voice_embedding"
    if not voice_dir.is_dir():
        warn("voice_embedding/", "directory not found, skipping")
        return 0

    dim = config.get("dim", 3072) if config else 3072
    voice_files = sorted(voice_dir.glob("*.safetensors"))
    if not voice_files:
        warn("voice_embedding/", "no .safetensors files found")
        pt_files = sorted(voice_dir.glob("*.pt"))
        if pt_files:
            status("voice_embedding/*.pt", False,
                   f"{len(pt_files)} .pt files found but Swift app needs .safetensors")
            return 1
        return 0

    errors = 0
    valid_count = 0
    for vf in voice_files:
        header, _ = parse_safetensors_header(vf)
        if header is None:
            status(vf.name, False, "failed to parse")
            errors += 1
            continue

        if "embedding" not in header:
            status(vf.name, False, "missing 'embedding' key (Swift uses fatalError)")
            errors += 1
            continue

        shape = header["embedding"]["shape"]
        # Shape should be [1, dim] or [dim]
        shape_ok = (shape == [1, dim] or shape == [dim] or
                    (len(shape) == 2 and shape[-1] == dim))
        if not shape_ok:
            status(vf.name, False, f"shape={shape}, expected dim={dim}")
            errors += 1
        else:
            valid_count += 1

    status(f"Voice embeddings ({len(voice_files)} files)", errors == 0,
           f"{valid_count} valid" + (f", {errors} invalid" if errors else ""))

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_model_dir(model_path):
    model_dir = Path(model_path).resolve()
    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}Validating: {model_dir}{Colors.RESET}")
    print(f"{'=' * 60}")

    if not model_dir.is_dir():
        print(f"{Colors.FAIL}ERROR: {model_dir} is not a directory{Colors.RESET}")
        return 1

    total_errors = 0

    # Stage 1
    total_errors += stage1_file_existence(model_dir)

    # Stage 2
    errs, config = stage2_config_parsing(model_dir)
    total_errors += errs

    # Stage 3
    total_errors += stage3_config_values(config)

    # Stage 4
    total_errors += stage4_weight_structure(model_dir, config)

    # Stage 5
    total_errors += stage5_weight_shapes(model_dir, config)

    # Stage 6
    total_errors += stage6_tokenizer(model_dir)

    # Stage 7
    total_errors += stage7_voice_embeddings(model_dir, config)

    # Summary
    print(f"\n{'=' * 60}")
    if total_errors == 0:
        print(f"{Colors.PASS}{Colors.BOLD}RESULT: ALL CHECKS PASSED{Colors.RESET}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}RESULT: {total_errors} ERROR(S) FOUND{Colors.RESET}")
    print(f"{'=' * 60}\n")

    return total_errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate model directory for VoxtralTTS Swift app compatibility"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to model directory (e.g., mlx_model_q4)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all model directories found at project root"
    )
    args = parser.parse_args()

    if args.all:
        # Find all directories with config.json or *.safetensors at project root
        project_root = Path(__file__).parent
        candidates = []
        for d in sorted(project_root.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                has_config = (d / "config.json").exists()
                has_weights = any(d.glob("*.safetensors"))
                if has_config or has_weights:
                    candidates.append(d)

        if not candidates:
            print("No model directories found at project root.")
            sys.exit(1)

        total = 0
        for d in candidates:
            total += validate_model_dir(d)
        sys.exit(1 if total > 0 else 0)

    elif args.model_path:
        errors = validate_model_dir(args.model_path)
        sys.exit(1 if errors > 0 else 0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
