#!/usr/bin/env bash
# build_models.sh — Convert Voxtral TTS HuggingFace model to MLX format
#
# Outputs three model variants:
#   mlx_model     — Full precision (fp16), ~8 GB
#   mlx_model_q4  — Uniform Q4 quantization, ~2.1 GB
#   mlx_model_q2  — Mixed quantization (Q4 LLM/acoustic + Q2 codec), ~1.6 GB
#
# Usage:
#   ./scripts/build_models.sh              # Build all three variants
#   ./scripts/build_models.sh fp16         # Build only full precision
#   ./scripts/build_models.sh q4           # Build only Q4
#   ./scripts/build_models.sh q2           # Build only mixed Q4+Q2
#   ./scripts/build_models.sh q4 q2        # Build Q4 and Q2
#
# Options:
#   --local-dir <path>   Use local HF model directory instead of downloading
#   --skip-voices        Skip voice embedding conversion
#   --dry-run            Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
LOCAL_DIR=""
SKIP_VOICES=false
DRY_RUN=false
TARGETS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --skip-voices)
            SKIP_VOICES=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        fp16|q4|q2)
            TARGETS+=("$1")
            shift
            ;;
        -h|--help)
            head -17 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default: build all
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(fp16 q4 q2)
fi

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
        echo "Activating .venv..."
        source "$PROJECT_DIR/.venv/bin/activate"
    else
        echo "Error: No .venv found. Run: python3 -m venv .venv && pip install -e '.[dev]'"
        exit 1
    fi
fi

LOCAL_FLAG=""
if [[ -n "$LOCAL_DIR" ]]; then
    LOCAL_FLAG="--local-dir $LOCAL_DIR"
fi

run_cmd() {
    echo ""
    echo ">>> $*"
    if [[ "$DRY_RUN" == "false" ]]; then
        "$@"
    fi
}

convert_voices() {
    local model_dir="$1"
    if [[ "$SKIP_VOICES" == "true" ]]; then
        return
    fi
    local voice_dir="$model_dir/voice_embedding"
    if [[ -d "$voice_dir" ]]; then
        # Check if .safetensors already exist
        local pt_count st_count
        pt_count=$(find "$voice_dir" -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
        st_count=$(find "$voice_dir" -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$pt_count" -gt 0 && "$st_count" -lt "$pt_count" ]]; then
            run_cmd python3 -m voxtral_tts.convert_voices "$model_dir"
        else
            echo "Voice embeddings already converted ($st_count .safetensors files)"
        fi
    fi
}

elapsed() {
    local start=$1
    local end
    end=$(date +%s)
    local dt=$((end - start))
    printf '%dm%02ds' $((dt / 60)) $((dt % 60))
}

total_start=$(date +%s)

for target in "${TARGETS[@]}"; do
    echo ""
    echo "========================================"
    case "$target" in
        fp16)
            echo "Building mlx_model (full precision)"
            echo "========================================"
            t=$(date +%s)
            run_cmd python3 -m voxtral_tts.convert \
                --output-dir "$PROJECT_DIR/mlx_model" \
                $LOCAL_FLAG
            convert_voices "$PROJECT_DIR/mlx_model"
            echo "Done mlx_model in $(elapsed $t)"
            ;;
        q4)
            echo "Building mlx_model_q4 (uniform Q4)"
            echo "========================================"
            t=$(date +%s)
            run_cmd python3 -m voxtral_tts.convert \
                --output-dir "$PROJECT_DIR/mlx_model_q4" \
                --quantize q4 \
                $LOCAL_FLAG
            convert_voices "$PROJECT_DIR/mlx_model_q4"
            echo "Done mlx_model_q4 in $(elapsed $t)"
            ;;
        q2)
            echo "Building mlx_model_q2 (mixed Q4 LLM/acoustic + Q2 codec)"
            echo "========================================"
            t=$(date +%s)
            run_cmd python3 -m voxtral_tts.convert \
                --output-dir "$PROJECT_DIR/mlx_model_q2" \
                --quantize-llm q4 --quantize-acoustic q4 --quantize-codec q2 \
                $LOCAL_FLAG
            convert_voices "$PROJECT_DIR/mlx_model_q2"
            echo "Done mlx_model_q2 in $(elapsed $t)"
            ;;
    esac
done

echo ""
echo "========================================"
echo "All done in $(elapsed $total_start)"
echo "========================================"

# Print output summary
echo ""
for target in "${TARGETS[@]}"; do
    case "$target" in
        fp16) dir="mlx_model" ;;
        q4)   dir="mlx_model_q4" ;;
        q2)   dir="mlx_model_q2" ;;
    esac
    full="$PROJECT_DIR/$dir"
    if [[ -d "$full" ]]; then
        size=$(du -sh "$full" 2>/dev/null | cut -f1)
        voices=$(find "$full/voice_embedding" -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $dir/  ${size}  (${voices} voices)"
    fi
done
