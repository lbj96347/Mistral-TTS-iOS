"""Transcription-based test for Voxtral TTS audio quality.

Generates audio, transcribes with mlx-whisper, and compares to input text.
Prints detailed diagnostics on failure.

Usage:
    python -m voxtral_tts.test_transcription --model-path mlx_model
    python -m voxtral_tts.test_transcription --model-path mlx_model --suite
    python -m voxtral_tts.test_transcription --model-path mlx_model --diagnose
"""

import argparse
import wave
import time
import re
import numpy as np
import mlx.core as mx


def save_wav(audio_array: np.ndarray, sample_rate: int, output_path: str):
    """Save float32 audio array as 16-bit WAV file."""
    audio = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_overlap_ratio(reference: str, hypothesis: str) -> float:
    """Compute word overlap ratio: fraction of reference words found in hypothesis."""
    ref_words = set(normalize_text(reference).split())
    hyp_words = set(normalize_text(hypothesis).split())
    if not ref_words:
        return 1.0 if not hyp_words else 0.0
    overlap = ref_words & hyp_words
    return len(overlap) / len(ref_words)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using edit distance."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / n


def diagnose_audio(audio: np.ndarray, sample_rate: int, num_frames: int):
    """Print audio signal diagnostics."""
    duration = len(audio) / sample_rate
    rms = np.sqrt(np.mean(audio ** 2))
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = zero_crossings / len(audio) * sample_rate if len(audio) > 1 else 0

    print(f"\n--- Audio Diagnostics ---")
    print(f"Duration:        {duration:.2f}s ({len(audio)} samples @ {sample_rate} Hz)")
    print(f"Frames generated: {num_frames}")
    print(f"Samples:         min={audio.min():.4f} max={audio.max():.4f} "
          f"mean={audio.mean():.4f} std={audio.std():.4f}")
    print(f"RMS energy:      {rms:.6f}")
    print(f"Zero-crossing rate: {zcr:.0f} Hz")

    issues = []
    if rms < 0.001:
        issues.append("SILENT: RMS < 0.001 — codec may be producing zero output")
    if audio.std() < 0.001:
        issues.append("FLAT: Very low variance — audio is essentially constant")
    if audio.std() > 0.5:
        issues.append("NOISY: Very high variance — may be noise, not speech")
    if duration < 0.5:
        issues.append("SHORT: < 0.5s — LLM may be terminating early")
    if num_frames < 5:
        issues.append(f"FEW FRAMES: Only {num_frames} frames — likely early termination")
    if zcr > 10000:
        issues.append("HIGH ZCR: > 10kHz — likely noise, not speech")
    if zcr < 100 and rms > 0.01:
        issues.append("LOW ZCR: < 100 Hz with non-zero RMS — unusual for speech")

    if issues:
        print(f"\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nNo obvious signal issues detected.")

    return {"duration": duration, "rms": rms, "zcr": zcr, "issues": issues}


def diagnose_weights(model):
    """Check key weight tensors for obvious problems."""
    print(f"\n--- Weight Diagnostics ---")
    checks = []

    # Check semantic codebook
    try:
        emb_sum = model.audio_tokenizer.quantizer.semantic_codebook.embedding_sum
        cluster = model.audio_tokenizer.quantizer.semantic_codebook.cluster_usage
        mx.eval(emb_sum, cluster)
        emb_sum_np = np.array(emb_sum.astype(mx.float32))
        cluster_np = np.array(cluster.astype(mx.float32))
        is_zero = np.allclose(emb_sum_np, 0)
        print(f"Semantic codebook embedding_sum: shape={emb_sum_np.shape}, "
              f"all_zeros={is_zero}, mean={emb_sum_np.mean():.6f}")
        print(f"Semantic codebook cluster_usage: min={cluster_np.min():.2f}, "
              f"max={cluster_np.max():.2f}, mean={cluster_np.mean():.2f}")
        if is_zero:
            checks.append("CRITICAL: semantic codebook embedding_sum is all zeros!")
    except Exception as e:
        checks.append(f"Could not check semantic codebook: {e}")

    # Check audio token embedding
    try:
        emb_weight = model.audio_token_embedding.embeddings.weight
        mx.eval(emb_weight)
        emb_np = np.array(emb_weight.astype(mx.float32))
        print(f"Audio token embedding: shape={emb_np.shape}, "
              f"mean={emb_np.mean():.6f}, std={emb_np.std():.6f}")
        if np.allclose(emb_np, 0):
            checks.append("CRITICAL: audio token embeddings are all zeros!")
    except Exception as e:
        checks.append(f"Could not check audio token embedding: {e}")

    # Check acoustic transformer semantic head
    try:
        sem_out = model.acoustic_transformer.semantic_codebook_output.weight
        mx.eval(sem_out)
        sem_np = np.array(sem_out.astype(mx.float32))
        print(f"Semantic codebook output: shape={sem_np.shape}, "
              f"mean={sem_np.mean():.6f}, std={sem_np.std():.6f}")
        if np.allclose(sem_np, 0):
            checks.append("CRITICAL: semantic_codebook_output weight is all zeros!")
    except Exception as e:
        checks.append(f"Could not check semantic output: {e}")

    if checks:
        print(f"\nWeight issues:")
        for c in checks:
            print(f"  - {c}")
    else:
        print(f"\nAll checked weights look non-trivial.")

    return checks


def transcribe_audio(wav_path: str, whisper_model: str = "mlx-community/whisper-small-mlx") -> str:
    """Transcribe a WAV file using mlx-whisper."""
    try:
        import mlx_whisper
    except ImportError:
        print("ERROR: mlx-whisper not installed. Run: pip install mlx-whisper")
        return ""

    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=whisper_model,
        language="en",
    )
    return result.get("text", "").strip()


def run_single_test(
    model,
    tokenizer,
    text: str,
    voice: str,
    output_path: str,
    whisper_model: str,
    threshold: float,
    verbose: bool,
    diagnose: bool,
) -> bool:
    """Run a single generate-transcribe-compare test. Returns True if passed."""
    print(f"\n{'='*60}")
    print(f"Input text:  \"{text}\"")
    print(f"Voice:       {voice or 'none'}")
    print(f"Output:      {output_path}")

    # Generate
    print(f"\nGenerating audio...")
    num_frames = 0
    audio_np = None
    sr = 24000

    for result in model.generate(
        text=text,
        tokenizer=tokenizer,
        voice=voice,
        temperature=0.0,
        top_p=1.0,
        max_audio_frames=500,
        verbose=verbose,
    ):
        audio_np = np.array(result.audio, dtype=np.float32)
        sr = result.sample_rate
        num_frames = result.samples // (sr // 12)  # approximate frame count from samples
        print(f"Generated {result.samples} samples, duration: {result.audio_duration}")
        print(f"Processing time: {result.processing_time_seconds:.1f}s, "
              f"RTF: {result.real_time_factor:.2f}x")

    if audio_np is None or len(audio_np) == 0:
        print(f"\nFAIL: No audio generated!")
        if diagnose:
            diagnose_weights(model)
        return False

    # Save WAV
    save_wav(audio_np, sr, output_path)

    # Diagnose audio signal
    diag = diagnose_audio(audio_np, sr, num_frames)

    if diagnose:
        diagnose_weights(model)

    # Transcribe
    print(f"\nTranscribing with Whisper...")
    transcription = transcribe_audio(output_path, whisper_model)
    print(f"Transcription: \"{transcription}\"")

    # Compare
    overlap = word_overlap_ratio(text, transcription)
    wer = compute_wer(text, transcription)
    exact = normalize_text(text) == normalize_text(transcription)

    print(f"\n--- Comparison ---")
    ref_words = normalize_text(text).split()
    hyp_words = normalize_text(transcription).split()
    matching = set(ref_words) & set(hyp_words)
    print(f"Word overlap:  {len(matching)}/{len(ref_words)} ({overlap*100:.1f}%)")
    print(f"WER:           {wer*100:.1f}%")
    print(f"Exact match:   {'Yes' if exact else 'No'}")

    passed = overlap >= threshold
    status = "PASS" if passed else "FAIL"
    print(f"\nResult: {status} (word overlap {overlap*100:.1f}% "
          f"{'>=':s} threshold {threshold*100:.1f}%)")

    if not passed and diag["issues"]:
        print(f"\nSuggested investigation:")
        print(f"  1. Check semantic codebook weights (embedding_sum should not be all-zeros)")
        print(f"  2. Check if predict_semantic routes through transformer layers")
        print(f"  3. Check acoustic embedding magnitude (37 embeddings summed)")
        print(f"  4. Try --diagnose flag for weight analysis")

    return passed


TEST_PHRASES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "One two three four five.",
    "Good morning.",
]


def main():
    parser = argparse.ArgumentParser(description="Transcription-based TTS test")
    parser.add_argument("--model-path", required=True, help="Path to MLX model directory")
    parser.add_argument("--text", default="Hello, how are you today?", help="Text to test")
    parser.add_argument("--voice", default=None, help="Path to voice embedding .pt file")
    parser.add_argument("--output", default="test_transcription.wav", help="Output WAV path")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-small-mlx",
                        help="Whisper model for transcription")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Word overlap threshold for pass (0-1)")
    parser.add_argument("--suite", action="store_true", help="Run multi-phrase test suite")
    parser.add_argument("--diagnose", action="store_true", help="Run weight diagnostics")
    parser.add_argument("--verbose", action="store_true", help="Verbose generation output")
    args = parser.parse_args()

    print(f"=== Voxtral TTS Transcription Test ===")
    print(f"Model: {args.model_path}")

    from .voxtral_tts import load
    model, tokenizer = load(args.model_path)
    print(f"Model loaded. Sample rate: {model.sample_rate}")

    if args.suite:
        phrases = TEST_PHRASES
    else:
        phrases = [args.text]

    passed = 0
    total = len(phrases)

    for i, text in enumerate(phrases):
        out_path = args.output if total == 1 else f"test_transcription_{i}.wav"
        ok = run_single_test(
            model, tokenizer, text, args.voice, out_path,
            args.whisper_model, args.threshold, args.verbose, args.diagnose,
        )
        if ok:
            passed += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed < total:
        print(f"STATUS: FAIL")
    else:
        print(f"STATUS: ALL PASSED")


if __name__ == "__main__":
    main()
