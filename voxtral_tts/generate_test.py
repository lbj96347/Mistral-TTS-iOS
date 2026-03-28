"""CLI test script for Voxtral TTS audio generation.

Usage:
    python -m voxtral_tts.generate_test --model-path mlx_model --text "Hello world"
    python -m voxtral_tts.generate_test --model-path mlx_model --text "Hello world" --voice mlx_model/voice_embedding/neutral_female.pt
    python -m voxtral_tts.generate_test --model-path mlx_model --text "Hello world" --output test.wav --verbose
"""

import argparse
import wave
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


def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Voxtral TTS")
    parser.add_argument("--model-path", required=True, help="Path to MLX model directory")
    parser.add_argument("--text", default="Hello world, this is a test.", help="Text to synthesize")
    parser.add_argument("--voice", default=None, help="Path to voice embedding .pt file")
    parser.add_argument("--output", default="test_output.wav", help="Output WAV file path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling threshold")
    parser.add_argument("--max-frames", type=int, default=500, help="Maximum audio frames")
    parser.add_argument("--verbose", action="store_true", help="Print progress info")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    from .voxtral_tts import load
    model, tokenizer = load(args.model_path)
    print(f"Model loaded. Sample rate: {model.sample_rate}")

    print(f"Generating: \"{args.text}\"")
    if args.voice:
        print(f"Voice: {args.voice}")

    for result in model.generate(
        text=args.text,
        tokenizer=tokenizer,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        max_audio_frames=args.max_frames,
        verbose=args.verbose,
    ):
        audio = np.array(result.audio, dtype=np.float32)
        print(f"Generated {result.samples} samples, duration: {result.audio_duration}")
        print(f"RTF: {result.real_time_factor:.2f}x, time: {result.processing_time_seconds:.1f}s")
        print(f"Peak memory: {result.peak_memory_usage:.2f} GB")

        # Audio stats
        print(f"Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, "
              f"mean={audio.mean():.4f}, std={audio.std():.4f}")

        save_wav(audio, result.sample_rate, args.output)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
