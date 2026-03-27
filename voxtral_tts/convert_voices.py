"""Convert voice embedding .pt files to .safetensors for Swift/MLX compatibility."""

import argparse
from pathlib import Path

import torch
import mlx.core as mx


def convert_voices(model_dir: str):
    voice_dir = Path(model_dir) / "voice_embedding"
    if not voice_dir.exists():
        print(f"No voice_embedding/ found in {model_dir}")
        return

    pt_files = sorted(voice_dir.glob("*.pt"))
    if not pt_files:
        print("No .pt files found")
        return

    print(f"Converting {len(pt_files)} voice embeddings...")
    for pt_file in pt_files:
        data = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor):
            arr = mx.array(data.float().numpy())
        else:
            raise ValueError(f"Unexpected type in {pt_file}: {type(data)}")

        out_path = pt_file.with_suffix(".safetensors")
        mx.save_safetensors(str(out_path), {"embedding": arr})
        print(f"  {pt_file.name} -> {out_path.name}  shape={arr.shape}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Model directory containing voice_embedding/")
    args = parser.parse_args()
    convert_voices(args.model_dir)
