"""
Download pretrained weights for Depth Anything V2.

For your 4GB VRAM (RTX 3050), use the SMALL variant (vits) at 518x518 input.
- Small (vits):  ~25M params,  ~0.7 GB VRAM  ← recommended
- Base  (vitb):  ~98M params,  ~1.8 GB VRAM  ← also fine
- Large (vitl):  ~335M params, ~5.5 GB VRAM  ← will OOM on 3050

Run:
    python download_weights.py
"""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Hugging Face hosts the official checkpoints.
WEIGHTS = {
    "vits": (
        "depth_anything_v2_vits.pth",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    ),
    "vitb": (
        "depth_anything_v2_vitb.pth",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    ),
}


def download(variant: str = "vits"):
    if variant not in WEIGHTS:
        raise ValueError(f"variant must be one of {list(WEIGHTS)}")
    fname, url = WEIGHTS[variant]
    out = MODELS_DIR / fname
    if out.exists():
        print(f"[ok] {fname} already present.")
        return out
    print(f"[..] downloading {fname}  ({url})")
    urllib.request.urlretrieve(url, out)
    print(f"[ok] saved to {out}")
    return out


if __name__ == "__main__":
    # Default: small variant for 4GB VRAM
    download("vits")
    # Uncomment if you want the base model too:
    # download("vitb")
    print("\nNext step: clone Depth-Anything-V2 source into ./external/")
    print("    git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2_repo")