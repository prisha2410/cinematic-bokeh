"""
Depth Anything V2 wrapper.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "depth_anything_v2_repo"
if EXTERNAL.exists() and str(EXTERNAL) not in sys.path:
    sys.path.insert(0, str(EXTERNAL))

_DepthAnythingV2 = None


def _lazy_import():
    global _DepthAnythingV2
    if _DepthAnythingV2 is None:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        _DepthAnythingV2 = DepthAnythingV2
    return _DepthAnythingV2


MODEL_CONFIGS = {
    "vits": dict(encoder="vits", features=64,  out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


class DepthAnythingV2Estimator:

    def __init__(
        self,
        variant: str = "vits",
        device: str = "cuda",
        input_size: int = 518,
        weights_dir: str | Path | None = None,
    ):
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"variant must be one of {list(MODEL_CONFIGS)}")
        self.variant = variant
        self.input_size = input_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        DepthAnythingV2 = _lazy_import()
        self.model = DepthAnythingV2(**MODEL_CONFIGS[variant])

        weights_dir = Path(weights_dir) if weights_dir else (ROOT / "models")
        ckpt_path = weights_dir / f"depth_anything_v2_{variant}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing weights at {ckpt_path}. Run `python download_weights.py` first."
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        # fp16 removed — input/weight dtype mismatch on RTX 3050; fp32 fits fine in 4GB

        print(f"[DepthAnythingV2] {variant} loaded on {self.device}")

    @torch.inference_mode()
    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Args:
            rgb_uint8: HxWx3 uint8 RGB image.
        Returns:
            depth: HxW float32, normalized to [0, 1]. Larger values = closer.
        """
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        depth = self.model.infer_image(bgr, self.input_size)
        d = depth.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d)
        return (d - lo) / (hi - lo)