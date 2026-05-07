"""I/O, normalization, and visualization helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ----------------------------- I/O -----------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Return RGB uint8 HxWx3 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    cv2.imwrite(str(path), bgr)


# ----------------------------- Normalization -----------------------------

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Works on any float depth map."""
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-8:
        return np.zeros_like(d)
    return (d - lo) / (hi - lo)


def colorize_depth(depth: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """Turn a [0,1] depth map into an RGB uint8 image for visualization."""
    d = normalize_depth(depth)
    rgba = cm.get_cmap(cmap)(d)
    return (rgba[..., :3] * 255).astype(np.uint8)


# ----------------------------- Visualization -----------------------------

def make_comparison_grid(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    depth_refined: np.ndarray,
    bokeh_baseline: np.ndarray,
    bokeh_refined: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save a 2x3 figure comparing baseline vs refined pipeline (the money shot for your report)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB")
    axes[0, 1].imshow(colorize_depth(depth_raw))
    axes[0, 1].set_title("Depth Anything V2 (raw)")
    axes[0, 2].imshow(colorize_depth(depth_refined))
    axes[0, 2].set_title("Refined depth (ours)")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(bokeh_baseline)
    axes[1, 1].set_title("Bokeh — raw depth")
    axes[1, 2].imshow(bokeh_refined)
    axes[1, 2].set_title("Bokeh — refined depth (ours)")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the masked region red for debug visualization."""
    out = rgb.astype(np.float32).copy()
    m = (mask > 0.5).astype(np.float32)[..., None]
    color = np.array([255, 60, 60], dtype=np.float32)
    out = out * (1 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)