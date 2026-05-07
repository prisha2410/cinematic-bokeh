"""
Quick verification that the renderer + refiner work, without needing
to download Depth Anything V2 weights.

Generates a synthetic scene with a "subject" rectangle in front of a
gradient background plus highlight points, then runs:
  - refine_depth_with_mask
  - render_bokeh        (multi-layer hex, the proper renderer)
  - render_bokeh_gaussian_baseline

Saves the outputs to assets/debug/synthetic_*.png.

Run:
    python verify.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np

from src.bokeh import render_bokeh, render_bokeh_gaussian_baseline
from src.refiner import refine_depth_with_mask
from src.utils import save_image, colorize_depth, make_comparison_grid


def make_synthetic(H=480, W=720, seed=0):
    rng = np.random.default_rng(seed)
    # Background: warm gradient with random noise
    bg = np.zeros((H, W, 3), dtype=np.float32)
    grad = np.linspace(0.2, 0.8, W).astype(np.float32)
    bg[..., 0] = grad * 200 + 30
    bg[..., 1] = grad * 150 + 40
    bg[..., 2] = grad * 100 + 60
    bg += rng.normal(0, 8, bg.shape).astype(np.float32)

    # Highlights (point lights) — these become the bokeh "balls"
    for _ in range(25):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(W // 2, W - 20)
        bg[cy - 3:cy + 4, cx - 3:cx + 4] = 250

    rgb = np.clip(bg, 0, 255).astype(np.uint8)

    # "Subject": dark figure on the left
    mask = np.zeros((H, W), dtype=np.float32)
    mask[80:H - 60, 60:W // 2 - 40] = 1.0
    rgb[mask > 0.5] = (40, 35, 50)

    # Depth: monotonically increasing background, subject at high depth value (= near).
    # Then we corrupt it with a "bleed" — soft transition at the subject edge.
    depth = np.tile(np.linspace(0.05, 0.5, W).astype(np.float32), (H, 1))
    depth[mask > 0.5] = 0.92
    # Bleed: 12-pixel soft falloff around the subject edge
    import cv2
    bleed = cv2.GaussianBlur(mask, (0, 0), sigmaX=12)
    depth = bleed * 0.85 + (1 - bleed) * depth
    return rgb, depth.astype(np.float32), mask


def main():
    out_dir = Path("assets/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth, mask = make_synthetic()

    refined = refine_depth_with_mask(depth, mask, rgb)

    focus = 0.92
    bokeh_base = render_bokeh_gaussian_baseline(rgb, depth, focus_depth=focus, max_blur_px=22)
    bokeh_ref = render_bokeh(rgb, refined, focus_depth=focus, max_blur_px=22,
                             kernel_shape="hexagonal", n_layers=7, highlight_boost=1.8)

    save_image(out_dir / "synthetic_input.png", rgb)
    save_image(out_dir / "synthetic_depth_raw.png", colorize_depth(depth))
    save_image(out_dir / "synthetic_depth_refined.png", colorize_depth(refined))
    save_image(out_dir / "synthetic_bokeh_baseline.png", bokeh_base)
    save_image(out_dir / "synthetic_bokeh_refined.png", bokeh_ref)

    make_comparison_grid(
        rgb=rgb,
        depth_raw=depth,
        depth_refined=refined,
        bokeh_baseline=bokeh_base,
        bokeh_refined=bokeh_ref,
        save_path=out_dir / "synthetic_comparison.png",
    )
    print(f"Saved synthetic verification outputs to {out_dir}/")


if __name__ == "__main__":
    main()