"""
Run the full pipeline on a single image.

Usage:
    python run.py --input assets/input/portrait.jpg
    python run.py --input assets/input/portrait.jpg --focus 540 720
    python run.py --input assets/input/portrait.jpg --kernel circle --max-blur 35
    python run.py --input assets/input/portrait.jpg --depth-variant vitb
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

from src.pipeline import BokehPipeline, BokehConfig
from src.evaluate import summarize
from src.utils import (
    load_image, save_image,
    colorize_depth, overlay_mask, make_comparison_grid,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output-dir", default="assets/output", type=str)
    p.add_argument("--debug-dir", default="assets/debug", type=str)
    p.add_argument("--focus", nargs=2, type=int, default=None,
                   metavar=("X", "Y"),
                   help="Pixel coords of the focus point. Default: subject median.")
    p.add_argument("--depth-variant", default="vits", choices=["vits", "vitb"])
    p.add_argument("--kernel", default="hexagonal", choices=["circle", "hexagonal"])
    p.add_argument("--max-blur", type=float, default=25.0)
    p.add_argument("--falloff", type=float, default=1.2)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    dbg_dir = Path(args.debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    cfg = BokehConfig(
        depth_variant=args.depth_variant,
        device="cpu" if args.no_cuda else "cuda",
        max_blur_px=args.max_blur,
        falloff=args.falloff,
        kernel_shape=args.kernel,
        n_layers=args.n_layers,
    )
    pipeline = BokehPipeline(cfg)

    rgb = load_image(in_path)
    focus = tuple(args.focus) if args.focus else None
    result = pipeline.run(rgb, focus_point=focus)

    stem = in_path.stem
    # Save the headline outputs
    save_image(out_dir / f"{stem}_bokeh_refined.png", result.bokeh_refined)
    save_image(out_dir / f"{stem}_bokeh_baseline.png", result.bokeh_baseline)

    # Save debug intermediates for the report
    save_image(dbg_dir / f"{stem}_depth_raw.png", colorize_depth(result.depth_raw))
    save_image(dbg_dir / f"{stem}_depth_refined.png", colorize_depth(result.depth_refined))
    save_image(dbg_dir / f"{stem}_mask_overlay.png", overlay_mask(result.rgb, result.mask))

    # Comparison grid (the figure for the report)
    make_comparison_grid(
        rgb=result.rgb,
        depth_raw=result.depth_raw,
        depth_refined=result.depth_refined,
        bokeh_baseline=result.bokeh_baseline,
        bokeh_refined=result.bokeh_refined,
        save_path=dbg_dir / f"{stem}_comparison.png",
    )

    # Metrics JSON
    metrics = summarize(result, save_json=dbg_dir / f"{stem}_metrics.json")
    print(json.dumps(metrics, indent=2))
    print(f"\nFocus depth: {result.focus_depth:.3f}")
    print(f"Saved outputs to {out_dir} and debug to {dbg_dir}")


if __name__ == "__main__":
    main()