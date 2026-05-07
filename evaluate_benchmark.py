"""
Run the full pipeline across the benchmark and emit a LaTeX-ready metrics table.
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import BokehPipeline, BokehConfig
from src.utils import load_image
from src.evaluate import depth_metrics, bokeh_metrics
from src.metrics_depth import evaluate_pair as eval_depth_pair, align_scale_shift
from src.metrics_bokeh import evaluate_pair as eval_bokeh_pair


REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_diode(pipeline: BokehPipeline) -> list[dict]:
    rgb_dir = ROOT / "data" / "diode" / "rgb"
    depth_dir = ROOT / "data" / "diode" / "depth"
    mask_dir = ROOT / "data" / "diode" / "depth_mask"
    if not rgb_dir.exists():
        print(f"[!] DIODE not found at {rgb_dir}. Run data/scripts/download_diode.py.")
        return []

    rows = []
    rgbs = sorted(rgb_dir.glob("*.png"))
    if not rgbs:
        print(f"[!] No DIODE RGBs found.")
        return []

    for rgb_path in tqdm(rgbs, desc="DIODE"):
        stem = rgb_path.stem
        depth_path = depth_dir / f"{stem}_depth.npy"
        mask_path = mask_dir / f"{stem}_depth_mask.npy"
        if not depth_path.exists() or not mask_path.exists():
            continue

        rgb = load_image(rgb_path)
        gt_depth = np.load(depth_path).astype(np.float32).squeeze()
        gt_mask = np.load(mask_path).astype(np.float32).squeeze()

        if gt_depth.shape != rgb.shape[:2]:
            gt_depth = cv2.resize(gt_depth, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.resize(gt_mask, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        result = pipeline.run(rgb)
        e_raw = eval_depth_pair(result.depth_raw, gt_depth, gt_mask)
        e_ref = eval_depth_pair(result.depth_refined, gt_depth, gt_mask)

        rows.append({
            "scene": stem,
            "raw": asdict(e_raw),
            "refined": asdict(e_ref),
        })
    return rows


def evaluate_ebb(pipeline: BokehPipeline) -> list[dict]:
    narrow_dir = ROOT / "data" / "ebb" / "narrow_aperture"
    wide_dir = ROOT / "data" / "ebb" / "wide_aperture"
    if not narrow_dir.exists():
        print(f"[!] EBB! not found at {narrow_dir}. Run data/scripts/download_ebb.py.")
        return []

    rows = []
    inputs = sorted(narrow_dir.glob("*.jpg"))
    if not inputs:
        print(f"[!] No EBB! images found.")
        return []

    for in_path in tqdm(inputs, desc="EBB!"):
        gt_path = wide_dir / in_path.name
        if not gt_path.exists():
            continue

        rgb = load_image(in_path)
        gt = load_image(gt_path)
        result = pipeline.run(rgb)

        sim_baseline = eval_bokeh_pair(result.bokeh_baseline, gt)
        sim_refined = eval_bokeh_pair(result.bokeh_refined, gt)

        rows.append({
            "scene": in_path.stem,
            "baseline": asdict(sim_baseline),
            "refined": asdict(sim_refined),
        })
    return rows


def evaluate_unsplash(pipeline: BokehPipeline) -> list[dict]:
    img_dir = ROOT / "data" / "unsplash"
    if not img_dir.exists() or not list(img_dir.glob("*.jpg")):
        print(f"[!] Unsplash photos not found at {img_dir}. Run data/scripts/download_unsplash.py.")
        return []

    rows = []
    for in_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Unsplash"):
        rgb = load_image(in_path)
        result = pipeline.run(rgb)
        d_raw = depth_metrics(result.depth_raw, result.mask)
        d_ref = depth_metrics(result.depth_refined, result.mask)
        b_base = bokeh_metrics(result.bokeh_baseline, result.mask)
        b_ref = bokeh_metrics(result.bokeh_refined, result.mask)
        rows.append({
            "scene": in_path.stem,
            "depth_raw": asdict(d_raw),
            "depth_refined": asdict(d_ref),
            "bokeh_baseline": asdict(b_base),
            "bokeh_refined": asdict(b_ref),
        })
    return rows


def average(rows, key: str, sub: str) -> float:
    vals = [row[key][sub] for row in rows
            if row.get(key) and isinstance(row[key].get(sub), (int, float))
            and np.isfinite(row[key][sub])]
    return float(np.mean(vals)) if vals else float("nan")


def render_latex(diode, ebb, unsplash) -> str:
    parts = []

    if diode:
        parts.append(r"""
\begin{table}[t]
\centering
\caption{Depth accuracy on DIODE (10 scenes). Lower is better for AbsRel, RMSE, log10; higher for $\delta_i$.}
\label{tab:diode}
\begin{tabular}{lcccccc}
\toprule
Method & AbsRel $\downarrow$ & RMSE $\downarrow$ & log10 $\downarrow$ & $\delta_1$ $\uparrow$ & $\delta_2$ $\uparrow$ & $\delta_3$ $\uparrow$ \\
\midrule
""")
        for label, key in [("Depth Anything V2 (raw)", "raw"),
                           ("\\quad + ours (refined)", "refined")]:
            row = [
                f"{average(diode, key, 'abs_rel'):.4f}",
                f"{average(diode, key, 'rmse'):.4f}",
                f"{average(diode, key, 'log10'):.4f}",
                f"{average(diode, key, 'delta_1'):.4f}",
                f"{average(diode, key, 'delta_2'):.4f}",
                f"{average(diode, key, 'delta_3'):.4f}",
            ]
            parts.append(label + " & " + " & ".join(row) + r" \\" + "\n")
        parts.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    if ebb:
        parts.append(r"""
\begin{table}[t]
\centering
\caption{Bokeh similarity on EBB! (10 paired images). PSNR/SSIM higher is better; LPIPS lower is better.}
\label{tab:ebb}
\begin{tabular}{lccc}
\toprule
Method & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\
\midrule
""")
        for label, key in [("Naive Gaussian on raw depth", "baseline"),
                           ("Multi-layer hex on refined (ours)", "refined")]:
            psnr = average(ebb, key, 'psnr')
            ssim_v = average(ebb, key, 'ssim')
            lpips_v = average(ebb, key, 'lpips')
            lpips_str = f"{lpips_v:.4f}" if np.isfinite(lpips_v) else "n/a"
            parts.append(f"{label} & {psnr:.3f} & {ssim_v:.4f} & {lpips_str} \\\\\n")
        parts.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    if unsplash:
        parts.append(r"""
\begin{table}[t]
\centering
\caption{Boundary quality on in-the-wild Unsplash photos (9 images). Higher is better for both metrics.}
\label{tab:unsplash}
\begin{tabular}{lcc}
\toprule
Method & Edge-IoU $\uparrow$ & Boundary $\nabla$ $\uparrow$ \\
\midrule
""")
        for label, key in [("Depth Anything V2 (raw)", "depth_raw"),
                           ("\\quad + ours (refined)", "depth_refined")]:
            iou = average(unsplash, key, "edge_iou")
            bg = average(unsplash, key, "boundary_gradient")
            parts.append(f"{label} & {iou:.4f} & {bg:.4f} \\\\\n")
        parts.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    return "".join(parts)


def render_markdown(diode, ebb, unsplash) -> str:
    out = ["# Benchmark Results\n"]

    if diode:
        out.append("## DIODE - depth accuracy (10 scenes)\n")
        out.append("| Method | AbsRel (lower=better) | RMSE (lower=better) | log10 (lower=better) | d1 (higher=better) | d2 (higher=better) | d3 (higher=better) |")
        out.append("|---|---|---|---|---|---|---|")
        for label, key in [("Depth Anything V2 (raw)", "raw"),
                           ("+ ours (refined)", "refined")]:
            cells = [f"{average(diode, key, k):.4f}"
                     for k in ["abs_rel", "rmse", "log10", "delta_1", "delta_2", "delta_3"]]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")

    if ebb:
        out.append("## EBB! - bokeh similarity (10 pairs)\n")
        out.append("| Method | PSNR (higher=better) | SSIM (higher=better) | LPIPS (lower=better) |")
        out.append("|---|---|---|---|")
        for label, key in [("Naive Gaussian on raw depth", "baseline"),
                           ("Multi-layer hex on refined (ours)", "refined")]:
            psnr = average(ebb, key, 'psnr')
            ssim_v = average(ebb, key, 'ssim')
            lpips_v = average(ebb, key, 'lpips')
            lpips_str = f"{lpips_v:.4f}" if np.isfinite(lpips_v) else "n/a"
            out.append(f"| {label} | {psnr:.3f} | {ssim_v:.4f} | {lpips_str} |")
        out.append("")

    if unsplash:
        out.append("## Unsplash - boundary quality (9 photos, no GT)\n")
        out.append("| Method | Edge-IoU (higher=better) | Boundary Grad (higher=better) |")
        out.append("|---|---|---|")
        for label, key in [("Depth Anything V2 (raw)", "depth_raw"),
                           ("+ ours (refined)", "depth_refined")]:
            iou = average(unsplash, key, "edge_iou")
            bg = average(unsplash, key, "boundary_gradient")
            out.append(f"| {label} | {iou:.4f} | {bg:.4f} |")
        out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["diode", "ebb", "unsplash"], default=None)
    p.add_argument("--depth-variant", default="vits", choices=["vits", "vitb"])
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    cfg = BokehConfig(
        depth_variant=args.depth_variant,
        device="cpu" if args.no_cuda else "cuda",
    )
    pipeline = BokehPipeline(cfg)

    diode = ebb = unsplash = []
    if args.only in (None, "diode"):
        diode = evaluate_diode(pipeline)
    if args.only in (None, "ebb"):
        ebb = evaluate_ebb(pipeline)
    if args.only in (None, "unsplash"):
        unsplash = evaluate_unsplash(pipeline)

    raw = {"diode": diode, "ebb": ebb, "unsplash": unsplash}
    # utf-8 encoding fixes UnicodeEncodeError on Windows (cp1252 can't handle arrows)
    (REPORT_DIR / "benchmark_results.json").write_text(
        json.dumps(raw, indent=2), encoding="utf-8")

    md = render_markdown(diode, ebb, unsplash)
    (REPORT_DIR / "benchmark_table.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    tex = render_latex(diode, ebb, unsplash)
    (REPORT_DIR / "benchmark_table.tex").write_text(tex, encoding="utf-8")

    print(f"\n[ok] Wrote: {REPORT_DIR / 'benchmark_table.md'}")
    print(f"[ok] Wrote: {REPORT_DIR / 'benchmark_table.tex'}")
    print(f"[ok] Wrote: {REPORT_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()