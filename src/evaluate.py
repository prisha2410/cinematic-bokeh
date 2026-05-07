"""
Before/after evaluation.

Two flavors:

1) Boundary-quality metrics on the depth map itself
   --------------------------------------------------
   - Edge-IoU between depth-Canny edges and segmentation-mask edges.
     Higher = depth discontinuities are better aligned with the silhouette.
   - Boundary gradient sharpness: mean magnitude of |∇depth| sampled in a
     thin band around the silhouette. Higher = sharper transition (less bleed).

2) Bokeh quality metrics
   --------------------------------------------------
   - Subject sharpness: variance-of-Laplacian inside the subject mask.
     The in-focus subject should be SHARP — higher is better.
   - Background blur: variance-of-Laplacian on the background.
     The defocused background should be SMOOTH — lower is better.
   - "Bleed score": gradient magnitude of the output image in a thin band
     just OUTSIDE the silhouette. A clean cinematic bokeh should have low
     gradient there (background already blurred). High = halos/bleed.

These give you tables for the report; the comparison_grid figure carries
the qualitative argument.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
import json


@dataclass
class DepthMetrics:
    edge_iou: float
    boundary_gradient: float


@dataclass
class BokehMetrics:
    subject_sharpness: float
    background_blur: float
    boundary_bleed: float


# --------------------------- depth metrics ---------------------------

def _canny01(x: np.ndarray, lo: int = 30, hi: int = 90) -> np.ndarray:
    x_u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return (cv2.Canny(x_u8, lo, hi) > 0).astype(np.uint8)


def depth_metrics(depth: np.ndarray, mask: np.ndarray) -> DepthMetrics:
    H, W = depth.shape
    mask01 = (mask > 0.5).astype(np.uint8)

    # Mask boundary as ground-truth silhouette
    mask_edge = _canny01(mask01.astype(np.float32), 50, 150)
    # Depth boundary
    depth_edge = _canny01(depth, 30, 90)

    # Allow small spatial tolerance (3px) when matching edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_edge_d = cv2.dilate(mask_edge, kernel)
    depth_edge_d = cv2.dilate(depth_edge, kernel)
    inter = np.logical_and(mask_edge_d, depth_edge_d).sum()
    union = np.logical_or(mask_edge_d, depth_edge_d).sum()
    edge_iou = float(inter / union) if union > 0 else 0.0

    # Boundary gradient sharpness in a thin band around the silhouette.
    band = cv2.dilate(mask_edge, kernel) > 0
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    boundary_grad = float(gmag[band].mean()) if band.sum() > 0 else 0.0

    return DepthMetrics(edge_iou=edge_iou, boundary_gradient=boundary_grad)


# --------------------------- bokeh metrics ---------------------------

def _var_lap(gray: np.ndarray, region: np.ndarray | None = None) -> float:
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    if region is None:
        return float(lap.var())
    vals = lap[region]
    if vals.size == 0:
        return 0.0
    return float(vals.var())


def bokeh_metrics(out_rgb: np.ndarray, mask: np.ndarray) -> BokehMetrics:
    gray = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask01 = (mask > 0.5)

    subject_sharp = _var_lap(gray, region=mask01)
    background_blur = _var_lap(gray, region=~mask01)

    # Boundary bleed: gradient magnitude in a thin band JUST OUTSIDE the subject.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    outer = cv2.dilate(mask01.astype(np.uint8), kernel) > 0
    band = outer & (~mask01)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    bleed = float(gmag[band].mean()) if band.sum() > 0 else 0.0

    return BokehMetrics(
        subject_sharpness=subject_sharp,
        background_blur=background_blur,
        boundary_bleed=bleed,
    )


# --------------------------- summary ---------------------------

def summarize(result, save_json: str | Path | None = None) -> dict:
    """Compute all metrics for both pipelines and optionally dump JSON."""
    d_raw = depth_metrics(result.depth_raw, result.mask)
    d_ref = depth_metrics(result.depth_refined, result.mask)
    b_base = bokeh_metrics(result.bokeh_baseline, result.mask)
    b_ref = bokeh_metrics(result.bokeh_refined, result.mask)

    summary = {
        "depth": {
            "raw": asdict(d_raw),
            "refined": asdict(d_ref),
            "delta_edge_iou": d_ref.edge_iou - d_raw.edge_iou,
            "delta_boundary_gradient": d_ref.boundary_gradient - d_raw.boundary_gradient,
        },
        "bokeh": {
            "baseline": asdict(b_base),
            "refined": asdict(b_ref),
            "delta_subject_sharpness": b_ref.subject_sharpness - b_base.subject_sharpness,
            "delta_background_blur": b_ref.background_blur - b_base.background_blur,
            "delta_boundary_bleed": b_ref.boundary_bleed - b_base.boundary_bleed,
        },
    }
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(summary, f, indent=2)
    return summary