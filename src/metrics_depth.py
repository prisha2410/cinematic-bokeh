"""
Standard monocular depth metrics, computed against DIODE ground truth.

Used for the table in your report that shows whether refinement actually
improves metric depth quality (not just boundary metrics).

Metrics implemented (the standard suite from Eigen et al. 2014 onwards):
  - AbsRel:  mean(|d_pred - d_gt| / d_gt)
  - RMSE:    sqrt(mean((d_pred - d_gt)^2))
  - log10:   mean(|log10(d_pred) - log10(d_gt)|)
  - delta_1: fraction of pixels with max(d_pred/d_gt, d_gt/d_pred) < 1.25
  - delta_2: ... < 1.25^2
  - delta_3: ... < 1.25^3

IMPORTANT: Depth Anything V2 outputs RELATIVE depth in [0, 1]
(actually inverse depth, but that's a unit choice). DIODE has METRIC depth
in meters. We must align them with a least-squares scale-and-shift fit
before computing metrics — this is the standard MiDaS / Depth Anything V2
evaluation protocol.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class DepthErrors:
    abs_rel: float
    rmse: float
    log10: float
    delta_1: float
    delta_2: float
    delta_3: float


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fit pred ≈ a * pred + b to gt, in least-squares sense, on valid pixels.
    Returns the aligned prediction (same shape as `pred`).

    This is the protocol used by MiDaS / DPT / Depth Anything V2 for
    cross-dataset evaluation (relative depth -> metric depth).
    """
    valid = mask > 0.5
    p = pred[valid].astype(np.float64).flatten()
    g = gt[valid].astype(np.float64).flatten()
    if p.size < 100:
        return pred  # not enough valid points to fit

    # solve [p, 1] @ [a, b] = g  in least-squares sense
    A = np.stack([p, np.ones_like(p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    aligned = a * pred + b
    return aligned


def compute_depth_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> DepthErrors:
    """
    Args:
        pred: HxW float, aligned to gt's metric scale.
        gt:   HxW float metric depth (meters).
        mask: HxW {0,1} validity mask. Invalid GT pixels are excluded.
        min_depth, max_depth: clamp range used in standard benchmarks.
    """
    valid = (mask > 0.5) & (gt > min_depth) & (gt < max_depth) & (pred > 0)
    if valid.sum() < 100:
        return DepthErrors(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    p = np.clip(pred[valid], min_depth, max_depth).astype(np.float64)
    g = np.clip(gt[valid], min_depth, max_depth).astype(np.float64)

    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    log10 = float(np.mean(np.abs(np.log10(p) - np.log10(g))))

    ratio = np.maximum(p / g, g / p)
    d1 = float((ratio < 1.25).mean())
    d2 = float((ratio < 1.25 ** 2).mean())
    d3 = float((ratio < 1.25 ** 3).mean())

    return DepthErrors(abs_rel, rmse, log10, d1, d2, d3)


def evaluate_pair(
    depth_pred: np.ndarray,   # [0, 1] inverse-depth-ish from Depth Anything V2
    depth_gt: np.ndarray,     # metric depth in meters
    mask_gt: np.ndarray,      # validity mask
) -> DepthErrors:
    """
    Convenience wrapper: convert inverse depth to depth, scale-shift align,
    compute standard errors.

    Depth Anything V2 outputs values in [0, 1] where larger = closer
    (i.e. inverse-depth-flavored). To compare with metric depth we:
      1) Convert: depth_proxy = 1 / max(pred, eps)
         (this lives in [1, ∞); shift/scale will absorb the constants)
      2) Fit a, b such that a * depth_proxy + b ≈ depth_gt on valid pixels.
      3) Compute errors on the aligned prediction.
    """
    eps = 1e-3
    depth_proxy = 1.0 / np.clip(depth_pred, eps, None)
    aligned = align_scale_shift(depth_proxy, depth_gt, mask_gt)
    return compute_depth_errors(aligned, depth_gt, mask_gt)