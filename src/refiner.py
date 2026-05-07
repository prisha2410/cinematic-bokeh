"""
Segmentation-guided depth boundary refinement.

THIS IS THE NOVEL CONTRIBUTION OF THE PROJECT.

Problem (motivated by reference papers):
- Depth Anything V2 (Yang et al., NeurIPS 2024) and DPT (Ranftl et al., 2021)
  produce smooth, low-frequency depth maps because the dense prediction head
  is trained with a scale-and-shift-invariant loss on aggregated features.
  This causes "depth bleeding" at high-frequency boundaries — hair, fingers,
  thin edges — where the predicted depth gradually transitions across the
  silhouette instead of snapping to it.
- For cinematic bokeh, this looks WRONG. A correct shallow-DoF rendering
  needs a sharp depth discontinuity at the subject silhouette so the
  background blur cleanly stops where the subject begins. Otherwise the
  hair gets blurred away and the silhouette looks "melted".

Idea:
- A semantic segmentation network produces a much sharper boundary than a
  monocular depth network, because it's solving a binary problem with a
  loss that directly penalizes silhouette errors.
- We use that mask as a *prior* on where a depth discontinuity SHOULD exist,
  then re-impose it on the depth map.

Method (three stages, each justified):

  (1) Subject-side flattening
      Inside the mask, replace the depth with a robust statistic
      (median or trimmed mean) of the masked region. Rationale: a single
      subject is approximately at one focal distance for cinematic bokeh
      purposes — small intra-subject depth variation should NOT cause
      visible blur on the in-focus subject.

  (2) Background-side hole filling
      Where the soft mask spilled depth from the subject onto the
      background (the "bleeding" region near edges), inpaint those
      pixels from the surrounding background depth so the background
      stays at its true distance up to the silhouette.

  (3) Edge-aware joint smoothing
      A guided / joint bilateral filter using the RGB image as guide,
      so the final depth has sharp edges aligned with image structure
      (hair strands, fabric edges) rather than blurry transitions.

Output: a depth map that is locally smooth on each side of the silhouette
but has a clean discontinuity along it — exactly what defocus rendering
needs to produce convincing bokeh.
"""
from __future__ import annotations
import numpy as np
import cv2


# ----------------------------- helpers -----------------------------

def _trimmed_mean(values: np.ndarray, trim: float = 0.1) -> float:
    """Mean after dropping the lowest and highest `trim` fraction."""
    if values.size == 0:
        return 0.0
    v = np.sort(values)
    k = int(len(v) * trim)
    if 2 * k >= len(v):
        return float(np.median(v))
    return float(v[k:len(v) - k].mean())


def _dilate(mask01: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask01, kernel)


def _erode(mask01: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask01
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.erode(mask01, kernel)


# ----------------------------- main API -----------------------------

def refine_depth_with_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    subject_flatten: str = "median",   # "median" | "trimmed_mean" | "none"
    bleed_radius: int = 8,             # how far we assume depth bled outward
    edge_sigma_color: float = 0.08,    # bilateral edge sensitivity (depth scale [0,1])
    edge_sigma_space: int = 9,         # bilateral spatial radius
    use_guided_filter: bool = True,    # True: joint filter w/ RGB guide
) -> np.ndarray:
    """
    Args:
        depth: HxW float32 depth in [0, 1] (larger = closer; matches Depth Anything V2).
        mask:  HxW float32 soft subject mask in [0, 1].
        rgb:   HxWx3 uint8 RGB image (used as guide for edge-aware filtering).

    Returns:
        refined: HxW float32 depth in [0, 1] with a sharp boundary at the silhouette.
    """
    assert depth.ndim == 2 and mask.ndim == 2 and rgb.ndim == 3
    H, W = depth.shape
    d = depth.astype(np.float32).copy()
    m_soft = np.clip(mask.astype(np.float32), 0.0, 1.0)
    m_hard = (m_soft > 0.5).astype(np.float32)

    # --- Stage 1: flatten subject depth -------------------------------------
    if subject_flatten != "none" and m_hard.sum() > 0:
        subj_vals = d[m_hard > 0.5]
        if subject_flatten == "median":
            subj_depth = float(np.median(subj_vals))
        else:
            subj_depth = _trimmed_mean(subj_vals, trim=0.1)
        # Soft replacement: closer to subj_depth where mask is high.
        d = m_soft * subj_depth + (1.0 - m_soft) * d

    # --- Stage 2: heal the "bleed" ring around the subject ------------------
    # Pixels that are NOT subject but lie within `bleed_radius` of one are
    # the ones likely contaminated by depth-network edge bleeding.
    dilated = _dilate(m_hard, bleed_radius)
    bleed_ring = ((dilated > 0.5) & (m_hard < 0.5)).astype(np.uint8)

    if bleed_ring.sum() > 0:
        # Inpaint those pixels using the surrounding (clean) background depth.
        d_uint16 = np.clip(d * 65535.0, 0, 65535).astype(np.uint16)
        # OpenCV inpaint wants 8-bit; do it on 8-bit then re-blend.
        d_u8 = (d * 255).astype(np.uint8)
        healed_u8 = cv2.inpaint(d_u8, bleed_ring, inpaintRadius=bleed_radius,
                                flags=cv2.INPAINT_TELEA)
        healed = healed_u8.astype(np.float32) / 255.0
        # Replace only the bleed ring; keep the rest exact.
        d = np.where(bleed_ring > 0, healed, d)

    # --- Stage 3: edge-aware joint smoothing --------------------------------
    # We want sharp edges aligned with the RGB image (hair strands, fabric)
    # while keeping each region piecewise-smooth.
    if use_guided_filter:
        try:
            # ximgproc is in opencv-contrib; fall back to bilateral if unavailable.
            from cv2 import ximgproc  # type: ignore
            guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            d = ximgproc.guidedFilter(
                guide=guide,
                src=d.astype(np.float32),
                radius=edge_sigma_space,
                eps=edge_sigma_color ** 2,
            )
        except Exception:
            d = cv2.bilateralFilter(
                d.astype(np.float32),
                d=2 * edge_sigma_space + 1,
                sigmaColor=edge_sigma_color,
                sigmaSpace=edge_sigma_space,
            )
    else:
        d = cv2.bilateralFilter(
            d.astype(np.float32),
            d=2 * edge_sigma_space + 1,
            sigmaColor=edge_sigma_color,
            sigmaSpace=edge_sigma_space,
        )

    # Final clamp to [0, 1].
    return np.clip(d, 0.0, 1.0)