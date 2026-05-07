"""
Physically-based cinematic bokeh.

We approximate a thin-lens defocus model:

    CoC(x, y) = |f^2 / (N * (D_focus - f))| * |D(x, y) - D_focus| / D(x, y)

where CoC is the diameter of the circle of confusion at each pixel.
For a stylized cinematic look we reduce this to a tunable form:

    CoC(x, y) = max_blur_px * |D(x, y) - D_focus| ** falloff

with `max_blur_px` set by the chosen aperture (lower N = bigger blur) and
`falloff` shaping how aggressively out-of-focus regions blur.

We render with a *scatter*-style, depth-sorted, multi-layer composite:
- Split the depth range into N discrete CoC layers.
- Blur each layer with the appropriate kernel size.
- Composite back-to-front using a defocused alpha so foreground blur
  spreads OVER the in-focus subject (this is what kills the classic
  "halo" artifact you see in naive gather-style bokeh).

Two kernel shapes are supported:
- "circle"     — modern fast lens look (round bokeh balls).
- "hexagonal"  — cinematic anamorphic / older lens look (6-blade aperture).
"""
from __future__ import annotations
import numpy as np
import cv2


# ----------------------------- kernels -----------------------------

def disk_kernel(radius: int) -> np.ndarray:
    """Anti-aliased disk of given radius. Sums to 1."""
    if radius < 1:
        k = np.ones((1, 1), dtype=np.float32)
        return k / k.sum()
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    r = np.sqrt(xx ** 2 + yy ** 2)
    k = np.clip(radius + 0.5 - r, 0.0, 1.0)  # 1 inside, smooth at the edge
    return (k / k.sum()).astype(np.float32)


def hex_kernel(radius: int) -> np.ndarray:
    """Hexagonal aperture kernel — flat-top hexagon, anti-aliased edge."""
    if radius < 1:
        k = np.ones((1, 1), dtype=np.float32)
        return k / k.sum()
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    # Hexagon: intersection of three slab inequalities.
    a = np.abs(yy)
    b = np.abs(0.5 * yy + (np.sqrt(3) / 2.0) * xx)
    c = np.abs(0.5 * yy - (np.sqrt(3) / 2.0) * xx)
    inside = np.maximum(np.maximum(a, b), c)
    k = np.clip(radius + 0.5 - inside, 0.0, 1.0)
    return (k / k.sum()).astype(np.float32)


KERNEL_FACTORIES = {"circle": disk_kernel, "hexagonal": hex_kernel}


# ----------------------------- CoC --------------------------------

def compute_coc(
    depth: np.ndarray,
    focus_depth: float,
    max_blur_px: float = 25.0,
    falloff: float = 1.2,
) -> np.ndarray:
    """
    Args:
        depth: HxW float32 in [0, 1]  (Depth Anything V2 convention: larger = closer).
        focus_depth: scalar in [0, 1], what's in focus.
        max_blur_px: maximum blur radius in pixels for the most defocused pixel.
        falloff: exponent on the |d - focus| term. >1 sharpens the in-focus
                 region; <1 makes the transition softer.

    Returns:
        coc: HxW float32 with per-pixel blur radius in pixels.
    """
    diff = np.abs(depth.astype(np.float32) - float(focus_depth))
    diff = diff / max(diff.max(), 1e-6)
    coc = max_blur_px * (diff ** falloff)
    return coc


# ------------------------- multi-layer renderer -------------------------

def render_bokeh(
    rgb: np.ndarray,
    depth: np.ndarray,
    focus_depth: float,
    max_blur_px: float = 25.0,
    falloff: float = 1.2,
    kernel_shape: str = "hexagonal",
    n_layers: int = 7,
    highlight_boost: float = 1.6,
    highlight_threshold: float = 0.85,
) -> np.ndarray:
    """
    Args:
        rgb: HxWx3 uint8 RGB image.
        depth: HxW float32 in [0, 1]. Larger = closer (Depth Anything V2 convention).
        focus_depth: scalar in [0, 1].
        max_blur_px: max CoC radius.
        falloff: see compute_coc.
        kernel_shape: "circle" or "hexagonal".
        n_layers: number of discrete CoC bins. 5–9 is a good range.
        highlight_boost: multiplier applied to bright pixels before blurring.
                         This is THE trick that makes bokeh "balls" look
                         cinematic — without it you just get soft blur.
        highlight_threshold: luminance threshold (in [0,1]) above which to boost.

    Returns:
        out_rgb: HxWx3 uint8.
    """
    if kernel_shape not in KERNEL_FACTORIES:
        raise ValueError(f"kernel_shape must be one of {list(KERNEL_FACTORIES)}")
    factory = KERNEL_FACTORIES[kernel_shape]

    H, W, _ = rgb.shape
    img = rgb.astype(np.float32) / 255.0

    # --- Highlight pre-emphasis (bright pixels become "bokeh balls") ------
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    boost_mask = np.clip((luma - highlight_threshold) / max(1.0 - highlight_threshold, 1e-6), 0, 1)
    boosted = img * (1.0 + (highlight_boost - 1.0) * boost_mask[..., None])

    # --- CoC per pixel, then quantize into layers --------------------------
    coc = compute_coc(depth, focus_depth, max_blur_px=max_blur_px, falloff=falloff)
    layer_radii = np.linspace(0, max_blur_px, n_layers).astype(np.int32)
    # Assign each pixel to the nearest layer index.
    idx = np.argmin(np.abs(coc[..., None] - layer_radii[None, None, :]), axis=-1)

    # Front/back ordering: pixels CLOSER than focus go in foreground layers
    # (composited on top after blurring), farther pixels go in background.
    is_foreground = depth > float(focus_depth)

    # --- Render each layer -------------------------------------------------
    # Premultiplied-alpha trick: blur (color * alpha) and (alpha) separately,
    # then divide. This prevents background colors from bleeding into the
    # in-focus subject.
    bg_accum = np.zeros_like(boosted)
    bg_alpha = np.zeros((H, W, 1), dtype=np.float32)
    fg_accum = np.zeros_like(boosted)
    fg_alpha = np.zeros((H, W, 1), dtype=np.float32)

    for li, r in enumerate(layer_radii):
        mask = (idx == li)
        if not mask.any():
            continue
        fg_layer_mask = mask & is_foreground
        bg_layer_mask = mask & (~is_foreground)

        kernel = factory(int(r))

        for layer_mask, accum, alpha_accum in [
            (bg_layer_mask, bg_accum, bg_alpha),
            (fg_layer_mask, fg_accum, fg_alpha),
        ]:
            if not layer_mask.any():
                continue
            a = layer_mask.astype(np.float32)[..., None]
            premult = boosted * a
            if r >= 1:
                premult_blurred = cv2.filter2D(premult, -1, kernel)
                a_blurred = cv2.filter2D(a, -1, kernel)
            else:
                premult_blurred = premult
                a_blurred = a
            # cv2.filter2D drops trailing singleton dims; restore them.
            if a_blurred.ndim == 2:
                a_blurred = a_blurred[..., None]
            if premult_blurred.ndim == 2:
                premult_blurred = premult_blurred[..., None]
            accum += premult_blurred
            alpha_accum += a_blurred

    # Resolve premultiplied alpha (avoid divide by zero).
    bg_color = bg_accum / np.clip(bg_alpha, 1e-6, None)
    fg_color = fg_accum / np.clip(fg_alpha, 1e-6, None)
    bg_a = np.clip(bg_alpha, 0, 1)
    fg_a = np.clip(fg_alpha, 0, 1)

    # Composite: background first, then foreground OVER it.
    out = bg_color * bg_a + img * (1.0 - bg_a)         # in-focus subject re-shows through bg blur
    out = fg_color * fg_a + out * (1.0 - fg_a)         # foreground blur sits on top

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ------------------------- baseline (gaussian) -------------------------

def render_bokeh_gaussian_baseline(
    rgb: np.ndarray,
    depth: np.ndarray,
    focus_depth: float,
    max_blur_px: float = 25.0,
) -> np.ndarray:
    """
    Naive baseline used in the report: per-pixel variable-sigma Gaussian blur.
    No highlight boost, no premultiplied alpha, no layer compositing.
    This is what you compare AGAINST in the figures — it shows the artifacts
    that the proper renderer + refined depth fix.
    """
    H, W, _ = rgb.shape
    img = rgb.astype(np.float32) / 255.0
    coc = compute_coc(depth, focus_depth, max_blur_px=max_blur_px, falloff=1.0)
    # Discrete sigma layers approximated by per-layer Gaussian blurs.
    sigmas = np.linspace(0.0, max_blur_px / 2.0, 6)
    out = img.copy()
    for s in sigmas[1:]:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(s))
        weight = np.exp(-((coc - 2 * s) ** 2) / (2 * (max_blur_px / 6) ** 2))[..., None]
        out = out * (1 - weight) + blurred * weight
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)