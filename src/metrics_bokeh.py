"""
Perceptual bokeh similarity metrics.

Computed on the EBB! benchmark, where we have ground-truth wide-aperture
photos. We compare our synthesized bokeh against the real bokeh photo of
the SAME scene.

Metrics:
  - PSNR (peak signal-to-noise ratio): pure pixel fidelity.
  - SSIM (structural similarity): perceptual proxy.
  - LPIPS (learned perceptual): the metric the AIM/EBB challenges actually
    use, computed via a small AlexNet/VGG distance. We use a TINY (≈9MB)
    pre-trained AlexNet-LPIPS model so it runs in seconds on CPU.

Note: pixel-perfect alignment is unrealistic — even matching real bokeh
photos have slight subject motion between the f/16 and f/1.8 captures.
Therefore PSNR/SSIM are reported as a LOWER BOUND of perceptual quality,
and LPIPS is the more meaningful number.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import cv2


@dataclass
class BokehSimilarity:
    psnr: float
    ssim: float
    lpips: float | None  # None if LPIPS unavailable


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean SSIM over color channels (Wang et al., 2004).
    Uses an 11x11 Gaussian window with sigma=1.5, K1=0.01, K2=0.03.
    """
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    win = cv2.getGaussianKernel(11, 1.5)
    window = win @ win.T

    def _ssim_ch(x, y):
        mu_x = cv2.filter2D(x, -1, window)
        mu_y = cv2.filter2D(y, -1, window)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sx2 = cv2.filter2D(x * x, -1, window) - mu_x2
        sy2 = cv2.filter2D(y * y, -1, window) - mu_y2
        sxy = cv2.filter2D(x * y, -1, window) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sxy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
        return float(np.mean(num / den))

    if a.ndim == 2:
        return _ssim_ch(a, b)
    return float(np.mean([_ssim_ch(a[..., c], b[..., c]) for c in range(a.shape[-1])]))


# --- LPIPS (optional) -----------------------------------------------------
# We try to import the `lpips` package; if absent, return None so the rest
# of the pipeline still works.

_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is not None:
        return _lpips_model
    try:
        import torch
        import lpips as _lpips_pkg
        # AlexNet variant — smallest (≈9 MB), fastest, and the one
        # the AIM bokeh challenges report.
        m = _lpips_pkg.LPIPS(net="alex", verbose=False).eval()
        _lpips_model = ("torch", m, torch)
        return _lpips_model
    except Exception:
        _lpips_model = ("unavailable", None, None)
        return _lpips_model


def lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    backend, model, torch = _get_lpips()
    if backend == "unavailable":
        return None

    def _to_tensor(x):
        # LPIPS expects [-1, 1] range, NCHW, float32
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        ta, tb = _to_tensor(a), _to_tensor(b)
        d = model(ta, tb).item()
    return float(d)


def evaluate_pair(synthesized: np.ndarray, ground_truth: np.ndarray) -> BokehSimilarity:
    """
    Compare synthesized bokeh to ground-truth bokeh photo.
    Both are HxWx3 uint8 RGB; we resize ground truth to match synthesized
    if dimensions differ.
    """
    if synthesized.shape != ground_truth.shape:
        gt = cv2.resize(ground_truth, (synthesized.shape[1], synthesized.shape[0]),
                        interpolation=cv2.INTER_AREA)
    else:
        gt = ground_truth
    return BokehSimilarity(
        psnr=psnr(synthesized, gt),
        ssim=ssim(synthesized, gt),
        lpips=lpips(synthesized, gt),
    )