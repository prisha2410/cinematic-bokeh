"""
End-to-end pipeline.

Connects: Depth Anything V2  →  MediaPipe segmentation  →  refinement  →  bokeh.

Single entrypoint:
    pipeline = BokehPipeline()
    result = pipeline.run(rgb, focus_point=(x, y))
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .depth_estimator import DepthAnythingV2Estimator
from .segmenter import SubjectSegmenter
from .refiner import refine_depth_with_mask
from .bokeh import render_bokeh, render_bokeh_gaussian_baseline


@dataclass
class BokehResult:
    rgb: np.ndarray
    depth_raw: np.ndarray
    depth_refined: np.ndarray
    mask: np.ndarray
    bokeh_baseline: np.ndarray
    bokeh_refined: np.ndarray
    focus_depth: float
    focus_point: tuple[int, int] | None


@dataclass
class BokehConfig:
    depth_variant: str = "vits"            # "vits" or "vitb"
    depth_input_size: int = 518
    device: str = "cuda"

    # Refinement
    bleed_radius: int = 8
    edge_sigma_color: float = 0.08
    edge_sigma_space: int = 9

    # Bokeh
    max_blur_px: float = 25.0
    falloff: float = 1.2
    kernel_shape: str = "hexagonal"
    n_layers: int = 7
    highlight_boost: float = 1.6
    highlight_threshold: float = 0.85


class BokehPipeline:
    def __init__(self, config: BokehConfig | None = None):
        self.cfg = config or BokehConfig()
        self.depth = DepthAnythingV2Estimator(
            variant=self.cfg.depth_variant,
            device=self.cfg.device,
            input_size=self.cfg.depth_input_size,
        )
        self.seg = SubjectSegmenter()

    # ----------------------- focus selection -----------------------

    @staticmethod
    def focus_from_point(depth: np.ndarray, point_xy: tuple[int, int], radius: int = 5) -> float:
        """Median depth in a small patch around the user-clicked point."""
        x, y = point_xy
        H, W = depth.shape
        x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        return float(np.median(depth[y0:y1, x0:x1]))

    @staticmethod
    def focus_from_mask(depth: np.ndarray, mask: np.ndarray) -> float:
        """Median depth over the subject mask — sensible default focus."""
        vals = depth[mask > 0.5]
        if vals.size == 0:
            return float(np.median(depth))
        return float(np.median(vals))

    # ----------------------- main entrypoint -----------------------

    def run(
        self,
        rgb: np.ndarray,
        focus_point: tuple[int, int] | None = None,
    ) -> BokehResult:
        # 1) Depth — Depth Anything V2
        depth_raw = self.depth.predict(rgb)

        # 2) Subject mask — MediaPipe
        mask_soft = self.seg.predict(rgb)
        mask = self.seg.sharpen_mask(mask_soft, threshold=0.5, feather_px=2)

        # 3) Refinement (the novel bit)
        depth_refined = refine_depth_with_mask(
            depth=depth_raw,
            mask=mask,
            rgb=rgb,
            bleed_radius=self.cfg.bleed_radius,
            edge_sigma_color=self.cfg.edge_sigma_color,
            edge_sigma_space=self.cfg.edge_sigma_space,
        )

        # 4) Pick focus depth
        if focus_point is not None:
            focus_depth = self.focus_from_point(depth_refined, focus_point)
        else:
            focus_depth = self.focus_from_mask(depth_refined, mask)

        # 5) Render: baseline (raw depth + naive gaussian) and refined (ours)
        bokeh_baseline = render_bokeh_gaussian_baseline(
            rgb=rgb,
            depth=depth_raw,
            focus_depth=focus_depth,
            max_blur_px=self.cfg.max_blur_px,
        )
        bokeh_refined = render_bokeh(
            rgb=rgb,
            depth=depth_refined,
            focus_depth=focus_depth,
            max_blur_px=self.cfg.max_blur_px,
            falloff=self.cfg.falloff,
            kernel_shape=self.cfg.kernel_shape,
            n_layers=self.cfg.n_layers,
            highlight_boost=self.cfg.highlight_boost,
            highlight_threshold=self.cfg.highlight_threshold,
        )

        return BokehResult(
            rgb=rgb,
            depth_raw=depth_raw,
            depth_refined=depth_refined,
            mask=mask,
            bokeh_baseline=bokeh_baseline,
            bokeh_refined=bokeh_refined,
            focus_depth=focus_depth,
            focus_point=focus_point,
        )