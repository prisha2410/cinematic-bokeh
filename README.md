# Semantically-Guided Depth Refinement for Cinematic Bokeh Synthesis

Single-image cinematic bokeh from a monocular RGB input — no stereo camera, no training.

**Pipeline:** Depth Anything V2 → MediaPipe Segmentation → Boundary Refinement *(ours)* → Multi-Layer Bokeh

---

## The Problem

Depth networks based on Vision Transformers process images in 14×14 pixel patches. At object boundaries, one patch straddles both subject and background — the predicted depth gets averaged across them. The result is a **"depth bleed"**: a gradual depth transition instead of a sharp edge at the subject silhouette.

When you apply defocus blur using this blurry depth map, the subject's edges (hair, fingers, clothing) get a visible glowing fringe — the classic **halo artifact** you see in smartphone portrait mode.

---

## What's Novel: The 3-Stage Depth Refinement Module

`src/refiner.py` — the core contribution. Uses the segmentation mask as a prior on where a sharp depth discontinuity *should* exist, then re-imposes it on the depth map:

| Stage | What it does | Why |
|---|---|---|
| **1. Subject Flattening** | Replace depth inside the mask with the median subject depth | Eliminates intra-subject depth noise that would partially blur the in-focus subject |
| **2. Bleed-Ring Inpainting** | Dilate mask by 8px → identify contaminated background pixels → inpaint from clean background | Kills the halo zone at the silhouette boundary |
| **3. Guided Filter** | Edge-aware smoothing using the RGB image as guide | Re-imposes fine boundary detail (hair strands, fabric edges) aligned with image structure |

The renderer (`src/bokeh.py`) adds two more techniques:
- **Premultiplied-alpha multi-layer compositing** — 7 discrete depth layers blended back-to-front, preventing color bleeding across depth boundaries
- **Highlight pre-emphasis** — bright pixels boosted 1.6× before blurring, producing the distinct bokeh "balls" you see from real lenses

---

## Results

Evaluated on 9 in-the-wild photos (Unsplash) using boundary-quality metrics:

| Method | Edge-IoU ↑ | Boundary Gradient ↑ |
|---|---|---|
| Depth Anything V2 (raw) | 0.1332 | 0.2004 |
| **+ Ours (refined)** | **0.1762** | **0.2193** |
| **Improvement** | **+32%** | **+9.4%** |

- **Edge-IoU** — spatial alignment between depth-map edges and true subject silhouette (3px tolerance)
- **Boundary Gradient** — sharpness of the depth transition at the subject boundary

Improvement is consistent across all 9 test images (portraits, animals, objects).

---

## Before / After

*Left: raw Depth Anything V2 depth → naive bokeh with halo artifact. Right: refined depth (ours) → clean bokeh.*

<img width="2075" height="1482" alt="portrait_03_curly_hair_comparison" src="https://github.com/user-attachments/assets/998510d3-e991-4a65-90c2-1dfa50f7e459" />


> **Depth maps (top row):** raw depth has a gradual, bleeding transition at the subject silhouette; refined depth has a sharp, flat subject plane with a clean boundary.
> **Bokeh outputs (bottom row):** raw depth causes colour bleed and a soft halo at the hair and shoulders; refined depth produces a clean cutout with no fringe.

---

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2_repo
    python download_weights.py

For sharper edge-aware filtering, use `opencv-contrib-python` instead of `opencv-python`.

---

## Running on a Single Image

    # Auto-focus on the segmented subject
    python run.py --input assets/input/your_photo.jpg

    # Click-to-focus at pixel (x=540, y=720)
    python run.py --input assets/input/your_photo.jpg --focus 540 720

    # Hexagonal kernel, stronger blur
    python run.py --input assets/input/your_photo.jpg --kernel hexagonal --max-blur 35

    # Larger depth model
    python run.py --input assets/input/your_photo.jpg --depth-variant vitb

---

## Outputs

| Path | Description |
|------|-------------|
| `assets/output/<stem>_bokeh_refined.png` | **Final cinematic result** |
| `assets/output/<stem>_bokeh_baseline.png` | Naïve baseline — raw depth + Gaussian blur |
| `assets/debug/<stem>_depth_raw.png` | Depth Anything V2 output (colorized) |
| `assets/debug/<stem>_depth_refined.png` | Refined depth map (colorized) |
| `assets/debug/<stem>_mask_overlay.png` | Segmentation mask overlay |
| `assets/debug/<stem>_comparison.png` | 2×3 comparison grid |
| `assets/debug/<stem>_metrics.json` | Per-image quantitative metrics |

---

## Benchmark Evaluation

| Dataset | Images | Ground Truth | Metrics |
|---------|--------|--------------|---------|
| **Unsplash** (in-the-wild) | 9 photos | None needed | Edge-IoU, Boundary Gradient |
| **EBB!** (AIM 2020) | 10 pairs | Real f/1.8 photographs | PSNR, SSIM, LPIPS |
| **DIODE** | 10 scenes | FARO laser-scanner depth | AbsRel, RMSE, δ₁/δ₂/δ₃ |

    python data/scripts/download_all.py
    python evaluate_benchmark.py
    python evaluate_benchmark.py --only unsplash

Results written to `report/`. See `data/MANIFEST.md` for citations and licensing.

---

## Project Layout

    cinematic-bokeh/
    ├── src/
    │   ├── refiner.py           # ★ Novel: segmentation-guided depth refinement
    │   ├── bokeh.py             # Multi-layer bokeh — disk + hex kernels, premult alpha
    │   ├── pipeline.py          # End-to-end orchestration
    │   ├── depth_estimator.py   # Depth Anything V2 wrapper
    │   ├── segmenter.py         # MediaPipe Selfie Segmentation
    │   ├── evaluate.py          # Boundary-quality metrics (no GT needed)
    │   ├── metrics_depth.py     # DIODE depth metrics (AbsRel, RMSE, δ)
    │   ├── metrics_bokeh.py     # EBB! bokeh metrics (PSNR, SSIM, LPIPS)
    │   └── utils.py             # I/O and visualization helpers
    ├── assets/{input,output,debug}/
    ├── models/
    ├── notebooks/demo.ipynb
    ├── report/
    ├── run.py
    ├── evaluate_benchmark.py
    └── download_weights.py

---

## References

1. Yang et al. — *Depth Anything V2*, NeurIPS 2024
2. Ranftl et al. — *Vision Transformers for Dense Prediction (DPT)*, ICCV 2021
3. Ranftl et al. — *MiDaS*, TPAMI 2020
4. Ignatov, Patel, Timofte — *EBB!*, CVPRW 2020
5. Vasiljevic et al. — *DIODE*, arXiv 2019
