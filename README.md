# Semantically-Guided Depth Refinement for Cinematic Bokeh Synthesis

Single-image cinematic bokeh from a monocular RGB input.
Pipeline: **Depth Anything V2 → MediaPipe segmentation → boundary refinement (ours) → physically-based bokeh**.

Hardware target: NVIDIA RTX 3050 4 GB (Intel i7-12650H, 16 GB RAM).

## Setup

```bash
# 1) Python env
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Clone Depth Anything V2 source (we import the model definition from it)
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2_repo

# 3) Download pretrained weights (small variant by default, ~100MB)
python download_weights.py
```

Optional (for sharper edge-aware filtering): `pip install opencv-contrib-python` instead of `opencv-python`.

## Run on one image

```bash
# Auto-focus on the segmented subject:
python run.py --input assets/input/your_photo.jpg

# Click-to-focus at pixel (x=540, y=720):
python run.py --input assets/input/your_photo.jpg --focus 540 720

# Stronger blur, hexagonal kernel:
python run.py --input assets/input/your_photo.jpg --kernel hexagonal --max-blur 35

# Use the larger 'base' depth model (still fits 4GB):
python run.py --input assets/input/your_photo.jpg --depth-variant vitb
```

Outputs:

- `assets/output/<stem>_bokeh_refined.png`  — the final cinematic result
- `assets/output/<stem>_bokeh_baseline.png` — naive baseline (raw depth + Gaussian)
- `assets/debug/<stem>_depth_raw.png`       — Depth Anything V2 output (colorized)
- `assets/debug/<stem>_depth_refined.png`   — refined depth (colorized)
- `assets/debug/<stem>_mask_overlay.png`    — segmentation overlay
- `assets/debug/<stem>_comparison.png`      — 2×3 grid for the report
- `assets/debug/<stem>_metrics.json`        — quantitative metrics

## Benchmark evaluation (for the report)

We evaluate on 30 images from three public datasets:

- **EBB!** (10 paired photos, AIM 2020) — bokeh similarity vs. real f/1.8 ground truth.
- **DIODE** (10 scenes, indoor + outdoor) — depth accuracy vs. FARO laser-scanner ground truth.
- **Unsplash** (10 photos) — in-the-wild boundary quality, no GT needed.

See `data/MANIFEST.md` for full citations and licensing.

```bash
# Download benchmark data (Unsplash and DIODE auto-download; EBB! prints
# manual instructions because Google Drive doesn't support unauthenticated bulk DLs):
python data/scripts/download_all.py

# Run the full benchmark and emit Markdown + LaTeX tables:
python evaluate_benchmark.py

# Or just one dataset:
python evaluate_benchmark.py --only diode
```

Outputs land in `report/`:
- `benchmark_table.md`  — Markdown table for the README/notebook
- `benchmark_table.tex` — LaTeX table for the report
- `benchmark_results.json` — raw per-image numbers

## Project layout
cinematic_bokeh/
├── src/
│   ├── depth_estimator.py   # Depth Anything V2 wrapper
│   ├── segmenter.py         # MediaPipe Selfie Segmentation
│   ├── refiner.py           # ★ Novel: segmentation-guided depth refinement
│   ├── bokeh.py             # Multi-layer bokeh w/ disk + hex kernels
│   ├── pipeline.py          # End-to-end orchestration
│   ├── evaluate.py          # Boundary-quality metrics (no GT needed)
│   ├── metrics_depth.py     # Depth metrics for DIODE (AbsRel, RMSE, δ)
│   ├── metrics_bokeh.py     # Bokeh metrics for EBB! (PSNR, SSIM, LPIPS)
│   └── utils.py             # I/O & visualization
├── data/
│   ├── MANIFEST.md          # Datasets, citations, licenses
│   ├── scripts/
│   │   ├── download_all.py
│   │   ├── download_ebb.py
│   │   ├── download_diode.py
│   │   └── download_unsplash.py
│   ├── ebb/                 # 10 EBB! pairs (downloaded)
│   ├── diode/               # 10 DIODE scenes w/ GT depth (downloaded)
│   └── unsplash/            # 10 Unsplash photos (downloaded)
├── assets/{input,output,debug}/
├── models/                  # Cached weights
├── notebooks/demo.ipynb     # Interactive playground
├── report/                  # Paper-mapping notes, figures, benchmark tables
├── run.py                   # Single-image CLI
├── evaluate_benchmark.py    # Full benchmark runner
├── verify.py                # Synthetic sanity check (no weights needed)
├── download_weights.py
└── requirements.txt

## Notes for the 4 GB VRAM budget

- Default `vits` model uses ~0.7 GB; `vitb` uses ~1.8 GB. Both fit comfortably.
- The model is loaded in `fp16`, so a 1080p image at `input_size=518` peaks well under 2 GB.
- MediaPipe runs on CPU (~0 GB VRAM), so the segmentation step is "free".
- The bokeh renderer is pure NumPy/OpenCV — no GPU needed.