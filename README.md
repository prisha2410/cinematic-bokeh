# Semantically-Guided Depth Refinement for Cinematic Bokeh Synthesis

Single-image cinematic bokeh from a monocular RGB input.

**Pipeline:** Depth Anything V2 → MediaPipe Segmentation → Boundary Refinement *(ours)* → Physically-Based Bokeh

---

## Setup

```bash
# 1. Create and activate a Python environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Clone Depth Anything V2 (we import the model definition from it)
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2_repo

# 3. Download pretrained weights (~100 MB, small variant by default)
python download_weights.py
```

> For sharper edge-aware filtering, use `opencv-contrib-python` instead of `opencv-python`.

---

## Running on a Single Image

```bash
# Auto-focus on the segmented subject
python run.py --input assets/input/your_photo.jpg

# Click-to-focus at a specific pixel (x=540, y=720)
python run.py --input assets/input/your_photo.jpg --focus 540 720

# Stronger blur with a hexagonal kernel
python run.py --input assets/input/your_photo.jpg --kernel hexagonal --max-blur 35

# Use the larger 'vitb' depth model
python run.py --input assets/input/your_photo.jpg --depth-variant vitb
```

---

## Outputs

| Path | Description |
|------|-------------|
| `assets/output/<stem>_bokeh_refined.png` | **Final cinematic result** |
| `assets/output/<stem>_bokeh_baseline.png` | Naïve baseline — raw depth + Gaussian |
| `assets/debug/<stem>_depth_raw.png` | Depth Anything V2 output (colorized) |
| `assets/debug/<stem>_depth_refined.png` | Refined depth map (colorized) |
| `assets/debug/<stem>_mask_overlay.png` | Segmentation mask overlay |
| `assets/debug/<stem>_comparison.png` | 2×3 comparison grid for the report |
| `assets/debug/<stem>_metrics.json` | Per-image quantitative metrics |

---

## Benchmark Evaluation

Evaluated on 30 images across three public datasets:

| Dataset | Images | Ground Truth | Purpose |
|---------|--------|--------------|---------|
| **EBB!** (AIM 2020) | 10 pairs | Real f/1.8 photographs | Bokeh similarity (PSNR, SSIM, LPIPS) |
| **DIODE** | 10 scenes | FARO laser-scanner depth | Depth accuracy (AbsRel, RMSE, δ) |
| **Unsplash** | 10 photos | None | In-the-wild boundary quality |

See `data/MANIFEST.md` for full citations and licensing.

```bash
# Download all benchmark data
# (EBB! prints manual instructions — Google Drive has no bulk-download API)
python data/scripts/download_all.py

# Run the full benchmark — emits Markdown and LaTeX tables
python evaluate_benchmark.py

# Run a single dataset only
python evaluate_benchmark.py --only diode
```

Results are written to `report/`: `benchmark_table.md`, `benchmark_table.tex`, and `benchmark_results.json`.

---

## Project Layout

```
cinematic_bokeh/
├── src/
│   ├── depth_estimator.py   # Depth Anything V2 wrapper
│   ├── segmenter.py         # MediaPipe Selfie Segmentation
│   ├── refiner.py           # ★ Novel: segmentation-guided depth refinement
│   ├── bokeh.py             # Multi-layer bokeh — disk + hex kernels
│   ├── pipeline.py          # End-to-end orchestration
│   ├── evaluate.py          # Boundary-quality metrics (no GT needed)
│   ├── metrics_depth.py     # DIODE depth metrics (AbsRel, RMSE, δ)
│   ├── metrics_bokeh.py     # EBB! bokeh metrics (PSNR, SSIM, LPIPS)
│   └── utils.py             # I/O and visualization helpers
├── data/
│   ├── MANIFEST.md
│   ├── scripts/
│   │   ├── download_all.py
│   │   ├── download_ebb.py
│   │   ├── download_diode.py
│   │   └── download_unsplash.py
│   ├── ebb/
│   ├── diode/
│   └── unsplash/
├── assets/{input,output,debug}/
├── models/                  # Cached weights
├── notebooks/demo.ipynb
├── report/
├── run.py
├── evaluate_benchmark.py
├── verify.py                # Synthetic sanity check (no weights needed)
├── download_weights.py
└── requirements.txt
```
