# How the three reference papers connect to your methodology

This is the scaffolding for the "Background" and "Method" sections of your report.
Each paper is mapped to the specific design decision it justifies, plus the exact
sentence-level claim you can make.

---

## Paper 1 — MiDaS (Ranftl et al., TPAMI 2020)

**Cite for:** Why monocular depth estimation generalizes across scenes at all,
and why the depth your pipeline consumes is *relative*, not metric.

**What to say in the report:**

> Monocular depth estimation became practical for in-the-wild images with MiDaS,
> which trained a single network on a mixture of complementary depth datasets
> using a scale-and-shift-invariant loss. This loss permits joint training across
> datasets with incompatible depth conventions (relative, disparity, metric) and
> is the reason modern monocular depth networks output *ordinal* or *affine-
> invariant* depth rather than metric distances. Our pipeline inherits this
> property: we never reason about absolute distances; instead we map the
> relative depth values into a focus-distance ordering for defocus rendering,
> which is sufficient for cinematic bokeh.

**Where it shows up in your code:**
- `src/depth_estimator.py` returns a normalized `[0, 1]` map.
- `src/bokeh.py:compute_coc` works on relative depth — the user's chosen
  focus is also expressed in `[0, 1]`.

**Limitation to acknowledge:** because depth is relative, the bokeh shape is
not strictly physically calibrated; it is *physically motivated* but
artistically tuned via `max_blur_px` and `falloff`.

---

## Paper 2 — DPT (Ranftl et al., ICCV 2021)

**Cite for:** The architectural backbone that explains *why* depth maps from
this family of models look smooth — and therefore why your refinement step
is needed.

**What to say in the report:**

> DPT (Dense Prediction Transformer) replaced the convolutional encoders of
> MiDaS with a Vision Transformer (ViT) backbone, reassembling token features
> from multiple transformer blocks into a multi-scale convolutional decoder.
> ViT tokens correspond to large image patches (typically 14×14 or 16×16
> pixels), and although the reassembly stage upsamples and fuses features,
> the effective receptive field at the decoder remains large. This produces
> globally consistent depth but smooths over high-frequency boundaries — a
> well-known limitation of dense ViT prediction that motivates our refinement
> step.

**Where it shows up in your code:**
- The blurry edges visible in `assets/debug/<stem>_depth_raw.png` are exactly
  this artifact — keep one raw-depth crop in the report as evidence.
- `src/refiner.py` is your fix.

**Bridge sentence to your contribution:**

> Whereas DPT's smoothness is a feature for global consistency, it becomes a
> liability for defocus rendering, which requires a sharp depth discontinuity
> at the subject silhouette. We address this with a post-hoc, segmentation-
> guided refinement step that does not require retraining the depth network.

---

## Paper 3 — Depth Anything V2 (Yang et al., NeurIPS 2024)

**Cite for:** Your specific depth backbone, and the most current state-of-the-art
context for the project.

**What to say in the report:**

> We use Depth Anything V2 as our monocular depth estimator. Building on the
> DPT architecture, Depth Anything V2 is trained on a large mixture of synthetic
> and pseudo-labeled real images, with synthetic data providing precise depth
> targets and unlabeled real images providing scene diversity through teacher-
> student distillation. The result is a family of ViT-based models (Small,
> Base, Large) that achieve state-of-the-art zero-shot generalization on
> in-the-wild monocular depth. We use the Small variant (≈25M parameters,
> ViT-S backbone, 518×518 inputs) on an RTX 3050 (4 GB VRAM) using fp16
> inference.

**Where it shows up in your code:**
- `src/depth_estimator.py` — instantiated with `variant="vits"` and `fp16`.
- `download_weights.py` — pulls the official Hugging Face checkpoint.

**What you keep, what you change:**

> We use Depth Anything V2 unchanged (no fine-tuning) and treat it as a black-
> box estimator of dense relative depth. Our contribution sits *downstream* of
> it: a segmentation-guided refinement step that targets the residual edge
> bleeding inherent to dense ViT prediction, and a multi-layer bokeh renderer
> that exploits the resulting sharp boundaries.

---

## Putting it together — the one-paragraph related work bridge

> Monocular depth estimation has progressed from convolutional, multi-dataset
> training (MiDaS, Ranftl et al. 2020) to transformer-based dense prediction
> (DPT, Ranftl et al. 2021), and most recently to large-scale synthetic-plus-
> pseudo-labeled training (Depth Anything V2, Yang et al. 2024). All three
> share a scale-and-shift-invariant training regime and a ViT- or CNN-based
> dense decoder, which yields globally consistent but locally smooth depth
> maps. This smoothness — a known property of dense ViT prediction — is
> incompatible with the sharp depth discontinuities required by defocus
> rendering at object silhouettes. We therefore propose a downstream,
> segmentation-guided depth refinement step that produces clean discontinuities
> at the subject boundary without retraining the depth network, and feed the
> refined depth into a physically-motivated multi-layer bokeh renderer.

---

## What is novel 

1. **The combination.** Depth Anything V2's monocular depth + MediaPipe's
   subject mask + a defocus renderer is, to our knowledge, not a published
   pipeline — and the specific failure mode (depth bleed at hair/silhouette
   ruining cinematic bokeh) has not been targeted with semantic guidance.

2. **The three-stage refiner** in `src/refiner.py`:
   - Subject-side flattening (treats the subject as one focal plane),
   - Background-side hole-filling around the bleed ring (kills haloing),
   - Edge-aware joint smoothing using the RGB image as guide
     (re-imposes hair-strand-level edges).

3. **The premultiplied-alpha multi-layer renderer** in `src/bokeh.py`,
   which is the technique that makes the visual difference dramatic enough
   to be obvious in figures — most amateur bokeh implementations skip it
   and end up with halos.

4. **A targeted evaluation protocol** (`src/evaluate.py`) that scores
   *boundary quality* specifically — the exact thing your refinement is
   designed to improve — rather than only generic depth error.