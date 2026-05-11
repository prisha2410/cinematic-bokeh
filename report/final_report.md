# Semantically-Guided Depth Refinement for Cinematic Bokeh Synthesis from Single Monocular Images

**Author:** Prisha Khalasi | **Date:** April 2026

---

## Abstract

Monocular depth estimators such as Depth Anything V2 produce globally
consistent depth maps but suffer from boundary bleeding at subject
silhouettes — a known limitation of dense Vision Transformer prediction.
When such depth maps drive defocus rendering, this bleeding causes
visible halos and incorrect blur at subject edges, defeating the
photographic illusion. We propose a lightweight, training-free pipeline
that fuses Depth Anything V2 with MediaPipe subject segmentation to
produce edge-accurate depth maps, then renders physically-based cinematic
bokeh using a multi-layer hexagonal aperture kernel. Our segmentation-
guided depth boundary refinement module improves Edge-IoU by 32% and
boundary gradient sharpness by 9.4% over the raw depth baseline on a
9-image in-the-wild benchmark, with visually compelling bokeh results
on portrait, animal, and object subjects.

---

## 1. Introduction

Cinematic bokeh — the aesthetic background blur produced by large-aperture
lenses — is one of the most sought-after photographic effects. Modern
smartphones approximate it using dual cameras or learned networks, but
single-image methods on consumer hardware remain limited. The core
challenge is accurate depth: to blur the background correctly, we need
a sharp depth discontinuity at the subject silhouette. Monocular depth
networks produce smooth, globally consistent depth but consistently fail
at fine boundaries such as hair, fingers, and clothing edges.

This project addresses that failure mode directly. We use Depth Anything
V2 as a strong depth prior, MediaPipe Selfie Segmentation to obtain a
sharp subject silhouette, and a novel three-stage refinement module to
snap the depth discontinuity to the true subject boundary. The refined
depth then drives a multi-layer bokeh renderer with physically-motivated
circle-of-confusion scaling and highlight pre-emphasis, producing
visually compelling results on an RTX 3050 (4 GB VRAM) with no training.

---

## 2. Related Work

Monocular depth estimation became practical for in-the-wild images with
MiDaS (Ranftl et al., 2020), which trained a single network on a mixture
of complementary depth datasets using a scale-and-shift-invariant loss.
This loss permits joint training across datasets with incompatible depth
conventions and is why modern monocular depth networks output relative
rather than metric depth. DPT (Ranftl et al., 2021) replaced MiDaS's
convolutional encoder with a Vision Transformer backbone, reassembling
token features from multiple transformer blocks into a dense prediction.
ViT tokens correspond to large image patches, and although the reassembly
stage upsamples features, the effective receptive field remains large —
producing globally consistent but boundary-smooth depth maps. Depth
Anything V2 (Yang et al., 2024) builds on the DPT architecture and
achieves state-of-the-art zero-shot generalization through large-scale
synthetic and pseudo-labeled training. Its authors explicitly note
residual boundary artifacts as a limitation, which our refinement module
directly addresses.

---

## 3. Method

### 3.1 Depth Estimation

We use Depth Anything V2 (ViT-Small variant, ~25M parameters) to estimate
a relative inverse-depth map from the input RGB image. The model runs at
518×518 input resolution in float32 on the RTX 3050, consuming
approximately 1.2 GB VRAM. The output is normalized to [0, 1] where
larger values indicate closer surfaces.

### 3.2 Subject Segmentation

MediaPipe Selfie Segmentation (model_selection=1) produces a soft [0,1]
subject probability mask at full image resolution, running entirely on
CPU and consuming no VRAM. The soft mask is sharpened with a threshold
of 0.5 and a 2-pixel Gaussian feather, then used as input to the
refinement module.

### 3.3 Segmentation-Guided Depth Refinement (Novel Contribution)

The refinement module operates in three stages:

**Stage 1 — Subject flattening.** The median depth value within the
subject mask replaces the raw depth inside the mask region. This treats
the subject as a single focal plane, which is appropriate for cinematic
bokeh and eliminates intra-subject depth noise that would cause
incorrect partial blur on the in-focus subject.

**Stage 2 — Bleed-ring inpainting.** A dilation of the subject mask
by 8 pixels defines a ring of background pixels likely contaminated by
depth bleeding from the subject. These pixels are inpainted using
OpenCV's TELEA algorithm from the surrounding clean background depth,
eliminating the soft halo region that causes edge artifacts in the bokeh.

**Stage 3 — Edge-aware joint smoothing.** A guided filter with the
RGB image as guide (radius=9, epsilon=0.0064) smooths the depth map
while preserving edges aligned with image structure (hair strands,
fabric edges). This re-imposes fine boundary detail that was lost in the
raw depth prediction.

### 3.4 Cinematic Bokeh Rendering

The refined depth drives a circle-of-confusion model:

    CoC(x,y) = max_blur * |D(x,y) - D_focus|^falloff

where max_blur=25px and falloff=1.2. We discretize the depth range into
7 CoC layers and render each with a hexagonal aperture kernel, compositing
back-to-front using premultiplied alpha to prevent background color
bleeding into the in-focus subject. Bright pixels (luminance > 0.85)
are pre-emphasized by a factor of 1.6 before blurring, producing
distinct bokeh balls from point-light highlights.

---

## 4. Evaluation

### 4.1 Dataset

We evaluate on 9 in-the-wild photographs from Unsplash (Unsplash License,
free for any use), covering portraits, animals, and objects. No ground-
truth depth is available for this set; we report boundary-quality proxy
metrics that directly measure the refinement's intended effect.

### 4.2 Metrics

**Edge-IoU** measures the spatial alignment between depth-map Canny edges
and segmentation-mask edges, with a 3-pixel tolerance. Higher values
indicate that depth discontinuities are better aligned with true subject
silhouettes.

**Boundary Gradient** measures the mean gradient magnitude of the depth
map in a thin band around the subject silhouette. Higher values indicate
a sharper depth transition (less bleeding).

### 4.3 Results

| Method | Edge-IoU (higher=better) | Boundary Grad (higher=better) |
|---|---|---|
| Depth Anything V2 (raw) | 0.1332 | 0.2004 |
| + ours (refined) | **0.1762** | **0.2193** |

Our refinement improves Edge-IoU by **32%** and boundary gradient by
**9.4%** over the raw Depth Anything V2 baseline. Both metrics improve
consistently across all 9 images in the benchmark.

### 4.4 Qualitative Results

Visual inspection of the before/after comparison grids (see assets/debug/)
confirms the quantitative findings. The portrait results (particularly
portrait_03_curly_hair and portrait_06_glasses) show clean subject
silhouettes with no visible halo, while the baseline produces soft
transitions and edge bleeding. The bokeh renderer produces distinct
hexagonal bokeh balls from background highlights — the highlight
pre-emphasis is the single largest contributor to the cinematic quality
of the output.

---

## 5. Limitations

**Segmentation dependency.** The pipeline requires a working subject
segmentation. MediaPipe Selfie Segmentation is optimized for people;
animal and object results are generally good but occasionally produce
incomplete masks.

**Relative depth.** Depth Anything V2 outputs relative, not metric,
depth. The bokeh effect is therefore artistically calibrated rather than
physically exact — the focus distance and blur radius are tunable
parameters rather than derived from a known focal length and aperture.

**Single focal plane.** The subject-flattening stage assumes the subject
occupies one depth. Scenes with multiple subjects at different distances
are not handled correctly.

---

## 6. Conclusion

We presented a lightweight, training-free pipeline for cinematic bokeh
synthesis from single monocular images. Our segmentation-guided depth
boundary refinement module addresses the known edge-bleeding limitation
of dense ViT-based depth estimators, improving Edge-IoU by 32% and
boundary gradient sharpness by 9.4% on an in-the-wild benchmark.
Combined with a multi-layer hexagonal bokeh renderer with highlight
pre-emphasis, the system produces visually compelling results within a
4 GB VRAM budget.

---

## References

1. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers
   for Dense Prediction. ICCV 2021.

2. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V.
   (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets
   for Zero-Shot Cross-Dataset Transfer. TPAMI 2020.

3. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H.
   (2024). Depth Anything V2. NeurIPS 2024.

4. Vasiljevic, I., et al. (2019). DIODE: A Dense Indoor and Outdoor
   DEpth Dataset. arXiv:1908.00463.

5. Ignatov, A., Patel, J., & Timofte, R. (2020). Rendering Natural
   Camera Bokeh Effect with Deep Learning. CVPRW 2020.
