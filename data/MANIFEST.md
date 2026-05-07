# Benchmark dataset

## Composition

We use a curated subset of three public datasets, totaling ~30 images:

| Dataset | Subset size | Role | License | Citation |
|---|---|---|---|---|
| **EBB!** (Everything's Better with Bokeh!) | 10 paired images | Bokeh quality (PSNR/SSIM/LPIPS vs. real wide-aperture ground truth) | CC BY-NC-SA 4.0 | Ignatov, Patel, Timofte (CVPRW 2020) |
| **DIODE** (val split, indoor + outdoor) | 10 images | Depth quality (RMSE, AbsRel, δ thresholds) on real ground truth | MIT | Vasiljevic et al. (arXiv 1908.00463) |
| **Unsplash Lite** | 10 portraits / object photos | In-the-wild aesthetic showcase + boundary-quality metrics | Unsplash License (free for any use) | Unsplash (2020+) |

Total: **30 images** chosen deliberately — small enough to inspect visually, large enough to make per-dataset metric averages meaningful.

## Why these three (and not others)

- **EBB!** is the only dataset that provides paired *(narrow-aperture sharp, wide-aperture real bokeh)* photos of the same scene. This lets us compute PSNR/SSIM/LPIPS between our synthesized bokeh and a real photographic bokeh — the strongest possible bokeh evaluation for a method like ours.
- **DIODE** provides dense, accurate ground-truth depth from a FARO laser scanner, with both indoor and outdoor scenes. This lets us measure the actual depth error on the Depth Anything V2 output AND on our refined depth, separating the contribution of refinement.
- **Unsplash Lite** photos are released under the Unsplash License, which permits free commercial and non-commercial use without attribution constraints — the cleanest license situation for a research demo. We use this for hero/aesthetic figures.

### What we deliberately did NOT use

- **FFHQ** — although widely cited, FFHQ has documented consent issues: the dataset persists images that have been removed from Flickr (~10%+ of images per Exposing.ai, 2023), and originals were collected without subject consent. For a 2026 academic project, we believe alternatives are preferable.
- **NYU Depth V2** — older sensor (Kinect v1), maxes out at ~10 m indoor only. DIODE supersedes it for our purposes.
- **Custom-photographed images** — we chose to keep the benchmark fully reproducible from public sources.

## How to obtain

```bash
# All three datasets in one go (downloads only the subsets we use):
python data/scripts/download_all.py
```

Or individually:

```bash
python data/scripts/download_ebb.py        # ~150 MB
python data/scripts/download_diode.py      # ~1.5 GB (val split only)
python data/scripts/download_unsplash.py   # ~50 MB (curated CC0 list)
```

After downloading, the layout will be:

data/
├── ebb/
│   ├── narrow_aperture/  *.jpg   # input (sharp, all-in-focus)
│   └── wide_aperture/    *.jpg   # ground truth (real bokeh)
├── diode/
│   ├── rgb/              *.png
│   ├── depth/            *.npy   # float32, meters
│   └── depth_mask/       *.npy   # validity, uint8
└── unsplash/
└── *.jpg

## Splits and reproducibility

We do not train anything, so there is no train/val/test split. The 30 images in our benchmark are listed explicitly in `data/benchmark_manifest.json` with their source URLs and SHA-256 hashes for reproducibility.

## Citations (BibTeX)

```bibtex
@inproceedings{ignatov2020rendering,
  title={Rendering natural camera bokeh effect with deep learning},
  author={Ignatov, Andrey and Patel, Jagruti and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={418--419},
  year={2020}
}

@article{vasiljevic2019diode,
  title={DIODE: A Dense Indoor and Outdoor DEpth Dataset},
  author={Vasiljevic, Igor and Kolkin, Nick and Zhang, Shanyi and Luo, Ruotian and Wang, Haochen and Dai, Falcon Z and Daniele, Andrea F and Mostajabi, Mohammadreza and Basart, Steven and Walter, Matthew R and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1908.00463},
  year={2019}
}

@misc{unsplash,
  author = {{Unsplash}},
  title = {Unsplash Lite Dataset},
  howpublished = {\url{https://unsplash.com/data}},
  note = {Unsplash License}
}
```