# Tips for getting beautiful (report-worthy) bokeh outputs

## Choose input photos that show off the technique

Pick photos where the failure mode is visually obvious and the fix is dramatic:

- **Hair against a busy background.** This is THE hero case. Wispy hair is exactly
  what monocular depth networks blur over, and the refinement makes the hair
  re-emerge cleanly. A portrait against trees or city lights is perfect.
- **Subject with point-light highlights behind it.** Christmas lights, distant
  street lamps, sun-through-leaves bokeh — the highlight boost in the
  renderer turns these into distinct hexagonal/circular bokeh "balls".
  *This is the single biggest aesthetic win.*
- **Strong foreground/background separation.** A subject 1–2 m from the
  camera with a background 5+ m away gives the depth network the easiest
  signal to work with.

Avoid:
- Flat-lit cluttered scenes with no depth cues — the depth map will be noisy.
- Backgrounds with lots of text or fine pattern — bokeh on text looks "wrong"
  to viewers because they expect to read it.

## Tune the rendering knobs deliberately

The defaults are tuned for portraits. For specific looks:

| Look | `kernel_shape` | `max_blur_px` | `falloff` | `highlight_boost` |
|---|---|---|---|---|
| Modern f/1.4 portrait | `circle` | 25–35 | 1.2 | 1.6 |
| Anamorphic / cinematic | `hexagonal` | 30–45 | 1.4 | 1.8 |
| Subtle "phone portrait mode" | `circle` | 12–18 | 1.0 | 1.2 |
| Dreamy / heavy mist | `circle` | 40–60 | 0.8 | 1.4 |

`falloff > 1` keeps more of the subject in focus and slams the background
out of focus quickly — feels more "cinematic". `falloff < 1` gives a gentle
gradient like a longer focal-length lens stopped slightly down.

## Post-processing that costs nothing but looks expensive

Apply these *after* `render_bokeh` to push the result over the line:

1. **Soft contrast S-curve** on the output luminance (slight). Real lenses
   roll off contrast in the bokeh — a gentle S-curve mimics this.
2. **Color grading: cool shadows, warm highlights** (the "teal-and-orange"
   cinematic look). One-line `cv2.LUT` or even just per-channel offsets work.
3. **A whisper of film grain** (Gaussian noise σ ≈ 2–3 in 8-bit) over the
   final image. This breaks up the smoothness of the bokeh and makes it
   read as "photographed" rather than "computed".
4. **Slight vignette.** A radial darkening of ≈10–15% at the corners makes
   the eye go to the in-focus subject.

I deliberately did *not* bake these into `bokeh.py` because they're subjective
and you may want different grades for different report figures. Add them in
your demo notebook so reviewers see the "before grading" technical result
and the "graded for the cover" hero shot side by side.

## Composing the report figures

For the headline figure, do **not** show only the final bokeh image. Show:

1. **Input** (top-left)
2. **Raw depth (Depth Anything V2)** with arrows pointing at bleed regions
3. **Refined depth (ours)** at the same locations — the arrows now point at
   sharp transitions
4. **Naive Gaussian bokeh on raw depth** with circles around halo artifacts
5. **Multi-layer hex bokeh on refined depth** — clean
6. (Optional) zoomed crop of the hair/edge area showing the difference at 100%

This 6-panel figure is what `make_comparison_grid` produces (with 5 of the 6
panels). For the report, drop the figure into the document and add the
arrow/circle annotations in your slides software (PowerPoint, Keynote, or
just Inkscape).

## Two more easy wins

- **Use the `vitb` variant** if you have time. The Small variant has slightly
  noisier depth in flat regions; Base is visibly cleaner and still fits in
  4 GB. The refinement helps both, but the absolute quality is higher.
- **Run on 1080p, not 4K.** Going higher than ~1080p on the long edge gives
  you minimal visible improvement in bokeh quality but quadruples runtime.
  If your input is huge, downsample to 1920px on the long side first.