# junction_test — classical raster → vector pipeline spike

Standalone verification of the **classical** vectorization pipeline for architectural
line drawings, following

> P. Dosch, K. Tombre, C. Ah-Soon, G. Masini,
> *"A complete system for the analysis of architectural drawings"*,
> IJDAR 3(2):102–116, 2000. (`inria-00099391`)

Sections 2–3 only (image processing + 2D feature extraction). **No 3D.** No CNN
(the old `junction_cnn/` experiment is dropped). Imports nothing from `rastervec/`.

Goal: measure whether this is accurate enough to build a real `rastervec` raster
stage on, and find which thresholds transfer.

## Install

```
.venv/Scripts/python.exe -m pip install -r junction_test/requirements.txt
```

(`opencv-python` and `pymupdf` are already in the repo `.venv`.)

## Run

```
# fast, no notebook: 5 synthetic drawings + metrics + PASS/FAIL
.venv/Scripts/python.exe -m junction_test.smoke

# full walk-through with visualisations
.venv/Scripts/python.exe -m jupyter notebook junction_test/explore.ipynb
#  or headless:
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace junction_test/explore.ipynb
```

## Pipeline (`pipeline.py::run`)

| # | function | Dosch 2000 | notes |
|---|---|---|---|
| 1 | `binarize` | §2.2 | Otsu ∪ soft threshold |
| 2 | `separate_text_graphics` | §2.2 | Fletcher & Kasturi CC size/shape |
| 2b | `reclaim_dashed_from_text` | §3.1 note | dashes wrongly binned as text → back to graphics |
| 3 | `thick_thin` | §2.2 | morphological geodesic reconstruction |
| 4 | `skeleton_and_dt` | §2.3 | `skimage.medial_axis` / `skeletonize` + distance transform |
| 5 | `skeleton_graph.build_graph` | §2.3 | node/chain graph, barb pruning |
| 6 | `polyapprox.approximate_*` | §2.3 | Rosin & West split-and-merge (default) or RDP |
| 7 | `polyapprox.detect_arcs` | §2.4 | consecutive-segment hypotheses + Taubin circle fit |
| 8 | `dashed.detect` | §3.1 | Dov Dori virtual-line grouping of short segments |
| 9 | `measure_width` | — | 2·median(distance transform along chain) |
| 10 | `regularize` | §2.3/2.5 | endpoint snap, collinear merge, junctions |
| 11 | `extract_remainder` | — | ink minus rendered geometry |
| 12 | `staircase.detect` | §3.2 | crude 5..30-tread run-regularity heuristic (dashed-line detector rotated 90°); metrics informational only |
| 13 | `symbols.recognize` | §3.3 | constraint-propagation network, 2/7 families (door, window); metrics informational only |

Out of scope for the spike (noted in the `rastervec` plan): tiling/merge (§2.1),
3D modeling and floor-to-floor matching (§4), post-vectorization junction-position
recompute.

## Metrics (`metrics.py`)

`mask_iou`, `coverage_pct`, `endpoint_hausdorff`, `junction_precision/recall/f1`,
`dashed_precision/recall`, `arc_count_error`, `segment_count_ratio`,
`staircase_precision/recall/f1`, `staircase_tread_count_err`,
`symbol_door_precision/recall/f1`, `symbol_window_precision/recall/f1`.
Ground truth comes from `synthetic.generate` (exact) or, for real PDFs, is
visual-only (IoU/coverage vs the input ink mask).

## Reading the results

- **Works well**: overall ink coverage, junction recall, straight-line geometry,
  arc/circle detection on clean isolated drawings.
- **Known-weak** (matches the paper — it uses interactive correction here):
  - dashed-line recovery is fragile, especially where dashes cross other lines;
  - Rosin & West over-splits polylines around junctions
    (`segment_count_ratio` > 2), inflating junction *precision* error;
  - dense full shop-drawing pages (lots of text / tables / hatching) degrade
    badly — the paper works on single clean drawings + tiling + a human in the loop;
  - staircase/symbol metrics are informational only, not part of the smoke gate
    (`smoke.py`): the staircase heuristic is a from-scratch stand-in for the
    paper's own (unavailable) Sánchez texture segmenter, and it itself calls its
    filter "crude and simplistic"; symbol recognition covers only 2 of the
    paper's 7 families (door, window), validated on synthetic data only —
    window recall in particular is fragile, since a jamb tick that crosses a
    wall gets split at the skeleton-graph junction, same junction-imprecision
    issue the paper documents in §2.5.
- `Params.max_work_px` downscales large inputs (the skeleton graph is pure-Python
  O(pixels)); real embedded rasters in `references/*.pdf` are small, whole-page
  renders need the cap.
