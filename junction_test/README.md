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

# stage-swap ablation (JUNCTION_ABLATION.md) -> ablation_results.csv
.venv/Scripts/python.exe -m junction_test.ablation [--quick]

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
`JUNCTION_ABLATION.md` §5 additions: `junction_type_accuracy` (+
`junction_type_confusion`), `x_passthrough_accuracy`,
`coincident_unrelated_false_merge`, `width_mae`, `dash_style_accuracy`,
`arc_radius_rel_err`, `curve_misfit_count`, `reconstruction_ssim`, `runtime_s`.
Ground truth comes from `synthetic.generate` (exact) or, for real PDFs, is
visual-only (IoU/coverage vs the input ink mask).

## Synthetic difficulty knobs (`SYNTHETIC_DATA.md`)

`synthetic.generate` keyword-only args, all default off so `smoke.py` is
unchanged: `weight_ladder` (AEC mm ladder → px at `dpi`), `color_layers`
(RGB layers in GT, luma-flattened into the raster), `dash_styles`
(dashed/hidden/center/phantom + real `dash_array`), `forced_crossings` (true X
pass-through), `coincident_unrelated` (grazing endpoint, `is_true_connection=False`),
`curved_walls` (large-radius arc + tangent stub), `residue_level` (white
rectangular holes punched through strokes + glyph-fragment speckle). GT junctions
carry `jtype` (L/T/X/Y/star/endpoint/coincident_unrelated), `arm_angles`, `members`.

## Ablation (`ablation.py`, `JUNCTION_ABLATION.md`)

Baseline vs one-stage swaps — `S2_medial_axis`, `S3_junction_repair` (erase disk
at deg≥3 nodes), `S8_lsd` (OpenCV LSD on the raster), `S9_hough`, `F1_rdp` —
scored over archetype × overlap-density × clean/noisy buckets → `ablation_results.csv`.
`explore.ipynb` §7 has the interactive view.

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
