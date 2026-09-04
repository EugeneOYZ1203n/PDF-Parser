# Junction & line-detection ablation

## Goal

Given a cleaned raster of AEC line work (**text already removed**, possibly leaving holes /
fragments), find the technique — or hybrid — that best recovers vector primitives
(line / arc / Bézier / rect) with **colour, stroke width, dash style** preserved, through
**many overlapping lines**, across the archetypes in `SYNTHETIC_DATA.md`. Output feeds the
real `rastervec` raster stage.

The current `pipeline.py::run` is ONE point in this space:

```
Otsu ∪ soft-threshold binarize
  -> Fletcher & Kasturi text/graphics split (+ reclaim dashes wrongly binned as text)
  -> morphological thick/thin (geodesic reconstruction)
  -> skimage skeletonize + distance transform
  -> skeleton_graph.build_graph  (degree map, barb pruning)
  -> Rosin-West polygonal approximation
  -> Taubin circle fit for arcs
  -> Dov Dori virtual-line grouping for dashed lines
  -> regularize (endpoint snap, collinear merge)
  -> junctions from >=2 non-collinear segment ends meeting at a snapped point
```

## §1 Background (condensed)

Two families:

- **Contour tracing** (Potrace / AutoTrace) — excellent curve fitting, but a line becomes a
  thin *filled rect*, there is no connectivity, and overlapping strokes merge into one blob.
- **Skeleton / centerline** (thinning + neighbourhood graph — what the spike does) — gives
  topology + width from the distance transform, but **distorts at junctions and
  endpoints**: barbs/spurs, crossings shifted or merged, X collapsed to a single node.

The overlapping-lines problem = **junction disambiguation**: at a crossing, decide X (two
strokes pass through) vs T (one ends on another) vs Y vs touch-but-unrelated. Levers:

- **opposite-arm angle pairing** — arms ≈180° apart belong to one stroke passing through;
- **tangent + curvature continuity** — follows a curved wall straight through a crossing;
- **width / colour / dash-style continuity** — the same stroke keeps its attributes;
- **global optimisation** — ILP (Liu, *Raster-to-Vector: Revisiting Floorplan
  Transformation*, ICCV 2017, ~90% precision/recall), MRF/CRF, min-cost flow over primitive
  hypotheses;
- **direction / frame fields** — PolyVector (Bessmeltsev & Solomon, SIGGRAPH 2019): two
  directions per pixel, so crossings are represented natively; trace streamlines.

Curves: polyline → smooth → fit straight, else circular arc (chord + sagitta, or Taubin),
else cubic Bézier (Schneider least-squares, as in Potrace); extend an arc through junctions
while tangent + curvature stay continuous (TIF2VEC; "stepwise recovery of arc segmentation").

Learned options: L-CNN / **HAWP** wireframe parsers (a `hawp/` checkout is already in the
repo), DeepFlux skeletonisation, *Deep Vectorization of Technical Drawings* (Egiazarian et
al., ECCV 2020, arXiv 2003.05471 — CNN cleaning → transformer estimates primitives incl.
quadratic Béziers with a width parameter → optimisation merges patch primitives).

Key references: Dosch, Tombre, Ah-Soon, Masini, *A complete system for the analysis of
architectural drawings*, IJDAR 3(2), 2000; Tombre, *Vectorization in Graphics Recognition:
To Thin or not to Thin*, ICPR 2000; Dori & Liu, *Sparse Pixel Vectorization*, PAMI 1999;
Chai & Dori, *Orthogonal Zig-Zag*; Hilaire & Tombre, *Robust and precise circular arc
detection*; Xia, Delon, Gousseau, *a-contrario junction detection*; Bessmeltsev & Solomon,
*Vectorization of Line Drawings via PolyVector Fields*.

## §2 Candidate methods (swap one stage at a time from the baseline, then test full combos)

Input to every method: a cleaned binary **per colour layer** (colour separation runs first;
results merged). Output: a primitive list → **identical post-processing (§4)** → metrics
(§5).

### Centerline / primitive extraction (S)
- **S1** skimage `skeletonize` (Zhang-Suen / Lee) + degree-map graph — *current baseline*.
- **S2** `medial_axis` + distance transform (subpixel width).
- **S3** S1 + **junction repair**: erase a disk (r ≈ local stroke width) around every
  deg≥3 node, re-pair the stubs by direction, reconnect. ("Thin, then fix junctions.")
- **S4** sparse-pixel vectorization (Dori & Liu): interval sampling + run-following medial
  tracking, direction-driven — designed to minimise junction distortion.
- **S5** Orthogonal Zig-Zag / contour-pair tracking (Chai & Dori) — walk between a stroke's
  two contours; width is free, junctions handled by zig-zag continuation.
- **S6** Delaunay / chordal-axis of contour points (Zou & Yan) — robust medial line + width.
- **S7** Potrace filled-outline polygons → chordal / medial axis → centerline + width.
- **S8** LSD / EDLines straight-segment detector **directly on the raster** (skip the
  skeleton for straights) + an arc / Bézier pass on the residual.
- **S9** Progressive Probabilistic Hough lines + Hough circles.
- **S10** structure-tensor / PolyVector **cross-field** + streamline tracing (overlaps
  resolved by the field).
- **S11** learned: **HAWP / L-CNN** wireframe parser (repo present); DeepFlux skeleton;
  Deep-Vectorization transformer (pretrained, then fine-tuned on `synthetic_aec/train`).

### Junction detection / classification (J)
- **J1** skeleton degree map — baseline, implicit in S1.
- **J2** morphological hit-or-miss junction templates on the skeleton.
- **J3** a-contrario junction detector (Xia ACJ) / Harris–Shi-Tomasi on the skeleton.
- **J4** **ring probe**: sample a circle of radius r around each candidate, cluster ink into
  angular sectors → arm count + arm angles (the design in `rastervec`'s `JunctionDetector`
  stub — `probe_directions`).
- **J5** CNN junction heatmap + per-arm direction regression (L-CNN / HAWP / Liu 2017),
  trained on `synthetic_aec` patches — the new GT junction types make this trainable.

### Overlap resolution through a junction (R)
- **R1** opposite-arm pairing by angle (Δ ≈ 180° ± tol).
- **R2** minimum-turning / tangent + curvature continuity (best for a curved wall crossing
  dimension lines).
- **R3** stroke-width continuity constraint.
- **R4** colour + dash-style continuity constraint.
- **R5** global optimisation over primitive hypotheses — ILP (Liu 2017), MRF / CRF, or
  min-cost flow — enforcing junction consistency + AEC priors (parallel / perpendicular /
  tangent).
- **R6** PolyVector streamline (resolution implicit in S10).
- **R7** learned line-verification / relation head (HAWP).

### Fitting (F)
- **F1** RDP polyline.
- **F2** Rosin–West split-and-merge — *current*.
- **F3** Rosin–West parameter-free (split to zero deviation, then merge back by
  significance).
- **F4** arc: Taubin (*current*) vs Kåsa vs RANSAC-circle vs sagitta / chord; arc extension
  through junctions on continuity.
- **F5** cubic **Bézier** least-squares (Schneider) for freeform curved walls / contours /
  roads.
- **F6** clothoid / Catmull-Rom spline for smooth site geometry.
- **F7** per-chain joint segment + arc + Bézier fit by minimum description length.

### Attribute recovery (W / D / C)
- **W1** width = 2·median(distance transform along the chain) — *current*.
- **W2** growing-circle probe (the `rastervec` `measure_width` stub).
- **W3** from the thick/thin reconstruction mask.
- **W4** per-segment (varying) width rather than one value per chain.
- **D1** Dov Dori virtual-line grouping of short colinear segments — *current*.
- **D2** coverage autocorrelation / run-length along a fitted host line.
- **D3** dash-**style** classification (solid / dashed / hidden / center / phantom) from the
  gap histogram.
- **D4** recover the exact `dash_array`.
- **C1** colour: HSV / Lab threshold clustering into an unknown number of layers; run
  S / J / R / F per layer; merge. Score layer-assignment accuracy.

### Text-removal residue (T)
- **T1** directional morphological closing before skeletonisation (bridge punched holes).
- **T2** inpainting (Telea / Navier–Stokes / learned) of the rectangular holes.
- **T3** endpoint gap-jumping by direction during graph build (Dosch "trace-through").
- **T4** post-skeleton glyph-fragment rejection by CC shape / isolation.

## §3 Ablation axes (buckets from `synthetic_aec/test`)

overlap density (low / med / high) · curvature (straight / circular / Bézier) · line-weight
spread · dash fraction & style mix · colour-layer count · residue level · noise & break
severity · DPI (150 / 300 / 600) · aspect ratio (square vs section) · near-parallel gap
(1–8 px) · hatching present.

## §4 Uniform post-processing (identical for every method)

endpoint snap (KD-tree, τ) → collinear / cocircular merge across deg-2 nodes → arc / Bézier
fit (residual-thresholded) → width per §W → dash style + `dash_array` per §D →
near-closed-loop → rect / polygon → drop `symbol_stroke`-region primitives via the
`extract_remainder` mask.

## §5 Metrics (extend `metrics.py`)

Keep: `mask_iou`, `coverage_pct`, `endpoint_hausdorff`, `junction_precision/recall/f1`,
`dashed_precision/recall`, `arc_count_error`, `segment_count_ratio`.

Add:
- **junction-type confusion** (L / T / X / Y / star / endpoint) and **X-junction
  pass-through accuracy** (at a GT X, fraction traced as two continuous strokes, not four
  stubs).
- **`coincident_unrelated` false-merge rate** (fraction wrongly joined).
- **width MAE** (per matched primitive) and width-classification accuracy vs the weight
  ladder.
- **dash-style F1** and **`dash_array` error** (on / off length relative error).
- **arc radius relative error**; **curve-misfit count** (a GT curve emitted as ≥3 straight
  segments); **Bézier Fréchet distance**.
- **colour-layer accuracy**.
- **per-role recall** (curved_wall, dimension_line, stair_tread, contour, …).
- **reconstruction SSIM / IoU** (render primitives vs input ink).
- **runtime**, **peak memory** (must survive a 600-DPI section — cf. `Params.max_work_px`).

## §6 Protocol

1. Freeze `synthetic_aec/test`.
2. Baseline = the current `pipeline.py`. Then, for each stage group S / J / R / F / W / D /
   T, swap in each alternative singly, holding the rest at baseline → one CSV row per
   (method, sample).
3. Take the top-2 per stage group and test their full-combo products.
4. Aggregate per §3 bucket; produce a ranking table. Note per-regime winners (e.g. S8+F4
   for straight-dominant sheets, S10 or S3+R2 for curved / heavily overlapping ones).
5. Qualitative pass on real embedded rasters from `references/*.pdf` (reconstruction SSIM
   only — no GT).
6. Harness: `junction_test/ablation.py` (mirrors `smoke.py`) → `ablation_results.csv`;
   `junction_test/methods/<id>.py`, one module per candidate behind a common
   `vectorize(binary, params) -> (segments, arcs, beziers)` interface; shared metrics.

## §7 Decision (filled in after the runs)

Results table + chosen recipe (a single method, or per-regime routing) → drives the real
`rastervec` raster-stage implementation.

## Implementation status

- Harness: `junction_test/ablation.py` (`METHODS` presets, bucketed `run_ablation`,
  `bucket_table`, `write_csv` → `ablation_results.csv`); `explore.ipynb` §7.
- Methods wired as `Params` overrides (no separate `methods/` package):
  `baseline` (S1), `S2_medial_axis`, `S3_junction_repair`, `S8_lsd`, `S9_hough`,
  `F1_rdp`. S4–S7, S10, S11 (learned), J2–J5, R1–R7 not yet implemented.
- Metrics §5: junction-type confusion/accuracy, X pass-through, coincident-unrelated
  false-merge, width MAE, dash-style accuracy, arc-radius rel-err, curve-misfit count,
  reconstruction SSIM, runtime — all in `metrics.evaluate`.
- Buckets §3: archetype (floor_plan / curved / mep_overlay / dimension / residue) ×
  overlap density (low/med/high, via `forced_crossings`) × clean/noisy × seeds.
  Not yet: DPI sweep, aspect-ratio > 4:1, per-primitive Bézier archetype.
