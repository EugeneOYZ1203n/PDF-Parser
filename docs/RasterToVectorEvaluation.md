# Raster-to-vector (P2) evaluation

How the benchmark scores what a Phase 2 backend (`Junction`, `DeepVectoriser`, `DeepTechVec`,
`ImplicitSketchVec`, `InvolutionStroke`) makes of a raster page. The headline metric is the
vector-level protocol of

> Liu Wenyin, Dov Dori, **"A protocol for performance evaluation of line detection
> algorithms"**, *Machine Vision and Applications* 9:240–250 (1997).
> DOI [10.1007/s001380050045](https://doi.org/10.1007/s001380050045) · open copy (the authors'
> own, archived):
> <https://web.archive.org/web/2010id_/http://www.cs.cityu.edu.hk/~liuwy/publications/Evl_MVA.pdf>
> · local copy `references/Liu_Dori_1997_line_detection_protocol.pdf` (gitignored).

with a handful of deliberate deviations (marked **⚠** below), plus the paper's own pixel-level
companion (PRI), SSIM/PSNR and two global sanity checks. Code: `rastervec/Evaluation/Evaluate/
liu_dori.py` (the protocol, pure), `vector_metrics.py` (orchestration + SSIM/length/direction),
`vector_render.py` (renders), `adapters.py` (labels → inputs), `benchmark_run_loading.py::
_score_vectors` (per page), `benchmark_report_sections.py::_add_vector_sections` + `charts.py`
(report), `Report/report_artifacts.py::_add_vector_eval_layers` (viewer overlays).

## 1. Inputs

**Predictions.** `scripts/generate_pipeline_report.py` with `benchmark: true`, for an input that is
a `scripts/label/master_label.py` folder with a `rasterised.pdf` (every page flattened to one
full-page image): besides the normal vectorised-PDF run, each page of `rasterised.pdf` gets a
**separate** pipeline run (same `p2`/`p3`). That page has no native vectors, so its output vectors
are Phase 2's. They are stored as `PageDump.raster_vectors` in `<run>/<doc>/dump.json`, together
with `PageDump.raster_dpi` — the resolution of that rasterised page, read off its embedded image
(`adapters.raster_dpi_for_page`). A failed raster run is logged and leaves no predictions.

**Ground truth.** The label folder's geometry annotations (`LabelSet.geometry_entries`):

| type | source | what |
|---|---|---|
| `vector_to_raster` | `source="auto"` | every drawing item of the original page (`raster_label.raster_geometry_for_page`) — incl. CAD vector-text glyph strokes, not native font text |
| `original_raster` | `source="manual"` | lines/curves hand-traced in `raster_label.py` on embedded images with no vector backing |

A page is scored against the **union** of both types (that is what P2 sees); D_v is also split per
type. `original_vector` is a text type only and never scored here.

**Text boxes.** Every text label's `cluster_bbox` on the page (all five text types) — used only to
split results into text / non-text (§5).

Scoring happens at benchmark time (`scripts/pipeline_report_benchmark.py`) from `dump.json`, so an
existing report folder can be re-scored without re-running the pipeline. For a dump without
`raster_dpi` (written before it existed) the dpi is read from the label folder's
`rasterised.pdf`, else `VectorMetricConfig.default_dpi` (300).

## 2. Lines: everything is a straight bar

Both sides are reduced to straight **bars** (`liu_dori.bars_from_entries`); nothing is chained,
merged or de-duplicated.

- **GT:** one bar per `"l"` annotation item (`re`/`qu` were already split into their 4 edges when
  the labels were made). A long CAD line stored as several items stays several GT lines.
- **Predictions:** one bar per `"l"` item of each P2 `Vector`, raw (Junction's items already went
  through its own 1 px Douglas–Peucker; nothing more is done to them).
- **Curves (`"c"`, either side):** sampled every `curve_sample_px` (0.5) raster pixels of arc
  length, RDP-simplified with ε = `rdp_eps_frac` (¼) × the item's W_eff, and **each resulting
  piece is its own bar**. ε ≤ W_eff/4 keeps every chord inside half the matching window.
- Zero-length bars are dropped (no length weight).

Every bar keeps its width, whether its source was dashed (`dashes` array non-empty), whether more
than 50 % of its length lies inside any one text label box, and its GT type.

### ⚠ Width floor

Every quantity in the protocol is relative to the GT line's width W(g), and CAD hairlines have
width 0 (or 0.1–0.25 pt). Both sides therefore use

    W_eff = max(W, floor_px · 72 / dpi)        floor_px = 2   (≈ 0.48 pt at 300 dpi)

where `dpi` is the rasterised page's resolution. A line can't be located more finely than the
raster P2 saw, so the floor ties the tolerance to that. Predicted widths get the same floor
(Junction always reports 1.0 pt, which makes its matching area wider than a hairline GT's — an
accepted consequence).

## 3. Matching (paper §3)

For every GT bar g and predicted bar k (candidates from a KD-tree on endpoints, O(N log N)):

1. **⚠ Angle gate:** the undirected angle between g and k must be ≤ `match_angle_tol_deg` (15°).
   The paper has no angle test; without it, short bars at corners and crossings (whose endpoints
   all sit inside each other's area) match perpendicular lines and inflate the fragmentation
   counts.
2. **Endpoint rule:** at least 2 of the 4 endpoints must lie inside the other line's area
   (distance to that segment ≤ its W_eff/2). Three cases: both of g's endpoints inside k, both of
   k's inside g, or one of each (partial overlap).
3. **Overlap segment** c = k ∩ g lies on k: all of k (k inside g), g's endpoints projected onto k
   (g inside k), or from k's inside endpoint to g's inside endpoint projected onto k. A
   zero-length c is not an overlap.
4. d1, d2 = distances of c's ends to g; **d_overlap** = Hausdorff distance between c and its
   projection onto g. It must be < W(g)/2.

## 4. Quality (paper §4.2, equation numbers are the paper's)

Per overlap c:

| factor | formula | eq. |
|---|---|---|
| endpoints Q_pt | exp(−(d1 + d2) / W(g)) | 12 |
| overlap distance Q_od | exp(−2·d_overlap / W(g)) | 13 |
| width Q_w | exp(−\|W(k) − W(g)\| / W(g)) | 14 |
| **⚠** Q_v(c) | (Q_pt · Q_od · Q_w)^(1/3) | 17 |

**⚠** The paper's line-style (Q_st) and line-shape (Q_sh) factors are dropped: every line is a
straight bar (Q_sh ≡ 1) and P2 never emits dashed lines, so keeping them would only inflate scores
(with the paper's 5th root) or penalise every dashed GT line for a style P2 cannot produce. W in
these factors is W_eff on both sides. Q_w is kept as in the paper — for Junction (fixed 1.0 pt) it
mostly reflects the drawing's line weights, which is why the report shows every factor separately.

Per GT line g, over D(g) = the predicted bars overlapping it:

- **Q_b(g)** = Σ Q_v(c)·l(c) / max(l(g), Σ l(c)) — basic quality: uncovered length counts 0,
  overlapping predictions can't push it above the line (Eq. 18)
- **Q_fr(g)** = √(Σ l(c)²) / Σ l(c) — fragmentation: 1 for one piece, → 1/√n for n equal pieces
  (Eq. 19)
- **Q_v(g)** = Q_b(g) · Q_fr(g) (Eq. 20); 0 when nothing matched

Per predicted line k, the same over G(k) = the GT bars it overlaps (Eqs. 22–24; Q_fr(k) measures
*consolidation* — one prediction spanning several GT lines), and its false-alarm factor
F_v(k) = 1 − Q_v(k) (Eq. 25).

Page / run:

- **D_v** = Σ_g Q_v(g)·l(g) / Σ_g l(g) — vector detection rate (Eq. 21; the paper prints the
  denominator as Σ_{k∈V_d} l(g), a typo for the GT length)
- **F_v** = Σ_k F_v(k)·l(k) / Σ_k l(k) — vector false alarm rate (Eq. 26)
- **VRI** = β·D_v + (1 − β)·(1 − F_v), β = 0.5 (Eq. 27)

**Fragmentation counts** — "how many predicted lines make up each GT line": |D(g)| per GT line
and |G(k)| per predicted line, as histograms (0 = missed / pure false alarm, 1, 2, 3–5, 6+) plus
the mean over matched lines.

## 5. Splits

| split | lines | metrics |
|---|---|---|
| text / non-text | both sides (> 50 % of the bar inside one text label box) | D_v, F_v, VRI |
| solid / dashed | GT only | D_v |
| GT type | GT only (`vector_to_raster` / `original_raster`) | D_v |

## 6. Pixel level and image similarity

- **D_p / F_p / PRI** (paper Eqs. 5, 10, 11): GT bars and predicted bars rendered black at
  W_eff at `pixel_render_dpi` (300) (`vector_render.render_ink_pair`); ink = gray <
  `ink_threshold` (128). D_p = |GT ink ∩ pred ink| / |GT ink|, F_p = 1 − |GT ink ∩ pred ink| /
  |pred ink|, PRI = α·D_p + (1 − α)(1 − F_p), α = 0.5. Unlike page-wide SSIM, only ink counts.
- **SSIM / PSNR**: unchanged — whole-page grayscale renders of the raw geometry at its real widths
  (stroke colour luminance), 150 dpi (`vector_render.render_pair`), SSIM with a Gaussian window
  (σ 1.5), PSNR capped at 100 dB.

## 7. Global sanity checks

- **Length ratio** — Σ raw flattened predicted length / Σ GT length (> 1: over-tracing or
  duplicates, < 1: misses).
- **Direction** — intersection of the two length-weighted 36-bin angle histograms (mod 180°);
  catches a systematically rotated or skewed page that per-line matching would just call "missed".

## 8. Aggregation

Micro-averaged: every ratio's numerator and denominator are summed over pages (and over inputs
for the grand aggregate) before dividing — a long-line-heavy page weighs more, exactly as within a
page. Histograms and ink-pixel counts are summed; SSIM is pixel-weighted and PSNR comes from the
pooled MSE.

## 9. Where results show up

- **`pipeline_report_benchmark.py` → `report.html`**, Vector group, one column per run:
  Headline (D_v, F_v, VRI, PRI, SSIM, PSNR) · Q_v components (Q_pt, Q_od, Q_w, Q_v(c); Q_b, Q_fr,
  Q_v for GT and for predictions) · Splits · Fragmentation |D(g)| and Consolidation |G(k)| ·
  Counts · Pixel level · Length · Direction. Charts in `charts/`: `vector_headline`,
  `vector_components`, `vector_fragmentation`, `vector_consolidation`, `vector_quality`,
  `vector_direction`. `benchmark.txt` has the same in text.
- **`pipeline_report_viewer.py`**, stage `vector_eval` (written by `generate_pipeline_report.py`
  in benchmark mode, unless `debug_layers: false`): `gt by Q_v` and `pred by Q_v` (every bar
  coloured red → yellow → green by its line's Q_v), `missed gt` (|D(g)| = 0), `false alarms`
  (|G(k)| = 0), `gt endpoints` and `pred endpoints` (a dot at both ends of every bar).

## 10. Known biases (accepted)

These follow from the choices above; read low numbers with them in mind.

- **Dashed GT** is one line: P2's solid dashes leave the gaps uncovered (Q_b ≤ dash / period) and
  fragment it (Q_fr ≈ 1/√n). See the solid/dashed split.
- **Duplicate GT** (CAD drawing the same line twice) is kept: one perfect prediction then overlaps
  two GT lines, Q_fr(k) ≈ 0.71, raising F_v.
- **CAD-split items**: one physical line stored as several GT items makes a clean single
  prediction look consolidated (|G(k)| > 1).
- **Curves**: GT and predicted curves are RDP'd independently, so their bar breakpoints don't line
  up; a perfectly traced curve still shows some fragmentation/consolidation.
- **Abutting items**: the paper's endpoint rule accepts a prediction whose end merely pokes into a
  GT line by less than a line width (a neighbouring collinear item), adding a short overlap and one
  to |D(g)|.
- **Text**: CAD text glyph strokes are GT, but the RasterText prelude erases text before tracing,
  so successful erasure lowers D_v; unerased native text becomes false alarms. See the text split.
- **Widths**: Junction's fixed 1.0 pt costs Q_w on thin GT and widens its matching area.

## 11. Verification

`tests/rastervec/Evaluation/Evaluate/test_liu_dori.py` reproduces the paper's Table 2 (single-bar
cases Good / Short / Long / Skew / Narrow / Style error / Errors / Fragmentary / Merged) with
`LiuDoriConfig(paper_mode=True)` — the paper exactly: no floor, no angle gate, the 5-factor 5th
root with Q_st/Q_sh, and its even-width rule (a 1-px offset on an even-width line counts as 0).
Every printed value matches within 0.01, except "Errors" (0.02 — the paper rounds a 2.54 px
distance to 3 px) and the "Merged" VRI: the paper prints 0.71, inconsistent with its own Eq. 27
for its printed D_v = 1 and F_v = 0.41 (= 0.795); every other column satisfies Eq. 27. The
production config's deviations are covered by synthetic cases in the same file.

## Tunables

`LiuDoriConfig` (`liu_dori.py`, reached via `VectorMetricConfig.liu_dori`): `floor_px` 2,
`curve_sample_px` 0.5, `rdp_eps_frac` 0.25, `match_angle_tol_deg` 15, `beta` 0.5, `alpha` 0.5,
`pixel_render_dpi` 300, `ink_threshold` 128, `paper_mode` (tests only). `VectorMetricConfig`
(`vector_metrics.py`): `curve_segments` 10, `direction_bins` 36, `render_dpi` 150,
`psnr_cap_db` 100, `default_width_pt` 0.5 (SSIM renders of width-less entries),
`default_dpi` 300.
