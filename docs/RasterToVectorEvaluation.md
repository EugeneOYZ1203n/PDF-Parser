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
(report), `Report/report_artifacts.py::_add_vector_eval_layers` (viewer overlays). §10 explains
how to read the numbers and go from a report table to a cause.

## 1. Inputs

**Predictions.** `scripts/generate_pipeline_report.py` with `benchmark: true` runs the pipeline
(same `p2`/`p3`) on each selected **input form** of every page (`inputs`, see
`scripts/report_configs/README.md` and `rastervec/Evaluation/forms.py`). Two forms are scored here:

| form | pipeline input | P2 predictions | GT scored against |
|---|---|---|---|
| `rasterised` | the label folder's own `rasterised.pdf` page, else `conversion.rasterise_page` at the config's `dpi` | every output vector (the page has no native vectors, so all of them are Phase 2's) | `vector_to_raster` ∪ `original_raster` |
| `original` | the source page | Phase 2's own vectors only (`PageDump.p2_vectors`, from the run's `extra["phase2_vectors"]`) — never P1/P3's native vectors | `original_raster` only (on the original page the `vector_to_raster` lines are real PDF vectors) |

The vectorised form has no Phase 2 geometry ground truth and is not vector-scored. Each form's
run lives in its own `<run>/<input>/<form>/dump.json`, with `PageDump.raster_dpi` — the resolution
of the largest embedded image on the run's input page (`adapters.raster_dpi_for_page`). A form run
that raises is logged and recorded as a failed page (`counts["run_failed"]`). Report folders made
before input forms existed (one `<input>/dump.json`, the rasterised run in its `raster_*` fields)
are still scored as the rasterised form.

`rasterise_page` (and, since this change, `master_label.py`'s own `rasterised.pdf`) renders the page
**unrotated** and re-applies `/Rotate`, so the run's coordinates match the labels' unrotated page
space. A `rasterised.pdf` made by `master_label.py` before that fix baked the rotation into the
image, which transposes every coordinate on a 90°/270° page — regenerate those pages (clear their
`rasterise_done` flags in the folder's `manifest.json` and re-run `master_label.py`).

**Ground truth.** The label folder's geometry annotations (`LabelSet.geometry_entries`):

| type | source | what |
|---|---|---|
| `vector_to_raster` | `source="auto"` | every drawing item of the original page (`raster_label.raster_geometry_for_page`) — incl. CAD vector-text glyph strokes, not native font text. Taken from the label folder / sidecar where that page has any saved; otherwise derived live from the original page at benchmark time (`_bench_ground_truth_by_type`), so a bare PDF, or a label folder whose geometry step never ran, is still scored |
| `original_raster` | `source="manual"` | lines/curves hand-traced in `raster_label.py` on embedded images with no vector backing |

A rasterised page is scored against the **union** of both types (that is what P2 sees), an
original page against `original_raster` only; D_v is also split per type. `original_vector` is a
text type only and never scored here.

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

- **`pipeline_report_benchmark.py` → `report.html`**: per input, a one-line-per-form summary, then
  one group per form; the Vector group of the rasterised and original forms has, one column per
  run: Headline (D_v, F_v, VRI, PRI, SSIM, PSNR) · Q_v components (Q_pt, Q_od, Q_w, Q_v(c); Q_b,
  Q_fr, Q_v for GT and for predictions) · Splits · Fragmentation |D(g)| and Consolidation |G(k)| ·
  Counts · Pixel level · Length · Direction. Each form's Counts group adds the run's own output
  counts, incl. Phase 2's per-colour-layer vectors / items and OCR blanks. Charts in `charts/`:
  `<key>__<form>__vector_{headline,components,fragmentation,consolidation,quality,direction}.png`.
  `benchmark.txt` has the same in text.
- **`pipeline_report_viewer.py`** on `<run>/<input>/<form>/`, stage `vector_eval` (written by
  `generate_pipeline_report.py` in benchmark mode, unless `debug_layers: false`): `gt by Q_v` and
  `pred by Q_v` (every bar coloured red → yellow → green by its line's Q_v), `missed gt`
  (|D(g)| = 0), `false alarms` (|G(k)| = 0), `gt endpoints` and `pred endpoints` (a dot at both ends
  of every bar).

## 10. Interpreting results

### Reading order

1. **Headline** — D_v and F_v first, VRI after. VRI = ½·D_v + ½·(1 − F_v), so the same VRI can
   mean "finds everything, half of it junk" or "finds half, all of it clean"; always read it with
   its two inputs.
2. **Q_v components** — *why* D_v / F_v are what they are (geometry vs. coverage vs.
   fragmentation).
3. **Fragmentation / Consolidation** — how GT and predicted lines pair up (1:1, 1:n, n:1, 0).
4. **Splits** — whether one population (text, dashed, one GT type) drags the total down.
5. **Viewer** (`vector_eval` stage) — the actual lines behind the numbers.

Two rows in the report are identities, not extra information: `Q_v(g) -- GT total` ≡ D_v and
`Q_v(k) -- pred total` ≡ 1 − F_v.

### What each number says

All in 0..1 with 1 = perfect, unless noted. Everything in the Liu & Dori block is weighted by
**length**, not by line count — one long missed wall line costs more than a dozen missed hatch
ticks.

| number | reads as | note |
|---|---|---|
| D_v | quality-weighted recall: how much of the GT length was found, *and how well* | not a hit rate — a found-but-fragmented line contributes < 1 |
| F_v | quality-weighted 1 − precision: how much predicted length is wrong, misplaced, overrunning its GT line or merging several | a prediction that overlaps nothing counts fully; an exact duplicate of a good prediction is **not** penalised here (it shows as GT fragmentation + length ratio > 1) |
| Q_pt, Q_od, Q_w, Q_v(c) | quality of the **overlaps that exist** (weighted by overlap length) | misses don't enter at all: high Q_* with low D_v = "what is found is precise, but much is not found" |
| Q_pt | endpoint agreement | low → lines too short / overshooting / stopping at the wrong junction |
| Q_od | positional agreement along the overlap | low → offset, skew, wobble (staircases) |
| Q_w | width agreement | mostly line-weight noise; Junction always reports 1.0 pt (§4) — don't rank backends on it |
| Q_b(g) / Q_b(k) | coverage × overlap quality, before fragmentation | low → uncovered length or poor geometry |
| Q_fr(g) | fragmentation of matched GT lines | ≈ 1/√n for n equal pieces: 0.71 ≈ 2 pieces, 0.58 ≈ 3, 0.5 ≈ 4 |
| Q_fr(k) | consolidation of matched predictions | same scale: one prediction spanning n GT lines |
| \|D(g)\| histogram | per GT line, how many predicted lines overlap it | `0` = missed, `1` = clean, `2`+ = fragmented. Counts lines, not length |
| \|G(k)\| histogram | per predicted line, how many GT lines it overlaps | `0` = pure false alarm, `2`+ = consolidated (often a GT artefact, §11) |
| mean (matched) | average of the above over lines with ≥ 1 match | excludes the `0` bucket |
| D_p, F_p, PRI | the same recall / false-alarm idea on **ink pixels** (§6) | blind to fragmentation, endpoint splits and line identity — "is the ink in the right place" |
| SSIM, PSNR | whole-page image similarity | dominated by blank paper: a sparse page scores high even with misses. Sanity only; PSNR is capped at 100 dB |
| length ratio | Σ pred length / Σ GT length (§7); 1 = match, not bounded | > 1 over-tracing / duplicates (both edges of a thick stroke, fill outlines, double-traced lines); < 1 misses |
| direction intersection | overlap of the two angle histograms | low, or a peak shifted by ~90° in the direction chart → a coordinate / rotation problem, not tracing |

### Diagnostic patterns

| symptom | likely cause | where to look |
|---|---|---|
| D_v low, D_p high | ink is there but fails the vector match: fragmented, wobbly (> 15° angle gate on short pieces), or ends in the wrong places | Q_fr(g), \|D(g)\| `2`+, Q_pt; viewer `gt by Q_v` (yellow/orange lines that look covered) |
| D_v low, D_p low | genuinely missed geometry | viewer `missed gt`; GT-type and text splits (erased text lowers D_v by design, §11) |
| F_v high, F_p low | predictions sit on GT ink but don't match it as lines: overshooting their GT line, spanning several GT lines, or staircase pieces failing the 15° angle gate | Q_b(k), Q_fr(k), \|G(k)\|; viewer `pred by Q_v` |
| D_v lowered by Q_fr(g), F_v fine, length ratio > 1 | duplicated / double-traced predictions (each copy scores Q_v(k) = 1, so only the GT side sees them) | \|D(g)\| `2`+, viewer `pred endpoints` |
| F_v high, F_p high | spurious geometry: noise, unerased text, fill outlines, paper texture | viewer `false alarms`; text-split F_v; \|G(k)\| `0` share |
| Q_pt low, Q_od high | right line, wrong extent | viewer `gt endpoints` vs `pred endpoints` |
| Q_od low | offset or skewed lines (or a dpi / coordinate mismatch if it is uniform across the page) | viewer, direction chart |
| \|G(k)\| `2`+ while Q_v(c) is high | one clean prediction across CAD-split or duplicated GT items | §11 — usually not a backend fault |
| dashed D_v ≪ solid D_v | expected: P2 traces dashes as solid pieces (§11) | compare dashed D_v between runs only, never against solid |
| text D_v low, non-text fine | RasterText erased the text before tracing — intended | non-text D_v / F_v is the tracing score |
| text F_v high | text that was not erased got traced | P2's `text_removal/*` debug layers |
| direction intersection low / histogram peaks moved ~90° | transposed coordinates (e.g. an old rotated `rasterised.pdf`, §1) | regenerate the page; distrust every other number for it |

### Reading the charts (`report.html`, `charts/`)

One bar group per metric, one colour per run; NaN (n/a) bars are simply absent.

| chart | what to look at |
|---|---|
| `vector_headline` | D_v, 1 − F_v, VRI, D_p, 1 − F_p, PRI, SSIM — all flipped so **taller = better**. Compare the vector pair (D_v, 1 − F_v) with the pixel pair (D_p, 1 − F_p): a pixel bar much taller than its vector bar means the ink is right but the line structure isn't (fragmentation / endpoints, see Diagnostic patterns). SSIM is usually near the top for every run — don't read differences there |
| `vector_components` | the ten Q_v factors left to right: per-overlap (Q_pt, Q_od, Q_w, Q_v(c)), then the GT side (Q_b, Q_fr, Q_v = D_v), then the prediction side (Q_b, Q_fr, Q_v = 1 − F_v). Find the **lowest bar in each block** — that factor is what limits it. Per-overlap bars high but Q_b(g) low → coverage (misses); Q_b fine but Q_fr low → fragmentation / consolidation |
| `vector_fragmentation` | share of GT lines (by count) per \|D(g)\| bucket `0`, `1`, `2`, `3-5`, `6+`. Ideal: everything in `1`. A tall `0` = missed lines; mass in `2`+ = broken lines |
| `vector_consolidation` | the same for predicted lines per \|G(k)\|. Tall `0` = false alarms; `2`+ = one prediction across several GT items (often CAD-split / duplicated GT, §11) |
| `vector_quality` | SSIM, length ratio, direction intersection — 1 = match for each. The y-axis is **not** clamped: a length-ratio bar above 1 is over-tracing, below 1 under-tracing |
| `vector_direction` | the GT's length-weighted angle histogram (grey bars, 0–180°) with one step line per run. Lines should follow the grey bars — CAD pages peak at 0° and 90°. A run peaking where GT doesn't (e.g. 45° noise, or swapped 0°/90° heights) traces systematically wrong directions |

The `aggregate__<form>__*` charts are the same over every shared input (micro-averaged, §8);
`<key>__<form>__*` are per input.

### Reading the viewer overlays (`vector_eval` stage)

Open `pipeline_report_viewer.py <run>/<input>/<form>` (or two runs side by side) with the source
page underneath. Every overlay is drawn per **bar** (§2 — a curve shows as several short bars), at
the bar's width.

| layer | colour | read as |
|---|---|---|
| `gt by Q_v` | per GT bar, red (Q_v 0) → yellow (0.5) → green (1), 10 steps | **D_v made visible**. The whole bar takes its line's Q_v, so a fully covered but fragmented line still shows yellow/orange (e.g. two halves → ≈ 0.7). Red = missed |
| `pred by Q_v` | per predicted bar, same scale | **1 − F_v made visible**. Red = false alarm; yellow on top of GT = poorly placed, overshooting or spanning several GT lines |
| `missed gt` | solid red | GT lines with \|D(g)\| = 0 only — the red subset of `gt by Q_v`, easier to spot alone |
| `false alarms` | solid orange | predicted lines with \|G(k)\| = 0 only |
| `gt endpoints` / `pred endpoints` | purple / cyan dots | both ends of every bar. Many cyan dots along one purple-delimited GT line = fragmentation; cyan short of purple = Q_pt loss; isolated cyan clusters = noise |

Workflow:

1. `gt by Q_v` alone: where is red/orange concentrated — text areas, dashed lines, hatching,
   curves, thin or pale lines, one colour layer? That names the failing population; check it
   against the matching split row.
2. Add `missed gt` + the source page: is the missed ink faint, in a colour layer that wasn't
   split out, erased as text, or under a fill?
3. `pred by Q_v` + `false alarms`: are red predictions on paper (noise, texture), on unerased
   text, or doubling real lines (thick strokes traced on both edges, fill outlines)?
4. Endpoints at high zoom on yellow lines: fragmentation (extra cyan dots mid-line) vs. extent
   errors (cyan dots short of / past the purple ones).
5. Two runs side by side: the same GT line green in one and red in the other is a real
   difference; a uniformly slightly shifted colour usually isn't (see Comparing runs).

Colour on the **GT** layer is always the GT line's own Q_v and on the **pred** layer the
prediction's own Q_v. Fragmented or duplicated predictions turn the GT line yellow too; a single
prediction spanning several consecutive GT items (consolidation) leaves every GT item green and
only the prediction yellow.

### "n/a"

- **D_v n/a** — no GT lines (page has no geometry labels of the scored types).
- **F_v n/a, hence VRI n/a** — no predicted lines: `p2: "Stub"`, a backend that output nothing, or
  a failed run (check the form's Counts group, `run_failed`). Predicting nothing is *not* VRI 0.5.
- **A split's row n/a** — that population is empty (no dashed GT, no text boxes, no
  `original_raster` labels…). Split F_v is always n/a for solid/dashed and GT type (GT-only
  splits).
- **A whole column n/a** — that run did not produce that form.

Guide values that follow from the formulas: every prediction missing gives D_v 0, F_v 1, VRI 0;
Q_fr ↔ piece count as in the table above. There are no absolute "good" thresholds — the GT
biases in §11 put a ceiling below 1 on real pages, so use the numbers to compare runs.

### Comparing runs

- Only compare the same input and form (`pipeline_report_benchmark.py` matches shared inputs by
  identity and lines up forms). **Rasterised and original forms are not comparable** with each
  other: they are scored on different GT (union vs. `original_raster` only, §1).
- Aggregates are micro-averaged by length (§8), so line-heavy pages dominate; check the per-input
  groups before trusting a grand aggregate shift.
- Rank on D_v and F_v (and the fragmentation histograms), not on VRI alone, and not on SSIM /
  PSNR.
- Differences of ~0.01–0.02 are within the noise of the accepted biases (curve breakpoints,
  abutting items, §11); look at the viewer before calling them real.

## 11. Known biases (accepted)

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

## 12. Verification

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
