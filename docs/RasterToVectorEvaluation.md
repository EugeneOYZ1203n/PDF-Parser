# Raster-to-vector (raster_vectorize) evaluation

How the benchmark scores what a raster_vectorize backend makes of a raster page. The headline
metric is the vector-level protocol of

> Liu Wenyin, Dov Dori, **"A protocol for performance evaluation of line detection
> algorithms"**, *Machine Vision and Applications* 9:240–250 (1997).
> DOI [10.1007/s001380050045](https://doi.org/10.1007/s001380050045) · open copy (the authors'
> own, archived):
> <https://web.archive.org/web/2010id_/http://www.cs.cityu.edu.hk/~liuwy/publications/Evl_MVA.pdf>

The protocol has a few deliberate deviations, each marked **⚠** below. On top of it come the
paper's own pixel-level companion (PRI), SSIM / PSNR and two global sanity checks.

Code, all under `raster_parser/evaluation/`:
- `evaluate/liu_dori.py`: the protocol (pure).
- `evaluate/vector_metrics.py`: orchestration, SSIM, length, direction.
- `evaluate/vector_render.py`: the renders.
- `evaluate/raster_benchmark.py`: ground truth, plus `score_vectors`, the single entry point the
  scripts call.
- `report/raster_report.py`: writes the run folder and the report sections.

The method lives in `raster_benchmark.score_vectors` alone; everything else only moves data, so a
different method replaces that function and its tests.

## 1. Inputs

A benchmark config with `"raster_benchmark": true` (`scripts/report_configs/README.md`) does three
extra things for each benchmark page:

1. It rasterises the original page at `raster_dpi` with `conversion.rasterise_page`. The result is
   one full-page image, rendered at rotation 0 with `/Rotate` re-applied, so coordinates stay in
   the source page's unrotated space.
2. It runs the pipeline on that image with the config's `p2`. The page has no native vectors, so
   every output vector comes from raster_vectorize.
3. It writes `<doc>/raster/`, holding the rasterised pages, that run's `dump.json` and
   `ground_truth_vector_to_raster.json`.

The ground truth is the **original page's own vectors**: every painted `Vector` (one with a
stroke colour or a fill), split into `"l"` / `"c"` entries, with `re` / `qu` turned into four
lines. Nothing is drawn by hand. The text label boxes (hand-labelled vector text plus native text)
are stored with each page and drive the text / non-text split (§5).

Limits of this ground truth:
- There is no hand-traced ground truth for embedded images.
- It is not cut to what the page render actually shows, so paper-coloured or covered paths still
  count as ground truth (§11).

## 2. Lines: everything is a straight bar

Both sides are reduced to straight **bars** (`liu_dori.bars_from_entries`); nothing is chained,
merged or de-duplicated.

- **GT:** one bar per `"l"` annotation item (`re`/`qu` were already split into their 4 edges when
  the labels were made). A long CAD line stored as several items stays several GT lines.
- **Predictions:** one bar per `"l"` item of each P2 `Vector`, raw: nothing is done to them
  beyond what the backend itself returned.
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
(a backend that reports a fixed width, e.g. 1.0 pt, gets a matching area wider than a hairline
GT's — an accepted consequence).

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
these factors is W_eff on both sides. Q_w is kept as in the paper — for a backend reporting a fixed
width it mostly reflects the drawing's line weights, which is why the report shows every factor separately.

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
| GT type | GT only (`vector_to_raster`) | D_v |

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

`pipeline_report_benchmark.py` adds a **Raster vectorization (Liu & Dori)** table whenever a run
folder holds `raster/` results. The table appears in the summary and per input, one column per
run. Its rows are:
- D_v, F_v, VRI
- D_v for text and for non-text lines
- D_p, F_p, PRI
- length ratio
- direction match
- SSIM, PSNR

`benchmark.txt` holds the same table as text. The finer protocol outputs (the Q factors and the
fragmentation histograms) are on the `LiuDoriResult` that `score_vectors` returns; they are not in
the report yet.

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
| Q_w | width agreement | mostly line-weight noise for a backend reporting a fixed width (§4) — don't rank backends on it |
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
| text D_v low, non-text fine | the P2 backend erased the text before tracing — intended | non-text D_v / F_v is the tracing score |
| text F_v high | text that was not erased got traced | P2's text-removal debug layers |
| direction intersection low / histogram peaks moved ~90° | transposed coordinates (e.g. an old rotated `rasterised.pdf`, §1) | regenerate the page; distrust every other number for it |

### "n/a"

- **D_v n/a** — no GT lines (page has no geometry labels of the scored types).
- **F_v n/a, hence VRI n/a** — no predicted lines: `p2: "Stub"`, a backend that output nothing, or
  a failed run (check the form's Counts group, `run_failed`). Predicting nothing is *not* VRI 0.5.
- **A split's row n/a** — that population is empty (no dashed GT, no text boxes…). Split F_v is always n/a for solid/dashed and GT type (GT-only
  splits).
- **A whole column n/a** — that run had no `raster_benchmark`.

Guide values that follow from the formulas: every prediction missing gives D_v 0, F_v 1, VRI 0;
Q_fr ↔ piece count as in the table above. There are no absolute "good" thresholds — the GT
biases in §11 put a ceiling below 1 on real pages, so use the numbers to compare runs.

### Comparing runs

- Only compare the same input (`pipeline_report_benchmark.py` matches shared inputs by key) at the
  same `raster_dpi`.
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
- **Text**: CAD text glyph strokes are GT, but a P2 backend that erases text before tracing
  lowers D_v by succeeding; unerased native text becomes false alarms. See the text split.
- **Widths**: a fixed predicted width (e.g. 1.0 pt) costs Q_w on thin GT and widens its matching
  area.

- **Invisible GT**: ground truth isn't cut to what the render shows yet (§1), so paths painted in
  the paper colour, or covered by fills, count as missed lines.

## 12. Verification

`tests/raster_parser/evaluation/evaluate/test_liu_dori.py` reproduces the paper's Table 2 (single-bar
cases Good / Short / Long / Skew / Narrow / Style error / Errors / Fragmentary / Merged) with
`LiuDoriConfig(paper_mode=True)` — the paper exactly: no floor, no angle gate, the 5-factor 5th
root with Q_st/Q_sh, and its even-width rule (a 1-px offset on an even-width line counts as 0).
Every printed value matches within 0.01, except "Errors" (0.02 — the paper rounds a 2.54 px
distance to 3 px) and the "Merged" VRI: the paper prints 0.71, inconsistent with its own Eq. 27
for its printed D_v = 1 and F_v = 0.41 (= 0.795); every other column satisfies Eq. 27. The
production config's deviations are covered by synthetic cases in the same file.
`test_raster_benchmark.py` covers the ground truth and `score_vectors`, and
`tests/scripts/test_raster_benchmark_report.py` the whole run (slow).

## Tunables

`LiuDoriConfig` (`liu_dori.py`, reached via `VectorMetricConfig.liu_dori`): `floor_px` 2,
`curve_sample_px` 0.5, `rdp_eps_frac` 0.25, `match_angle_tol_deg` 15, `beta` 0.5, `alpha` 0.5,
`pixel_render_dpi` 300, `ink_threshold` 128, `paper_mode` (tests only). `VectorMetricConfig`
(`vector_metrics.py`): `curve_segments` 10, `direction_bins` 36, `render_dpi` 150,
`psnr_cap_db` 100, `default_width_pt` 0.5 (SSIM renders of width-less entries),
`default_dpi` 300.
