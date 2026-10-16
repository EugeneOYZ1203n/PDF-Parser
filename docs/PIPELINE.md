# v2VectorClass, end to end

`P3_Vector_Parsing/v2VectorClass/` (`p3="v2VectorClass"`) finds text drawn as vector paths, stroked
or filled, reads it with PaddleOCR and returns the rest as drawing. This document follows one
page through it. Terms in *italics* are defined in [Glossary.md](Glossary.md); every threshold
is a constant in `v2VectorClass/config.py`, named here in `CAPS`.

```
Phase 1 vectors
  │
  ├─ 1. classify_vectors      per (layer, colour, width) bucket: oversize → collinear → pattern
  │                           → seqno merge → spatial cluster → length outliers → crossings
  │        ├─ dropped ─────────────────────────────────────────────────────────────┐
  │        └─ text candidate clusters + global angles                              │
  ├─ 2. render + detect       per cluster: B/W render, PaddleOCR detector → quads   │
  ├─ 3. rotation + crop       quad long-edge angle, snapped; upright crop; split    │
  ├─ 4. recognise             0/180 classifier + recognise; retries at 90/180/270   │
  ├─ 5. ownership             non-blank quads own the cluster's vectors ─ rest ─────┤
  │                                                                                 ▼
  └─ texts (source="ocr")                                       drawing (sorted by seqno)
```

## What goes in and out

- **In:** Phase 1's vectors only (one `Vector` per `get_drawings()` path, in unrotated page
  space) and the `RunSettings`. Phase 3 never sees Phase 2's output.
- **Out:** `(drawing, texts)`. `texts` are OCR `Text`s with `source="ocr"`, a `direction` from
  the final reading angle, `quad_points` (the detected quad, reordered so p0 → p1 runs along
  the reading direction), the envelope `bbox`, `confidence` and `seqno` (the cluster's lowest).
  `drawing` is every vector not owned by a read, in paint order (`seqno`). Vectors that became
  text are not returned.

## 1. Classification (`classify_vectors.py`)

Vectors are split into *buckets* by layer, then colour (stroke + fill), then stroke width
(`layer_color_separation.width_key`: rounded to 0.01 pt; fill-only `"f"` vectors share one
`None` key, so one word's glyphs aren't split by a leftover width). Nothing is ever compared
across buckets. Each bucket runs these steps in order; each step either keeps entries or
*drops* them to drawing:

| # | Step | Drops to drawing |
|---|---|---|
| 0 | Oversize | a vector whose bbox covers at least `MAX_VECTOR_PAGE_AREA_FRAC` (30 %) of the page: borders, frames, background fills |
| 1 | Collinear drawing | a *collinear group* of straight vectors with more than `COLLINEAR_DRAWING_MIN_COUNT` (50) members and a length std below `COLLINEAR_DRAWING_MAX_STD_PT` (5 pt): long dashed or repeated lines |
| 2 | Pattern lattice | repeated motifs on a regular lattice (stipple, hatch tiles, symbol arrays): see below |
| 3 | Seq overlap merge | nothing: chains vectors in `seqno` order into *groups* while each stays within `SEQ_OVERLAP_TOLERANCE_PX` of the group's bbox |
| 4 | Spatial cluster | nothing: single-linkage merge of groups within `SPATIAL_CLUSTER_THRESHOLD` (10 pt) whose matching sides agree within `SPATIAL_SIZE_TOLERANCE` (30 %), giving *clusters* |
| 5 | Length outliers | per cluster, strokes in a *parallel group* more than `LENGTH_OUTLIER_STD` (2) std from the pooled length mean |
| 6 | Crossings | per cluster, vectors made only of `"l"` items that at least `MIN_CROSSINGS` (4) foreign pieces properly cross, when they lie on the *dominant grid* (`GRID_ANGLE_TOL_DEG`, `GRID_DOMINANCE`); other crossed vectors stay |

Steps 3 and 4 never grow a group or cluster past `MAX_CLUSTER_PAGE_AREA_FRAC` (30 %) of the
page, which bounds each cluster's OCR render.

**Pattern lattice** (`pattern_lattice.py`). Vectors with the same item kinds and the same
per-item lengths (in `PATTERN_SIM_LENGTH_TOL_PT` buckets, rotation-invariant) form a
*similarity bucket*. In a bucket of more than `PATTERN_MIN_BUCKET` members, groups are
flood-filled over lattice sites from a seed: v1 is the nearest neighbour, v2 the nearest one not
parallel to it (none: a 1D lattice), and a vector joins a site only within `PATTERN_GRID_TOL_PT`
of the global grid point. A group is drawing when it has more than `PATTERN_MAX_GROUP` (10)
members (1D: `PATTERN_MAX_GROUP_1D`, 50, so a row of repeated glyphs stays text) **and** the
links between adjacent members cross, on average, fewer than `PATTERN_LINK_FOREIGN_MEAN_LIMIT`
foreign vectors, so a repeated letter with other text between its copies stays. When more than
`PATTERN_BUCKET_DRAWING_GROUPS` groups of one similarity bucket drop, the bucket's other large
groups drop too.

The clusters left after step 6 are the *text candidates*. Classification also returns the
page's *global angles*: the mean angle of every collinear group with at least
`GLOBAL_ANGLE_MIN_GROUP_SIZE` (2) members, in every bucket, deduplicated within `ANGLE_TOL_DEG`.

## 2. Render and detect (`parse.py`, `paddle_engine.py`)

Clusters are processed in chunks of `DETECT_RENDER_CHUNK_SIZE` (8) so only a few renders exist
at once. Each cluster is drawn alone, black on white, with its real stroke widths
(`commons/renderer/png.render_vector_cluster`), on a canvas of its bbox plus half its widest
stroke plus `RENDER_PADDING_EXTRA_PT` (15 pt). The DPI starts at `OCR_DPI` (300) and is raised
until the shorter side reaches `MIN_RENDER_SIDE_PX` (100 px), up to `MAX_RENDER_DPI`
(`ocr_prep.render_cluster_with_dynamic_dpi`). The render is not rotated. PaddleOCR's text
detector (PP-OCRv4, `en`) returns one rotated *quad* per text line it finds.

## 3. Rotation and the upright crop

The rotation comes from the quad alone; there is no Hough transform or minimum-area rectangle:

1. The angle of the quad's longer side (`quad_long_edge_angle`, in [-90, 90)).
2. Snapped to the nearest global angle when within `QUAD_ANGLE_SNAP_TOL_DEG` (5°), else kept.
3. A quad more than `MAX_CROP_ASPECT` (25) times longer than tall is split along its length,
   cutting in the gaps between the cluster's vectors where possible; each piece is read as its
   own text.
4. `upright_crop` rotates the region around the quad by that angle (`cv2.warpAffine`, a pure
   rotation) and cuts the quad out, expanded by `CROP_EXPAND_FRACTION` plus a
   `CROP_BORDER_PX` white border.

## 4. Recognition and retries

All crops of the page are recognised in batches of `OCR_BATCH_SIZE` (128):

- **Pass 1:** PaddleOCR's 0/180 angle classifier flips upside-down crops, then the recogniser
  reads them. A read's *score* is its confidence, halved for a single character
  (`SINGLE_CHAR_PENALTY`), 0 when blank.
- **Retries:** every crop scoring below `RETRY_CONFIDENCE_THRESHOLD` (0.8) is also read turned
  +90, +180 and +270 (no classifier). The attempt with the best *selection score* wins: score ×
  `ENGLISH_WORD_MULTIPLIER` (1.2) per dictionary English word in the read (pyenchant `en_US`,
  alphabetic tokens of at least `ENGLISH_WORD_MIN_LEN` letters). Ties go to the earlier attempt,
  so pass 1 is always a candidate. The retry trigger itself uses the raw score.
- **Final angle** = (snapped) quad angle + the classifier's flip (when pass 1 won) + 90° × the
  winning retry.

A crop that is blank on every attempt produces no text.

## 5. Ownership: text or drawing

Each vector of an OCR'd cluster is tested against the non-blank quads detected in **its own
cluster** (for a split quad, the whole detected quad). A quad *owns* a vector, cheapest test
first:

1. bbox fully inside the quad → owned;
2. bbox area larger than the quad → not owned;
3. fewer than `TEXT_SEGMENT_OVERLAP_FRAC` (50 %) of its pieces touch the quad → not owned;
4. otherwise owned when more than `TEXT_INK_INSIDE_FRAC` (50 %) of its ink (path length) lies
   inside the quad.

Owned vectors are text and left out of the output; the rest join the drawing. Drawing = every
vector classification dropped + every unowned vector of the OCR'd clusters, sorted by `seqno`.

## Settings, parallelism and timing

| `RunSettings` | Effect |
|---|---|
| `workers > 1` | detection (one job per cluster render) and recognition (one job per batch, pass 1 and each retry turn) run on one spawn process pool per page (`ocr_pool.py`); inputs are written to `dump_dir/v2VectorClass/` as `.npz` and removed afterwards. Batches are the same as with one worker and results come back in order, so the output is identical. A call with one job runs in-process. |
| `ocr_dpi`, `use_angle_cls` | not used (DPI per cluster from `OCR_DPI`; the classifier always runs on pass 1) |
| `on_debug_layer`, `on_debug_image` | debug layers and images, listed in the folder's `README.md` |
| `on_step_time` | sub-step seconds as `("phase3", name, s)`: `classify_*` per step, `ocr_render`, `ocr_detect`, `ocr_crop`, `ocr_recognize`, `ocr_assemble`, `quad_ownership`, `drawing`, `debug_render`. The pipeline report stores them in `dump.json` (`sub_durations`) and the benchmark shows them as `phase3 › <name>` rows. |

## Compared with v1VectorClass

| | v1VectorClass | v2VectorClass |
|---|---|---|
| Text it can find | filled glyphs only (`"f"` paths that aren't white, boxes or rules) | any vector text, stroked or filled |
| Grouping | words by draw order + proximity | buckets by layer / colour / width, drawing removed by geometry (collinear, lattice, outliers, crossed grids), clusters by draw order + spatial gap |
| Detection | none: each word group is one crop | PaddleOCR's detector on each cluster render |
| Orientation | horizontal / vertical per group, `rotate` 0 or 90 | any angle: the quad's long edge, snapped to the page's line angles, plus classifier flip and retries |
| Retries | none | +90 / 180 / 270 for low scores, English-word scoring |
| Text vs drawing | decided before OCR (fill filter) | decided after OCR (quad ownership) |
| `ocr_dpi`, `use_angle_cls` | used | not used |
| `workers` | not used | OCR process pool |
| Extra dependencies | none | `scipy`, `pyenchant` |
