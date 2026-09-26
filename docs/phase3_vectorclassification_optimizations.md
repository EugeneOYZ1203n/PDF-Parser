# Phase 3 `VectorClassification`: runtime optimizations, 2026-09

**Status: implemented.** This document records a session-long runtime-optimization pass over
`rastervec/P3_Vector_Parsing/VectorClassification/` (the classify → FAST-filter → OCR backend,
now `core.registry.DEFAULT_P3`) — what was slow, why, and exactly what changed. It's a changelog,
not a proposal: every item below has landed and is covered by the test suite
(`tests/rastervec/P3_Vector_Parsing/VectorClassification/`).

The audit that produced this list started from a simple question — "where does this backend's
time actually go, and is any of it wasted?" — and found two recurring patterns: the classify
stage did several times more clustering work than necessary per bucket, and the OCR stage never
batched or parallelized across clusters despite the machinery (Pool 2, `config.OCR_BATCH_SIZE`)
already existing for exactly that. Fixing the OCR-batching item required either duplicating the
sibling `FastIntoPaddle` backend's reference pattern or removing it — it was removed as a separate
piece of this session's work (see the note at the end of this doc), and its pattern is what this
document's biggest item is built on.

## 1. Page-wide OCR batching + Pool-2 dispatch

**Files**: `paddle_engine.py`, `parse.py`, `config.py` (`DETECT_RENDER_CHUNK_SIZE`).

**Before**: `parse()`'s main loop processed one FAST-surviving cluster at a time — render → detect
→ deskew/crop → recognize → up to 3 retry passes — each step a separate PaddleOCR engine call
sized to that one cluster's own handful of quads. A page with hundreds of small text clusters made
hundreds of separate detect calls and hundreds of separate recognize calls, each paying PaddleOCR's
own fixed per-call overhead (Python↔Paddle marshalling, internal resize/padding) regardless of how
little work that call actually did. None of it used `compute` (the optional Pool-2 process-pool
proxy) even though `parse()` already accepted it and threaded it into the FAST stage.

**After**: the OCR pass is staged across the whole page instead of looped per cluster:

1. **Render** every surviving cluster's own image, in-process (rendering needs this process's own
   shared `fitz` document; Pool-2 workers never import `fitz`/`pymupdf`) — but in bounded chunks of
   `config.DETECT_RENDER_CHUNK_SIZE` clusters at a time, not the whole page at once (see the memory
   note below).
2. **Detect** — one PaddleOCR detector call per cluster, now dispatched across Pool-2 workers per
   chunk via `paddle_engine.py::_detect_job` when a caller passes `compute` (`compute.starmap`),
   instead of always running serially in the calling process. This is new capability beyond any
   existing precedent (`FastIntoPaddle`'s own detect stage never dispatched to Pool 2, only its
   recognize stage did).
3. **Deskew + crop** every detected quad (cheap CPU work, stays in-process) and accumulate every
   chunk's quads into one flat, **page-wide** pool.
4. **Recognize** the whole page's quads in `config.OCR_BATCH_SIZE`-sized batches — a handful of
   batched calls instead of one call per cluster — each batch dispatched via
   `paddle_engine.py::_recognize_crops_job` through `compute.apply` when given, mirroring
   `FastIntoPaddle/paddle_engine.py::recognize_segments`'s `recognize_fn` hook (the reference
   pattern this was built from, before that module was removed).
5. **Blank-recognition retry sweep** (+90/180/270) — also now batched across every still-blank crop
   on the *whole page* at each pass, not just one cluster's, via the same chunked dispatch
   (`_recognize_crops_raw_job`).

**A real memory issue found and fixed during verification**: an early version of this change
rendered *every* surviving cluster before detecting any of them, holding every cluster's raw image
array in memory simultaneously. On a real reference PDF (~1800 text-candidate clusters, one large
title-block cluster rendering to ~195MB as a raw array), this crashed PaddleOCR's own detector with
an opaque allocator error under memory pressure. The fix: stages 1-3 run in bounded chunks
(`DETECT_RENDER_CHUNK_SIZE = 8`), so only one chunk's worth of cluster renders is ever live at once;
recognition (stage 4/5) is unaffected since crops are far smaller than full cluster renders and
still batch across the whole page. Separately, `debug_out["cluster_detections"]` (raw per-cluster
bgr + quads, read back only by `scripts/debug_image_savers.py` for debug-image dumping) is now only
accumulated when a caller actually passed `debug_out` — it used to be built unconditionally on
every run, which is the same latent memory-accumulation pattern independent of the chunking fix
above.

**Verified**: the new Pool-2 dispatch path has a dedicated regression test
(`test_parse_dispatches_detect_and_recognize_through_compute`, a fake compute-pool stand-in
asserting the exact `compute.starmap`/`compute.apply` call shape) since no prior test exercised
`compute` at all. A real end-to-end run against a reference PDF (`--no-fast`, since this
environment has no FAST model weights) confirmed the chunked design completes without the earlier
crash — one real page, ~1800 classification clusters, `p2=Stub p3=VectorClassification texts=776
vectors=0`, phase3 sub-step durations `{'classify': 0.81s, 'fast': 0.04s, 'ocr_render': 19.8s,
'ocr_detect': 355.9s, 'ocr_recognize': 78.6s}` with no `compute` pool (fully serial). Re-run
against the same page with a real 2-worker `compute_pool` (`core.parallel.pool.compute_pool(2)`,
not a test double) produced the identical `texts=776` output — confirming the Pool-2 path is
correct, not just non-crashing — in `phase3: 322.0s` total, down from `455.5s` serial (~29% less
wall-clock time from just 2 workers on the detect stage, the dominant cost by far in both runs).

## 2. Vectorized FAST mask sampling

**File**: `fast_filter.py`.

**Before**: `detect_text_fast` scored every vector, across every cluster, with its own call to
`_sample_mask` — a Python function that sliced the FAST page mask and summed it, once per vector.
On a page with thousands of small stroke-vectors, this was thousands of separate small numpy
slice-and-`.sum()` calls, each cheap but paying real per-call Python/numpy dispatch overhead.

**After**: one summed-area table (`_build_integral_image`, two `np.cumsum` passes) is built once
over the page mask; every vector across every cluster is then scored in a handful of vectorized
numpy operations (`_vector_mask_scores`) — gathering every vector's clipped pixel bbox into
parallel coordinate arrays, computing all region sums via the standard 4-corner integral-image
lookup in one shot, and re-splitting the flat per-vector score array back into the
`vector_scores_by_cluster` shape `detect_text_fast`'s caller-facing behavior already expects.
Verified against the existing `test_fast_filter.py` suite (per-vector pass/fail semantics
unchanged) with no signature change at the `detect_text_fast` level.

## 3. Tile-candidate spatial index

**File**: `fast_detect.py`.

**Before**: `detect_tiled`'s candidate-bbox filtering (`_tile_has_candidate`) checked every tile
against every candidate bbox — O(tiles × candidates).

**After**: candidates are bucketed once into a spatial hash grid (`_build_candidate_grid`, cell
size = the tile grid's own `block_size`, same spirit as `commons.helpers.clustering.cluster_spatial`
but simplified since tile geometry is already a fixed regular grid) so a tile's candidate check is
a small dict lookup instead of a full scan — O(tiles + candidates). A low-priority item (tile
counts are bounded by page size, so this was unlikely to dominate today), included since it was
cheap and low-risk once the surrounding code was already being touched.

## 4. `classify_vectors.py::_collect_dropped` — analyzed, intentionally left unchanged

Originally flagged as "dead-weight work repeated per bucket per page," since the current 2-step
classification chain (seqno-overlap merge + spatial clustering) never produces a `role="dropped"`
category — `drawing_vectors` downstream is populated entirely by the later FAST stage. Re-examined
against the actual code rather than just the original finding: the function's cost is a handful of
step/category string-equality checks per bucket, not a scan over vectors — genuinely negligible,
several orders of magnitude below the other items on this list. Hardcoding a `drawing_vectors=[]`
short-circuit would save nothing measurable and would silently break if a future classification
step ever reintroduces dropping. Left as-is.

## Already landed earlier this session (for a complete record)

These four were implemented before the items above, in an earlier pass over the same backend:

- **`cluster_filters.py::cluster_spatial_groups` ran the shared spatial-hash clustering 3x per
  bucket, unconditionally** — two of the three passes existed purely to populate debug-only
  categories that were never gated behind `verbose`. Removed; the constrained pass is the only one
  that ever ran for real.
- **Repeated bbox recomputation instead of caching** — `cluster_filters.py::_matched_side_value`
  now takes precomputed bboxes (cached once per group in `cluster_spatial_groups`, via a
  `bbox_by_id` dict) instead of recomputing `union_bbox` on every neighbor-pair examined;
  `parse.py`'s per-quad loop no longer calls `pixel_to_page_bbox` twice for the same quad.
- **`core/parallel/pool.py::warmup()` warmed the wrong classes** — it only warmed the deprecated
  top-level `OCR/` module's `FastDetector`/`PaddleRecBackend`, not each P3 backend's own duplicated
  copies. Split into `_warm_fast_detectors()`/`_warm_paddle_engines()`, each iterating every backend
  that has its own copy (now `VectorClassification` and the legacy `OCR/` module — `FastIntoPaddle`
  was removed since, `LegacyRecreation` has no FAST stage).
- **`paddle_engine.py::_hough_ink_mask`/`_minarea_ink_mask` each independently grayscaled the same
  crop** — shared into one `_grayscale` call in `hough_deskew`, computed once per quad instead of
  twice.

## A related but separate change: `FastIntoPaddle` removed

In the same session, the sibling `FastIntoPaddle` P3 backend was removed entirely (it was
redundant with `VectorClassification`, and its `paddle_engine.py::recognize_segments`/
`_recognize_crops_job` pattern is what item 1 above is built from). `core.registry.DEFAULT_P3` is
now `"VectorClassification"`. This is an architectural change, not a performance one, so it isn't
detailed further here — see `CLAUDE.md`'s `P3_Vector_Parsing/` bullet (the `FastIntoPaddle` entry
has been removed from it) and `docs/PIPELINE.md` for the current state.

## Verification

- `Evaluation/Evaluate/timing.py`'s per-substep breakdown (`phase3.classify`/`fast`/`ocr_render`/
  `ocr_detect`/`ocr_recognize`/`drawing`) is the way to measure any of this empirically — run
  `scripts/pipeline_report_benchmark.py` (or `generate_pipeline_report.py` with `benchmark: true`)
  on a shared reference PDF before/after a change and compare those named rows directly.
- For the Pool-2 items (page-wide detect/recognize dispatch, the classify-stage 3x-redundancy
  removal), compare `Evaluation/Evaluate/benchmark.py --compute-workers N` against a
  `--compute-workers 0` baseline on the same input, both in wall-clock and in the same per-substep
  breakdown.
- This step needs a real, non-gitignored reference PDF and PaddleOCR's models on disk, so it's a
  manual measurement to run yourself rather than a number this document can assert.
