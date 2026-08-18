# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A raster-to-vector pipeline project for architectural/engineering shop drawings
(see `references/*.pdf`, gitignored sample PDFs). Two independent pieces live here, sharing no
imports:

- **`inspector/`** — a Tkinter + PyMuPDF desktop tool for visually inspecting what's inside a PDF
  (text, images, annotations, vector drawings, as toggleable overlays). This was step 0, built to
  visually validate what PyMuPDF extracts before writing real extraction logic. It's done and
  should not need further changes for `rastervec` work.
- **`rastervec/`** — the actual extraction pipeline: native text, vector drawings (including
  reconstructing CAD "text-as-filled-vector-paths" back into real text), and raster-image line
  diagrams (via a CNN junction detector + line tracing), consolidated into text/line objects and
  reassembled into a PDF for evaluation. Built stage by stage (Reader → Native → Vector → Raster);
  currently Reader, Native, and Vector are implemented, including Vector-stage OCR
  (`Renderer.render_vector_cluster` + `RenderOCR`, via PaddleOCR) — the Raster stage and
  `Renderer.render_raster_region` remain interface-only stubs — see "rastervec architecture" below.

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt        # install deps
.venv/Scripts/python.exe -m inspector.app [path/to.pdf]             # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.pipeline --pdf PATH --page N  # run the extraction pipeline demo (CLI)
.venv/Scripts/python.exe -m rastervec.debug_app [path/to.pdf]       # run the pipeline debug app (GUI)
.venv/Scripts/python.exe -m pytest tests/ -v                         # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC DST --dpi 300  # flatten a PDF to pure raster
```

venv is **Python 3.12** (`py -3.12 -m venv .venv`), not 3.14 — `rastervec`'s OCR (paddleocr/
paddlepaddle) and the still-unbuilt Raster stage (opencv-python) don't ship Windows wheels for 3.14
yet. On this dev machine's paddlepaddle build, the default mkldnn-accelerated CPU inference path
hits an unimplemented PIR attribute-conversion error, so `rastervec/helpers/render_ocr.py` sets
`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` at import time (before paddleocr/paddlex read their
flags) to force the plain "paddle" run mode instead — fine for OCR's small, pre-cropped cluster
renders. If a future paddlepaddle release fixes this, that env-var default can be dropped.

## `inspector/` architecture

Five modules, one package:

- **`layers.py`** — the extensibility core. `OverlayItem` is the normalized shape every extractor
  returns (bbox always in PDF page coordinates; `quad`/`points` optionally for non-axis-aligned
  geometry; `attrs` for machine-filterable values; `metadata` for human-readable hover info).
  `LayerSpec` (one top-level checkbox) and `SubFilterSpec` (a sub-checkbox group under a layer) are
  declarative — `build_layers(pdf_model)` wires the four current layers (text/images/annotations/
  drawings) to their extractor functions in `pdf_model.py`. `filter_items()` is the one shared
  filtering function all layers use (AND across sub-filter groups, OR within a group, empty
  selection = no restriction). Adding a new layer means adding one `LayerSpec` + one extractor
  function — nothing in `app.py`, `overlay_canvas.py`, or `control_panel.py` needs to change.
- **`pdf_model.py`** — the only module that calls into `fitz` for extraction. `PdfDocument` wraps
  the open document; `extract_text_items`/`extract_image_items`/`extract_annot_items`/
  `extract_drawing_items` each return `list[OverlayItem]` for one page; `collect_drawing_colors`
  scans a page's `get_drawings()` once to populate the dynamic stroke/fill color sub-filters.
- **`overlay_canvas.py`** — `PageView`: the left-pane Tk `Canvas` showing the rendered page pixmap
  with overlay shapes drawn on top, plus page nav/zoom controls and a hover tooltip.
- **`control_panel.py`** — `ControlPanel`: the right-pane checkbox tree built from the `LAYERS`
  registry, with collapsible sub-filter groups (checkboxes or color swatches).
- **`app.py`** — `InspectorApp` wires the two panels together, owns `AppState` (current page/zoom,
  per-page extraction and color caches), and drives the redraw cycle. `REFERENCES_DIR` resolves to
  the repo-root `references/` folder (one level above the `inspector/` package).

### Coordinate spaces — read this before touching geometry, in either package

PyMuPDF's extraction APIs (`get_text`, `get_drawings`, `get_image_info`, `annots()`) all return
coordinates in the page's **unrotated MediaBox space**, regardless of the page's `/Rotate` value.
`page.get_pixmap()` and `page.rect`, however, are already in **rotated display space** (rotation
baked in). `inspector/app.py._get_display_matrix()` builds the single transform
(`page.rotation_matrix * zoom_matrix`) that both the pixmap and every overlay must go through to
land in the same canvas space — never compute a separate scale/rotation by hand, or overlays will
drift from the underlying page image on any rotated page (several `references/*.pdf` pages are
rotated 90/270). `rastervec` keeps unrotated MediaBox space as its canonical space through every
stage, only converting to display space at final render/reconstruction — see `rastervec/models.py`'s
module docstring.

For text specifically: a word's axis-aligned bbox from `get_text("words")` only equals its
along-direction/normal-direction extents when the text is horizontal. `make_oriented_quad`
(`rastervec/geometry.py`, ported from `inspector/pdf_model.py._make_oriented_quad`) projects the
bbox corners onto the text's actual direction vector (from the matching span's `dir`) to build a
correctly oriented quad for rotated/vertical text — don't reintroduce a bbox-width/height shortcut
there, in either package.

### Extending `inspector/` with a new layer

1. Write an extractor `def extract_x_items(page: fitz.Page) -> list[OverlayItem]` in `pdf_model.py`.
2. Add a `LayerSpec(key=..., extractor=pdf_model.extract_x_items, subfilters=[...])` to the list in
   `layers.build_layers()`.
3. For a new filterable attribute, add a `SubFilterSpec` and register its `attr_getter` in
   `layers._GETTERS` (a `SubFilterSpec` not present in `_GETTERS` renders in the UI but silently
   filters nothing — this bit the `close_path`/`has_mask` filters before it was fixed, so don't
   forget this step).

## `rastervec/` architecture

Stage classes, one per pipeline phase, plus shared helpers — designed so each stage is testable
independently of the others (every stage's *output* is a plain dataclass from `models.py`, no
`fitz` objects, except `Page.fitz_page` which `Reader` must hand to `Native`/etc.):

- **`models.py`** — all shared dataclasses (`PageMeta`, `Page`, `TextWord`, `TextRun`, and
  forward-declared `VectorPath`/`DrawingVector`/`TextVectorResult`/`RasterImage`/`JunctionPoint`/
  `LineVector`/`ReconstructedPage` for stages not yet implemented).
- **`geometry.py`** — pure-math helpers ported from `inspector/pdf_model.py` (`point_angle`,
  `line_length`, `quad_angle`, `matrix_rotation`, `matrix_scale`, `make_oriented_quad`, etc.), so
  both packages use the same already-verified math without duplicating it.
- **`logging_setup.py`** — stdlib `logging` only. `configure_logging(level)` once at startup;
  `get_logger("stage_name")` returns `logging.getLogger("rastervec.stage_name")` per module.
- **`reader.py` — `Reader`** *(implemented)*: opens a PDF, hands out `Page` objects one at a time
  (`get_page(index)`, `iter_pages(indices=None)`), each carrying a `PageMeta` snapshot (mediabox,
  rotation, dimensions) plus the live `fitz.Page`.
- **`native.py` — `Native`** *(implemented)*: `extract_text(page) -> list[TextWord]`, via
  `get_text("dict")` for span metadata (font/size/color/direction) joined to `get_text("words")`
  geometry by max bbox-overlap (`_match_word_to_span`), producing correctly oriented quads even for
  rotated text (`_build_oriented_quad`). Split into small private methods
  (`_extract_spans`/`_extract_words`/`_match_word_to_span`/`_build_oriented_quad`/`_to_text_word`)
  so each is independently testable against a synthetic `fitz.Page`.
- **`vector.py` — `Vector`** *(implemented)*: `extract_paths(page) -> list[VectorPath]` walks
  `page.fitz_page.get_drawings()`, emitting one `VectorPath` per drawing item (`l`/`re`/`qu`/`c`),
  tagged with its parent drawing's `seq` (drawing index) plus stroke/fill color, width, dashes,
  closed, layer, and item-level `bbox`/`points`. `separate_by_layer`/`separate_by_color` group paths
  by `layer` (`""` for none) / by `stroke_color` if set else `fill_color` else `None`.
  `classify(paths, page, cluster_order=None) -> list[list[VectorPath]]` chains the full
  classification pipeline (the pipeline (`pipeline.py`) instead calls each piece individually, one
  per debug-app stage):
  1. `filter_layout_panels` drops single-item `re`/`qu` drawings (page borders/title-block panels,
     which never share a `seq` with anything else — a real CAD text-as-vector-paths drawing is one
     `seq` with *many* glyph items, so this never touches real text).
  2. `filter_large_bbox` drops paths whose own bbox covers more than `large_bbox_area_fraction`
     (default 0.2) of the page — border/frame geometry caught by size instead of item-count.
  3. `cluster(paths, page, order=None)` runs up to 4 clustering/grouping operations —
     `cluster_spatial`, `cluster_spatial_union_find`, `cluster_by_seq`, `group_overlapping`,
     `cluster_groups_by_dimension`, or `"none"` to skip an ordinal position — in `order` (default
     `Vector.CLUSTER_STEPS`, i.e. just one layer of `cluster_spatial`, the other 3 ordinal positions
     `"none"`), returning one groups-list **snapshot per step** (`snapshots[i]` = the result after
     applying `order[i]`), not just the final result — this is what lets the debug app's single
     "Clustering" stage show/compare every step's output, and let the user reorder the operations
     interactively (see `debug_app.py` below). Internally, `_apply_cluster_step` dispatches each
     step key to its underlying method:
     - `cluster_spatial` — high-tolerance (`spatial_threshold`, default 8.0) single-linkage
       grouping by bbox gap, via `Clustering.cluster_spatial`. Since this method's signature takes
       a *flat* path list (not groups), `_apply_cluster_step` re-flattens whatever grouping exists
       so far before re-clustering — it's always a from-scratch spatial pass, never a groups-in/
       groups-out refinement of its input, so reordering it away from 1st genuinely re-clusters
       everything spatially at that later point.
     - `cluster_spatial_union_find` — same distance rule as `cluster_spatial` (bbox gap ≤
       `spatial_threshold`, same grid-bucketed union-find), but applied to the *incoming groups
       themselves* (by each group's aggregate bbox via `geometry.union_bbox`) rather than the raw
       paths — reuses `Clustering.cluster_spatial` at the group level the same way
       `cluster_groups_by_dimension` reuses `Clustering.cluster_by_dimension`. Unlike
       `cluster_spatial`, this never flattens/re-derives from scratch: each incoming group is one
       atomic unit, so groups an earlier operation already formed only ever get merged together
       here, never re-split.
     - `cluster_by_seq` — tighter pass within each incoming group, splitting by drawing
       sequence-number proximity (`seq_max_gap`), via `Clustering.cluster_by_seq`.
     - `group_overlapping` merges paths whose bboxes overlap OR are within a small gap tolerance of
       each other (`max(0.5% of the page's smaller dimension, 3px)` — real glyph/symbol strokes are
       often a pixel or two apart, not truly touching); a path fully contained in (or equal to)
       another's bbox is left separate regardless of tolerance — via `Clustering.group_by_overlap`.
       Scoping stays per-cluster: only paths already in the same incoming group are ever compared,
       since `group_by_overlap` applies its pairwise merge independently to each incoming group.
     - `cluster_groups_by_dimension` merges groups whose *overall* bbox width/height are close
       (`group_dimension_tolerance`) — reuses `Clustering.cluster_by_dimension` at the group level
       (each incoming group, not each path, is one item to compare; `geometry.union_bbox` gives
       each group's aggregate bbox), then flattens each resulting super-group back to a flat path
       list.
  4. `filter_large_group_bbox` drops whole groups whose *aggregate* bbox covers more than
     `large_bbox_area_fraction` of the page — the same per-path rule `filter_large_bbox` applies
     earlier, reapplied per-group now that clustering has run (a group can end up oversized even
     when no single member path was).
  5. `filter_aspect_ratio` drops whole groups shaped like a long thin line/ruler (aggregate bbox
     aspect ratio > `max_aspect_ratio`, default 10.0) — real drawing content, never a text
     candidate. This is the *last* classification step — `classify()` returns whatever survives
     it, unmodified.

  There is deliberately **no drawing-vs-text heuristic** at the end of `classify()` — an earlier
  version had one (`classify_clusters`/`_looks_like_text`, judging a cluster "text" by member count
  + size + fill fraction) but it was removed: every path/group any filter step (1/2/4/5, plus
  whatever clustering step lands where within the chosen order) drops along the way is drawing
  content (`pipeline.py`'s per-stage wiring folds those drops into `drawing_vectors`, not `Vector`
  itself — see `pipeline.py` below), and everything that survives every filter is handed to OCR
  (`pipeline.py`'s `ocr_text_clusters` stage) as-is — OCR success/failure (did it read any text) is
  the actual signal for whether a cluster was text, not a pre-filter guess. `build_drawing_vectors
  (paths) -> list[DrawingVector]` re-aggregates same-`seq` paths back into one `DrawingVector` per
  original drawing (bbox union + first path's style), i.e. the unit a downstream renderer would
  draw. All thresholds (`spatial_threshold`, `seq_max_gap`, `large_bbox_area_fraction`,
  `max_aspect_ratio`, `group_dimension_tolerance`) are `Vector.__init__` params — tune per-PDF if a
  specific page's default classification looks wrong; there's no per-PDF auto-tuning (the
  `group_overlapping` gap tolerance is the one threshold that's computed from the page's own
  dimensions instead, not a `Vector.__init__` param).

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across buckets**:
  `pipeline.py`'s `_iter_groups`/`_run_clustering` key every filter/cluster stage's work by
  `GroupKey = (layer, color)` (from `color_separation`'s output), and `Vector.cluster()`/the filter
  methods are only ever called with one bucket's paths/groups at a time — two paths in different
  layers, or with different stroke/fill colors, are never spatially merged, sequence-merged,
  overlap-merged, or dimension-merged together, regardless of how close they are on the page.
- **`helpers/clustering.py` — `Clustering`** *(cluster_spatial/cluster_by_dimension/cluster_by_seq/
  group_by_overlap implemented, `cluster_hsv` still a stub)*: pure-Python (no scipy/sklearn) spatial
  hash grid + union-find for `cluster_spatial` (buckets items into grid cells sized by `threshold`,
  unions items in neighboring cells whose `geometry.rect_gap` ≤ `threshold` — `Vector.
  cluster_spatial_union_find` reuses this same method at the group level, treating each group as one
  atomic item), then O(k²) pairwise
  union-find within each resulting group (`_split_group_pairwise`, shared by all three of the
  following) for `cluster_by_dimension` (relative width/height closeness — reused both per-path,
  early on, and per-group, late in `Vector`'s pipeline), `cluster_by_seq` (sorted-seq gap split), and
  `group_by_overlap` (merges items whose bboxes overlap or are within an optional `tolerance` of
  each other, via module-level `_bboxes_close_or_overlapping` — `geometry.rect_gap` already returns
  0.0 for overlapping/touching boxes, so one gap check covers both "touching" and "merely nearby";
  `_bbox_fully_contains` keeps a fully-contained/equal pair from ever merging regardless of
  tolerance). Has safety caps (`_MAX_CELLS_PER_ITEM`, `_MAX_GROUP_SIZE_FOR_PAIRWISE`) so a huge
  unfiltered bbox or a very dense cluster degrades to "keep as one cluster" (logged) instead of
  hanging — verified against a 78k-path reference PDF in ~3.5s. `Raster.separate_by_color` (HSV
  pixel clustering, not yet implemented) is designed to reuse `cluster_hsv` once Raster is built.
- **`raster.py`, `helpers/clustering.py`'s `cluster_hsv`, `helpers/masking.py`, `helpers/junction.py`,
  `evaluation.py`, `Renderer.render_raster_region`** *(interface stubs)*: full method signatures and
  docstrings exist, bodies `raise NotImplementedError` — everything the not-yet-built Raster stage
  needs. `Raster.separate_by_color` (HSV pixel clustering) is designed to reuse `cluster_hsv` once
  Raster is built.
- **`renderer.py` — `Renderer`** *(rendering helpers, not a pipeline stage)*: `path_color_hex(path)`
  returns a path's real PDF stroke/fill color as hex (used by both the debug app and OCR input
  rendering) — any B/W-style simplification stays purely internal to classification, never
  substituted into a rendered/displayed color. `render_vector_cluster(paths, page, dpi)`
  *(implemented)* isolates a cluster onto a fresh single-page PyMuPDF document sized to the
  cluster's own bbox (plus padding — `max(4pt, largest member's stroke_width)` — so edge strokes
  aren't clipped), redraws each path with its own real stroke/fill/width/dashes via `fitz.Shape`
  (translated so the bbox's top-left lands at the padding offset), then rasterizes at `dpi` and
  returns a PIL `Image` — reusing PyMuPDF's own path/curve/fill rendering rather than
  reimplementing rasterization by hand. A path with neither `stroke_color` nor `fill_color` set is
  skipped outright (never handed to `Shape.finish()`): `finish()` emits a stroke operator whenever
  `fill` is `None` regardless of `color`, falling back to the current graphics-state color (default
  black) instead of staying invisible, so skipping is the only way to keep a genuinely colorless
  path from getting an unwanted black stroke in the render. `page` is accepted but unused (paths
  are already absolute page-space coordinates) — kept for interface parity with
  `render_raster_region` and because `RenderOCR.ocr_cluster` calls both through the same shape.
  `render_raster_region` (raster-image OCR input) is still a stub.
- **`helpers/render_ocr.py` — `RenderOCR`** *(implemented)*: multi-rotation render + OCR +
  confidence-voting, shared by the Vector stage (OCR'ing rendered vector-text clusters, the only
  case actually reachable today) and, once built, the Raster stage (OCR'ing raster image regions —
  `ocr_cluster` already branches on `list[VectorPath]` vs. `RasterImage`, but the raster branch
  hits `Renderer.render_raster_region`'s `NotImplementedError` until that stage exists).
  `render_rotations(image, n=8)` returns `n` evenly-spaced rotations (white-filled, `expand=True`
  so nothing gets clipped). `ocr(image)` runs one image through a lazily-built, module-scope-cached
  PaddleOCR engine (`_ENGINE_CACHE`, keyed by `lang` — constructing a fresh `RenderOCR()` per stage
  run, as `pipeline.py` does, is cheap after the first real OCR call) configured with
  `use_doc_orientation_classify=use_doc_unwarping=use_textline_orientation=False` (callers already
  hand in an isolated, upright render, so none of that preprocessing is needed) and joins
  PaddleOCR's `rec_texts`/`rec_scores`/`rec_polys` output (it can detect more than one text box per
  image) left-to-right into one `(text, confidence, bbox_corners)` result. `combine_rotation_results`
  groups per-rotation readings by normalized-text similarity (`difflib.SequenceMatcher.ratio() >=
  0.75`), sums each group's confidence to pick the winning group, then reports that group's
  *average* confidence (not the sum) as the final score. `ocr_cluster(cluster, page, renderer, dpi=
  300, n_rotations=8)` is the shared entrypoint: render → `render_rotations` → `ocr` each rotation →
  `combine_rotation_results`, returning a `TextVectorResult` whose `rotation_used` is the angle of
  whichever single rotation had the highest individual confidence (not necessarily a member of the
  winning combined-text group — just a simple, well-defined answer to "which orientation did this
  come from").
- **`evaluation.py` — `Evaluation`** *(interface stub)*: the pipeline's actual intended final stage
  — `reconstruct_page`/`build_pdf` reassemble consolidated text/line objects back into a PDF for
  evaluation — not yet registered in `Pipeline.STAGES` since not implemented.
- **`pipeline.py`** — shared stage-running machinery, used by both the CLI (`main()` in this file)
  and `debug_app.py`. `PipelineContext` accumulates state across stages for one page run (`page`,
  `native_words`, `vector_paths`, `paths_by_layer`, `paths_by_layer_color`, `filter_layout_panels`,
  `filter_large_bbox`, `clustering_order` (the chosen 4-step order, `None` = `Vector.CLUSTER_STEPS`
  default), `clustering`, `filter_large_group_bbox`, `filter_aspect_ratio`, `drawing_vectors`,
  `text_clusters`, `ocr_text_clusters`, and a field per future stage's output);
  `StageSpec(key, label, run)` is one stage; `Pipeline.STAGES` is the ordered list: `reader`,
  `native`, `vector_extract`, `layer_separation`, `color_separation`, `filter_layout_panels`,
  `filter_large_bbox`, `clustering`, `filter_large_group_bbox`, `filter_aspect_ratio`,
  `drawing_vectors`, `ocr_text_clusters` (12 stages total, mirroring `Vector.classify()`'s steps
  above plus the OCR step after it — the single `clustering` stage replaces what used to be 4
  separate cluster/group `StageSpec` entries, and there is no stage between the two filters and
  `drawing_vectors`, since `cluster_groups_by_dimension` is now just one of the 4 operations
  selectable *within* `clustering`, not a stage of its own).
  `Pipeline.run_page(reader, page_index, clustering_order=None, final_stage=None)` runs every stage
  in order, threading `clustering_order` onto `ctx` before `_run_clustering` executes, and wraps
  each stage in `try/except` so one stage failing/not-yet-being-implemented is recorded as a
  `StageOutput(status="error", error=...)` rather than crashing the run or any caller — this is what
  lets the debug app show "this stage failed" instead of dying. `final_stage`, if given, must be one
  of `Pipeline.stage_keys()` (raises `ValueError` otherwise) — the loop stops right after that
  stage's `StageOutput` is appended, so every later `StageSpec.run` is never even called (not merely
  hidden from the UI afterward). This is how both `pipeline.py`'s CLI (`--final-stage`) and
  `debug_app.py` (`--final-stage`, threaded through as `DebugApp.final_stage`, fixed for the app's
  whole lifetime rather than living in `DebugAppState`) let you skip `ocr_text_clusters` — and the
  PaddleOCR engine it would otherwise build — entirely while iterating on earlier stages.
  `filter_layout_panels`/`filter_large_bbox`/`filter_large_group_bbox`/`filter_aspect_ratio` each
  produce a `dict[GroupKey, VectorStageBuckets]` (`GroupKey = (layer, color)`); `VectorStageBuckets`
  splits that group's paths into `this_stage` (what *this* stage decided — singleton `[path]` groups
  for a path-level filter's drops, whole dropped groups for a group-level filter), `previous`
  (decided by an earlier stage in this same sequence — accumulated stage to stage), and `pending`
  (still undecided, flows into the next stage — a flat `list[VectorPath]` for the two path-level
  filters, a `list[list[VectorPath]]` of kept groups for the two group-level filters).
  `clustering` produces a `dict[GroupKey, ClusteringStageResult]` instead —
  `ClusteringStageResult(order, steps, previous)`, where `order` is the 4-step order actually used
  for that group, `steps` is `Vector.cluster()`'s list of per-step group-snapshots (`steps[i]` = the
  groups after applying `order[i]`), and `previous` carries forward every path dropped by
  `filter_layout_panels`/`filter_large_bbox` before clustering ran (so the debug app can still show
  "previously classified as Drawing" during the clustering stage). `_run_clustering` builds this by
  calling `vector.cluster(buckets.pending, ctx.page, ctx.clustering_order)` per `(layer, color)`
  group, reading `buckets.pending`/`.previous` off `ctx.filter_large_bbox`.
  `_run_filter_large_group_bbox`/`_run_filter_aspect_ratio` then read from `ctx.clustering[key].
  steps[-1]` (the final clustering snapshot) as their input groups, and `ctx.clustering[key].
  previous` as their starting `previous` bucket. `_run_drawing_vectors` is the one place that
  reconciles the "filtering out = classified as Drawing" rule: for every group, it folds the *entire*
  `previous` chain (every filter's drops, accumulated through `filter_layout_panels` →
  `filter_large_bbox` → clustering's carried-forward `previous` → `filter_large_group_bbox` →
  `filter_aspect_ratio`) into one `drawing_paths` list before calling `Vector.build_drawing_vectors`
  — whatever `filter_aspect_ratio`'s final round still has `pending` is *not* folded in here (there's
  no drawing-vs-text heuristic left in `Vector` to decide that): it's stashed as-is on
  `ctx.text_clusters` instead, for `ocr_text_clusters` to consume — every survivor becomes a text
  candidate, and OCR success/failure is what actually determines whether it was text.
  `_run_ocr_text_clusters` OCRs each of `ctx.text_clusters` via a fresh `Renderer()` + `RenderOCR()`
  (cheap to construct per run — the expensive part, PaddleOCR's engine, is cached at module scope in
  `helpers/render_ocr.py`, not per-`RenderOCR()`-instance) and stores the `list[TextVectorResult]`
  on `ctx.ocr_text_clusters`. Each cluster is a real multi-rotation OCR round-trip, so this loop is
  wrapped in a `tqdm` progress bar (`desc="OCR text clusters"`) — both the CLI and the debug app
  (which runs `Pipeline.run_page()` synchronously before its window becomes interactive) call this
  stage the same way, so the terminal bar is the only progress feedback available while it runs.
- **`debug_app.py`** — Tk GUI (`python -m rastervec.debug_app [pdf]`): page pixmap on a canvas, a
  page-nav bar (reusing the same `page.rotation_matrix * zoom_matrix` display-transform rule as
  `inspector/app.py`), and a stage-nav bar (`< Prev Stage | Stage i/N: Label | Next Stage >`) that
  cycles **only over `Pipeline.STAGES`** — i.e. only stages that are actually implemented, not the
  full 17-stage pipeline spec. Per-page `StageOutput` results are cached in `DebugAppState.
  stage_cache`, keyed by `(page_index, tuple(clustering_order))` rather than just `page_index` — a
  different chosen clustering order is a different pipeline run, so it gets its own cache slot
  (nothing is invalidated; flipping back to a previously-visited order is free after its first
  compute). Each stage's overlay is drawn by a small function registered in `_STAGE_RENDERERS`,
  keyed by `StageSpec.key`, taking a `RenderContext` (`canvas`, `matrix`, `output`, `tooltip`,
  `side_panel`, `filters`, `on_change`, `epoch_box`, plus `clustering_order`/`set_clustering_order`
  for the one stage that needs them). Vector paths are always drawn in their **real PDF stroke/fill
  color** (`_path_color_hex`) — any black/white simplification a classification step might use
  internally is never substituted into the overlay itself; synthetic colors (`_THIS_STAGE_COLOR`
  blue, `_PREVIOUS_COLOR` grey, `_PENDING_GROUP_COLOR` green, `_CENTROID_COLOR` red, plus one color
  per clustering step in `_CLUSTER_STEP_COLORS`) are only ever used for bbox/centroid *outlines*,
  never in place of a path's real color. Stages with filterable sub-groups (`vector_extract` by path
  kind, `layer_separation` by layer, `color_separation` by layer+color, `drawing_vectors` by
  dashed/solid) build `ttk.Checkbutton`s per sub-group into `ctx.side_panel`.

  The 4 remaining **filter** stages (`filter_layout_panels`, `filter_large_bbox`,
  `filter_large_group_bbox`, `filter_aspect_ratio`) share `_render_filter_stage_buckets`, driven by
  three checkboxes — "Classified as Drawing this round (N)", "Previously classified as Drawing (N)",
  "Not yet classified (N)" — toggling visibility of `VectorStageBuckets.this_stage`/`previous`/
  `pending` (never one checkbox per dropped item; a filter's drops are always drawing content, never
  text, hence the wording).

  The old separate cluster/group stages are gone; `clustering` is a single stage rendered by
  `_render_clustering_stage`, built around 4 `ttk.Combobox` dropdowns in `ctx.side_panel` — one per
  ordinal position (`_ORDINAL_LABELS`: "1st" … "4th") — each listing all 6 entries of
  `_CLUSTER_STEP_LABELS` (the 5 real operations — `cluster_spatial`, `cluster_spatial_union_find`,
  `cluster_by_seq`, `group_overlapping`, `cluster_groups_by_dimension` — plus `"none"`, to skip that
  ordinal position; `Vector.CLUSTER_STEPS`, the *default* order, is just one layer of
  `cluster_spatial` with the other 3 positions `"none"`). The 4 dropdowns always represent a valid
  permutation *of the real steps actually chosen*: reassigning one dropdown to a real step another
  dropdown already holds swaps the two dropdowns' values, but `"none"` is exempt from that swap (any
  number of dropdowns can be `"none"` at once, and any number of dropdowns can differ — only a real
  step already in use elsewhere triggers a swap). On any change,
  `DebugApp._set_clustering_order(order)` updates `DebugAppState.
  clustering_order` and re-triggers `_get_stage_outputs()` under the new cache key, then
  `redraw_overlay()`. Below the dropdowns, one collapsible section per ordinal position shows: a
  checkbox ("Clusters after Nth operation (count)") toggling that step's `ClusteringStageResult.
  steps[i]` groups on/off, a live stats line from `_cluster_size_stats_text` (count and
  min/median/mean/max member count, recomputed from `steps[i]` — so reordering the dropdowns updates
  every later step's stats immediately, since `steps[i]` for `i > 0` depends on the operations before
  it), and each visible group is drawn as a bbox outline plus centroid dot in that step's assigned
  `_CLUSTER_STEP_COLORS` entry. A final checkbox, "Previously classified as Drawing (N)", toggles
  `ClusteringStageResult.previous` (paths already dropped by `filter_layout_panels`/
  `filter_large_bbox`), same as the equivalent checkbox in the filter-stage renderer.

  Hover precision: a group only counts as hovered when the cursor is over an actual *member path's*
  own bbox, not merely anywhere inside the group's aggregate bbox (`_bind_bucket_hover`'s
  `_group_hit` does the cheap aggregate-bbox reject first, then confirms against each member's bbox),
  padded by a small screen-pixel tolerance (`_HOVER_TOLERANCE_PX = 5.0`, converted to page-space via
  `geometry.matrix_scale(ctx.matrix)`) so near-misses right at a bbox edge still register.
  `_bind_bucket_hover` takes a variadic `*category_groups: tuple[list[list[VectorPath]], dict]` —
  one `(groups, filter_state)` pair per visible category — so the same function serves both the
  2-category filter-stage renderer and the 5-category (4 clustering steps + previous) clustering-
  stage renderer without duplicating hit-testing logic.

  Both renderer families share the same viewport-culling + chunked-drawing machinery: these stages can have
  tens of thousands of clusters/paths, which was laggy even with tag-based visibility toggling
  (thousands of live canvas items). `_SpatialIndex` (a uniform grid over `(bbox, payload)` entries)
  is built once per category per stage-visit; only the subset whose bbox overlaps the current
  viewport (`_visible_page_rect`, page-space via the inverse display matrix, padded 20%) is ever
  drawn, redrawing the delta on scroll/resize (`_recull`, wired through `DebugApp.
  _on_viewport_changed` from the scrollbars' commands and the canvas's `<Configure>` binding) and
  streaming even that delta in via small `after_idle` chunks (`_RECULL_CHUNK_SIZE`) so no single draw
  blocks the UI. A per-category `generation` counter plus `ctx.epoch_box` (bumped once per
  `redraw_overlay()` call) let a deferred chunk callback detect it's been superseded by a newer
  stage/page/zoom change or scroll-triggered recull and stop instead of drawing ghost items — bumping
  the epoch, then deleting via a single tag-based `canvas.delete("overlay")`, are both required at
  the top of `redraw_overlay()` (an id-list snapshot would miss items a still-in-flight chunk adds
  after the snapshot). Hover (`_bind_bucket_hover`) is computed on demand against the full in-memory
  group bboxes regardless of what's currently drawn, capped at `_MAX_HOVER_HIGHLIGHT_PATHS` member
  paths highlighted and cleared via tag-based `canvas.delete("overlay_hover")`. Checkbox/toggle state
  is persisted per-stage in `DebugAppState.filter_state` (keyed by `StageSpec.key`).

  `ocr_text_clusters` gets its own renderer, `_render_ocr_text_clusters_stage` — text clusters per
  page are few compared to raw path counts, so it skips the viewport-culling/spatial-index
  machinery above and just draws each *visible* `TextVectorResult`'s bbox with a plain `tag_bind`
  hover (`_render_native_stage`'s pattern), showing the OCR'd text, confidence, `rotation_used`,
  and member-path count in the tooltip, plus a click binding to select that cluster for the
  inspector panel below. "Visible" is gated by two bottom-pinned checkboxes, "Show passed (N)"/
  "Show failed (N)" (`ctx.filters["show_passed"]`/`["show_failed"]`, default both on) — a cluster
  "passed" if `TextVectorResult.text.strip()` is non-empty (`_ocr_passed`); toggling either
  triggers a full `ctx.on_change()` redraw (not a tag-visibility flip like the filter stages,
  since it also changes which cluster is selectable), and the currently-selected cluster snaps to
  the first still-visible one if the toggle hides it.

  Clicking a cluster's bbox (or the app defaulting to cluster 0 the first time the stage is
  visited) opens an inspector section. Its Prev/Next nav row and Left/Right arrow-key binding
  (unbound again at the top of `redraw_overlay()`, alongside Motion/Leave, so a stale binding from
  a previous stage-visit can't fire) cycle between *clusters*, not rotations — rotation detail
  isn't something this panel surfaces; `TextVectorResult.rotation_used` is just displayed as a
  fact. `_ocr_cluster_preview` renders the selected cluster once, at the single rotation
  `result.rotation_used` (not all `n_rotations`), and re-OCRs just that one rendered image purely
  to recover the detected-text bbox in that image's own pixel space for the overlay — `text`/
  `confidence` are read directly off the already-computed `TextVectorResult`, never re-derived from
  that extra OCR call. One render + one OCR call per cluster (not the 8 `RenderOCR.ocr_cluster`
  itself tries and discards) is still a real PaddleOCR round-trip, so it's cached on
  `DebugAppState.ocr_detail_cache`, keyed by `(page_index, clustering_order, cluster_index)` — a
  separate dict from `filter_state` (which is keyed only by stage name, fine for cheap UI toggles,
  wrong for data that's only valid for one exact page/order/cluster) so switching pages or
  clustering orders never serves another page's cached render. The inspector shows: the render
  (`_ocr_preview_photo`, resized up to `_OCR_PREVIEW_MAX_SIDE` px, with the detected bbox corners
  drawn via `PIL.ImageDraw.polygon`), the Prev/Next/"Cluster i/N" nav row (wrapping around
  `visible_indices`, so hidden — filtered-out by the passed/failed checkboxes — clusters are
  skipped when cycling), and a read-only `tk.Text` with `text`/`confidence`/`rotation_used`/path
  count/pass-or-fail.

`scripts/rasterize_pdf.py` (outside `rastervec/`, a one-off utility not a pipeline stage): flattens
every page of a PDF to an image and rebuilds a pure-raster PDF from those images — used as test
input for the Raster stage, independent of the Vector stage.

`tests/` mirrors `rastervec/`'s layout; `tests/conftest.py`'s `synthetic_pdf_factory` builds small
in-memory PDFs via `fitz.open()`/`insert_text`/`set_rotation` — preferred over `references/*.pdf`
for unit tests since those are gitignored and give no exact expected values to assert against.

### Adding the next `rastervec` stage

Three things, all following the Reader/Native pattern:
1. Define the stage's dataclass(es) in `models.py` if not already forward-declared there, and
   replace the stub file's `NotImplementedError` bodies with real logic split into small private
   methods per sub-step (e.g. `_extract_x`/`_match_y`) so each is independently testable.
2. Add one `StageSpec` to `Pipeline.STAGES` in `pipeline.py` (a `_run_<stage>(ctx)` function that
   reads whatever `PipelineContext` fields it needs and stores its own result back onto `ctx`).
3. Add one entry to `debug_app.py`'s `_STAGE_RENDERERS` — a `_render_<stage>_stage(ctx: RenderContext)`
   function drawing that stage's overlay onto `ctx.canvas` (via `ctx.matrix`), and, if the stage has
   filterable sub-groups, building checkboxes into `ctx.side_panel` backed by `ctx.filters`.
Also add a `tests/rastervec/test_<stage>.py` using the synthetic PDF fixtures, and new third-party
dependencies (scikit-learn/scipy/torch/opencv-python/paddleocr) to `requirements.txt` only when the
stage that needs them is actually implemented.
