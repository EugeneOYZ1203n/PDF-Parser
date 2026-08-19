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

  Classification is a single configurable chain of up to **8 pipeline steps**, run via `cluster()`.
  Every ordinal position independently picks any one of 4 **filter** steps or 7
  **clustering/grouping** steps, or `"none"` to skip that position — and any step may repeat at more
  than one position (there's no uniqueness constraint):
  - `filter_layout_panels` drops single-item `re`/`qu` drawings (page borders/title-block panels,
    which never share a `seq` with anything else — a real CAD text-as-vector-paths drawing is one
    `seq` with *many* glyph items, so this never touches real text). Recomputed dynamically from
    whatever paths are still present when this step actually runs in the chain, not the original
    full population.
  - `filter_large_bbox` drops paths whose own bbox covers more than `max_area_fraction` (override
    param; default `large_bbox_area_fraction`, 0.2) of the page — border/frame geometry caught by
    size instead of item-count. Same dynamic-recompute behavior as `filter_layout_panels`.
  - `filter_large_group_bbox` / `filter_aspect_ratio` — the group-level counterparts, dropping whole
    groups whose *aggregate* bbox is oversized (`max_area_fraction`, same param name/default as
    `filter_large_bbox`) or shaped like a long thin line/ruler (`max_aspect_ratio`, default 10.0) —
    real drawing content, never a text candidate.
  - `cluster_spatial` — high-tolerance (`threshold` override; default `spatial_threshold`, 8.0)
    single-linkage grouping by bbox gap, via `Clustering.cluster_spatial`. Since this method's
    signature takes a *flat* path list (not groups), `_apply_pipeline_step` re-flattens whatever
    grouping exists so far before re-clustering — always a from-scratch spatial pass at whatever
    ordinal position it runs.
  - `cluster_spatial_union_find` — same distance rule as `cluster_spatial` (bbox gap ≤ `threshold`,
    same grid-bucketed union-find), but applied to the *incoming groups themselves* (by each group's
    aggregate bbox via `geometry.union_bbox`) rather than the raw paths — reuses
    `Clustering.cluster_spatial` at the group level the same way `cluster_groups_by_dimension`
    reuses `Clustering.cluster_by_dimension`. Unlike `cluster_spatial`, this never
    flattens/re-derives from scratch: each incoming group is one atomic unit, so groups an earlier
    step already formed only ever get merged together here, never re-split.
  - `cluster_by_seq` — tighter pass within each incoming group, splitting by drawing
    sequence-number proximity (`max_gap` override; default `seq_max_gap`), via
    `Clustering.cluster_by_seq`.
  - `group_overlapping` merges members whose bboxes overlap OR are within a small gap `tolerance`
    override (default `max(0.5% of the page's smaller dimension, 3px)`, via
    `Vector.default_overlap_tolerance`) of each other; a member fully contained in (or equal to)
    another's bbox is left separate regardless of tolerance — via `Clustering.group_by_overlap`. A
    second param, `bbox_scope` (`"path"` default or `"cluster"`), picks what's actually compared:
    `"path"` flattens every path across all incoming groups into one pool first and compares by each
    path's own bbox (a genuine from-scratch merge, e.g. the strokes making up one glyph, often a
    pixel or two apart rather than truly touching — not scoped to whatever grouping already
    existed); `"cluster"` instead treats each incoming group as one atomic unit compared by its own
    aggregate bbox, mirroring `cluster_spatial_union_find`'s group-atomic approach but with the
    overlap/tolerance/containment rule instead of a flat gap threshold. `Clustering.group_by_overlap`
    itself only ever *splits* within whatever single group it's handed — it's `Vector.
    group_overlapping`'s `bbox_scope` branching that decides what that one group actually contains
    (`[flat_paths]` for `"path"`, `[groups]` for `"cluster"`) before calling it; without the `"path"`
    flattening, `group_overlapping` used to be a silent no-op whenever it ran on the initial
    per-path singleton groups (e.g. as the sole/first active step), since a group of size 1 can
    never be split further and the tolerance param had no visible effect.
  - `cluster_groups_by_dimension` merges groups whose *overall* bbox width/height are close
    (`tolerance` override; default `group_dimension_tolerance`, 0.35) — reuses
    `Clustering.cluster_by_dimension` at the group level (each incoming group, not each path, is one
    item to compare; `geometry.union_bbox` gives each group's aggregate bbox; the whole incoming
    `groups` list is wrapped as one outer group, `[groups]`, so this is a genuine merge across
    everything currently pending, same "wrap as one group" trick `group_overlapping`'s `"cluster"`
    scope reuses), then flattens each resulting super-group back to a flat path list.
  - `cluster_by_item_path_count` / `cluster_by_item_bbox` cluster by a path's **original item**
    (its `seq`, i.e. the drawing it was extracted from) rather than anything about its current
    cluster: `_compute_item_stats(paths)` computes, once per `cluster()` call from the full incoming
    path population (before any step runs, so it never depends on what an earlier step already did
    to the grouping), each `seq`'s path count and aggregate bbox. `cluster_by_item_path_count`
    flattens to one pool and splits by absolute gaps in sorted item-path-count (`max_gap` override;
    default `item_count_max_gap`, 2 — an absolute integer difference, more intuitive than a relative
    one for small counts) via `Clustering.cluster_by_seq` reused generically (its `get_seq` callable
    just needs to return an `int`, not a real sequence number). `cluster_by_item_bbox` flattens to
    one pool and merges by relative item-bbox width/height closeness (`tolerance` override; default
    `item_bbox_tolerance`, 0.35) via `Clustering.cluster_by_dimension`, same generic reuse.

  There is deliberately **no drawing-vs-text heuristic** anywhere in this chain — an earlier version
  had one (`classify_clusters`/`_looks_like_text`, judging a cluster "text" by member count + size +
  fill fraction) but it was removed: every group any filter step drops along the way is drawing
  content (`pipeline.py`'s per-stage wiring folds those drops into `drawing_vectors`, not `Vector`
  itself — see `pipeline.py` below), and everything that survives the whole chain is handed to OCR
  (`pipeline.py`'s `ocr_text_clusters` stage) as-is — OCR success/failure (did it read any text) is
  the actual signal for whether a cluster was text, not a pre-filter guess.

  `cluster(paths, page, order=None, step_params=None) -> tuple[kept_snapshots, dropped_snapshots]`
  runs the chain: `order` defaults to `PIPELINE_STEPS` — `("filter_layout_panels",
  "filter_large_bbox", "cluster_spatial", "none", "none", "none", "filter_large_group_bbox",
  "filter_aspect_ratio")`, reproducing the original fixed 5-step pipeline. Both returned lists have
  one entry per step: `kept_snapshots[i]` is the surviving groups after step `i`,
  `dropped_snapshots[i]` is only what step `i` itself dropped (always empty for a pure
  clustering/grouping step or `"none"` — only the 4 filter steps ever drop; a caller wanting the
  cumulative drop total up to step `i` sums `dropped_snapshots[0:i+1]`). `step_params`, if given,
  maps a step key to a dict of that step's own keyword overrides, keyed by step name (not ordinal
  position) so a step repeated at more than one position still shares one set of overrides across
  every position it appears at. Internally, `_apply_pipeline_step` dispatches each step key to its
  underlying method, returning `(kept, dropped)` — the 2 path-level filters use `_filter_step`
  (flattens current groups, calls the filter method on that flat list, reconstitutes kept groups
  with their still-kept members while each dropped path becomes its own singleton group) and the 2
  group-level filters use `_group_filter_step` (identity-compares the filter method's already-kept
  group objects against the incoming list to recover what was dropped) — both shared helpers, not
  duplicated per filter. `classify(paths, page, cluster_order=None) -> list[list[VectorPath]]` is a
  thin convenience wrapper returning just `cluster()`'s final kept snapshot, for callers that don't
  need the drop bookkeeping — `pipeline.py`'s own stage wiring calls `cluster()` directly instead, to
  keep every step's drops for the debug app and `drawing_vectors`. `build_drawing_vectors(paths) ->
  list[DrawingVector]` re-aggregates same-`seq` paths back into one `DrawingVector` per original
  drawing (bbox union + first path's style), i.e. the unit a downstream renderer would draw. All
  thresholds (`spatial_threshold`, `seq_max_gap`, `large_bbox_area_fraction`, `max_aspect_ratio`,
  `group_dimension_tolerance`, `item_count_max_gap`, `item_bbox_tolerance`) are `Vector.__init__`
  params — tune per-PDF if a specific page's default classification looks wrong; there's no per-PDF
  auto-tuning (the `group_overlapping` gap tolerance is the one threshold computed from the page's
  own dimensions instead, not a `Vector.__init__` param).

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across buckets**:
  `pipeline.py`'s `_iter_groups`/`_run_clustering` key the whole chain's work by `GroupKey = (layer,
  color)` (from `color_separation`'s output), and `Vector.cluster()` is only ever called with one
  bucket's paths at a time — two paths in different layers, or with different stroke/fill colors,
  are never spatially merged, sequence-merged, overlap-merged, dimension-merged, or item-merged
  together, regardless of how close they are on the page.
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
  `render_reconstructed_page(page_meta, *, native_words=None, drawing_vectors=None,
  ocr_results=None, zoom=1.0)` *(implemented, debug-app-only preview — not OCR input, not
  `evaluation.py`'s real reconstruction stage)*: redraws whatever elements are passed onto a fresh
  blank page sized/rotated to match `page_meta`, then rasterizes at `zoom` the same way
  `DebugApp.render()` rasterizes the real page pixmap, so the two are pixel-comparable at the same
  zoom. `drawing_vectors` are redrawn from each `DrawingVector`'s own real member `VectorPath`s
  (reusing the module-level `_draw_vector_path` helper `render_vector_cluster` also uses, factored
  out so both share the same stroke/fill/skip-blank-path logic — `render_vector_cluster` calls it
  with a translation offset into its isolated small canvas, this calls it with `dx=dy=0` since
  paths are already in the reconstruction's absolute page-space coordinates), never just their
  aggregate bbox. `native_words`/`ocr_results` are inserted as real text via `page.insert_text` —
  necessarily approximate: font family isn't preserved (always PyMuPDF's base14 `"helv"`). Rotation
  is exact at any angle: since `insert_text`'s own `rotate` param only accepts multiples of 90,
  rotation is applied instead via its `morph=(fixpoint, matrix)` param — `(origin, fitz.Matrix(1,
  1).prerotate(angle))`, PyMuPDF's mechanism for arbitrary-angle text (a `cm` transform applied
  before drawing) — a "does this look roughly right" preview, not a
  byte-accurate reconstruction. `native_words` position at each `TextWord.origin` if set, else the
  bbox's bottom-left corner as a baseline approximation; color unpacked from the packed-int
  `TextWord.color` via `_text_color`. `ocr_results` position at each `TextVectorResult.bbox`'s
  bottom-left with `fontsize` derived from the bbox height. A blank/whitespace-only `text` is
  skipped outright (never handed to `insert_text`, which can be finicky with empty strings).
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
  `native_words`, `vector_paths`, `paths_by_layer`, `paths_by_layer_color`, `clustering_order` (the
  chosen up-to-8-step order, `None` = `Vector.PIPELINE_STEPS` default; a step may repeat),
  `clustering_params` (`dict[str, dict[str, float | str]]` of per-step param overrides, keyed by
  step name not ordinal position — `None`/missing entries fall back to each `Vector` method's own
  instance default; almost always numeric, except `group_overlapping`'s `bbox_scope`, a string),
  `clustering`, `drawing_vectors`, `text_clusters`, `ocr_text_clusters`, and a field per future
  stage's output); `StageSpec(key, label, run)` is one stage; `Pipeline.STAGES` is the ordered list:
  `reader`, `native`, `vector_extract`, `layer_separation`, `color_separation`, `clustering`,
  `drawing_vectors`, `ocr_text_clusters` (8 stages total -- the single `clustering` stage now
  encompasses the entire filter/cluster chain, including what used to be 4 separate
  `filter_layout_panels`/`filter_large_bbox`/`filter_large_group_bbox`/`filter_aspect_ratio`
  `StageSpec` entries -- there's nothing between `color_separation` and `clustering`, or between
  `clustering` and `drawing_vectors`, since every filter/cluster step is now just one of the up to 8
  operations selectable within `clustering`).
  `Pipeline.run_page(reader, page_index, clustering_order=None, clustering_params=None,
  final_stage=None)` runs every stage in order, threading `clustering_order`/`clustering_params`
  onto `ctx` before `_run_clustering` executes, and wraps
  each stage in `try/except` so one stage failing/not-yet-being-implemented is recorded as a
  `StageOutput(status="error", error=...)` rather than crashing the run or any caller — this is what
  lets the debug app show "this stage failed" instead of dying. `final_stage`, if given, must be one
  of `Pipeline.stage_keys()` (raises `ValueError` otherwise) — the loop stops right after that
  stage's `StageOutput` is appended, so every later `StageSpec.run` is never even called (not merely
  hidden from the UI afterward). This is how both `pipeline.py`'s CLI (`--final-stage`) and
  `debug_app.py` (`--final-stage`, threaded through as `DebugApp.final_stage`, fixed for the app's
  whole lifetime rather than living in `DebugAppState`) let you skip `ocr_text_clusters` — and the
  PaddleOCR engine it would otherwise build — entirely while iterating on earlier stages.
  `clustering` produces a `dict[GroupKey, ClusteringStageResult]` (`GroupKey = (layer, color)`) --
  `ClusteringStageResult(order, steps, dropped)`, where `order` is the (up to 8-step) order actually
  used for that group, `steps` is `Vector.cluster()`'s `kept_snapshots` (`steps[i]` = the surviving
  groups after applying `order[i]`, so `steps[-1]` is the final result), and `dropped` is `Vector.
  cluster()`'s `dropped_snapshots` (`dropped[i]` = only what step `i` itself dropped -- always empty
  for a pure clustering step or `"none"`, since every dropped group was "classified as Drawing" the
  moment a filter step dropped it; a caller wanting the cumulative drop total up to step `i` sums
  `dropped[0:i+1]`). `_run_clustering` builds this by calling `vector.cluster(paths, ctx.page,
  order, step_params=ctx.clustering_params)` directly per `(layer, color)` group from
  `_iter_groups(ctx.paths_by_layer_color)` -- there's no intermediate filter-stage bucket to read
  from any more, since filtering is just steps inside this one call now. `_run_drawing_vectors` is
  the one place that reconciles the "filtering out = classified as Drawing" rule: for every group's
  `ClusteringStageResult`, it folds every entry of `dropped` (all 8 steps' drops, whichever were
  actually filters) into one `drawing_paths` list before calling `Vector.build_drawing_vectors` --
  whatever `steps[-1]` (the chain's final kept groups) still holds is *not* folded in here (there's
  no drawing-vs-text heuristic left in `Vector` to decide that): it's stashed as-is on
  `ctx.text_clusters` instead, for `ocr_text_clusters` to consume -- every survivor becomes a text
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
  stage_cache`, keyed by `(page_index, tuple(clustering_order), _params_cache_key(clustering_params))`
  rather than just `page_index` — a different chosen clustering order or a different param override
  is a different pipeline run, so each gets its own cache slot (nothing is invalidated; flipping
  back to a previously-visited order/params combo is free after its first compute).
  `_params_cache_key(params)` flattens the nested dict into a sorted, hashable tuple of
  `(step, tuple(sorted(kv.items())))` pairs so it can be a dict key. Each stage's overlay is drawn by
  a small function registered in `_STAGE_RENDERERS`, keyed by `StageSpec.key`, taking a
  `RenderContext` (`canvas`, `matrix`, `output`, `tooltip`, `side_panel`, `filters`, `on_change`,
  `epoch_box`, `page`, plus `clustering_order`/`set_clustering_order` and `clustering_params`/
  `set_clustering_params` for the one stage that needs them). Vector paths are always drawn in their
  **real PDF stroke/fill
  color** (`_path_color_hex`) -- any black/white simplification a classification step might use
  internally is never substituted into the overlay itself; synthetic colors (`_DROPPED_COLOR` grey,
  `_CENTROID_COLOR` red, plus one color per clustering step in `_CLUSTER_STEP_COLORS`) are only ever
  used for bbox/centroid *outlines*, never in place of a path's real color. Stages with filterable
  sub-groups (`vector_extract` by path kind, `layer_separation` by layer, `color_separation` by
  layer+color, `drawing_vectors` by dashed/solid) build `ttk.Checkbutton`s per sub-group into
  `ctx.side_panel`.

  The 4 filter steps no longer have separate stages of their own -- `clustering` is a single stage
  rendered by `_render_clustering_stage`, built around **8** `ttk.Combobox` dropdowns in
  `ctx.side_panel`, one per ordinal position (`_ORDINAL_LABELS`: "1st" ... "8th") -- each listing
  all 12 entries of `_CLUSTER_STEP_LABELS`: the 4 filter steps (`filter_layout_panels`,
  `filter_large_bbox`, `filter_large_group_bbox`, `filter_aspect_ratio`), the 7 clustering/grouping
  operations (`cluster_spatial`, `cluster_spatial_union_find`, `cluster_by_seq`, `group_overlapping`,
  `cluster_groups_by_dimension`, `cluster_by_item_path_count`, `cluster_by_item_bbox`), plus
  `"none"` to skip that ordinal position (`Vector.PIPELINE_STEPS`, the *default* order, reproduces
  the original fixed pipeline: both filters, one layer of `cluster_spatial`, then both group
  filters). Every dropdown is **fully independent** -- unlike the earlier 4-dropdown design, there
  is no swap-on-conflict/uniqueness constraint any more: a step may be selected at more than one
  ordinal position (e.g. running `cluster_spatial` twice), so `_apply_order_change` just reads all 8
  dropdowns' current values straight into the new order and calls `ctx.set_clustering_order`, which
  persists it and re-triggers `_get_stage_outputs()` under the new cache key, then `redraw_overlay()`.

  Directly below each dropdown, `_render_param_rows` builds one row per tunable param that step
  accepts, from the `_CLUSTER_STEP_PARAMS` registry (`(param_name, display_label, kind)` tuples,
  keyed by step -- `"none"`/`filter_layout_panels` have none). `kind` is `"float"`/`"int"` for a
  plain `ttk.Entry` (parsed on commit via `_parse_param_text`; an unparseable value is silently
  ignored -- the entry just keeps showing what the user typed, uncommitted, rather than raising or
  reverting) or `"choice:opt1,opt2"` for a readonly `ttk.Combobox` of those exact string values
  instead (used only for `group_overlapping`'s `bbox_scope`, committed immediately on selection, no
  parsing needed). Each row pre-fills with the current override from `DebugAppState.
  clustering_params` if set, else the step's real default via `_cluster_step_param_default` (mirrors
  each `Vector` method's own instance-attribute default exactly -- `group_overlapping`'s tolerance is
  the one page-dependent default, via `Vector.default_overlap_tolerance`; its `bbox_scope` default is
  just `Vector.group_overlapping`'s own default argument value, `"path"`, not an instance attribute).
  A valid commit updates `DebugAppState.clustering_params[step][param]` via `DebugApp.
  _set_clustering_params` and triggers a full `redraw_overlay()`, which rebuilds the whole side panel
  from scratch and re-runs `_get_stage_outputs()` -- cached per `(page_index, clustering_order,
  clustering_params)` via `_params_cache_key` (keyed by step name, not ordinal position, so
  `cluster_spatial` and `cluster_spatial_union_find` -- which share `Vector.spatial_threshold` as
  their un-overridden default -- can still hold independent overrides if both appear in the same
  order, and a step repeated at more than one position shares one set of overrides across every
  position it appears at). This flows all the way down to `Vector.cluster(..., step_params=...)` ->
  `_apply_pipeline_step(..., params=...)`, which passes each override as a keyword arg to the
  underlying `Vector` method.

  Below all 8 dropdown+params blocks, a separator, then **two** checkboxes per ordinal position: "N:
  Step kept (count)" toggles that step's `ClusteringStageResult.steps[i]` (its surviving groups, bbox
  + centroid, in that step's assigned `_CLUSTER_STEP_COLORS` entry) plus a live stats line from
  `_cluster_size_stats_text` (count and min/median/mean/max member count, recomputed from `steps[i]`
  -- so reordering the dropdowns updates every later step's stats immediately, since `steps[i]` for
  `i > 0` depends on the operations before it), and "N: Step dropped (count)" toggles that step's
  `ClusteringStageResult.dropped[i]` (only ever non-empty for a filter step -- always 0 for a pure
  clustering step or `"none"` -- drawn grey, `_DROPPED_COLOR`, no centroid). There's no longer a
  single cumulative "previously classified as Drawing" checkbox -- each step's own drops are shown
  separately now that filters can sit anywhere in the chain; toggling more than one step's "dropped"
  checkbox at once reconstructs whatever cumulative view is needed.

  The side panel itself is now **scrollable**: `ctx.side_panel` (a plain `ttk.Frame`, unchanged from
  every renderer's point of view) lives inside a `tk.Canvas` + `ttk.Scrollbar` wrapper built once in
  `DebugApp._build_layout` (`self.side_panel_canvas`, holding `self.side_panel` via `create_window`)
  -- necessary once the clustering stage alone can pack in 8 dropdown+params blocks plus up to 16
  kept/dropped checkboxes, comfortably exceeding the panel's fixed height. The inner frame's
  `<Configure>` updates the canvas's `scrollregion`; the canvas's own `<Configure>` keeps the inner
  frame exactly as wide as the visible canvas so its children can still fill/wrap horizontally, only
  the height scrolls. Mouse-wheel scrolling is bound globally only while the cursor is actually over
  the side panel (bound in `<Enter>`, unbound in `<Leave>`) so it doesn't hijack scrolling elsewhere
  in the app. `redraw_overlay()` resets the scroll position to the top (`yview_moveto(0.0)`)
  whenever it rebuilds the panel for a newly-visited stage, so switching stages never leaves you
  scrolled partway down unrelated content.

  Hover precision: a group only counts as hovered when the cursor is over an actual *member path's*
  own bbox, not merely anywhere inside the group's aggregate bbox (`_bind_bucket_hover`'s
  `_group_hit` does the cheap aggregate-bbox reject first, then confirms against each member's bbox),
  padded by a small screen-pixel tolerance (`_HOVER_TOLERANCE_PX = 5.0`, converted to page-space via
  `geometry.matrix_scale(ctx.matrix)`) so near-misses right at a bbox edge still register.
  `_bind_bucket_hover` takes a variadic `*category_groups: tuple[list[list[VectorPath]], dict]` --
  one `(groups, filter_state)` pair per visible category -- so the same function serves the
  clustering stage's up to 16 categories (kept + dropped per each of up to 8 steps) without
  duplicating hit-testing logic.

  This renderer shares the same viewport-culling + chunked-drawing machinery other high-volume
  stages use: these stages can have tens of thousands of clusters/paths, which was laggy even with
  tag-based visibility toggling (thousands of live canvas items). `_SpatialIndex` (a uniform grid
  over `(bbox, payload)` entries)
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

  **Reconstructed-PDF toggle**: `native`, `drawing_vectors`, and `ocr_text_clusters` each open their
  renderer with a call to the shared `_add_reconstruction_toggle(ctx, build_image_fn)`, which packs
  a "Show Reconstructed PDF"/"Show Original PDF" `ttk.Button` at the top of `ctx.side_panel` (state
  persisted per-stage in `ctx.filters["show_reconstructed"]`, same mechanism as every other
  checkbox here, so leaving and returning to a stage remembers whether it was showing). Clicking it
  flips the state and calls `ctx.on_change()` (a full `redraw_overlay()`); when the state is on,
  `_add_reconstruction_toggle` calls the caller-supplied zero-arg `build_image_fn()` (each stage's
  own small wrapper around `Renderer.render_reconstructed_page`, passing just that stage's own
  captured elements — `native_words` for `native`, `drawing_vectors` for `drawing_vectors`,
  `ocr_results=passed` — only the OCR'd-successfully results, via `_ocr_passed` — for
  `ocr_text_clusters`; `zoom` is recovered from `geometry.matrix_scale(ctx.matrix)[0]`, since the
  reconstruction's own `page.get_pixmap()` call handles its own rotation internally and only needs
  the plain zoom magnitude, not the combined rotation+zoom `ctx.matrix`) and draws the resulting PIL
  image as one `tk.PhotoImage`-backed canvas item tagged `"overlay"` — drawn *on top of* the real
  page pixmap (never hidden or replaced) rather than swapped in for it, since the reconstruction
  covers the exact same rect at the exact same zoom and so fully occludes the original without
  risking the base `"page_image"` canvas item's separate lifecycle (that item is only ever
  (re)created by `DebugApp.render()` on page/zoom changes, not on every stage switch — deleting/
  replacing it from a stage renderer would leave the canvas blank after leaving that stage).
  `_add_reconstruction_toggle` returns whether reconstruction is currently showing; each of the 3
  stage renderers checks this immediately after building the toggle and `return`s early when it's
  `True`, skipping their normal overlay entirely — the toggle replaces the overlay, it doesn't layer
  under it, so the reconstructed image is the only thing drawn. The toggle is skipped altogether
  (no button appears) if `ctx.page` is `None`, since reconstruction needs `ctx.page.meta`.

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
