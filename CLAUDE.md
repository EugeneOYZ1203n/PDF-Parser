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
  currently Reader, Native, and Vector are implemented, Raster/Renderer remain interface-only stubs —
  see "rastervec architecture" below.

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt        # install deps
.venv/Scripts/python.exe -m inspector.app [path/to.pdf]             # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.pipeline --pdf PATH --page N  # run the extraction pipeline demo (CLI)
.venv/Scripts/python.exe -m rastervec.debug_app [path/to.pdf]       # run the pipeline debug app (GUI)
.venv/Scripts/python.exe -m pytest tests/ -v                         # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC DST --dpi 300  # flatten a PDF to pure raster
```

venv is **Python 3.12** (`py -3.12 -m venv .venv`), not 3.14 — later `rastervec` stages need
opencv-python and PaddleOCR/paddlepaddle, which don't ship Windows wheels for 3.14 yet.

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
  `filter_layout_panels` drops single-item `re`/`qu` drawings (page borders/title-block panels, which
  never share a `seq` with anything else). `filter_background_fill` drops paths sharing the page's
  dominant fill color when that color covers ≥30% of page area (a full-bleed background rect, not
  real content). `cluster_spatial`/`cluster_by_dimension`/`cluster_by_seq` are thin wrappers around
  the matching `Clustering` (`helpers/clustering.py`) methods, pre-bound to `VectorPath`'s
  `bbox`/`seq` and this `Vector` instance's thresholds; `classify_clusters(clusters) ->
  (drawing_paths, text_clusters)` runs `_looks_like_text` on each cluster (≥2 members, every member's
  bbox ≤`text_max_dim`, ≥`text_min_fill_fraction` filled) to bucket it as a text-candidate vs. a
  drawing path — this is how CAD "text-as-filled-vector-paths" gets separated from real drawing
  geometry before OCR (OCR itself is a later, not-yet-implemented stage). `classify(paths) ->
  (drawing_paths, text_clusters)` chains all four in sequence and is the one-call convenience API;
  the pipeline (`pipeline.py`) calls the four sub-steps individually instead, one per debug-app stage.
  `build_drawing_vectors(paths) -> list[DrawingVector]` re-aggregates same-`seq` paths back into one
  `DrawingVector` per original drawing (bbox union + first path's style), i.e. the unit a downstream
  renderer would draw. All thresholds (`spatial_threshold`, `dimension_tolerance`, `seq_max_gap`,
  `text_max_dim`, `text_min_fill_fraction`) are `Vector.__init__` params — tune per-PDF if a specific
  page's default classification looks wrong; there's no per-PDF auto-tuning.
- **`helpers/clustering.py` — `Clustering`** *(cluster_spatial/cluster_by_dimension/cluster_by_seq
  implemented, `cluster_hsv` still a stub)*: pure-Python (no scipy/sklearn) spatial hash grid +
  union-find for `cluster_spatial` (buckets items into grid cells sized by `threshold`, unions items
  in neighboring cells whose `geometry.rect_gap` ≤ `threshold`), then O(k²) pairwise union-find within
  each resulting group for `cluster_by_dimension` (relative width/height closeness) and a sorted-seq
  gap split for `cluster_by_seq`. Has safety caps (`_MAX_CELLS_PER_ITEM`, `_MAX_GROUP_SIZE_FOR_PAIRWISE`)
  so a huge unfiltered bbox or a very dense cluster degrades to "keep as one cluster" (logged) instead
  of hanging — verified against a 78k-path reference PDF in ~3.5s. Shared by `Vector.classify` now;
  `Raster.separate_by_color` (HSV pixel clustering, not yet implemented) is designed to reuse
  `cluster_hsv` once Raster is built.
- **`raster.py` / `renderer.py`, remaining `helpers/*.py`** *(interface stubs)*: full method
  signatures and docstrings exist, bodies `raise NotImplementedError`. `helpers/render_ocr.py`'s
  `RenderOCR` is designed to be shared between OCR'ing rendered vector-text clusters (from
  `Vector.classify`'s text candidates) and OCR'ing raster image regions once both are implemented.
- **`pipeline.py`** — shared stage-running machinery, used by both the CLI (`main()` in this file)
  and `debug_app.py`. `PipelineContext` accumulates state across stages for one page run (`page`,
  `native_words`, `vector_paths`, `paths_by_layer`, `paths_by_layer_color`, `filter_layout_panels`,
  `filter_background_fill`, `cluster_spatial`, `cluster_by_dimension`, `cluster_by_seq`,
  `drawing_vectors`, and a field per future stage's output); `StageSpec(key, label, run)` is one
  stage; `Pipeline.STAGES` is the ordered list (`reader`, `native`, `vector_extract`,
  `layer_separation`, `color_separation`, `filter_layout_panels`, `filter_background_fill`,
  `cluster_spatial`, `cluster_by_dimension`, `cluster_by_seq`, `drawing_vectors`);
  `Pipeline.run_page(reader, page_index)` runs every stage, wrapping each in `try/except` so one
  stage failing/not-yet-being-implemented is recorded as a `StageOutput(status="error", error=...)`
  rather than crashing the run or any caller — this is what lets the debug app show "this stage
  failed" instead of dying. The two filter stages and three cluster stages (spatial → dimension →
  seq) each get their own `StageSpec` rather than being folded into one "classification" stage, so
  the debug app can step through each decision individually, mirroring how the original pipeline
  spec described them. Each of these five stages' data is a `dict[GroupKey, VectorStageBuckets]`
  (`GroupKey = (layer, color)`); `VectorStageBuckets` splits that group's paths into `this_stage`
  (what *this* stage decided — singleton `[[path]]` groups for a filter's drops, real cluster
  groupings for a cluster stage), `previous` (decided by an earlier stage in this same 5-stage
  sequence — carried forward stage to stage), and `pending` (still undecided, flows into the next
  filter stage; always `[]` once clustering starts, since each cluster stage consumes its whole
  input in one pass). `drawing_vectors`'s runner calls `Vector.classify_clusters` on the final
  `cluster_by_seq` groupings itself (no separate "classification" field needed) to get the
  `drawing_paths` it aggregates.
- **`debug_app.py`** — Tk GUI (`python -m rastervec.debug_app [pdf]`): page pixmap on a canvas, a
  page-nav bar (reusing the same `page.rotation_matrix * zoom_matrix` display-transform rule as
  `inspector/app.py`), and a stage-nav bar (`< Prev Stage | Stage i/N: Label | Next Stage >`) that
  cycles **only over `Pipeline.STAGES`** — i.e. only stages that are actually implemented, not the
  full 17-stage pipeline spec. Per-page `StageOutput` results are cached (`DebugAppState.stage_cache`)
  so cycling stages/zoom doesn't re-run extraction. Each stage's overlay is drawn by a small function
  registered in `_STAGE_RENDERERS`, keyed by `StageSpec.key`, taking a `RenderContext` (`canvas`,
  `matrix`, `output`, `tooltip`, `side_panel`, `filters`, `on_change`, `hover_overlay_ids`). Vector
  paths are always drawn in their **real PDF stroke/fill color** (`_path_color_hex`) — any
  black/white simplification (e.g. `filter_background_fill`'s dominant-color heuristic) is purely
  internal to `Vector`'s classification logic and never substituted into the overlay itself.
  Stages with filterable sub-groups (`vector_extract` by path kind, `layer_separation` by layer,
  `color_separation` by layer+color, `drawing_vectors` by dashed/solid) build `ttk.Checkbutton`s per
  sub-group into `ctx.side_panel`. The five filter/cluster stages (`filter_layout_panels`,
  `filter_background_fill`, `cluster_spatial`, `cluster_by_dimension`, `cluster_by_seq`) share one
  renderer, `_render_vector_stage_buckets`, driven by exactly **three** checkboxes regardless of how
  many clusters/paths exist — "classified this stage", "previously classified", "not yet classified"
  — toggling visibility of the `VectorStageBuckets.this_stage`/`previous`/`pending` collections
  respectively (never one checkbox per cluster/path). `this_stage` groups draw a solid blue bbox,
  `previous` groups a dashed grey bbox, `pending` paths their real color with no grouping; hovering a
  bbox (`_draw_cluster_group`) shows its exact coordinates and count in the tooltip and highlights
  every member path (in that path's real color, at extra width) via transient `overlay_hover`-tagged
  canvas items collected in `ctx.hover_overlay_ids`, cleared on `<Leave>`. Checkbox/toggle state is
  persisted per-stage in `DebugAppState.filter_state` (keyed by `StageSpec.key`), calling
  `ctx.on_change()` (== `redraw_overlay`) on toggle to redraw with the new filter applied.

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
