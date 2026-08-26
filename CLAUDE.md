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
  (`Renderer.render_vector_cluster` + `RenderOCR`, via a pluggable `OcrBackend` — PaddleOCR or
  Tesseract, see `helpers/ocr_backend.py`) — the Raster stage and `Renderer.render_raster_region`
  remain interface-only stubs — see "rastervec architecture" below.

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt        # install deps
.venv/Scripts/python.exe -m inspector.app [path/to.pdf]             # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.pipeline --pdf PATH --page N  # run the extraction pipeline demo (CLI)
.venv/Scripts/python.exe -m rastervec.debug_app [path/to.pdf]       # run the pipeline debug app (GUI)
.venv/Scripts/python.exe -m pytest tests/ -v                         # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC DST --dpi 300  # flatten a PDF to pure raster
powershell -ExecutionPolicy Bypass -File scripts/setup_tesseract.ps1  # install Tesseract OCR engine binary (Windows, via winget)
bash scripts/setup_tesseract.sh                                      # install Tesseract OCR engine binary (Linux/WSL, via apt)
```

venv is **Python 3.12** (`py -3.12 -m venv .venv`), not 3.14 — `rastervec`'s OCR (paddleocr/
paddlepaddle) and the still-unbuilt Raster stage (opencv-python) don't ship Windows wheels for 3.14
yet. On this dev machine's paddlepaddle build, the default mkldnn-accelerated CPU inference path
hits an unimplemented PIR attribute-conversion error, so `rastervec/helpers/ocr_backend.py` sets
`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` at import time (before paddleocr/paddlex read their
flags) to force the plain "paddle" run mode instead — fine for OCR's small, pre-cropped cluster
renders. If a future paddlepaddle release fixes this, that env-var default can be dropped.

`TesseractOcrBackend` (`helpers/ocr_backend.py`, an alternative to `pipeline.py`'s active
`RenderOCR()`/`PaddleOcrBackend()` default — pass `backend=TesseractOcrBackend()` for word-level
instead of Paddle's line-level detection) needs the actual Tesseract OCR **engine binary** installed separately — `pytesseract` (in
`requirements.txt`) is only a thin wrapper around a `tesseract`/`tesseract.exe` it shells out to,
and does not ship the binary itself. `_resolve_tesseract_cmd()` finds it automatically across both
target environments (this project's Windows dev machine and a Linux/WSL box) with no per-machine
setup needed in the common case: `RASTERVEC_TESSERACT_CMD` env var first if set, then a `PATH`
lookup (covers `apt install tesseract-ocr` on Linux/WSL, which already puts it on `PATH`), then a
handful of common Windows install paths (`C:\Program Files\Tesseract-OCR\tesseract.exe` etc., for
an installer like UB-Mannheim's that doesn't add itself to `PATH`) as a last resort. Only set
`RASTERVEC_TESSERACT_CMD` yourself for a non-standard install location none of that finds. Without
the binary present anywhere, constructing a `TesseractOcrBackend` and calling `.detect()` raises
`pytesseract.TesseractNotFoundError`.

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

See `Glossary.md` (repo root) for standardized group/cluster/global-group/similarity-group
terminology used throughout this section.

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

  Classification is a **single fixed, non-configurable 12-step pipeline**, run in order by
  `cluster()` (each step implemented as a plain function in `helpers/vector_classification.py`; see
  that module's docstring for the exhaustive per-step description):
  1. `filter_large_items` — drop items whose own bbox's max dimension exceeds a fraction of the
     page's smaller side (border/frame geometry).
  2. `compute_vector_signatures` — informational: per-signature occurrence counts, reused below.
  3. `remove_duplicate_runs` + `combine_overlapping_seq` — drop long runs of exact-duplicate shapes,
     then chain-merge the rest by `seq` order into "groups" (see Glossary.md).
  4. `filter_tiny_groups` / 5. `filter_large_groups` — drop undersized/oversized groups.
  6. `cluster_spatial_groups` — single-linkage spatial merge of groups into clusters, constrained to
     groups sharing a similar-length parallel side; also tracks `lineage` (which groups compose each
     cluster) for every later step and for `StepResult.cluster_groups`.
  7. `filter_mixed_fill_rule_clusters` — drop clusters mixing fill/stroke paint styles.
  8. `compute_group_stats` — informational per-cluster stats (member/signature counts, bbox).
  9. `filter_perimeter_only_clusters` — drop border/ring-only clusters.
  10. `filter_density_clusters` — drop clusters too sparse across their own bbox grid.
  11. `filter_constant_spacing_clusters` — drop clusters where most members belong to a
      near-perfectly-regular repeated same-shape sub-group (hatching, tick marks).
  12. `filter_low_variety_clusters` — drop clusters below a member-count-scaled minimum
      distinct-shape-type count.

  There is deliberately **no drawing-vs-text heuristic** anywhere in this chain — every group/cluster
  any filter step drops along the way is drawing content (`pipeline.py`'s `_run_drawing_vectors`
  folds every `role="dropped"` category into `drawing_vectors`), and everything that survives the
  whole chain is a *text candidate*, handed to `unique_clusters`/`fast_text_detect`/`ocr_compare` —
  OCR success/failure is the actual signal for whether a cluster was text, not a pre-filter guess.

  `cluster(paths, page) -> list[StepResult]` runs the fixed chain in order; each `StepResult` holds
  every named `CategoryResult` that step produced (`role="kept"` always present and fed to the next
  step; `role="dropped"` is a side channel folded into `drawing_vectors`; `role="info"` is
  display-only). `steps[-1].categories["kept"]` is the final surviving clusters. The last step's
  `StepResult.cluster_groups` (keyed by `id(cluster)`) records which of step 6's pre-spatial "groups"
  each survivor is composed of, via the `lineage` dict step 6 builds internally.
  `classify(paths, page) -> list[list[VectorPath]]` is a thin convenience wrapper returning just the
  final kept groups. `build_drawing_vectors(paths) -> list[DrawingVector]` re-aggregates same-`seq`
  paths back into one `DrawingVector` per original drawing. `group_similar_clusters(clusters) ->
  list[list[list[VectorPath]]]` groups text-candidate clusters by whole-page geometric similarity
  (see "similarity group" in Glossary.md and the `unique_clusters` pipeline stage below). All
  thresholds (`MAX_DIMENSION_FRACTION`, `SPATIAL_CLUSTER_THRESHOLD`, `SPATIAL_SIZE_TOLERANCE`,
  `PERIMETER_MARGIN_FRACTION`, `DENSITY_*`, `PATTERN_*`, `LOW_VARIETY_*`, `UNIQUE_CLUSTER_TOLERANCE`)
  are module-level constants in `vector.py` — tune per-PDF if a specific page's default
  classification looks wrong; there's no runtime/UI way to change them or the step order.

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across buckets**:
  `pipeline.py`'s `_iter_groups`/`_run_clustering` key the whole chain's work by `GroupKey = (layer,
  color)` (from `color_separation`'s output), and `Vector.cluster()` is only ever called with one
  bucket's paths at a time — two paths in different layers, or with different stroke/fill colors,
  are never spatially merged together, regardless of how close they are on the page.
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
  substituted into a rendered/displayed color. `render_vector_cluster(paths, dpi)` *(implemented)*
  isolates a cluster onto a fresh single-page PyMuPDF document sized to the cluster's own bbox (plus
  padding — `max(4pt, largest member's stroke_width)`, computed by the private `_cluster_frame`
  helper — so edge strokes aren't clipped), redraws each path with its own real
  stroke/fill/width/dashes via `fitz.Shape`, then rasterizes at `dpi` and returns a PIL `Image` —
  reusing PyMuPDF's own path/curve/fill rendering rather than reimplementing rasterization by hand.
  A path with neither `stroke_color` nor `fill_color` set is skipped outright (never handed to
  `Shape.finish()`): `finish()` emits a stroke operator whenever `fill` is `None` regardless of
  `color`, falling back to the current graphics-state color (default black) instead of staying
  invisible, so skipping is the only way to keep a genuinely colorless path from getting an unwanted
  black stroke in the render. `pixel_to_page_bbox(paths, dpi, pixel_points)` inverts
  `_cluster_frame`'s same transform to map pixel-space points (e.g. Paddle's detected text-region
  corners, from a render of that exact `paths`/`dpi`) back into PDF page space — used by
  `RenderOCR.ocr_cluster` to compute a `TextVectorResult.ocr_bbox`. `render_page_paths(paths,
  page_meta, dpi)` is the whole-page counterpart (every given path drawn onto one page-sized canvas,
  no isolation/padding, no rotation applied) — used as FAST's own detection input by
  `fast_text_detect` (see below). `render_raster_region` (raster-image OCR input) is still a stub.
  `render_reconstructed_page(page_meta, *, native_words=None, drawing_vectors=None,
  ocr_results=None, zoom=1.0)` *(implemented, debug-app-only preview — not OCR input, not
  `evaluation.py`'s real reconstruction stage)*: redraws whatever elements are passed onto a fresh
  blank page sized/rotated to match `page_meta`, then rasterizes at `zoom` the same way
  `DebugApp.render()` rasterizes the real page pixmap, so the two are pixel-comparable at the same
  zoom. `drawing_vectors` are redrawn from each `DrawingVector`'s own real member `VectorPath`s
  (reusing the module-level `_draw_vector_path` helper `render_vector_cluster` also uses), never
  just their aggregate bbox. `native_words`/`ocr_results` are inserted as real text via
  `page.insert_text` — necessarily approximate: font family isn't preserved (always PyMuPDF's
  base14 `"helv"`). Font size and baseline are derived from `fitz.Font("helv")`'s own
  ascender/descender metrics rather than treating the bbox height as the fontsize and the bbox's
  bottom edge as the baseline outright (a font's em-square is taller than its rendered bbox, and the
  baseline sits `ascender * fontsize` below the bbox's *top* edge, not at its bottom): `fontsize =
  (bbox_height) / (ascender - descender)`, `baseline_y = bbox_top + ascender * fontsize`. For
  `ocr_results` specifically, placement is per-word when `TextVectorResult.words` is populated
  (one `_place_text` call per `OcrWord`, each scaled/baselined into its own bbox instead of one
  string stretched across the whole cluster bbox — meaningful with Tesseract's word-level
  detection; falls back to the single-bbox `result.text`/`result.bbox` path when `words` is
  `None`/empty, e.g. Paddle's line-level boxes or `native_words`, which has no per-word concept).
  Either way, that height-derived fontsize is then shrunk further if needed so the
  text actually fits the bbox it was read from *widthwise* too, via `fitz.Font.text_length(text,
  fontsize)` against `bbox_width` -- the height-only fontsize can otherwise overflow a narrow
  cluster/group bbox for a long OCR'd string. Rotation is
  exact at any angle: since `insert_text`'s own `rotate` param only accepts multiples of 90, rotation
  is applied instead via its `morph=(fixpoint, matrix)` param — `(bbox_center, fitz.Matrix(1,
  1).prerotate(angle))`, PyMuPDF's mechanism for arbitrary-angle text (a `cm` transform applied
  before drawing). The fixpoint is the bbox's own center, not the baseline origin — using origin as
  the fixpoint (an earlier version of this code did) rotates the text around its own left edge
  instead of turning it in place, drifting visibly off the bbox at any non-zero angle — a "does this
  look roughly right" preview, not a byte-accurate reconstruction. A
  blank/whitespace-only `text` is skipped outright (never handed to `insert_text`, which can be
  finicky with empty strings).
- **`helpers/ocr_backend.py`** *(implemented)*: the strategy-pattern split behind OCR engine choice
  — `OcrBackend` (Protocol: `detect(image) -> OcrDetection`), `OcrBox` (one detected text box:
  `text`/`confidence`/`corners`/`is_word`, always already mapped into the *caller's* original image
  pixel space — no engine-specific rotation quirks leak past this module), `OcrDetection` (`boxes` +
  page-level `rotation`). Two concrete backends:
  - `PaddleOcrBackend` *(the original/default engine)*: PP-OCRv6 via PaddleOCR, orientation
    classifiers on (`use_doc_orientation_classify=use_textline_orientation=True,
    use_doc_unwarping=False`), engine lazily built and cached per `lang` at class scope
    (`_ENGINE_CACHE`). Detected boxes are **line/region-level**, never word-level
    (`OcrBox.is_word=False`). `_page_rotation(page)` combines Paddle's own `doc_preprocessor_res
    .angle` (0/90/180/270 document-orientation classification) with the majority vote of
    `textline_orientation_angles` (0/180 per-line flip correction). **Confirmed bug fix**: when
    `use_doc_orientation_classify=True`, PaddleOCR's own doc-preprocessor sub-pipeline actually
    rotates the input image by `doc_preprocessor_res.angle` degrees *before* running text detection
    (confirmed by reading paddlex's `doc_preprocessor/pipeline.py`/`ocr/pipeline.py` source), so
    every `rec_poly` it returns is in that rotated image's own pixel space (size-swapped for
    90/270) — not the space of the image actually passed in; `detect()` corrects every corner back
    via `_undo_doc_rotation` (exact per-point inverse of `cv2.getRotationMatrix2D`'s rotation,
    derived for each of the 4 possible angles) before returning.
  - `TesseractOcrBackend` *(alternative engine, not active by default — see pipeline.py below)*: via
    `pytesseract`, **word-level** detection (`image_to_data`, one row per word,
    `OcrBox.is_word=True` — aggregate line/block/paragraph rows, which carry `conf == -1`, are
    filtered out). Rotation comes from `image_to_osd`, wrapped in `try/except
    pytesseract.TesseractError` since OSD needs enough text to classify orientation and routinely
    fails on the small, sparse cluster crops this only ever sees — falls back to `rotation=0`
    rather than raising. Needs the actual `tesseract.exe` binary installed separately (see
    Commands section above); constructing one and calling `.detect()` without it raises
    `pytesseract.TesseractNotFoundError`.
- **`helpers/render_ocr.py` — `RenderOCR`** *(implemented)*: render + detect, backend-agnostic —
  shared by the Vector stage (OCR'ing rendered vector-text clusters, the only case actually
  reachable today) and, once built, the Raster stage (OCR'ing raster image regions — `ocr_cluster`
  already branches on `list[VectorPath]` vs. `RasterImage`, but the raster branch hits `Renderer.
  render_raster_region`'s `NotImplementedError` until that stage exists). `__init__(self, backend:
  OcrBackend | None = None)` defaults to `PaddleOcrBackend()`. `ocr_boxes(image)`/`ocr(image)` wrap
  `self.backend.detect(image)` (`ocr` joins every detected box left-to-right into one `(text,
  confidence, bbox_corners)` result, used by the debug app's OCR inspector).
  `ocr_cluster(cluster, page, renderer, dpi=300)` is the shared entrypoint: render the cluster
  **once**, upright, run **one** `backend.detect()` call, and build a `TextVectorResult` whose
  `rotation_used` comes from `OcrDetection.rotation`, whose `ocr_bbox` (union of every detected
  box, mapped back to page space via `Renderer.pixel_to_page_bbox`) is `None` when nothing was
  detected or the input was a `RasterImage`, and whose new `words: list[OcrWord] | None` field
  (models.py) is one `OcrWord` per individually detected box, each independently mapped through
  `pixel_to_page_bbox` — line-granularity for Paddle, word-granularity for Tesseract; `None` when
  nothing was detected or the input wasn't a vector-path cluster. `pipeline.py`'s
  `_run_ocr_compare` merges `.words` across joined group readings the same way it already joins
  `.text` (left-to-right by bbox) when building a fallback `resolved` reading.
- **`evaluation.py` — `Evaluation`** *(interface stub)*: the pipeline's actual intended final stage
  — `reconstruct_page`/`build_pdf` reassemble consolidated text/line objects back into a PDF for
  evaluation — not yet registered in `Pipeline.STAGES` since not implemented.
- **`pipeline.py`** — shared stage-running machinery, used by both the CLI (`main()` in this file)
  and `debug_app.py`. `PipelineContext` accumulates state across stages for one page run: `page`,
  `native_words`, `vector_paths`, `paths_by_layer`, `paths_by_layer_color`, `clustering` (`dict
  [GroupKey, ClusteringStageResult]`, `GroupKey = (layer, color)`; `ClusteringStageResult.steps` is
  exactly `Vector.cluster()`'s return value), `text_clusters` + `cluster_groups` (every bucket's
  final "kept" clusters flattened, plus their merged group lineage), `similarity_groups` +
  `cluster_similarity_id` (whole-page similarity grouping, see "similarity group" in Glossary.md),
  `fast_result` + `fast_passed`/`fast_dropped`, `ocr_comparisons` + `ocr_results` + `ocr_failed`,
  `rotation_checks`, `drawing_vectors`. `StageSpec(key, label, run)` is one stage; `Pipeline.STAGES`
  is the ordered list: `reader`, `native`, `vector_extract`, `layer_separation`, `color_separation`,
  `clustering`, `text_candidates`, `unique_clusters`, `fast_text_detect`, `ocr_compare`,
  `rotation_verify`, `drawing_vectors` (12 stages total).

  `_run_clustering` calls `vector.cluster(paths, ctx.page)` per `(layer, color)` bucket from
  `_iter_groups(ctx.paths_by_layer_color)`. `_run_text_candidates` gathers every bucket's final "kept" clusters into
  `ctx.text_clusters` and merges every bucket's `StepResult.cluster_groups` into `ctx.cluster_groups`.
  `_run_unique_clusters` groups `ctx.text_clusters` by whole-page geometric similarity (`Vector.
  group_similar_clusters`) into `ctx.similarity_groups`/`ctx.cluster_similarity_id`.

  `_run_fast_text_detect` renders **one** whole-page image (`Renderer.render_page_paths`) of every
  path in `ctx.vector_paths` (drawing content included, not just `ctx.text_clusters`' paths), and
  runs `FastDetector.detect_tiled` once on it. `detect_tiled` (`helpers/fast_detect.py`) doesn't run
  FAST on the whole render in one direct pass -- it upscales the render by `TILED_SCALE_FACTOR` (5x),
  splits it into non-overlapping `TILED_BLOCK_SIZE`-square tiles (the last row/column right-padded
  with white), and detects each tile at `TILED_ROTATION_COUNT` (4) evenly-spaced rotations, rotating
  each resulting mask back to the tile's own orientation and averaging the 4 before stitching every
  tile's mask back into one full-resolution mask (then resized back down to the render's own original
  size, so callers sample it exactly like `detect()`'s output, at the same pixel coordinates) --
  necessary because `detect()`'s own preprocessing always downsizes to a 640px short side regardless
  of input size, so one direct whole-page pass throws away most of a large page's resolution. Wrapped
  in a `tqdm` bar (`desc="FAST text detection"`) since a real page can mean hundreds of tile x
  rotation passes. For each text-candidate cluster, `_sample_mask` scores it against the page mask
  (sampled at the cluster's own bbox region, whether or not that region also includes non-candidate
  paths); then — per similarity group — the final score is `min(score)` across every member of that
  group (so one low-scoring instance drags every geometrically-identical cluster down with it). A
  cluster passes if its final score exceeds `FAST_COMBINED_KEEP_THRESHOLD` (0.1); `FastPageResult`
  carries the render/mask, the final `scores`, `passed`/`dropped`, and `detect_seconds` timing. A
  page with zero vector paths never even constructs `FastDetector`'s underlying torch model.

  `_run_ocr_compare` constructs `RenderOCR()` — Paddle (`PaddleOcrBackend`, `RenderOCR`'s own
  default), whose text detector already returns line-level boxes (not one per whole paragraph), so
  `TextVectorResult.words` ends up one entry per detected line, and reconstruction places/scales
  each line into its own bbox rather than one string across the whole cluster; pass
  `backend=TesseractOcrBackend()` instead for word-level detection. OCRs each
  FAST-passed cluster whole first (`RenderOCR.ocr_cluster`); a reading above
  `OCR_CLUSTER_CONFIDENCE_THRESHOLD` (0.9) is trusted outright as `resolved`. Otherwise every
  composing group (`ctx.cluster_groups`, skipping the redundant re-render when there's only one
  group) is OCR'd on its own; readings below `OCR_GROUP_CONFIDENCE_THRESHOLD` (0.8) are dropped, and
  unless at least one surviving group reading itself clears `OCR_CLUSTER_CONFIDENCE_THRESHOLD`, the
  whole cluster is a failure (`resolved` is blank, its full path list collected into `ctx.
  ocr_failed`) — otherwise the surviving group readings are joined left-to-right by bbox into one
  `resolved` reading, `.words` concatenated the same left-to-right way. Each `ClusterOcrComparison`
  records `cluster_reading`, `group_readings`, `resolved`, and every OCR call's wall-clock duration
  (`cluster_seconds`/`group_seconds`). This loop is wrapped in a `tqdm` progress bar
  (`desc="OCR compare"`).

  `_run_rotation_verify` (new layer between `ocr_compare` and `drawing_vectors`): for every
  `ocr_comparisons` entry with real (non-blank) `resolved` text, compares the text's own natural
  width/height aspect ratio (`_text_aspect_ratio` — `fitz.Font("helv").text_length(text,
  fontsize=1.0)` over `ascender - descender`, fontsize-invariant so no bbox/fontsize input is
  needed) against `resolved.bbox`'s aspect ratio as-is, and again against that same bbox rotated 90
  deg (width/height swapped, i.e. `1 / bbox_ratio`). If the rotated comparison is a meaningfully
  closer match (`error_unrotated - error_rotated > ROTATION_VERIFY_IMPROVEMENT_MARGIN`, 0.15 —
  avoids flipping on a near-tie), `resolved.rotation_used` is corrected by `+90 % 360` **in place**:
  since `resolved` is the exact same `TextVectorResult` object `ctx.ocr_comparisons`/`ctx.
  ocr_results` (and the debug app's cached `ocr_compare` `StageOutput`) already hold, the fix is
  visible to every later consumer — the reconstruction toggle, `drawing_vectors`, eventually
  `evaluation.py` — automatically, with nothing to rebuild. One `RotationCheck` (`cluster`, `text`,
  `bbox`, `before_rotation`, `after_rotation`, `applied`, `error_unrotated`, `error_rotated`) is
  recorded per checked cluster into `ctx.rotation_checks`, blank/failed readings and degenerate
  (zero-area) bboxes are skipped outright (nothing to check).

  `_run_drawing_vectors` folds three sources into one `drawing_paths` list before calling `Vector.
  build_drawing_vectors`: every `role="dropped"` category from every classification-chain step,
  `ctx.fast_dropped` (FAST found no text signal), and `ctx.ocr_failed` (OCR resolution failed) —
  whatever `ctx.ocr_results` still holds real text for is the only content that doesn't end up in
  `drawing_vectors`. `Pipeline.run_page(reader, page_index, final_stage=None)` wraps each stage in
  `try/except` (`StageOutput(status="error", ...)` on failure, never crashing the run) and, if
  `final_stage` is given, stops right after that stage's output is appended — e.g. `--final-stage
  fast_text_detect` skips `ocr_compare` (and the PaddleOCR engine it would otherwise build) entirely.
- **`debug_app.py`** — Tk GUI (`python -m rastervec.debug_app [pdf]`): page pixmap on a canvas, a
  page-nav bar (reusing the same `page.rotation_matrix * zoom_matrix` display-transform rule as
  `inspector/app.py`), and a stage-nav bar (`< Prev Stage | Stage i/N: Label | Next Stage >`) that
  cycles **only over `Pipeline.STAGES`**. Per-page `StageOutput` results are cached in
  `DebugAppState.stage_cache`, keyed by `page_index`. Each stage's overlay is drawn by a small
  function registered in `_STAGE_RENDERERS`, keyed by `StageSpec.key`, taking a `RenderContext`
  (`canvas`, `matrix`, `output`, `tooltip`, `side_panel`, `filters`, `on_change`, `epoch_box`,
  `page`). Vector paths are always drawn in their **real PDF stroke/fill color** (`_path_color_hex`)
  — any black/white simplification a classification step might use internally is never substituted
  into the overlay itself; synthetic colors (`_DROPPED_COLOR` grey, `_CENTROID_COLOR` red, one color
  per step in `_CLUSTER_STEP_COLORS`) are only ever used for bbox/centroid *outlines*. Stages with
  filterable sub-groups (`vector_extract` by path kind, `layer_separation` by layer,
  `color_separation` by layer+color, `drawing_vectors` by dashed/solid) build `ttk.Checkbutton`s per
  sub-group into `ctx.side_panel`.

  `_render_clustering_stage` shows every one of the fixed 13 classification steps' kept/dropped
  categories as its own toggleable checkbox (bbox + centroid, `_CLUSTER_STEP_COLORS`-keyed, plus a
  live `_cluster_size_stats_text` stats line for each kept category); `_render_unique_clusters_stage`
  colors each similarity group distinctly. This renderer (and other high-volume stages) shares
  viewport-culling + chunked-drawing machinery, since these stages can have tens of thousands of
  clusters/paths: `_SpatialIndex` (a uniform grid over `(bbox, payload)` entries) is built once per
  category per stage-visit; only the subset whose bbox overlaps the current viewport
  (`_visible_page_rect`, page-space via the inverse display matrix, padded 20%) is ever drawn,
  redrawing the delta on scroll/resize (`_recull`) and streaming even that delta in via small
  `after_idle` chunks (`_RECULL_CHUNK_SIZE`) so no single draw blocks the UI. A per-category
  `generation` counter plus `ctx.epoch_box` (bumped once per `redraw_overlay()` call) let a deferred
  chunk callback detect it's been superseded and stop instead of drawing ghost items. Hover precision:
  a group only counts as hovered when the cursor is over an actual *member path's* own bbox, not
  merely anywhere inside the group's aggregate bbox (`_bind_bucket_hover`'s `_group_hit`), padded by
  a small screen-pixel tolerance (`_HOVER_TOLERANCE_PX`). Checkbox/toggle state is persisted
  per-stage in `DebugAppState.filter_state` (keyed by `StageSpec.key`).

  **Reconstructed-PDF compare toggle**: `native`, `ocr_compare`, `rotation_verify`, and
  `drawing_vectors` each open their renderer with a call to the shared
  `_add_reconstruction_toggle(ctx, build_own_fn, build_combined_fn)`, which packs a "Show
  Reconstructed Compare"/"Hide Reconstructed Compare" `ttk.Button` at the top of `ctx.side_panel`
  (state persisted per-stage in `ctx.filters["show_reconstructed"]`). Clicking it flips the state and
  calls `ctx.on_change()`; when on, a "This layer"/"Combined" `ttk.Radiobutton` pair also appears
  (state in `ctx.filters["reconstruction_scope"]`, default `"own"`), picking whether `build_own_fn()`
  (just that stage's own captured elements — `native_words` for `native`, `ocr_results=[c.resolved
  for c in passed]` for `ocr_compare`/`rotation_verify`, `drawing_vectors` for `drawing_vectors`) or
  `build_combined_fn()` (that stage's own elements plus every earlier stage's, looked up via
  `_stage_output_data(ctx, key)` scanning `ctx.all_outputs` — e.g. `drawing_vectors`'s combined view
  draws `native_words` + `ocr_results` + `drawing_vectors` together; `native`'s combined view is
  identical to its own, since nothing precedes it) is called to build the image. `ctx.all_outputs`
  (every `StageOutput` for the current page, in `Pipeline.STAGES` order) is populated once in
  `redraw_overlay` alongside the rest of `RenderContext`. `rotation_verify`'s own build functions
  read `ocr_compare`'s `resolved` readings via `_stage_output_data` rather than its own
  `ctx.output.data` (a `list[RotationCheck]`, not readings) — since `_run_rotation_verify` corrects
  `rotation_used` **in place** on those same `TextVectorResult` objects, this already shows the
  corrected orientation with no extra plumbing.

  The built image isn't drawn as a flat overlay — it's a **drag-anywhere compare slider**: left of
  the divider shows the reconstruction, right of it shows the real original page pixmap (the base
  canvas layer beneath, never touched — the slider just crops how much of the reconstruction
  image covers it). Divider position is `ctx.filters["reconstruction_slider"]` (a 0..1 fraction,
  default 0.5). `ButtonPress-1`/`B1-Motion` bound directly on `ctx.canvas` (cleaned up centrally in
  `redraw_overlay`'s unbind block, alongside `Motion`/`Leave`/`Left`/`Right`, so a stage switch can't
  leave a stale drag handler attached) recompute the fraction from `canvas.canvasx(event.x)` and
  call a local `_apply_slider`, which re-crops the **already-built** PIL image (a cheap operation)
  and updates the existing canvas image item + divider line via `itemconfig`/`coords` directly —
  no full stage redraw, no re-render, so dragging stays smooth even though building the
  reconstruction itself (a `Renderer.render_reconstructed_page` call) can be relatively expensive.
  `_add_reconstruction_toggle` returns whether the compare view is currently showing; each of the 4
  stage renderers `return`s early when it's `True`, skipping their normal overlay. Skipped
  altogether if `ctx.page` is `None`.

  `fast_text_detect` (`_render_fast_text_detect_stage`) shows `FastPageResult`'s single
  all-vectors render+mask, "Render"/"Detection heatmap" toggles, a page-level FAST detect-time
  readout, and per-cluster hover rects resized to the current debug-app zoom (via
  `geometry.matrix_scale(ctx.matrix)`, not a fixed DPI/72 factor) reporting the final
  (post-similarity-group-min) score and pass/fail.

  `ocr_compare` (`_render_ocr_compare_stage`) — text clusters per page are few compared to raw path
  counts, so it skips the viewport-culling/spatial-index machinery above and just draws each
  *visible* comparison's `resolved.bbox` with a plain `tag_bind` hover, plus a click binding to
  select that cluster for the inspector panel below. "Visible" is gated by "Show passed (N)"/"Show
  failed (N)" (a comparison "passed" if `resolved.text.strip()` is non-empty, `_ocr_passed`). A
  page-level OCR timing readout (total + average across every `cluster_seconds`/`group_seconds`
  call) sits at the top. Two more checkboxes, "Show Paddle OCR bbox" and "Show group bbox", draw
  `resolved.ocr_bbox`/each `group_readings[i].ocr_bbox` and each `group_readings[i].bbox` alongside
  the main bbox.

  Clicking a cluster's bbox (or the app defaulting to cluster 0) opens an inspector section: its
  Prev/Next nav row and Left/Right arrow-key binding cycle between *visible* clusters.
  `_ocr_cluster_preview` renders the selected cluster once, at `resolved.rotation_used`, and re-OCRs
  just that one rendered image purely to recover the detected-text bbox in that image's own pixel
  space for the overlay — `text`/`confidence` are read directly off the already-computed
  `TextVectorResult`, never re-derived from that extra OCR call. Cached on `DebugAppState.
  ocr_detail_cache`, keyed by `(page_index, cluster_index)`. The inspector text panel shows
  RESOLVED, the raw CLUSTER (whole) reading plus its own timing, and — only when the fallback ran —
  each raw GROUPS reading tagged `[kept]`/`[dropped]` against `OCR_GROUP_CONFIDENCE_THRESHOLD`
  (imported from `pipeline.py`), plus a note if the fallback gate itself failed.

  `rotation_verify` (`_render_rotation_verify_stage`) draws each checked cluster's `RotationCheck.
  bbox` as a hover rect — orange + thicker outline if `applied` (the fix flipped `rotation_used` by
  90 deg), grey + thin if not — with a tooltip showing the OCR'd text, both aspect-ratio errors, and
  the before/after rotation. A summary label at the top shows how many fixes were applied out of how
  many clusters were checked.

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
