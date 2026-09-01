# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A raster-to-vector pipeline project for architectural/engineering shop drawings
(see `references/*.pdf`, gitignored sample PDFs). `rastervec/` is the only package here —
the standalone PDF-layer inspector tool that used to live at the repo root now lives inside it,
at `rastervec/Evaluation/inspector/` (see below).

`rastervec/` is the actual extraction pipeline: native text, vector drawings (including
reconstructing CAD "text-as-filled-vector-paths" back into real text), consolidated into text/line
objects and reassembled into a PDF for evaluation. Built stage by stage (Reader → Native Text →
Vector → Vector Classification → OCR); currently Reader, Native Text, Vector, and Vector
Classification are implemented, including OCR (`renderer.render_vector_cluster` +
`RenderOCR`/`OcrBackend`, PaddleOCR-only — see `OCR/Paddle_OCR/ocr_backend.py`). A raster-image
line-diagram stage (CNN junction detector + line tracing) was scoped out for now — no `Raster`
module exists in the current tree. The `Evaluation/` package holds the pipeline's still-unbuilt
final reconstruction stage (`evaluation.py`, interface-only), a benchmarking suite for the Vector
Classification + OCR pipeline (`Conversion/` — native text → vector-text PDF; `Labelling/` —
manual + automatic ground-truth labelling; `Evaluate/` — accuracy metrics against those labels;
all three implemented), and the inspector tool — see "rastervec architecture" below.
`junction_cnn/` and `hawp/` at the repo root are unrelated, independent experiments; nothing in
`rastervec/` imports from them.

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt                        # install deps
.venv/Scripts/python.exe -m rastervec.Evaluation.inspector.inspector [path/to.pdf]  # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.pipeline --pdf PATH --page N                 # run the extraction pipeline demo (CLI)
.venv/Scripts/jupyter lab rastervec/notebooks/pipeline_stage_visualization.ipynb   # per-stage pipeline visualization (needs jupyter + matplotlib)
.venv/Scripts/python.exe -m pytest tests/ -v                                        # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC DST --dpi 300                 # flatten a PDF to pure raster
.venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.manual_label PDF --page N --out labels.json  # manual cluster-label editor (GUI)
.venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.view_auto_labels PDF --page N               # view auto_label output in that editor
```

venv is **Python 3.12** (`py -3.12 -m venv .venv`), not 3.14 — `rastervec`'s OCR (paddleocr/
paddlepaddle) doesn't ship Windows wheels for 3.14 yet. On this dev machine's paddlepaddle build,
the default mkldnn-accelerated CPU inference path hits an unimplemented PIR attribute-conversion
error, so `rastervec/OCR/Paddle_OCR/ocr_backend.py` sets
`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` at import time (before paddleocr/paddlex read their
flags) to force the plain "paddle" run mode instead — fine for OCR's small, pre-cropped cluster
renders. If a future paddlepaddle release fixes this, that env-var default can be dropped.

OCR is PaddleOCR-only — `TesseractOcrBackend` was removed (along with `pytesseract` and the
`scripts/setup_tesseract.*` install scripts) since a single backend was simpler to maintain and
Tesseract wasn't in active use.

## `rastervec/Evaluation/inspector/` architecture

A standalone Tkinter + PyMuPDF desktop tool for visually inspecting what's inside a PDF (text,
images, annotations, vector drawings, as toggleable overlays) — predates `rastervec`'s own
extraction pipeline and shares no imports with it; it was built as step 0, to visually validate
what PyMuPDF extracts before writing real extraction logic elsewhere in `rastervec`. It now lives
inside `rastervec/` (under `Evaluation/`, alongside the not-yet-built benchmarking suite) since it
remains a useful dev-facing inspection tool, but its own five modules are otherwise unchanged:

- **`layers.py`** — the extensibility core. `OverlayItem` is the normalized shape every extractor
  returns (bbox always in PDF page coordinates; `quad`/`points` optionally for non-axis-aligned
  geometry; `attrs` for machine-filterable values; `metadata` for human-readable hover info).
  `LayerSpec` (one top-level checkbox) and `SubFilterSpec` (a sub-checkbox group under a layer) are
  declarative — `build_layers(pdf_model)` wires the four current layers (text/images/annotations/
  drawings) to their extractor functions in `pdf_model.py`. `filter_items()` is the one shared
  filtering function all layers use (AND across sub-filter groups, OR within a group, empty
  selection = no restriction). Adding a new layer means adding one `LayerSpec` + one extractor
  function — nothing in `inspector.py`, `overlay_canvas.py`, or `control_panel.py` needs to change.
- **`pdf_model.py`** — the only module that calls into `fitz` for extraction. `PdfDocument` wraps
  the open document; `extract_text_items`/`extract_image_items`/`extract_annot_items`/
  `extract_drawing_items` each return `list[OverlayItem]` for one page; `collect_drawing_colors`
  scans a page's `get_drawings()` once to populate the dynamic stroke/fill color sub-filters.
- **`overlay_canvas.py`** — `PageView`: the left-pane Tk `Canvas` showing the rendered page pixmap
  with overlay shapes drawn on top, plus page nav/zoom controls and a hover tooltip.
- **`control_panel.py`** — `ControlPanel`: the right-pane checkbox tree built from the `LAYERS`
  registry, with collapsible sub-filter groups (checkboxes or color swatches).
- **`inspector.py`** (the package's entry point, `python -m rastervec.Evaluation.inspector.inspector
  [pdf]` — renamed from the original standalone tool's `app.py`) — `InspectorApp` wires the two
  panels together, owns `AppState` (current page/zoom, per-page extraction and color caches), and
  drives the redraw cycle. `REFERENCES_DIR` resolves to the repo-root `references/` folder (three
  levels above the `inspector/` package: `inspector` → `Evaluation` → `rastervec` → repo root).

### Coordinate spaces — read this before touching geometry, anywhere in `rastervec/`

PyMuPDF's extraction APIs (`get_text`, `get_drawings`, `get_image_info`, `annots()`) all return
coordinates in the page's **unrotated MediaBox space**, regardless of the page's `/Rotate` value.
`page.get_pixmap()` and `page.rect`, however, are already in **rotated display space** (rotation
baked in). `inspector.py`'s `_get_display_matrix()` builds the single transform
(`page.rotation_matrix * zoom_matrix`) that both the pixmap and every overlay must go through to
land in the same canvas space — never compute a separate scale/rotation by hand, or overlays will
drift from the underlying page image on any rotated page (several `references/*.pdf` pages are
rotated 90/270). The rest of `rastervec` keeps unrotated MediaBox space as its canonical space
through every stage, only converting to display space at final render/reconstruction — see
`rastervec/models.py`'s module docstring.

For text specifically: a word's axis-aligned bbox from `get_text("words")` only equals its
along-direction/normal-direction extents when the text is horizontal. `make_oriented_quad`
(`rastervec/helpers/geometry.py`, ported from `inspector/pdf_model.py._make_oriented_quad` when
the inspector tool was standalone) projects the bbox corners onto the text's actual direction
vector (from the matching span's `dir`) to build a correctly oriented quad for rotated/vertical
text — don't reintroduce a bbox-width/height shortcut there, anywhere it's used.

### Extending the inspector with a new layer

1. Write an extractor `def extract_x_items(page: fitz.Page) -> list[OverlayItem]` in `pdf_model.py`.
2. Add a `LayerSpec(key=..., extractor=pdf_model.extract_x_items, subfilters=[...])` to the list in
   `layers.build_layers()`.
3. For a new filterable attribute, add a `SubFilterSpec` and register its `attr_getter` in
   `layers._GETTERS` (a `SubFilterSpec` not present in `_GETTERS` renders in the UI but silently
   filters nothing — this bit the `close_path`/`has_mask` filters before it was fixed, so don't
   forget this step).

## `rastervec/` architecture

See `rastervec/Glossary.md` for standardized group/cluster/global-group/similarity-group
terminology used throughout this section. `rastervec/` is organized into one folder per pipeline
concern (`Reader/`, `Native_Text/`, `Vector/`, `Vector_Classification/`, `OCR/`, `Evaluation/`),
plus cross-cutting modules that don't belong to one concern (`models.py`, `output_types.py`,
`logging_setup.py`, `pipeline.py`, `renderer/`) and a `helpers/` package for
utilities shared across more than one concern (`geometry.py`, `clustering.py`). Every stage is
testable independently of the others (every stage's *output* is a plain dataclass from
`models.py`, no `fitz` objects, except `Page.fitz_page` which `Reader` must hand to `Native`/etc.):

- **`models.py`** — all shared dataclasses (`PageMeta`, `Page`, `TextWord`, `TextRun`, `VectorPath`,
  `DrawingVector`, `OcrWord`, `TextVectorResult`, `ClusterOcrResult`, and forward-declared
  `RasterImage`/`JunctionPoint`/`LineVector`/`ReconstructedPage` for the pipeline's still-unbuilt
  final reconstruction stage, see `Evaluation/evaluation.py` below).
- **`output_types.py`** — pydantic DTOs (`TextDTO`, `VectorDTO`, `NativePDFElements`) mirroring what
  a raw PyMuPDF `get_text("words")` word / `get_drawings()` drawing look like, built from the
  dataclasses above — the serialization/export shape for external consumers, not a replacement for
  the dataclasses used mid-pipeline.
- **`logging_setup.py`** — stdlib `logging` only. `configure_logging(level)` once at startup;
  `get_logger("stage_name")` returns `logging.getLogger("rastervec.stage_name")` per module.
- **`helpers/geometry.py`** — pure-math helpers, originally ported from the inspector tool's
  `pdf_model.py` (`point_angle`, `line_length`, `quad_angle`, `matrix_rotation`, `matrix_scale`,
  `make_oriented_quad`, `rect_gap`, `union_bbox`, etc.), shared by `Native_Text/` and `Vector/` (and
  the inspector) so none of them duplicate this math independently.
- **`helpers/clustering.py` — `Clustering`** *(implemented)*: pure-Python (no scipy/sklearn) spatial
  hash grid + union-find for `cluster_spatial` (buckets items into grid cells sized by `threshold`,
  unions items in neighboring cells whose `geometry.rect_gap` ≤ `threshold` —
  `Vector_Classification/clusters/cluster_filters.py`'s `cluster_spatial_groups` reuses this same
  method at the group level, treating each group as one atomic item), then O(k²) pairwise
  union-find within each resulting group (`_split_group_pairwise`, shared by all three of the
  following) for `cluster_by_dimension` (relative width/height closeness), `cluster_by_seq`
  (sorted-seq gap split), and `group_by_overlap` (merges items whose bboxes overlap or are within an
  optional `tolerance` of each other, via module-level `_bboxes_close_or_overlapping` —
  `geometry.rect_gap` already returns 0.0 for overlapping/touching boxes, so one gap check covers
  both "touching" and "merely nearby"; `_bbox_fully_contains` keeps a fully-contained/equal pair
  from ever merging regardless of tolerance). Has safety caps (`_MAX_CELLS_PER_ITEM`,
  `_MAX_GROUP_SIZE_FOR_PAIRWISE`) so a huge unfiltered bbox or a very dense cluster degrades to
  "keep as one cluster" (logged) instead of hanging — verified against a 78k-path reference PDF in
  ~3.5s. `cluster_by_dimension`/`cluster_by_seq`/`group_by_overlap` aren't currently called by the
  fixed Vector Classification chain below, kept for reuse (own tests, own callers).
- **`Reader/reader.py` — `Reader`** *(implemented)*: opens a PDF, hands out `Page` objects one at a
  time (`get_page(index)`, `iter_pages(indices=None)`), each carrying a `PageMeta` snapshot
  (mediabox, rotation, dimensions) plus the live `fitz.Page`.
- **`Native_Text/native.py` — `Native`** *(implemented)*: `extract_text(page) -> list[TextWord]`,
  via `get_text("dict")` for span metadata (font/size/color/direction) joined to `get_text("words")`
  geometry by max bbox-overlap (`_match_word_to_span`), producing correctly oriented quads even for
  rotated text (`_build_oriented_quad`). Split into small private methods
  (`_extract_spans`/`_extract_words`/`_match_word_to_span`/`_build_oriented_quad`/`_to_text_word`)
  so each is independently testable against a synthetic `fitz.Page`.
- **`Vector/vector.py` — `Vector`** *(implemented)*: `extract_paths(page) -> list[VectorPath]` walks
  `page.fitz_page.get_drawings()`, emitting one `VectorPath` per drawing item (`l`/`re`/`qu`/`c`),
  tagged with its parent drawing's `seq` (drawing index) plus stroke/fill color, width, dashes,
  closed, layer, and item-level `bbox`/`points`. `separate_by_layer`/`separate_by_color` delegate to
  `Vector/Layer_Color_Separation/layer_color_separation.py`'s module-level functions of the same
  name, which group paths by `layer` (`""` for none) / by `stroke_color` if set else `fill_color`
  else `None`.
- **`Vector_Classification/`** — classification of extracted paths into text candidates vs. drawing
  content. `classification.py`'s `VectorClassifier` is the orchestrator (`cluster`, `classify`,
  `build_drawing_vectors`, plus `CategoryResult`/`StepResult` and every threshold constant); the
  fixed 12-step chain itself is split by processing level into three submodules (each step
  implemented as a plain function; see each submodule's own docstring for the exhaustive per-step
  description):
  - `items/item_filters.py` (step 1-2, plus the shared `_bbox_of`/`_max_dimension`/`_dims` bbox
    helpers reused by the other two submodules): `filter_large_items` — drop items whose own bbox's
    max dimension exceeds a fraction of the page's smaller side (border/frame geometry).
    `compute_vector_signatures` — informational: per-signature occurrence counts, reused below.
  - `groups/group_filters.py` (steps 3-5, 8): `remove_duplicate_runs` + `combine_overlapping_seq` —
    drop long runs of exact-duplicate shapes, then chain-merge the rest by `seq` order into "groups"
    (see Glossary.md). `filter_tiny_groups` / `filter_large_groups` — drop undersized/oversized
    groups. `compute_group_stats` — informational per-cluster stats (member/signature counts, bbox).
  - `clusters/cluster_filters.py` (steps 6-7, 9-12, plus `group_similar_clusters`, not one of the
    numbered steps): `cluster_spatial_groups` — single-linkage spatial merge of groups into
    clusters, constrained to groups sharing a similar-length parallel side; also tracks `lineage`
    (which groups compose each cluster) for every later step and for `StepResult.cluster_groups`.
    `filter_mixed_fill_rule_clusters` — drop clusters mixing fill/stroke paint styles.
    `filter_perimeter_only_clusters` — drop border/ring-only clusters. `filter_density_clusters` —
    drop clusters too sparse across their own bbox grid. `filter_constant_spacing_clusters` — drop
    clusters where most members belong to a near-perfectly-regular repeated same-shape sub-group
    (hatching, tick marks). `filter_low_variety_clusters` — drop clusters below a
    member-count-scaled minimum distinct-shape-type count. `group_similar_clusters` — whole-page
    similarity grouping of text-candidate clusters (see "similarity group" in Glossary.md).

  There is deliberately **no drawing-vs-text heuristic** anywhere in this chain — every group/cluster
  any filter step drops along the way is drawing content (`pipeline.py`'s `_run_drawing_vectors`
  folds every `role="dropped"` category into `drawing_vectors`), and everything that survives the
  whole chain is a *text candidate*, handed to `unique_clusters`/`fast_text_detect`/`ocr_compare` —
  OCR success/failure is the actual signal for whether a cluster was text, not a pre-filter guess.

  `VectorClassifier.cluster(paths, page) -> list[StepResult]` runs the fixed chain in order; each
  `StepResult` holds every named `CategoryResult` that step produced (`role="kept"` always present
  and fed to the next step; `role="dropped"` is a side channel folded into `drawing_vectors`;
  `role="info"` is display-only). `steps[-1].categories["kept"]` is the final surviving clusters.
  The last step's `StepResult.cluster_groups` (keyed by `id(cluster)`) records which of step 6's
  pre-spatial "groups" each survivor is composed of, via the `lineage` dict step 6 builds
  internally. `classify(paths, page) -> list[list[VectorPath]]` is a thin convenience wrapper
  returning just the final kept groups. `build_drawing_vectors(paths) -> list[DrawingVector]`
  re-aggregates same-`seq` paths back into one `DrawingVector` per original drawing.
  `group_similar_clusters(clusters) -> list[list[list[VectorPath]]]` groups text-candidate clusters
  by whole-page geometric similarity (see the `unique_clusters` pipeline stage below). All
  thresholds (`MAX_DIMENSION_FRACTION`, `SPATIAL_CLUSTER_THRESHOLD`, `SPATIAL_SIZE_TOLERANCE`,
  `PERIMETER_MARGIN_FRACTION`, `DENSITY_*`, `PATTERN_*`, `LOW_VARIETY_*`, `UNIQUE_CLUSTER_TOLERANCE`)
  are module-level constants in `classification.py` — tune per-PDF if a specific page's default
  classification looks wrong; there's no runtime/UI way to change them or the step order.

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across buckets**:
  `pipeline.py`'s `_iter_groups`/`_run_clustering` key the whole chain's work by `GroupKey = (layer,
  color)` (from `color_separation`'s output), and `VectorClassifier.cluster()` is only ever called
  with one bucket's paths at a time — two paths in different layers, or with different stroke/fill
  colors, are never spatially merged together, regardless of how close they are on the page.
- **`OCR/FAST_Text_Detect/fast_detect.py` — `FastDetector`** *(implemented)*: see the `pipeline.py`
  bullet below for `detect`/`detect_tiled`.
- **`OCR/Rotation_Correction/rotation_correction.py` — `RotationCheck`, `_run_rotation_verify`**
  *(implemented)*: see the `pipeline.py` bullet below.
- **`OCR/Paddle_OCR/ocr_backend.py`** *(implemented, PaddleOCR-only — `TesseractOcrBackend` was
  removed, see Commands section above)*: `OcrBackend` (Protocol: `detect(image) -> OcrDetection`),
  `OcrBox` (one detected text box: `text`/`confidence`/`corners`/`is_word`, always already mapped
  into the *caller's* original image pixel space), `OcrDetection` (`boxes` + page-level `rotation`).
  `PaddleOcrBackend`: PP-OCRv6 via PaddleOCR, orientation classifiers on
  (`use_doc_orientation_classify=use_textline_orientation=True, use_doc_unwarping=False`), engine
  lazily built and cached per `lang` at class scope (`_ENGINE_CACHE`). Detected boxes are
  **line/region-level**, never word-level (`OcrBox.is_word=False`). `_page_rotation(page)` combines
  Paddle's own `doc_preprocessor_res.angle` (0/90/180/270 document-orientation classification) with
  the majority vote of `textline_orientation_angles` (0/180 per-line flip correction). **Confirmed
  bug fix**: when `use_doc_orientation_classify=True`, PaddleOCR's own doc-preprocessor sub-pipeline
  actually rotates the input image by `doc_preprocessor_res.angle` degrees *before* running text
  detection (confirmed by reading paddlex's `doc_preprocessor/pipeline.py`/`ocr/pipeline.py`
  source), so every `rec_poly` it returns is in that rotated image's own pixel space (size-swapped
  for 90/270) — not the space of the image actually passed in; `detect()` corrects every corner back
  via `_undo_doc_rotation` (exact per-point inverse of `cv2.getRotationMatrix2D`'s rotation, derived
  for each of the 4 possible angles) before returning.
- **`OCR/Paddle_OCR/render_ocr.py` — `RenderOCR`** *(implemented)*: render + detect, backend-agnostic
  (though PaddleOCR is the only backend now) — OCR's rendered vector-text clusters (`ocr_cluster`
  only ever handles `list[VectorPath]` clusters; there is no Raster stage in this project to OCR
  raster image regions). `__init__(self, backend: OcrBackend | None = None)` defaults to
  `PaddleOcrBackend()`. `ocr_boxes(image)`/`ocr(image)` wrap `self.backend.detect(image)` (`ocr`
  joins every detected box left-to-right into one `(text, confidence, bbox_corners)` result).
  `ocr_cluster(cluster, page, dpi=300)` is the shared
  entrypoint (it calls `rastervec.renderer`'s module functions directly — no `renderer` param):
  render the cluster **once**, upright, run **one** `backend.detect()` call, and build a
  `TextVectorResult` whose `rotation_used` comes from `OcrDetection.rotation`, whose `ocr_bbox`
  (union of every detected box, mapped back to page space via `renderer.pixel_to_page_bbox`) is
  `None` when nothing was detected, and whose `words: list[OcrWord] | None` field (models.py) is one
  `OcrWord` per individually detected box, each independently mapped through `pixel_to_page_bbox`
  (line-granularity, since Paddle is the only backend); `None` when nothing was detected.
- **`Evaluation/evaluation.py` — `Evaluation`** *(interface stub, not yet implemented)*: the
  pipeline's actual intended final stage — `reconstruct_page`/`build_pdf` will reassemble
  consolidated text/line objects back into a PDF for evaluation — not yet registered in
  `Pipeline.STAGES`. Distinct from the `Evaluation/Evaluate/` benchmarking subpackage below, which
  is implemented and scores the existing Vector Classification + OCR pipeline, not this stub.
- **`Evaluation/Conversion/conversion.py` — `convert_page_to_vector_text`** *(implemented)*: turns
  one page's native text into vector-drawn text (`get_svg_image()` → `fitz.open(filetype="svg")` →
  `convert_to_pdf()` → `show_pdf_page` onto a page sized from the source's own `PageMeta.mediabox`/
  `rotation`) — confirmed by a spike (see the module's own docstring) that PyMuPDF renders text as
  filled SVG `<path>`s, never an SVG `<text>` element or an embedded raster fallback, so the
  round-tripped PDF's `get_drawings()` holds real vector path content and `get_text()` comes back
  empty. Turns a native-text PDF into a known-answer test case for Vector_Classification, since the
  ground-truth text is whatever the original native words said.
- **`Evaluation/Labelling/`** *(implemented)*: ground-truth labelling for vector-text regions.
  `label_schema.py`'s `LabelEntry` (`page_index`, `cluster_bbox`, `cluster_signature`, `text`,
  `source: "manual"|"auto"`, `expected_rotation`) + `LabelSet` are the sidecar JSON format
  (`save_labels`/`load_labels`); `cluster_signature`'s meaning depends on `source` —
  `"manual"` entries use `cluster_signature(cluster)`, a deterministic member-count + rounded-bbox
  string identifying a real clustered-run's cluster across repeated pipeline runs (VectorPath
  objects have no identity across runs); `"auto"` entries use a
  `f"line:{page_index}:{block_no}:{line_no}"` native-text line-region id instead, since there's no
  clustered run backing them (see below). `auto_label.py`'s `auto_label_pdf` is deliberately
  independent of the pipeline being evaluated — it reads *only* the original PDF's own
  `Native.extract_records` (never runs Conversion or any classification/clustering), groups words
  by `(block_no, line_no)` into line-level ground-truth regions (bbox via `helpers.geometry.
  union_bbox`, text joined left-to-right by word `bbox[0]`), and sets `expected_rotation` from each
  line's own text angle rounded to the nearest quarter-turn. This independence matters: an earlier
  version derived labels from the *converted* page's own surviving classification clusters, which
  meant a native word the classification chain's own filter steps wrongly dropped never became a
  label at all — silently excluded from ground truth rather than scored as a miss. Ground truth
  must not depend on what the system under test decided. `manual_label.py`'s `ManualLabelApp`
  (`python -m rastervec.Evaluation.Labelling.manual_label PDF --page N --out labels.json`) *does*
  need real clusters (a human has to click something), so it's the one place that still runs the
  real pipeline — via `pipeline.run_page_context(reader, page_index, final_stage="text_candidates")`
  — with a `_get_display_matrix` / `Tooltip` for the page-space → canvas-space transform and hover
  tooltip, both ported from the former `debug_app.py` when it was removed. It's also a light cluster
  *editor* (the pipeline's clustering isn't always right): scroll + `Zoom -`/`Zoom +` + Ctrl-wheel
  zoom, and two edit modes — **cluster mode** (left-click toggles a whole cluster; `Group` merges
  the selected clusters, `Ungroup` splits one back into its pre-spatial `ctx.cluster_groups`
  "groups", or one-path-per-cluster if it was already edited) and **path mode** (left-click toggles
  an individual `VectorPath`; `Group` builds a new cluster from exactly the selected paths). In either
  mode a left-click-drag draws a rubber-band box that adds every intersecting cluster/path to the
  selection, or removes them all if they were already selected (one drag both selects and deselects
  an area); a click that barely moves still does the single-item toggle. `Ctrl+Z`
  undoes the last group/ungroup. Right-click a cluster (cluster mode) to type its ground-truth text
  and `expected_rotation` (turns green once labelled); hovering shows the assigned text.
  `LabelEntry`s loaded from `--out` that match no live cluster (every `source="auto"` entry, plus
  manual entries left stale by an edit) draw as dashed grey boxes, so the same window doubles as an
  auto-label viewer. Save/window-close writes the label file — not unit-testable (a real Tk event
  loop), smoke-test steps are in its own module docstring. `view_auto_labels.py`
  (`python -m rastervec.Evaluation.Labelling.view_auto_labels PDF --page N [--out labels.json]`)
  runs `auto_label_pdf`, merges its entries onto any existing `--out` file (default: a temp file),
  and opens `ManualLabelApp` on it.
- **`Evaluation/Evaluate/evaluate.py` — `evaluate_pipeline`** *(implemented)*: scores a completed
  pipeline run's `ClusterOcrResult`/`DrawingVector`/`RotationCheck` lists against a `LabelSet`.
  Matches each label to a predicted OCR reading by `bbox_iou` (greedy highest-IoU-first, one-to-one)
  above `iou_threshold`; a `RotationCheck` with `applied=True` for a matched cluster overrides that
  reading's `rotation_used` for the rotation-accuracy check. Returns an `EvaluationResult`:
  `characters_found_pct` (char-count-weighted, not label-count-weighted), `character_accuracy`/
  `character_error_rate` (mean `difflib.SequenceMatcher` ratio per matched pair — stdlib, no
  string-distance dependency in `requirements.txt`), `rotation_accuracy` (matched-pair rotation vs.
  `expected_rotation`), `bbox_accuracy` (mean IoU), `classification_precision`/`_recall` (matched =
  true positive, unmatched label = false negative i.e. text the pipeline dropped as drawing content,
  unmatched prediction = false positive), `drawing_vector_count`. Deliberately decoupled from
  `PipelineContext`/a real PDF — callers pass their own OCR/drawing-vector/rotation-check lists, so
  this module is testable against small hand-built inputs. Optional `clustering`/`fast_dropped`/
  `ocr_failed` params (the rest of `PipelineContext`, all default `None`) turn on a stage-attributed
  "loss funnel": `_attribute_miss` checks, in pipeline order, whether an unmatched label's bbox
  overlaps a `role="dropped"` classification-step category (`"classification:<step label>"`, the
  *earliest* matching step), else a `fast_dropped` cluster (`"fast_text_detect"`), else an
  `ocr_failed` cluster (`"ocr_blank"`), else `"not_found"` (never appeared in any known bucket — a
  Conversion-fidelity gap or extraction issue, not a classification/OCR decision) — populated into
  `EvaluationResult.miss_attributions` (`list[MissAttribution]`, empty when `clustering` is
  omitted, so passing none of these three keeps the exact old behavior).
- **`Evaluation/Evaluate/benchmark.py`** *(implemented)* — the CLI wiring Conversion → auto_label →
  a real full pipeline run → `evaluate_pipeline` together: `python -m
  rastervec.Evaluation.Evaluate.benchmark --pdf PATH [--pdf PATH2 ...] --pages 0,1,2
  [--iou-threshold 0.3]`. `run_one_page` builds ground truth with no pipeline run
  (`auto_label_pdf`), converts, runs the entire `Pipeline.STAGES` chain including real PaddleOCR
  via `pipeline.run_page_context`, then scores it with every miss-attribution param wired through.
  `format_report`/`aggregate_results` (mean of each numeric metric + total miss-reason counts
  across every page) are pure and unit-tested; `main()`'s actual OCR-backed path is a manual smoke
  test only (same reasoning as `manual_label.py`'s Tk UI — real PaddleOCR, first run downloads
  models — matches the existing `RASTERVEC_RUN_OCR_TESTS`-gated convention for OCR-dependent
  tests).
- **`renderer/` — module-level functions, no `Renderer` class** *(rendering helpers, not a pipeline
  stage)*: a package split by output concern — `png.py` (rasterize vector paths for OCR / FAST
  input), `pdf.py` (`render_reconstructed_page`), `svg.py` (`render_page_svg`, a thin
  `get_svg_image()` wrapper), and `_shapes.py` (shared). Import straight from `rastervec.renderer`
  (`from rastervec.renderer import render_vector_cluster`, etc.).
  `_shapes.path_color_hex(path)` returns a path's real PDF stroke/fill color as hex (used by both the
  visualization notebook and OCR input rendering) — any B/W-style simplification stays purely
  internal to classification, never substituted into a rendered/displayed color.
  `_shapes.replay_drawing_paths(shape, paths, *, dx, dy)` is the accuracy-critical helper shared by
  png/pdf: it regroups `paths` by their parent drawing (`VectorPath.seq`), replays every item of a
  drawing into the `fitz.Shape`, then calls `shape.finish()` **once per drawing** carrying that
  drawing's real `even_odd` / `line_join` / `line_cap` / stroke+fill opacity (ported from
  `archive/raster_parser/rendering/pdf_render/reconstruct.py`). This is why a multi-contour filled
  glyph (an "o", "e", "8", "A" — outer contour + inner counter, one drawing, `even_odd`) renders
  with its counter as a white hole instead of filled solid; drawing each `VectorPath` primitive on
  its own and calling `finish(closePath=True)` per primitive (the pre-split behaviour) filled every
  counter solid — a direct hit to OCR of vector text. `finish()` is per drawing but `commit()` is
  left to the caller (one commit per render). A drawing whose paths carry neither `stroke_color` nor
  `fill_color` is skipped outright: `finish()` emits a stroke operator whenever `fill` is `None`
  regardless of `color`, falling back to the default-black graphics state instead of staying
  invisible. `even_odd` / `line_cap` / `line_join` are drawing-level fields now copied onto every
  `VectorPath` of a drawing (like `fill_rule`), defaulted so existing constructions are unaffected;
  `Vector.extract_records` normalises a tuple `lineCap` from `get_drawings()` to a plain int.
  `png.render_vector_cluster(paths, dpi)` *(implemented)*
  isolates a cluster onto a fresh single-page PyMuPDF document sized to the cluster's own bbox (plus
  padding — `max(4pt, largest member's stroke_width)`, computed by the private `_cluster_frame`
  helper — so edge strokes aren't clipped), replays each drawing's items via `replay_drawing_paths`,
  then rasterizes at `dpi` and returns a PIL `Image` — reusing PyMuPDF's own path/curve/fill
  rendering rather than reimplementing rasterization by hand.
  `png.pixel_to_page_bbox(paths, dpi, pixel_points)` inverts
  `_cluster_frame`'s same transform to map pixel-space points (e.g. Paddle's detected text-region
  corners, from a render of that exact `paths`/`dpi`) back into PDF page space — used by
  `RenderOCR.ocr_cluster` to compute a `TextVectorResult.ocr_bbox`. `png.render_page_paths(paths,
  page_meta, dpi)` is the whole-page counterpart (every given path drawn onto one page-sized canvas,
  no isolation/padding, no rotation applied) — used as FAST's own detection input by
  `fast_text_detect` (see below).
  `pdf.render_reconstructed_page(page_meta, *, native_words=None, drawing_vectors=None,
  ocr_results=None, zoom=1.0)` *(implemented, visualization-notebook preview — not OCR input, not
  `evaluation.py`'s real reconstruction stage)*: redraws whatever elements are passed onto a fresh
  blank page sized/rotated to match `page_meta`, then rasterizes at `zoom` the same way the
  notebook's `page_raster()` rasterizes the real page pixmap, so the two are pixel-comparable at the
  same zoom. `drawing_vectors` are redrawn from each `DrawingVector`'s own real member `VectorPath`s
  (via the shared `_shapes.replay_drawing_paths`, so multi-contour fills keep their holes here too),
  never just their aggregate bbox. `native_words`/`ocr_results` are inserted as real text via
  `page.insert_text` — necessarily approximate: font family isn't preserved (always PyMuPDF's
  base14 `"helv"`). Font size and baseline are derived from `fitz.Font("helv")`'s own
  ascender/descender metrics rather than treating the bbox height as the fontsize and the bbox's
  bottom edge as the baseline outright (a font's em-square is taller than its rendered bbox, and the
  baseline sits `ascender * fontsize` below the bbox's *top* edge, not at its bottom): `fontsize =
  (bbox_height) / (ascender - descender)`, `baseline_y = bbox_top + ascender * fontsize`. For
  `ocr_results` specifically, placement is per-word when `TextVectorResult.words` is populated
  (one `_place_text` call per `OcrWord`, each scaled/baselined into its own bbox instead of one
  string stretched across the whole cluster bbox; falls back to the single-bbox
  `result.text`/`result.bbox` path when `words` is `None`/empty, e.g. Paddle's line-level boxes or
  `native_words`, which has no per-word concept).
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
- **`pipeline.py`** — shared stage-running machinery, used by the CLI (`main()` in this file),
  `run_page_context`, and the visualization notebook. `PipelineContext` accumulates state across
  stages for one page run: `page`,
  `native_words`, `vector_paths`, `paths_by_layer`, `paths_by_layer_color`, `clustering` (`dict
  [GroupKey, ClusteringStageResult]`, `GroupKey = (layer, color)`; `ClusteringStageResult.steps` is
  exactly `VectorClassifier.cluster()`'s return value), `text_clusters` + `cluster_groups` (every
  bucket's final "kept" clusters flattened, plus their merged group lineage), `similarity_groups` +
  `cluster_similarity_id` (whole-page similarity grouping, see "similarity group" in Glossary.md),
  `fast_result` + `fast_passed`/`fast_dropped`, `regrouped_clusters`, `cluster_ocr_results` +
  `ocr_results` + `ocr_failed`, `rotation_checks`, `drawing_vectors`. `StageSpec(key, label, run)` is
  one stage; `Pipeline.STAGES` is the ordered list: `reader`, `native`, `vector_extract`,
  `layer_separation`, `color_separation`, `clustering`, `text_candidates`, `unique_clusters`,
  `fast_text_detect`, `spatial_regroup`, `ocr_compare`, `rotation_verify`, `drawing_vectors` (13
  stages total).

  `_run_clustering` calls `VectorClassifier().cluster(paths, ctx.page)` per `(layer, color)` bucket
  from `_iter_groups(ctx.paths_by_layer_color)`. `_run_text_candidates` gathers every bucket's final
  "kept" clusters into `ctx.text_clusters` and merges every bucket's `StepResult.cluster_groups`
  into `ctx.cluster_groups`. `_run_unique_clusters` groups `ctx.text_clusters` by whole-page
  geometric similarity (`VectorClassifier.group_similar_clusters`) into `ctx.similarity_groups`/
  `ctx.cluster_similarity_id`.

  `_run_fast_text_detect` renders **one** whole-page image (`renderer.render_page_paths`) of every
  path in `ctx.vector_paths` (drawing content included, not just `ctx.text_clusters`' paths), and
  runs `FastDetector.detect_tiled` once on it. `detect_tiled` (`OCR/FAST_Text_Detect/fast_detect.py`) doesn't run
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
  cluster passes if its final score exceeds `FAST_COMBINED_KEEP_THRESHOLD` (0.2); `FastPageResult`
  carries the render/mask, the final `scores`, `passed`/`dropped`, and `detect_seconds` timing. A
  page with zero vector paths never even constructs `FastDetector`'s underlying torch model.

  `_run_spatial_regroup` re-merges every FAST-passed cluster whose member paths actually overlap
  (or touch within `SPATIAL_REGROUP_TOLERANCE_PX`) some other cluster's member paths
  (`_clusters_overlap`, via `Clustering.cluster_spatial`) — deliberately ignores which `(layer,
  color)` bucket or `unique_clusters` similarity group a cluster originally came from, unlike every
  earlier clustering step in this pipeline (see `Vector_Classification/classification.py`'s module
  docstring: normal classification never merges across `(layer, color)` buckets). Two nearby
  FAST-passed clusters that classification/FAST happened to keep as separate pieces are stitched
  back into one piece here (`ctx.regrouped_clusters`) before OCR sees them.

  `_run_ocr_compare` constructs `RenderOCR()` (PaddleOCR) and OCRs each `regrouped_clusters` cluster
  directly with one `RenderOCR.ocr_cluster` call each (no fallback tiers, no similarity-group
  reuse) — wrapped in a `tqdm` progress bar (`desc="OCR compare"`). A cluster's reading counts as
  failed if its text comes back blank; its full path list is collected into `ctx.ocr_failed`
  (folded into `drawing_vectors`), in addition to being kept (blank) in `ctx.ocr_results`. Each
  `ClusterOcrResult` records `cluster`, `resolved`, and the OCR call's wall-clock duration
  (`ocr_seconds`).

  `_run_rotation_verify` (`OCR/Rotation_Correction/rotation_correction.py`'s `_run_rotation_verify`,
  a new layer between `ocr_compare` and `drawing_vectors`): for every `cluster_ocr_results` entry
  with real (non-blank) `resolved` text, compares the text's own natural width/height aspect ratio
  (`_text_aspect_ratio` — `fitz.Font("helv").text_length(text, fontsize=1.0)` over `ascender -
  descender`, fontsize-invariant so no bbox/fontsize input is needed) against `resolved.bbox`'s
  aspect ratio as-is, and again against that same bbox rotated 90 deg (width/height swapped, i.e.
  `1 / bbox_ratio`). If the rotated comparison is a meaningfully closer match (`error_unrotated -
  error_rotated > ROTATION_VERIFY_IMPROVEMENT_MARGIN`, 0.15 — avoids flipping on a near-tie),
  `RotationCheck.resolved` is a *new* `TextVectorResult` (via `dataclasses.replace`, not a mutation
  of `ocr_compare`'s own object) with `rotation_used` corrected by `+90 % 360` (`applied=True`) —
  `ocr_compare`'s own `resolved` reading (held by `ctx.cluster_ocr_results`/`ctx.ocr_results`) is
  never mutated; consumers wanting the corrected orientation (this stage's own reconstruction view,
  `drawing_vectors`) read `RotationCheck.resolved` instead. One `RotationCheck` (`cluster`, `text`,
  `bbox`, `before_rotation`, `after_rotation`, `applied`, `error_unrotated`, `error_rotated`,
  `resolved`) is recorded per checked cluster into `ctx.rotation_checks`; blank/failed readings and
  degenerate (zero-area) bboxes are skipped outright (nothing to check).

  `_run_drawing_vectors` folds three sources into one `drawing_paths` list before calling
  `VectorClassifier.build_drawing_vectors`: every `role="dropped"` category from every
  classification-chain step, `ctx.fast_dropped` (FAST found no text signal), and `ctx.ocr_failed`
  (OCR resolution failed) — whatever `ctx.ocr_results` still holds real text for is the only
  content that doesn't end up in `drawing_vectors`. `Pipeline.run_page(reader, page_index,
  final_stage=None)` wraps each stage in `try/except` (`StageOutput(status="error", ...)` on
  failure, never crashing the run) and, if `final_stage` is given, stops right after that stage's
  output is appended — e.g. `--final-stage fast_text_detect` skips `ocr_compare` (and the PaddleOCR
  engine it would otherwise build) entirely. Module-level `run_page_context(reader, page_index,
  final_stage=None)` runs that same stage sequence but returns the `PipelineContext` itself instead
  of the `list[StageOutput]`, for callers that want to read pipeline state directly (e.g.
  `ctx.text_clusters`) rather than each stage's `StageOutput.data` — used by `Evaluation/
  Labelling/manual_label.py` and `Evaluation/Evaluate/benchmark.py` instead of either hand-rolling
  a partial `Pipeline.STAGES` sequence themselves.
- **`notebooks/pipeline_stage_visualization.ipynb`** — the static replacement for the former
  `debug_app.py` Tk GUI (deleted). One `Pipeline._run_stages(ctx, FINAL_STAGE)` run on a single
  configured `(PDF_PATH, PAGE_INDEX)` gives both the accumulated `ctx.*` fields and each stage's
  `StageOutput` (`status`/`error`). A `visualize(stage_key, categories)` helper renders, per stage:
  one **original page** raster, then for **every** overlay category a pair — the category's geometry
  drawn alone on white (**isolated**) and the same geometry on the original page (**overlay**) — so
  a single-overlay stage is 3 images and an N-overlay stage is `1 + 2N`. `clustering` expands every
  one of the 12 steps' every category (`kept` + `dropped`/`info` side categories, merged across all
  `(layer, color)` buckets, `f"{step_i}_{name}"` keyed, per-step colour from `CLUSTER_STEP_COLORS`),
  and `color_separation` expands every `(layer, color)` bucket — a dense page can be 100+ images.
  Overlay drawing is the notebook's own `PIL.ImageDraw` polyline/polygon port (polylines/polygons via
  `page.fitz_page.rotation_matrix * fitz.Matrix(zoom, zoom)`, the ex-`_get_display_matrix` rule; it
  draws per-primitive and does not do the renderer package's per-drawing even-odd replay);
  the four reconstruction stages (`native`, `ocr_compare`, `rotation_verify`, `drawing_vectors`)
  use `renderer.render_reconstructed_page(...)` for their isolated panel, and `fast_text_detect`
  uses `FastPageResult.page_image` + a red-channel `page_mask` heatmap blend. `FINAL_STAGE` stops
  the run early — set it before `fast_text_detect` to skip needing the FAST weights file, before
  `ocr_compare` to skip building PaddleOCR. Inline (matplotlib) only, nothing written to disk.

`scripts/rasterize_pdf.py` (outside `rastervec/`, a one-off utility not a pipeline stage): flattens
every page of a PDF to an image and rebuilds a pure-raster PDF from those images — not currently
consumed by anything in `rastervec/` (kept for possible future raster-image work).

`tests/rastervec/` mirrors `rastervec/`'s own folder layout (e.g. `tests/rastervec/Reader/
test_reader.py` for `rastervec/Reader/reader.py`, `tests/rastervec/Vector_Classification/
test_classification.py` for `Vector_Classification/classification.py`, `tests/rastervec/renderer/
test_png.py` for `rastervec/renderer/png.py`); modules that stay at
`rastervec/`'s top level (`output_types.py`, `pipeline.py`) keep their tests at
`tests/rastervec/`'s top level too. `tests/conftest.py`'s `synthetic_pdf_factory` builds small
in-memory PDFs via `fitz.open()`/`insert_text`/`set_rotation` — preferred over `references/*.pdf`
for unit tests since those are gitignored and give no exact expected values to assert against.

### Adding a new `rastervec` module or pipeline stage

Three things, all following the existing stage folders' pattern:
1. Define the module's dataclass(es) in `models.py` if they don't exist yet, and give the module
   its own folder under `rastervec/` (or a new file inside an existing one, e.g. a new submodule
   under `Vector_Classification/`) with real logic split into small private methods per sub-step
   (e.g. `_extract_x`/`_match_y`) so each is independently testable.
2. Add one `StageSpec` to `Pipeline.STAGES` in `pipeline.py` (a `_run_<stage>(ctx)` function that
   reads whatever `PipelineContext` fields it needs and stores its own result back onto `ctx`).
3. Add a per-stage cell to `notebooks/pipeline_stage_visualization.ipynb` — a markdown header plus
   a code cell that builds the stage's overlay `categories` list from `ctx` / `outputs[<key>].data`
   and calls `visualize("<key>", categories)`.
Also add tests under the matching `tests/rastervec/` subfolder using the synthetic PDF fixtures,
and new third-party dependencies to `requirements.txt` only when the stage that needs them is
actually implemented.
