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
line-diagram stage (CNN junction detector + line tracing) was scoped out — no `Raster` module,
`helpers/masking.py`, `helpers/junction.py`, or their `models.py` dataclasses exist; recover them
from git history if that work ever starts. The `Evaluation/` package holds a benchmarking suite
for the Vector Classification + OCR pipeline (`Conversion/` — native text → vector-text PDF;
`Labelling/` — manual + automatic ground-truth labelling; `Evaluate/` — accuracy metrics against
those labels; all three implemented) plus the inspector tool — see "rastervec architecture" below.
`junction_cnn/` and `hawp/` at the repo root are unrelated, independent experiments; nothing in
`rastervec/` imports from them.

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt                        # install deps
.venv/Scripts/python.exe -m rastervec.Evaluation.inspector.inspector [path/to.pdf]  # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.pipelines.current --pdf PATH --page N        # run the current extraction pipeline (CLI; also .pipelines.legacy)
.venv/Scripts/jupyter lab rastervec/notebooks/pipeline_stage_visualization.ipynb   # per-stage pipeline visualization (needs jupyter + matplotlib)
.venv/Scripts/python.exe -m pytest tests/ -v                                        # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC [DST] --dpi 300               # flatten a PDF to pure raster (DST defaults to outputs/rasterize/)
.venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.manual_label PDF --page N [--out labels.json]  # manual cluster-label editor (GUI; --out defaults to outputs/labels/)
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

**Package layout note:** every folder under `rastervec/` is a PEP 420 namespace package
(no `__init__.py`) except `renderer/`, `Reader/Parallel/`, `pipelines/` and
`pipelines/sub_pipelines/`, which keep a docstring-only `__init__.py`. `tests/` keeps its full
`__init__.py` tree (pytest `importmode=prepend` + shared basenames).

**Pipelines:** the pipeline is defined in `rastervec/pipelines/` as flat, readable block
sequences (`native = extract_native_text(page)` etc.) — see the `pipelines/` bullet below.
`pipeline.py` (the old `PipelineContext` + `STAGES` + `_run_stages` machinery) is gone; the
new rule is **new capability = one more named call in a `pipelines/` file**. Deskew + line/word
segmentation before OCR is a Radon transform (`skimage.transform.radon`) in `OCR/radon.py`
(an OCR-preprocessing concern, not a pipeline-orchestration `sub_pipelines/*.py` module),
replacing the old `OCR/Paddle_OCR/ink_segment.py`. There is one OCR backend now
(`PaddleRecBackend`, recognition-only over Radon-segmented word crops); the old light/heavy split
and `PaddleOcrBackend` full-detection path are gone.

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
concern (`Reader/`, `native_text.py`, `Vector/`, `Vector_Classification/`, `OCR/`, `Evaluation/`),
plus cross-cutting modules that don't belong to one concern (`models.py`, `output_types.py`,
`logging_setup.py`, `paths.py`, `pipelines/`, `renderer/`) and a `helpers/` package for
utilities shared across more than one concern (`geometry.py` — pure tuple math; `fitz_geometry.py`
— the same for live `fitz` objects; `clustering.py`; `iterutils.py`). Every stage is testable
independently of the others (every stage's *output* is a plain dataclass from `models.py`, no
`fitz` objects, except `Page.fitz_page` which `Reader` must hand to `Native`/etc.):

- **`models.py`** — all shared dataclasses (`PageMeta`, `Page`, `TextWord`, `VectorPath`,
  `DrawingVector`, `VectorRecord`, `OcrWord`, `TextVectorResult`, `ClusterOcrResult`).
- **`output_types.py`** — pydantic DTOs (`TextDTO`, `VectorDTO`, `NativePDFElements`) mirroring what
  a raw PyMuPDF `get_text("words")` word / `get_drawings()` drawing look like, built from the
  dataclasses above — the serialization/export shape for external consumers, not a replacement for
  the dataclasses used mid-pipeline.
- **`logging_setup.py`** — stdlib `logging` only. `configure_logging(level)` once at startup;
  `get_logger("stage_name")` returns `logging.getLogger("rastervec.stage_name")` per module.
- **`paths.py`** — `REPO_ROOT`, `OUTPUTS_DIR` (repo-root `outputs/`, gitignored), and
  `output_dir(name, *subparts)` (mkdir-p `outputs/<name>/…`). Every script/notebook that writes
  files sends them under one `outputs/<source>/` subfolder by default: `benchmark_notebook/`
  (benchmark notebook), `benchmark_cli/` (`benchmark.py --reconstruct-dir` default), `labels/`
  (`manual_label.py` / `view_auto_labels.py --out` default), `rasterize/` (`rasterize_pdf.py` dst
  default).
- **`helpers/geometry.py`** — pure-math helpers, originally ported from the inspector tool's
  `pdf_model.py` (`point_angle`, `line_length`, `quad_angle`, `matrix_rotation`, `matrix_scale`,
  `make_oriented_quad`, `rect_gap`, `union_bbox`, etc.), shared by `native_text.py` and `Vector/` (and
  the inspector) so none of them duplicate this math independently.
- **`helpers/clustering.py` — `Clustering`** *(implemented)*: pure-Python (no scipy/sklearn) spatial
  hash grid + union-find for `cluster_spatial` (buckets items into grid cells sized by `threshold`,
  unions items in neighboring cells whose `geometry.rect_gap` ≤ `threshold` —
  `Vector_Classification/clusters/cluster_filters.py`'s `cluster_spatial_groups` reuses this same
  method at the group level, treating each group as one atomic item, and `pipelines._steps.spatial_regroup`
  reuses it too, merging purely on bbox proximity regardless of layer/color), then O(k²) pairwise
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
- **`Reader/reader.py` — `Reader`** *(implemented)*: opens a PDF (a bad path raises `ValueError`,
  not a raw fitz error), hands out `Page` objects one at a time (`get_page(index)`,
  `iter_pages(indices=None)`), each carrying a `PageMeta` snapshot (mediabox, rotation normalised to
  [0,360), dimensions) plus the live `fitz.Page`. `page.meta.index` is always the source-PDF page
  index and round-trips through `get_page`.
- **`native_text.py`** *(implemented)*: `extract_native_text(page) -> list[TextWord]` — one
  `TextWord` per `get_text("words")` word (geometry + `block_no`/`line_no`/`word_no`), font/size/
  colour/direction/`wmode` joined from the best-overlapping `get_text("dict")` span (`_Span`
  dataclass), above `_MIN_SPAN_OVERLAP`. Produces correctly oriented quads even for rotated text
  (`_oriented_quad` → `geometry.make_oriented_quad`). Split into small private module functions
  (`_extract_spans`/`_extract_words`/`_match_word_to_span`/`_oriented_quad`/`_to_word`) so each is
  independently testable against a synthetic `fitz.Page`.
- **`Vector/vector.py`** *(implemented)*: `extract_paths(page) -> list[VectorPath]` walks
  `page.fitz_page.get_drawings()`, emitting one `VectorPath` per drawing item (`l`/`re`/`qu`/`c`),
  tagged with its parent drawing's `seq` (drawing index) plus stroke/fill color, width, dashes,
  closed, layer, and item-level `bbox`/`points`. `separate_by_layer`/`separate_by_color` delegate to
  `Vector/layer_color_separation.py`'s module-level functions of the same
  name, which group paths by `layer` (`""` for none) / by `stroke_color` if set else `fill_color`
  else `None`.
- **`Vector_Classification/`** — classification of extracted paths into text candidates vs. drawing
  content. `classification.py`'s module functions are the orchestrator (`cluster`, `classify`,
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
  any filter step drops along the way is drawing content (`pipelines._steps.build_drawing_output`
  folds every `role="dropped"` category into `drawing_vectors`), and everything that survives the
  whole chain is a *text candidate*, handed to `unique_clusters`/`fast_text_detect`/`ocr_compare` —
  OCR success/failure is the actual signal for whether a cluster was text, not a pre-filter guess.

  `cluster(paths, page) -> list[StepResult]` runs the fixed chain in order; each
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
  `pipelines/sub_pipelines/vector_classification.py` key the whole chain's work by `GroupKey = (layer,
  color)` (from `color_separation`'s output), and `classification.cluster()` is only ever called
  with one bucket's paths at a time — two paths in different layers, or with different stroke/fill
  colors, are never spatially merged together, regardless of how close they are on the page.
- **`OCR/fast_detect.py` — `FastDetector`** *(implemented)*: see the `pipelines/_steps.py`
  entry below (`detect_text_fast`) for `detect`/`detect_tiled` usage.
- **`OCR/radon.py`** *(implemented)*: replaces the deleted
  `OCR/Paddle_OCR/ink_segment.py` — Radon-transform deskew + line/word split. See the
  `pipelines/` bullet below.
- **`OCR/Paddle_OCR/crop_normalize.py`** *(implemented, PIL only)*: `normalize_line_crop` — PIL port
  of `archive`'s `normalise_crop_for_ocr` (asymmetric 5%/30%-of-height white pad + resize to a fixed
  48px recognition line height, aspect preserved, width ≤ 1024).
- **`OCR/Paddle_OCR/ocr_backend.py` + `render_ocr.py`** — see the `pipelines/` bullet below
  (`PaddleRecBackend`, the one recognition-only backend; `RenderOCR.recognize_segmented` /
  `ocr_cluster` / `ocr`). Text *detection* is the Radon segmentation step
  (`OCR/radon.py`), not PaddleOCR.
- **`Evaluation/conversion.py`** *(implemented)*: three functions, each re-expressing a
  page's content as vector paths (`get_drawings()`) for a Vector_Classification known-answer test —
  **none ever rewrites pre-existing vector-path geometry** (an earlier version SVG-round-tripped the
  *whole* page, re-encoding the very CAD-text/drawing vectors a human labelled). Shared mechanics:
  `insert_pdf` for a verbatim page copy; `apply_redactions` with `PDF_REDACT_LINE_ART_NONE` to delete
  native text while leaving line art bit-identical, or `PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED` +
  `PDF_REDACT_IMAGE_REMOVE` to leave a text-only page; that text-only page's `get_svg_image()` →
  `fitz.open(filetype="svg")` → `convert_to_pdf()` (PyMuPDF emits each glyph as a filled `<path>`,
  never an SVG `<text>` or a raster fallback); everything at rotation 0 (canonical unrotated MediaBox
  space) with the original `/Rotate` re-applied at the end.
  - `convert_page_text_only` — native text as vector paths, **every drawing + image removed**. The
    benchmark's **auto**-ground-truth input.
  - `convert_page_drawings_only` — the original page with native text objects removed, **every
    drawing kept byte-for-byte** (images removed). The benchmark's **manual**-ground-truth input.
  - `convert_page_to_vector_text` — both overlaid (text-as-vectors on the untouched drawings). No
    longer used by the benchmark; kept as a general utility + for its tests.
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
  `native.extract` (never runs Conversion or any classification/clustering), groups words
  by `(block_no, line_no)` into line-level ground-truth regions (bbox via `helpers.geometry.
  union_bbox`, text joined in reading-direction order — x for horizontal lines, y for vertical),
  and sets `expected_rotation` to the most common quarter-turn among the line's words. This
  independence matters: an earlier
  version derived labels from the *converted* page's own surviving classification clusters, which
  meant a native word the classification chain's own filter steps wrongly dropped never became a
  label at all — silently excluded from ground truth rather than scored as a miss. Ground truth
  must not depend on what the system under test decided. `manual_label.py`'s `ManualLabelApp`
  (`python -m rastervec.Evaluation.Labelling.manual_label PDF --page N [--out labels.json]`;
  `--out` defaults to `outputs/labels/<stem>.json`) *does*
  need real clusters (a human has to click something), so it's the one place that still runs the
  real pipeline — via `classify_vectors(vectors.paths, page, verbose=True)` (sub_pipelines)
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
  undoes the last group/ungroup. The **inline label bar** below the toolbar — a persistent Text
  entry + Rotation dropdown + `Apply`/`Delete` (an earlier version used chained `simpledialog`
  popups that could vanish behind the topmost hover tooltip) — writes to `_label_targets()`:
  **every selected cluster** (one `Apply` labels them all with the same text+rotation) or, if
  nothing is selected, the **single** cluster right-clicked (which drops any selection and
  pre-fills the fields from its existing label). `Apply` (or Enter) upserts the manual
  `LabelEntry`(s), then clears the selection, empties Text and resets Rotation to 0; `Delete`
  removes the targeted label(s) the same way. The `<`/`>`
  buttons (or `PageUp`/`PageDown`) move between pages of the same PDF without relaunching — one
  `LabelSet` spans every page (`_load_page` reruns `classify_vectors` per page, `_page_entries()`
  scopes overlays/hover/label-bar to the current `page_index`), and the labels are saved on every
  page change. `LabelEntry`s loaded from `--out` that match no live cluster on the current page
  (every `source="auto"` entry, plus manual entries left stale by an edit) draw as dashed grey
  boxes, so the same window doubles as an auto-label viewer. Save/window-close writes the label
  file — not unit-testable (a real Tk event loop), smoke-test steps are in its own module
  docstring. `view_auto_labels.py`
  (`python -m rastervec.Evaluation.Labelling.view_auto_labels PDF --page N [--out labels.json]`)
  runs `auto_label_pdf`, merges its entries onto any existing `--out` file (default:
  `outputs/labels/<stem>_p<N>_auto_labels.json`), and opens `ManualLabelApp` on it.
- **`Evaluation/Evaluate/metrics.py` — `evaluate_metrics`** *(implemented, the current scorer)*: an
  **independent** metric suite — each metric is a separate reduction over one shared many-to-many
  `OverlapGraph` between ground-truth `GtRegion`s and `Prediction`s (per (gt, pred) edge: intersection
  area, IoU, `gt_coverage`, `pred_coverage`; plus an N:1 `assigned_preds_by_gt` so a gt line covered
  by several predicted clusters is scored as one), rather than the single greedy 1:1 match every
  `evaluate.py` metric shares. Pure / no `rastervec.pipeline` import. Every metric is a
  `Ratio(numerator, denominator)` holding **absolute page counts**; `MetricSuiteResult` bundles the 16
  ratio fields (+ 2 derived `*_f1`) + `per_stage_miss_counts` + `counts`. `aggregate_suite`
  **micro-averages** — `Ratio(Σ numerators, Σ denominators)` over the applicable pages, NOT the mean of
  per-page ratios; a `nan` denominator ("not applicable": empty gt, no candidates, zero misses,
  `clustering=None`) is excluded. Phase-1 metrics (20): page char/word multiset recall/precision/f1
  (`text_metrics.normalize_text` folds case + whitespace first), `region_concat_char_accuracy_all_gt`
  / `region_concat_char_accuracy_overlapping` (the only edit-distance metrics — per-gt-region
  `1 - CER` via `text_metrics.levenshtein` over the region's reading-order-concatenated overlapping
  predictions; `_all_gt` counts every gt region, `_overlapping` only those with a prediction),
  `pred_text_fully_contained_in_
  overlapping_gt_rate` / `gt_text_word_coverage_by_overlapping_preds`, `per_gt_best_single_pred_iou_
  mean` / `per_gt_union_pred_iou_mean` / `undetected_gt_area_ratio`, `rotation_accuracy_localized_gt`,
  `classification_recall_gt_reached_ocr` / `classification_precision_candidate_is_text`, and
  `gt_miss_attributed_to_{classification,fast,ocr_blank,not_found}_frac` + `per_stage_miss_counts`
  (`attribute_miss`). `METRIC_GROUPS` is the display-order source
  for `benchmark.format_report` + the notebook charts. `overlay_boxes` / `overlay_boxes_split` are
  data-only helpers (no rendering) returning `(bbox, rgb[, dashes])` for the benchmark's `boxes.pdf`
  pred-vs-GT overlay. **`EVAL_METRICS.md`** documents every metric's formula, both normalisation
  rules (text + aggregation), and the ~36-metric catalogue not yet built.
- **`Evaluation/Evaluate/adapters.py`** *(implemented)*: the only `Evaluation/Evaluate/` module that
  imports `rastervec.pipeline`. `gt_regions_from_labelset` / `predictions_from_cluster_ocr` /
  `text_candidate_boxes` (union bbox per `ctx.regrouped_clusters` cluster, blank OCR included; falls
  back to resolved bboxes for the archive legacy path) / `build_eval_inputs(ctx)` turn a `LabelSet` +
  `PipelineContext` into `metrics.evaluate_metrics` arguments.
- **`Evaluation/Evaluate/text_metrics.py`** *(implemented)*: `normalize_text` (upper-case, trim,
  collapse internal whitespace to one space), `char_multiset` / `word_tokens`, and pure-Python
  `levenshtein` / `char_error_rate` / `word_error_rate` (the standard text-diff — **not**
  `difflib.SequenceMatcher`). `levenshtein` is used by `metrics.py`'s two
  `region_concat_char_accuracy_*` metrics.
- **`Evaluation/Labelling/label_schema.py::split_labelset_by_source`** splits a mixed-source
  `LabelSet` into `{"auto": …, "manual": …}` so the benchmark can score auto-derived and
  human-entered ground truth as separate runs.
- **`Evaluation/Evaluate/golden_schema.py` + `golden_regression.py`** *(implemented)*: a
  hand-curated golden/regression test suite, separate from the `LabelSet`/benchmark machinery
  above — one `GoldenCase` (pydantic, `golden_schema.py`, pure) is a single curated snapshot of
  one cluster's/segmentation's output at one of three stages (`"classification"`, `"fast"`,
  `"word_split"`), labelled `"positive"` (from that stage's kept/passed pool) or `"negative"`
  (from its dropped pool; `"word_split"` has no structural pass/drop signal, so both labels there
  are purely the curator's own visual judgement) — stored in one shared `GoldenCaseBank`
  (`upsert_case`/`save_cases`/`load_cases`, JSON at `outputs/regression_cases/cases.json`, not
  one file per PDF, since a bank only ever holds a handful of self-describing cases rather than
  an exhaustive per-PDF labelling). `golden_regression.py` (the pipeline-facing half, alongside
  `adapters.py` the only other `Evaluation/Evaluate/` module importing `rastervec.pipelines`)
  builds each stage's candidate pool from the same `PipelineResult` fields the visualization
  notebook's own `render_*` functions read (`text_clusters`/`clustering`'s `role="dropped"`
  categories, `fast_passed`/`fast_dropped`, `regrouped_clusters` zipped with `segmentations`),
  `capture_case` turns a picked `Candidate` into a `GoldenCase` (`label_schema.cluster_signature`
  for debug provenance only), and `compare_case`/`run_regression` replay a bank against fresh
  pipeline runs — matching is **IoU-based**, not signature-based (an unrelated member-count shift
  would make an exact-equality signature falsely read as "not found"): the current pool entry
  closest by `bbox_iou` to the stored bbox must clear `DEFAULT_IOU_THRESHOLD` (0.85 — deliberately
  tighter than `metrics.MetricConfig.iou_edge_min`'s 0.10, tuned loose for benchmark-style
  detection matching, since this wants "did this basically not change"), then
  classification/fast compare `role` exactly and `word_split` greedily IoU-pairs stored vs.
  current word boxes. `run_regression` runs the pipeline once per distinct `(pdf_path,
  page_index)` among a bank's cases, not once per case.
- **`notebooks/golden_case_curation.ipynb`** *(implemented)*: the curation notebook over
  `golden_regression.py` — run the pipeline once (`verbose=True`), then per stage a
  browse-candidates / pick-an-index / capture-and-save cell triplet, then a shared **Replay**
  section (`run_regression` + `format_regression_report`) that re-checks every case in the bank
  (from any prior session, not just the current one) — the same cell to re-run later as a
  regression check.
- **`Evaluation/Evaluate/variants.py`** *(implemented)*: `PipelineVariant` (name, `engine`
  current/legacy, `enable_fast`) + the `VARIANTS` registry (`current` [default],
  `current_nofast`, `legacy`) + `DEFAULT_VARIANTS` + `resolve_variant`. Adding an ablation = one
  `VARIANTS` entry; `benchmark_jobs.run_page_task` reads it and threads `enable_fast` into
  `rastervec.pipelines.current.run_pipeline`.
- **`Evaluation/Evaluate/legacy_adapter.py`** *(implemented)*: runs archive's
  `raster_parser.main_pipeline_extract.extract` unmodified and reshapes its output into
  `ClusterOcrResult`s for `metrics.evaluate_metrics` (the `legacy` variant). Archive's OCR code
  targets **PaddleOCR 2.x** but the venv ships **paddleocr 3.4.x** (needed by the current
  pipeline's PP-OCRv5), so `_ensure_archive_importable` also installs
  `Evaluation/Evaluate/_paddle_compat.py`'s shim: it swaps `paddleocr.PaddleOCR` for
  `_PaddleOCRv2Compat`, which translates archive's removed ctor kwargs (`use_gpu`/`show_log`
  dropped; `drop_score`→`text_rec_score_thresh`, `det_limit_side_len`→`text_det_limit_side_len`,
  `det_limit_type`→`text_det_limit_type`, `use_angle_cls`→`use_textline_orientation`), reshapes
  3.x `predict()` results back to 2.x `[[[box,(text,conf)],...]]`, and exposes a
  `text_recognizer` callable over `paddleocr.TextRecognition` for `region_ocr._batch_recognize`.
  Nothing in `archive/` is touched — only the symbol it imports. Shim install is idempotent and
  scoped to a legacy run.
- **`Reader/Parallel/`** *(implemented)*: parallelism for the benchmark. `pool.py` — `worker_init`
  (pins `OMP`/`MKL`/`OPENBLAS` to 1 per worker), `default_worker_count`, `warmup` (builds the
  PaddleOCR heavy + light-rec + FAST caches in the *calling* process, so a spawn pool started next
  finds the models on disk and no worker races the first-run download —
  `PaddleRecBackend.warmup()` / `FastDetector.warmup()`
  classmethods force the existing lazy `_engine()`/`_model()` path), and
  `run_parallel(items, fn, *, workers, desc)` — an input-order map that is a plain serial loop when
  `workers <= 1` and a spawn `ProcessPoolExecutor` otherwise. Processes not threads: the PaddleOCR
  engine + FAST model module caches are unlocked shared singletons and PyMuPDF is not reentrant.
  `benchmark_jobs.py` — the picklable per-page job: `PageTask` (pdf/page/manual entries/`variant`/
  reconstruct dir/…) → `run_page_task` (resolves `task.variant` via `variants.resolve_variant`,
  dispatches to the current or legacy runner) → `PageResult` (`variant`, auto + manual
  `MetricSuiteResult`, stage durations, formatted report blocks, a few PNG-bytes `ShowcaseSample`s,
  `error`). **Each variant is run twice per page** on disjoint inputs — `convert_page_text_only`
  scored vs the `auto` labels, `convert_page_drawings_only` scored vs the `manual` labels (the
  manual run only when the page has manual labels) — so the two GT sources are scored against
  physically separate runs and can't contaminate each other's precision. Each of the (up to) four
  runs per page is wrapped in its own `try/except`: a failed run leaves that field `None` + a
  `report_blocks` line; a whole-job failure lands in `PageResult.error`. `run_benchmark(tasks, *,
  workers, desc)` is the thin `run_parallel(tasks, run_page_task, …)` wrapper used by both
  `benchmark.py` and the notebook. Per-variant reconstruct output goes to
  `RECONSTRUCT_DIR/<stem>_p<N>_<variant>/`.
- **`Evaluation/Evaluate/benchmark.py`** *(implemented)* — the CLI wiring Conversion → auto_label →
  a real full pipeline run → `metrics.evaluate_metrics` together: `python -m
  rastervec.Evaluation.Evaluate.benchmark --pdf PATH [--pdf PATH2 ...] --pages 0,1,2
  [--iou-threshold 0.1] [--reconstruct-dir DIR] [--workers N] [--variants current,legacy]`
  (`--iou-threshold` = `MetricConfig.iou_edge_min`; `--workers N>1` runs pages across
  `Reader/Parallel`'s spawn pool; `--variants` selects which `variants.VARIANTS` to run and compare;
  `--reconstruct-dir` defaults to `outputs/benchmark_cli/reconstructions/`).
  `run_one_page` is a thin wrapper over `Reader/Parallel/benchmark_jobs.run_page_task` (auto labels,
  returns that page's `MetricSuiteResult`); `main()` runs one `PageTask` product per selected variant
  and prints `format_aggregate_comparison` + `format_variant_timing_comparison` across them.
  `--reconstruct-dir` writes the per-page output PDFs (see the notebook bullet). `format_report` /
  `aggregate_results` / `format_aggregate` / `format_aggregate_comparison` /
  `format_variant_timing_comparison` / `summarize_stage_timings` are pure and unit-tested; `main()`'s
  actual OCR-backed path is a manual smoke test only (real PaddleOCR, first run downloads models —
  matches the existing `RASTERVEC_RUN_OCR_TESTS`-gated convention for OCR-dependent tests).
  `notebooks/benchmark_vector_classification.ipynb` is the interactive counterpart: it scores every
  variant in an editable `VARIANTS_TO_RUN` list (default `["current",
  "legacy"]`) over a `collect_dataset` tree (mixed `.pdf` + `manual_label.py` sidecar `.json`).
  `build_tasks(variant)` makes one `PageTask` per deduped `(pdf, page)`; the run cell loops
  `run_benchmark(build_tasks(v), …)` per variant into `*_by_variant` dicts; `collect_results` splits
  each variant's `PageResult`s into auto/manual metric lists + stage timings + the showcase pool and
  writes a per-variant report `.txt` (all output under `outputs/benchmark_notebook/`). Comparison
  cells print `format_aggregate_comparison` (metric
  rows × variant columns) and `format_variant_timing_comparison` (per-stage median seconds ×
  variant, + delta-vs-first). Per page × variant (when `RECONSTRUCT_DIR` is set) the job writes one
  folder `RECONSTRUCT_DIR/<stem>_p<N>_<variant>/` with five PDFs: `input_auto.pdf` /
  `input_manual.pdf` (the two disjoint pipeline inputs), `current.pdf` / `legacy.pdf` (each a
  text-only reconstruction of that run's two halves merged), and `boxes.pdf` — the pred-vs-GT
  overlay via `metrics.overlay_boxes_split` → `renderer.render_boxes_pdf`: **dashed** = auto GT,
  **solid** = manual GT, **dotted** = a prediction; green = matched, red = a GT no prediction
  reached, yellow = a prediction over no GT. A showcase cell plots `SHOWCASE_N` of the
  `ShowcaseSample` PNGs from the first `current_*` variant, sampled ~50/50 between non-blank (PASS)
  and blank (FAIL) OCR readings.
- **`renderer/` — module-level functions, no `Renderer` class** *(rendering helpers, not a pipeline
  stage)*: a package split by output concern — `png.py` (rasterize vector paths for OCR / FAST
  input), `pdf.py` (`render_reconstructed_page`, `render_reconstructed_pdf`, and `render_boxes_pdf`
  — colored rectangle outlines, each entry `(bbox, rgb)` or `(bbox, rgb, dashes)` where `dashes` is
  a PyMuPDF dash string / `None`), `svg.py` (`render_page_svg`, a thin
  `get_svg_image()` wrapper), and `_shapes.py` (shared). Import straight from `rastervec.renderer`
  (`from rastervec.renderer import render_vector_cluster`, etc.). `notebook.py` is the exception:
  notebook-only display plumbing (`RenderResult`, `visualize`, `draw_paths`/`draw_polys`/
  `draw_bboxes`, `page_setup`, ...) for `pipeline_stage_visualization.ipynb`, deliberately **not**
  re-exported through `renderer/__init__.py` — it imports matplotlib, and this package is imported
  by the real pipeline itself (`render_vector_cluster`, `render_reconstructed_page`, ...), so
  folding it into the package's own `__init__` would drag matplotlib into every pipeline run's
  import graph. Import it directly (`from rastervec.renderer.notebook import ...`); every stage
  module's own `render_<stage_name>` function does this lazily, inside the function body, for the
  same reason.
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
  `vector.extract_records` normalises a tuple `lineCap` from `get_drawings()` to a plain int.
  `png.render_vector_cluster(paths, dpi)` *(implemented)*
  isolates a cluster onto a fresh single-page PyMuPDF document sized to the cluster's own bbox plus
  an asymmetric OCR render border, computed by the private `_cluster_frame` helper: each side is at
  least `max(4pt, largest member's stroke_width)` (clipping safety), then further widened per axis
  by a fraction of the cluster's own bbox height — tight vertically (5%, keeps glyphs filling the
  frame) and generous horizontally (30%, keeps edge glyphs from clipping) — ported from
  `archive/raster_parser`'s Type-2 full native-to-OCR pipeline
  (`parsing/parser.py::normalise_crop_for_ocr`), which padded an already-rasterized crop with
  `cv2.copyMakeBorder` before resizing it for PaddleOCR; here the same ratios expand the render
  frame itself, in PDF-point space, before the page is rasterized, rather than as a post-render
  pixel-fill step. `render_vector_cluster` then replays each drawing's items via `replay_drawing_paths`,
  then rasterizes at `dpi` and returns a PIL `Image` — reusing PyMuPDF's own path/curve/fill
  rendering rather than reimplementing rasterization by hand.
  `png.pixel_to_page_bbox(paths, dpi, pixel_points)` inverts
  `_cluster_frame`'s same transform to map pixel-space points (e.g. Paddle's detected text-region
  corners, from a render of that exact `paths`/`dpi`) back into PDF page space — used by
  `RenderOCR.ocr_cluster` to compute a `TextVectorResult.ocr_bbox`.
  `png.page_points_to_pixel(paths, dpi, page_points)` is its exact forward inverse (page space →
  that render's pixel space) — used by the visualization notebook to draw Paddle's returned
  page-space boxes back onto the rendered cluster image. `png.render_page_paths(paths,
  page_meta, dpi)` is the whole-page counterpart (every given path drawn onto one page-sized canvas,
  no isolation/padding, no rotation applied) — used as FAST's own detection input by
  `fast_text_detect` (see below).
  `pdf.render_reconstructed_page(page_meta, *, native_words=None, drawing_vectors=None,
  ocr_results=None, text_boxes=None, zoom=1.0)` *(implemented, visualization-notebook preview — not
  OCR input, not `evaluation.py`'s real reconstruction stage)*: redraws whatever elements are passed
  onto a fresh blank page sized/rotated to match `page_meta`, then rasterizes at `zoom` the same way
  the
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
  finicky with empty strings). `text_boxes` is a generic
  `list[tuple[text, bbox, rotation]]` fed through the same `_place_text` path — the ground-truth
  reconstruction (label text) uses it, keeping the renderer decoupled from `label_schema`. Both
  `render_reconstructed_page` and `pdf.render_reconstructed_pdf(...) -> bytes` (the PDF-bytes
  variant, for a selectable-text comparison file — used by the benchmark notebook) share one
  private `_build_reconstructed_doc`.
- **`pipelines/`** — the pipeline, as flat readable block-sequence files. This is where a
  contributor reads to learn what runs; **new capability = one more named call added here.**
  - **`pipelines/current.py`** — `run_pipeline(pdf_path, page_index, *, enable_fast=True,
    verbose=False) -> PipelineResult`, plus `STEP_NAMES` and the `python -m
    rastervec.pipelines.current --pdf … --page N [-v] [--no-fast]` CLI. Delegates to
    `_common.run_current_pipeline`, whose body is the literal 9-step sequence: `read` → `native`
    (`extract_native_text`) → `vectors` (`extract_vectors`) → `classify` (`classify_vectors`) →
    `fast` (`detect_text_fast`) → `regroup` (`spatial_regroup`) → `segment` (`segment_for_ocr` —
    Radon deskew + line/word split) → `ocr` (`recognize`) → `drawing` (`build_drawing_output`).
    `STEP_NAMES` are the `step_durations` keys (replacing the old 12 `stage_keys()`).
  - **`pipelines/legacy.py`** — `run_pipeline(...)` wrapping `Evaluation/Evaluate/legacy_adapter`
    (archive's unmodified pipeline), reshaped into a `PipelineResult` (only the OCR-scored
    fields are meaningful).
  - **`pipelines/result.py`** — `PipelineResult` (one dataclass): final-output fields always
    populated (`page`, `native_words`, `drawing_vectors`, `ocr_results`, `cluster_ocr_results`,
    `text_clusters`, `regrouped_clusters`, `clustering` (`dict[GroupKey, ClusteringStageResult]`),
    `cluster_groups`, `fast_dropped`, `ocr_failed`, `step_durations`, `engine`); intermediate
    fields (`vector_paths`, `paths_by_layer[_color]`, `similarity_groups`,
    `cluster_similarity_id`, `fast_passed`, `fast_result`, `segmentations`, `step_outputs`, …)
    are `None` unless `verbose=True`. Also the new home for `GroupKey`, `ClusteringStageResult`,
    `FastPageResult`, `StepOutcome`. **Never pickled** — it holds numpy arrays / PIL images.
    The pipeline closes its `Reader` before returning, so `result.page.fitz_page` is `None`;
    `PipelineResult.open_page()` reopens the source PDF (via `page.doc_path` / `page.meta.index`)
    and yields a live `fitz.Page` for rasterization.
  - **`pipelines/_steps.py`** — the thin step functions the pipeline files call: `read_page`,
    `extract_native_text`/`extract_vectors` (re-exports), `detect_text_fast` (whole-page
    `render_page_paths` + `FastDetector.detect_tiled`, per-cluster `_sample_mask` scoring min'd
    across the similarity group, `> FAST_COMBINED_KEEP_THRESHOLD` passes; `enable_fast=False` is
    a pass-through), `spatial_regroup` (`cluster_spatial` merge of touching clusters regardless of
    layer/color, similarity-id carry-forward), `build_drawing_output` (folds
    classification drops + FAST drops + OCR blanks into `DrawingVector`s in source draw order).
  - **`pipelines/_common.py`** — `run_current_pipeline` + `StepTimer` (records per-step
    wall-clock; on `verbose=True` a failing step is logged + recorded as a `StepOutcome` and
    suppressed so partial state survives; on a normal run it propagates — the benchmark wraps
    each run). **Simplification vs the old pipeline:** no never-crash-per-stage behaviour on the
    non-verbose path.
  - **`pipelines/sub_pipelines/vector_classification.py`** — `classify_vectors(vector_paths,
    page, *, verbose=False) -> ClassificationResult`: `separate_by_layer` → `separate_by_color`
    → per-`(layer, color)`-bucket `_classify_bucket` → gather every bucket's surviving "kept"
    clusters (no copy — `manual_label` keys Ungroup on `id(cluster)`) → `group_similar_clusters`
    → collect every `role="dropped"` group as drawing content. `_classify_bucket` is the fixed
    12-step chain, one named `filter_*`/`compute_*`/`cluster_spatial_groups` call + one
    `steps.append` per step (moved verbatim out of `classification.cluster()`, which is now a
    one-line shim into it).
  - **`pipelines/sub_pipelines/ocr.py`** — `segment_for_ocr(clusters, *, dpi=300)` (Radon
    segment each cluster render — the detection step) and `recognize(segmentations, clusters,
    page, *, backend=None, similarity_id=None) -> OcrResult` (one real `RenderOCR.
    recognize_segmented` per similarity group, reused for the rest; blank reading folds into
    `failed`; per-call `ocr_seconds`).
  - **`OCR/radon.py`** — replaces `ink_segment.py`; lives under `OCR/` (a top-level file, matching
    `OCR/fast_detect.py`'s precedent) rather than `pipelines/sub_pipelines/`, since it's an
    OCR-preprocessing concern, not pipeline orchestration. `segment_cluster(image,
    dpi_used) -> ClusterSegmentation` (`skew_deg`, `line_spacing_px`, `word_crops` (deskewed
    grayscale), `word_corners` (4 corners each, in the *original* render's pixel space),
    `render_dpi`, verbose-only `deskewed_gray`/`profile`). `estimate_skew` scores each swept
    Radon projection angle by `sum(projection**2)` (Postl criterion — sharpest profile is
    parallel to the text baseline), coarse full sweep + fine sweep (`RADON_*` config), maps to a
    `(-90, 90]` deskew angle; `_rotation` builds the exact forward/inverse affine so word-box
    corners map back to the original render. `split_words` splits the column (x-extent) profile
    on a `gap_threshold` via `_split_on_gaps`/`_ink_runs`/`_group_runs` — `segment_cluster`
    computes that threshold once per cluster (`_cluster_gap_threshold`, floored by
    `RADON_MIN_GAP_PX`), as the median of every line's inter-run gaps (`_line_gaps`) pooled
    together, so every line in the cluster splits on the same shared threshold rather than each
    line recomputing its own from just its own (often noisy, small-sample) gaps. Each word's
    y-extent is then taken from ink within just that word's own column slice, not the whole
    line's ink bbox — so two words on the same line with different
    glyph heights (e.g. one with a descender, one without) get genuinely different, tight
    `word_corners` boxes rather than sharing the line's full ink height. The 0-vs-180 (and
    90-vs-270) flip Radon can't resolve is left to `RenderOCR.recognize_segmented`.
    `render_radon(res: PipelineResult) -> RenderResult` (notebook-only, appended at the bottom of
    this file) re-renders each segmented cluster's original pre-deskew image via
    `renderer.render_vector_cluster` and draws its real `word_corners` polygons on top — see the
    `notebooks/pipeline_stage_visualization.ipynb` bullet below.

  `PipelineResult.to_native_pdf_elements()` is the serialization boundary (ported from the old
  `PipelineContext`).
- **`OCR/Paddle_OCR/ocr_backend.py`** — `OcrBox`, the `OcrBackend` Protocol (one method,
  `recognize_crops(crops: list[np.ndarray]) -> list[OcrBox]`), and `PaddleRecBackend` (the only
  implementation): PaddleOCR standalone `TextRecognition` (`config.OCR_REC_MODEL`, default
  `PP-OCRv5_mobile_rec`), `normalize_line_crop` each crop → one batched `predict` → one `OcrBox`
  per crop in input order. Engine cached at class scope, `warmup()`. **No text detection here** —
  that's the Radon step. `PaddleOcrBackend`/`LightPaddleOcrBackend`/`DocImgOrientationClassification`
  and their geometry helpers were all deleted.
- **`OCR/Paddle_OCR/render_ocr.py` — `RenderOCR`** *(implemented)*: `recognize_segmented(seg,
  cluster, page) -> TextVectorResult` — `backend.recognize_crops(seg.word_crops)` upright and
  again on the 180-rotated crops, keep the higher length-weighted-confidence set (that settles
  the flip), map each surviving word's `seg.word_corners` through `renderer.pixel_to_page_bbox`,
  join left-to-right, `rotation_used = round(skew/90)*90 + flip`. `ocr_cluster(cluster, page,
  dpi=300)` = `render_cluster_for_ocr` → `segment_cluster` → `recognize_segmented` (kept for
  non-pipeline callers); `ocr(image)` is the raw-image convenience used by the inspector.
- **`notebooks/pipeline_stage_visualization.ipynb`** *(implemented, on the new `pipelines/` API)*:
  one `res = run_pipeline(PDF_PATH, PAGE_INDEX, enable_fast=…, verbose=True)` run, then one
  `visualize(stage_key, render_<stage_name>(res), step_outputs=…, original=…, matrix=…)` cell per
  pipeline step — `visualize` and the generic pixel-drawing plumbing it shares across every stage
  (`RenderResult`, `draw_paths`/`draw_polys`/`draw_bboxes`, `page_setup`, ...) live in
  `renderer/notebook.py`; the stage-specific part (*what* to draw) is one `render_<stage_name>`
  function living next to that stage's own code (`native_text.render_native`,
  `Vector.vector.render_vectors`, `Vector_Classification.classification.render_layers` /
  `render_layer_color_buckets` / `render_clustering_steps` / `render_text_candidates`,
  `OCR.fast_detect.render_fast`, `pipelines._steps.render_regroup` / `render_drawing`,
  `OCR.radon.render_radon`, `OCR.Paddle_OCR.render_ocr.render_ocr_results`). "Segment (Radon)" and
  "PaddleOCR" are two separate sections/cells (`segment` and `ocr` are already two distinct
  `STEP_NAMES`) rather than one combined cell, so the Radon step's own pass/fail/timing is now
  visible too. `render_text_candidates` reports similarity grouping as a plain original-vs-unique
  cluster count in its note, not a per-similarity-group image overlay (there can be dozens).
  `VARIANT` picks a `current`-engine `variants.VARIANTS` entry (`current` / `current_nofast`) for
  its `enable_fast`. The pipeline always runs all 9 steps (PaddleOCR included); writes no files.

`scripts/rasterize_pdf.py` (outside `rastervec/`, a one-off utility not a pipeline stage): flattens
every page of a PDF to an image and rebuilds a pure-raster PDF from those images — not currently
consumed by anything in `rastervec/` (kept for possible future raster-image work).

`tests/rastervec/` mirrors `rastervec/`'s own folder layout (e.g. `tests/rastervec/Reader/
test_reader.py` for `rastervec/Reader/reader.py`, `tests/rastervec/Vector_Classification/
test_classification.py` for `Vector_Classification/classification.py`, `tests/rastervec/renderer/
test_png.py` for `rastervec/renderer/png.py`); modules that stay at
`rastervec/`'s top level (`output_types.py`) keep their tests at
`tests/rastervec/`'s top level too. `tests/conftest.py`'s `synthetic_pdf_factory` builds small
in-memory PDFs via `fitz.open()`/`insert_text`/`set_rotation` — preferred over `references/*.pdf`
for unit tests since those are gitignored and give no exact expected values to assert against.

### Adding a new `rastervec` module or pipeline stage

Three things, all following the existing pattern:
1. Define the module's dataclass(es) in `models.py` if they don't exist yet, and give the module
   its own concern folder under `rastervec/` (or a new file in an existing one) with real logic
   split into small private functions per sub-step so each is independently testable. Give the
   folder one documented high-level entrypoint function.
2. Add one **named call** to the relevant `pipelines/` file — a new line in
   `_common.run_current_pipeline` (wrapped in `with timer("<name>"):`, add the name to
   `STEP_NAMES`), or a step in a `sub_pipelines/*.py` block sequence — plus a thin adapter in
   `_steps.py` if it needs one. Add its output to the `PipelineResult` constructor (always-on
   or verbose-only). No registry, no `StageSpec`.
3. Add a `render_<stage_name>(res: PipelineResult) -> RenderResult` function next to the stage's
   own code (reading `res.<field>`, lazily importing `RenderResult`/`renderer.notebook` inside the
   function body so matplotlib stays out of the pipeline's own import graph), plus a thin cell in
   `notebooks/pipeline_stage_visualization.ipynb` calling
   `visualize(stage_key, render_<stage_name>(res), step_outputs=outputs, original=ORIGINAL,
   matrix=MATRIX)`.
Also add tests under the matching `tests/rastervec/` subfolder using the synthetic PDF fixtures,
and new third-party dependencies to `requirements.txt` only when the stage that needs them is
actually implemented.
