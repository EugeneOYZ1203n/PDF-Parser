# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A raster-to-vector pipeline project for architectural/engineering shop drawings
(see `references/*.pdf`, gitignored sample PDFs). `rastervec/` is the only package here —
the standalone PDF-layer inspector tool that used to live at the repo root now lives inside it,
at `rastervec/Evaluation/inspector/` (see below).

`rastervec/` is the actual extraction pipeline, organized as **three pluggable phases behind one
shared harness**, not a fixed step sequence: **Phase 1** (`P1_Reading_Native/`, always the same)
opens the PDF and extracts native text + raw vectors + page/embedded images; **Phase 2**
(`P2_Raster_To_Vec/`, pluggable — `Stub` no-op, or `Junction`, a ported classical raster→vector
pipeline) turns Phase 1's images into additional vectors; **Phase 3**
(`P3_Vector_Parsing/`, pluggable — `VectorClassification`, `FastIntoPaddle`, or
`LegacyRecreation`) takes Phase 1's + Phase 2's vectors and produces the final vectors + OCR'd
text. `core/` is the orchestrator + registry + the stable public API surface, and `commons/` is
the shared foundation (dataclasses, geometry/rendering/logging/paths helpers) every phase builds
on. See "`rastervec/` architecture" below for the full breakdown, including the hard rule that
sibling P2/P3 backends share **zero** code with each other — each is fully self-contained,
duplicating whatever infra it needs rather than importing a sibling's.

A separate, unrelated **`legacy`** engine (`pipelines/legacy.py` + `Evaluation/Evaluate/
legacy_adapter.py`) runs `archive/raster_parser`'s own pre-`rastervec` pipeline unmodified, kept
only as a benchmark comparison baseline — it's not one of the three phases and ignores `p2`/`p3`
entirely (see the `variants.py`/`legacy_adapter.py` bullets below). `P3_Vector_Parsing/
LegacyRecreation/` is a *different* thing: a genuine from-scratch port of that same old
algorithm onto `commons.models` types, selectable as a normal P3 backend.

Several top-level `rastervec/` folders from before this phase split still exist but are now dead
code, not imported by anything live: `rastervec/OCR/`, `rastervec/Vector/`,
`rastervec/Vector_Similarity/`, `rastervec/Reader/` (superseded by `P1_Reading_Native/reader.py`
+ `core/parallel/`), and `rastervec/pipelines/current.py`/`_steps.py`/`_common.py` (superseded by
`core/pipeline.py` + the P3 backends' own `parse.py`/`steps.py`). `rastervec/pipelines/legacy.py`
is the one live exception in that folder — it's what `pipeline: "legacy"` actually calls. These
dead folders haven't been deleted yet; don't build on them.

The `Evaluation/` package holds a benchmarking suite (`Conversion/` — native text → vector-text
PDF; `Labelling/` — manual + automatic ground-truth labelling; `Evaluate/` — accuracy metrics
against those labels) plus the inspector tool — see "rastervec architecture" below.
`junction_cnn/` and `hawp/` at the repo root are unrelated, independent experiments (the CNN
junction-detector approach `junction_cnn/` explored was superseded by the classical, no-ML
pipeline ported into `P2_Raster_To_Vec/Junction/junction_test/` — see that bullet); nothing in
`rastervec/` imports from `junction_cnn/`/`hawp/` directly (`P2_Raster_To_Vec/Junction/` vendors
its own copy of the classical pipeline instead).

## Commands

```
.venv/Scripts/python.exe -m pip install -r requirements.txt                        # install deps
.venv/Scripts/python.exe -m rastervec.Evaluation.inspector.inspector [path/to.pdf]  # run the PDF layer inspector
.venv/Scripts/python.exe -m rastervec.core.pipeline --pdf PATH --page N [--p2 Stub] [--p3 FastIntoPaddle]  # run the pluggable pipeline (P1 -> P2_REGISTRY[p2] -> P3_REGISTRY[p3])
.venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json      # config-driven per-stage report (PDFs + stats + dump.json per source PDF)
.venv/Scripts/python.exe scripts/pipeline_report_viewer.py <run>/<stem> [<run2>/<stem>]  # Tkinter viewer: source page + toggleable stage-PDF overlays (1-2 folders side by side)
.venv/Scripts/python.exe scripts/pipeline_report_benchmark.py --run DIR1 [--run DIR2]  # multiclass-score + chart benchmark report folders (1 or 2) on shared inputs
.venv/Scripts/python.exe -m pytest tests/ -v                                        # run rastervec's test suite
.venv/Scripts/python.exe scripts/rasterize_pdf.py SRC [DST] --dpi 300               # flatten a PDF to pure raster (DST defaults to outputs/rasterize/)
.venv/Scripts/python.exe scripts/label/master_label.py PDF [--dpi 300]              # full native+vector+raster label workflow, one outputs/labels/<stem>_label/ folder per PDF
.venv/Scripts/python.exe scripts/label/native_label.py PDF --page N [--out ...]     # auto-derive native-text ground truth (GUI-free)
.venv/Scripts/python.exe scripts/label/vector_label.py PDF --page N [--out ...]     # manual vector-text labelling (GUI)
.venv/Scripts/python.exe scripts/label/raster_label.py PDF [--out ...]              # manual embedded-image line/curve/text labelling (GUI)
.venv/Scripts/python.exe scripts/label/view_native_labels.py PDF --page N [--out ...]  # view native_label output in the vector_label editor
.venv/Scripts/python.exe scripts/label/label_viewer.py PDF_OR_LABEL_FOLDER          # read-only viewer over a master_label.py folder's three label sets
```

venv is **Python 3.10** (`.venv/pyvenv.cfg` → 3.10.11):
`py -3.10 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`. OCR runs
**paddleocr 2.x** (`paddleocr>=2.9,<3` + `paddlepaddle>=2.6,<3`, resolves to 2.10.0 / 2.6.2) on
the **PP-OCRv4** model family — the last family 2.x ships, and the API surface `archive/`'s
`raster_parser` OCR was written against, so the `legacy` benchmark variant needs no compatibility
shim. paddleocr 2.x is not numpy-2 compatible (`numpy>=1.24,<2` → 1.26.4) and does not pull in
`paddlex`. To move models: `config.OCR_VERSION` / `config.OCR_LANG`.

Two Windows-specific gotchas, both handled in-code: (1) `torch` must be imported **before**
`paddle`/`paddleocr` in a process — a paddle-first process fails torch's DLL load (`shm.dll`,
WinError 127, clashing OpenMP runtimes). The pipeline runs `fast` (torch) before `ocr`
(paddleocr); `pool.warmup()` warms FAST first; `PaddleRecBackend._engine()` does `import torch`
right before `from paddleocr import PaddleOCR`. (2) `torch` is pinned to `2.13.0+cpu` from the
PyTorch CPU index (`--extra-index-url` in `requirements.txt`) — the plain PyPI Windows wheel is
CUDA-enabled and won't load without a CUDA runtime.

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
(`PaddleRecBackend`, recognition-only over Radon-segmented word crops — paddleocr 2.x's
`PaddleOCR(...).text_recognizer` batch call, PP-OCRv4); the old light/heavy split and
`PaddleOcrBackend` full-detection path are gone.

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

See `docs/Glossary.md` for standardized group/cluster/global-group/similarity-group
terminology used throughout this section. `rastervec/` is organized into five buckets:
`commons/` (shared foundation), `core/` (orchestrator + registry + public API),
`P1_Reading_Native/` (the one, always-run extraction phase), `P2_Raster_To_Vec/` (pluggable
raster→vector backends), `P3_Vector_Parsing/` (pluggable vector-parsing/OCR backends) — plus
`Evaluation/`, `notebooks/`, `weights/` alongside them (benchmarking/dev tooling, not phase
code). **Sibling P2 backends (`Stub`/`Junction`) and sibling P3 backends
(`VectorClassification`/`FastIntoPaddle`/`LegacyRecreation`) import nothing from each other** —
each is fully self-contained, duplicating its own copy of any infra it needs (a FAST-style text
detector, a PaddleOCR engine wrapper, layer/color/width separation, Radon deskew, ...) rather
than sharing one. This is deliberate: it lets each backend be rewritten or torn out without ever
touching a sibling's files. `commons/` is the only layer every phase may depend on, and it stays
thin — pure foundation (dataclasses, geometry math, rendering primitives, logging, paths,
generic parallel-pool mechanics), never phase-specific business logic.

- **`commons/`** — the shared foundation:
  - **`models/`** (`page.py`, `text.py`, `vector.py`, `segment.py`, `image.py`) — all shared
    dataclasses: `PageMeta`/`Page`, `Text`, `Vector` (one drawing-item primitive, replacing the
    old `VectorPath`/`DrawingVector` split), `Segment`/`SegmentMeta`, `Image` (a raster image
    handed from Phase 1 to a Phase 2 backend — `array`, `bbox` in page space, source metadata).
  - **`output_types.py`** — pydantic DTOs (`TextDTO`, `VectorDTO`, `NativePDFElements`) mirroring
    what a raw PyMuPDF `get_text("words")` word / `get_drawings()` drawing look like, built from
    the dataclasses above — the serialization/export shape for external consumers.
  - **`logging_setup.py`** — stdlib `logging` only. `configure_logging(level)` once at startup;
    `get_logger("name")` returns `logging.getLogger("rastervec.name")` per module.
  - **`paths.py`** — `REPO_ROOT`, `OUTPUTS_DIR` (repo-root `outputs/`, gitignored), and
    `output_dir(name, *subparts)` (mkdir-p `outputs/<name>/…`).
  - **`helpers/`** — `geometry.py` (pure tuple math: `point_angle`, `rect_gap`, `union_bbox`,
    `compute_origin`, `make_oriented_quad`, etc.), `fitz_geometry.py` (the same for live `fitz`
    objects), `clustering.py` (`Clustering` — spatial hash grid + union-find `cluster_spatial`,
    `cluster_by_dimension`, `cluster_by_seq`, `group_by_overlap`), `iterutils.py`.
  - **`renderer/`** — module-level rendering functions (no `Renderer` class), split by concern:
    `png.py` (`render_vector_cluster`, `render_page_paths`, pixel↔page transforms), `pdf.py`
    (`render_reconstructed_page`/`_pdf`, and the three generic primitives every backend's own
    debug renderer builds on: `render_boxes_pdf` — colored bbox outlines; `render_text_pdf` — a
    `Text` list placed at its own bbox/rotation; `render_vectors_pdf` — a `Vector` list replayed
    recoloured), `svg.py`, `_shapes.py` (`replay_drawing_paths`), `stages.py` (the fixed
    `phase1`/`phase2`/`final`/`reconstructed` layer renderers `core.pipeline` results always get
    — see the `generate_pipeline_report.py` bullet below for the *additional*, per-backend debug
    layers each P2/P3 module renders itself).
- **`core/`** — the orchestrator, registry, and stable public API:
  - **`pipeline.py`** — `run_pipeline(pdf_path, page_index=0, *, p2="Stub",
    p3="FastIntoPaddle", enable_fast=True, verbose=False, compute=None, progress_counter=None) ->
    PipelineResult`. Body: `phase1 = P1.read_and_extract(...)` → `p2_vectors, p2_texts =
    P2_REGISTRY[p2](phase1.images, phase1.page)` → `p3_vectors, p3_texts =
    P3_REGISTRY[p3](phase1.vectors, p2_vectors, phase1.page, **forwarded_kwargs)` (forwarded
    kwargs — `enable_fast`/`verbose`/`compute`/`progress_counter`/`debug_out` — are only passed
    to a backend whose own signature declares that parameter, via `inspect.signature`). CLI:
    `python -m rastervec.core.pipeline --pdf PATH --page N [--p2 Stub] [--p3 FastIntoPaddle]
    [--no-fast] [-v]`. No `stop_after`/partial-run support — always a full Phase1→P2→P3 run.
  - **`registry.py`** — `P2_REGISTRY`/`P3_REGISTRY` (name → backend callable),
    `resolve_p2`/`resolve_p3` (`ValueError` listing valid names on a miss), `DEFAULT_P2="Stub"`,
    `DEFAULT_P3="FastIntoPaddle"`. Also `P2_RENDER_DEBUG`/`P3_RENDER_DEBUG` — a *separate*,
    optional registry of each backend's own `render_debug(debug_out, page_meta) ->
    list[(stage, label, hex, pdf_bytes)]` function (a backend with nothing to render, e.g. Stub,
    simply isn't in these dicts). Adding a new backend = one folder implementing the
    `Phase2Backend`/`Phase3Backend` interface, plus one line in each applicable dict here.
  - **`interfaces.py`** — the `Phase2Backend`/`Phase3Backend` `Protocol`s (structural, not
    enforced at runtime): `Phase2Backend.__call__(images, page) -> (vectors, texts)`;
    `Phase3Backend.__call__(vectors_p1, vectors_p2, page) -> (vectors, texts)`.
  - **`result.py`** — `PipelineResult`: `page` (`page.fitz_page` is `None` — use
    `PipelineResult.open_page()` to reopen the source PDF), `texts`, `vectors`, `step_durations`,
    `p2`, `p3`, `extra: dict` (verbose-only: `extra["phase1"]` — the whole `Phase1Result`,
    `extra["phase2_vectors"]`/`["phase2_texts"]`, and `extra["p2_debug"]`/`["p3_debug"]` — each
    backend's own opaque `debug_out` stash, read back by that same backend's `render_debug`),
    `step_outputs` (per-step `StepOutcome`, verbose-only). No `engine` field and no per-stage
    named-Optional fields the way the old (now-dead) `pipelines/result.py::PipelineResult` had —
    intermediates live entirely in the generic `extra` bucket since the three P3 backends'
    shapes genuinely differ.
  - **`api.py`** — `extract`/`extract_svg`, the stable export surface for other codebases to
    import (thin wrappers over `run_pipeline`).
  - **`parallel/`** — `pool.py` (Pool 1/Pool 2 mechanics: `worker_init`, `warmup`, `run_parallel`,
    `compute_pool` — moved here from the old `Reader/Parallel/pool.py`, unchanged) and
    `benchmark_jobs.py` (`PageTask`/`run_page_task`/`PageResult` — see the `Reader/Parallel/`
    bullet below for its actual current behaviour, which calls `core.pipeline.run_pipeline`, not
    the old `pipelines.current.run_pipeline`).
- **`P1_Reading_Native/`** — the one, always-run extraction phase:
  - **`reader.py` — `Reader`** *(implemented)*: opens a PDF (a bad path raises `ValueError`),
    hands out `Page` objects (`get_page(index)`, `iter_pages(indices=None)`), each carrying a
    `PageMeta` snapshot plus the live `fitz.Page`.
  - **`dataset.py`** — page-dataset iteration helper.
  - **`native_text.py`** *(implemented)*: `extract_native_text(page) -> list[Text]` — one `Text`
    (`source="native"`) per `get_text("words")` word, font/size/colour/direction joined from the
    best-overlapping `get_text("dict")` span. Same logic as before the phase split (see git
    history for the full per-function breakdown if needed — `_extract_spans`/`_extract_words`/
    `_match_word_to_span`/`_to_word`).
  - **`vector_extract.py`** — raw, unclassified `extract_vectors(page) -> list[Vector]`
    (formerly `Vector/vector.py::extract_paths`) — walks `page.fitz_page.get_drawings()`, one
    `Vector` per drawing item, tagged with `seq`/stroke/fill/width/dashes/layer. No
    classification here — that's entirely a P3 backend's job.
  - **`image_extract.py`** — `extract_images(page) -> list[Image]`: the whole-page raster
    (`page.get_pixmap` at a configured DPI) plus any embedded raster images
    (`page.get_image_info(xrefs=True)`) — feeds Phase 2.
  - **`phase1.py`** — `read_and_extract(pdf_path, page_index) -> Phase1Result(page, texts,
    images, vectors)`, the one entrypoint `core.pipeline` calls.
- **`P2_Raster_To_Vec/`** — pluggable raster→vector backends, selected by `p2=`:
  - **`Stub/stub.py`** — `extract(images, page) -> ([], [])`, a documented no-op. The default
    (`p2="Stub"`) — matches the old (pre-phase-split) pipeline's behaviour of never deriving
    vectors from raster content.
  - **`Junction/`** — a ported classical (no-ML) raster→vector pipeline, vendored from branch
    `junction-classification`'s `junction_test/` package (Dosch et al.-style: binarize →
    text/graphics separation → dashed-line reclaim → thick/thin split → skeletonize+distance-
    transform → graph build → polygon approx → arc detection → dashed-line detection → width
    measurement → regularize → remainder extraction → staircase/symbol recognition). `adapter.py`
    is the `Phase2Backend` entrypoint: runs `junction_test.pipeline.run(gray, params)` per
    `Image`, maps its pixel-space `Segment`/`Arc` primitives into page-space `Vector`s via a
    fraction-of-frame `to_page` mapper (dpi-agnostic, correct regardless of the pipeline's own
    internal downscale ratio). Returns no `Text` — OCR text-box extraction inside `junction_test`
    stays off (`Params(run_ocr=False)`); Phase 3 OCRs the merged vector pool for real. **This
    module uses `cv2`** (ported as-is from `junction_test`) — there is no longer a repo-wide
    "`rastervec/` never imports `cv2`" rule; other modules still don't need it, this one does.
    `adapter.py::render_debug` (registered in `P2_RENDER_DEBUG`) renders the earliest
    raster-processing stages coarsely (mask ink-bbox as a single box, since they're numpy masks
    not vector geometry) and the later, genuinely vector-shaped stages (graph chains, junctions,
    fitted segments/arcs) as their own bboxes.
- **`P3_Vector_Parsing/`** — pluggable vector-parsing/OCR backends, selected by `p3=`, each
  implementing `parse(vectors_p1, vectors_p2, page, **kwargs) -> (vectors, texts)`:
  - **`VectorClassification/`** — the original Vector Classification chain, restored verbatim
    from what used to be the repo's `Vector_Classification/` package (before the phase split
    archived it, then this backend un-archived it into its own self-contained copy). `parse.py`
    combines `vectors_p1 + vectors_p2` into one flat pool, then: `classify_vectors.py`'s fixed
    12-step chain (`item_filters.py` steps 1-2, `group_filters.py` steps 3-5+8,
    `cluster_filters.py` steps 6-7+9-12 — same per-step logic as before, see
    `classification.py`'s module docstring for the exhaustive per-step description; a
    `StepResult`/`CategoryResult` per step, `role="kept"`/`"dropped"`/`"info"`, every
    `"dropped"` category folds into drawing output) → this folder's own `fast_filter.py`
    (`detect_text_fast`, a whole-page FAST mask scored per surviving cluster) → this folder's own
    `radon.py` (`segment_clusters` — full pixel-space word-level Radon deskew + line/word
    splitting, the original algorithm) → `fast_filter.py`'s `group_similar_segments`/
    `elect_unique_segments` (whole-page duplicate-segment dedup + representative election) →
    `ocr.py` (`recognize_unique_words` + `restore_word_texts`, over this folder's own
    `paddle_engine.py::PaddleRecBackend`, recognition-only). `parse.py::render_debug`
    (`P3_RENDER_DEBUG["VectorClassification"]`) renders one kept/dropped layer per classification
    step plus fast/segment/ocr/drawing layers, from whatever `parse()` stashed into `debug_out`.
  - **`FastIntoPaddle/`** — the pipeline that had been `rastervec/pipelines/current.py` before
    the phase split, now self-contained here. `parse.py` combines `vectors_p1 + vectors_p2`, then
    `steps.py`'s chain: `similarity_group` (`similarity.py::vector_similarity_group`, shape-
    similarity independent of position/size/rotation) → `filter_vectors_fast` (per-vector FAST
    coverage score over this folder's own `fast_detect.py`) → `reclassify_by_similarity` (pulls a
    similarity group's FAST-dropped members up to "pass" under a fraction threshold) →
    `separate_by_layer_color_width` (this folder's own `layer_color_separation.py`) →
    `cluster_buckets`/`cluster_bucket_spatial` (union-find spatial merge per bucket) →
    `detect_text_paddle_per_cluster` (one PaddleOCR-detector render per cluster, this folder's
    own `paddle_engine.py::PaddleDetectBackend`) → `reassign_by_overlap` (vector→detection
    assignment by bbox-coverage) → `rotate_paddle_detections` (this folder's own `radon.py::
    sweep_rotation` refinement + crop) → `recognize_segments` (`paddle_engine.py::
    PaddleRecBackend`, recognition-only). `parse.py::render_debug`
    (`P3_RENDER_DEBUG["FastIntoPaddle"]`) renders one layer per named step
    (`similarity`/`fast`/`reclassify`/`clusters`/`paddle_detect`/`assignment`/`rotate`/`ocr`/
    `drawing`).
  - **`LegacyRecreation/`** — a genuine from-scratch port (not a wrapper) of
    `archive/raster_parser`'s own Type-2 algorithm onto `commons.models` types, selectable as a
    normal P3 backend (distinct from the separate `legacy`/`legacy_adapter.py` engine axis, which
    shells out to the real unmodified archive codebase — see that bullet below). `parse.py`:
    `filters.py::filter_text_vectors` classifies filled vectors into glyph-ink candidates vs.
    everything else (drawing) → `wordgrouping.py::cluster_by_seqno` groups glyph candidates by
    content-stream draw-order adjacency into `WordGroup`s → each group is rendered
    (`commons.renderer.render_vector_cluster`) and OCR'd (this folder's own
    `paddle_engine.py::PaddleRecBackend`, recognition-only). `parse.py::render_debug`
    (`P3_RENDER_DEBUG["LegacyRecreation"]`) renders `filter_fill`/`group_words`/`ocr`/`drawing`
    layers.

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across
  buckets**, in every P3 backend that separates by layer/color at all — two vectors in different
  layers, or with different stroke/fill colors, are never spatially merged together regardless of
  page proximity.
- **`rastervec/OCR/`** (top-level: `fast_detect.py`, `radon.py`, `Paddle_OCR/ocr_backend.py` +
  `render_ocr.py`) — **dead code**, superseded by each P3 backend's own duplicated copy
  (`P3_Vector_Parsing/VectorClassification/{fast_detect,radon,paddle_engine,ocr}.py` and
  `P3_Vector_Parsing/FastIntoPaddle/{fast_detect,radon,paddle_engine,steps}.py` — see the
  `P3_Vector_Parsing/` bullet above for what each backend actually does now). Not deleted yet;
  don't build on it.
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
- **`Evaluation/Labelling/`** *(implemented)*: ground-truth labelling, three kinds — native
  (auto-derived from native text), vector (human-labelled real `Vector`s), raster (human-labelled
  embedded-image content with no vector backing) — bundled per PDF by `scripts/label/
  master_label.py` into one `outputs/labels/<stem>_label/` folder (see that bullet). `label_schema.py`
  is the shared sidecar-JSON schema, kept in the package (not moved to `scripts/`) since it's
  imported broadly outside the labelling tools themselves (`adapters.py`, `generate_pipeline_report.py`,
  `Reader/dataset.py`, benchmark plumbing). `LabelEntry` (`page_index`, `cluster_bbox`,
  `cluster_signature`, `label_id`, `text`, `source: "native"|"vector"|"raster"`,
  `expected_rotation`, `vector_signatures`) + `GeometryAnnotation` (`page_index`, `kind: "l"|"c"`,
  `points`, `color`/`fill`/`width`/`dashes`/`opacity` mirroring the matching `Vector` fields,
  `source: "auto"|"manual"`) + `LabelSet` (`entries` + `geometry_entries`) are the format
  (`save_labels`/`load_labels`). `label_id` is the stable identity a labelling tool edits in place
  (`vector_label`/`raster_label`: `uuid4().hex`; `native_label`: the deterministic
  `f"line:{page_index}:{block_no}:{line_no}"`); `cluster_signature` is informational/debug-only now.
  `vector_signatures_for(vectors)` / `path_signature(v)` (SHA1 of absolute page-space geometry +
  `seqno` + paint attrs — run-stable, collision-resistant, **not** translation-invariant unlike
  `item_filters.vector_signature`) let a label re-match its exact paths across a fresh
  `extract_vectors` run; empty for `source="raster"` (no backing vectors).
  `geometry_annotations_for_vector(v)` decomposes one `Vector`'s raw `items` into `source="auto"`
  `GeometryAnnotation`s carrying its real paint attrs (`"l"`/`"c"` as-is, `"re"`/`"qu"` → their 4
  edges as 4 separate lines) — the raster-geometry auto-labelling step's building block.
  `split_labelset_by_source` still buckets into the benchmark's long-standing `"auto"`/`"manual"`
  GT-class vocabulary (`source="native"` → `"auto"`; `source in ("vector", "raster")` →
  `"manual"`), distinct from the finer three-way `LabelSource` used at labelling time.
  - **`native_label.py`** (library; the interactive-tool convention below still applies to its thin
    CLI wrapper) — `native_label_pdf(pdf_path, page_index)` is deliberately independent of the
    pipeline being evaluated: it reads *only* the original PDF's own `native.extract` (never runs
    Conversion or any classification/clustering), groups words by `(block_no, line_no)` into
    line-level ground-truth regions (bbox via `helpers.geometry.union_bbox`, text joined in
    reading-direction order), and sets `expected_rotation` to the most common quarter-turn among
    the line's words, `vector_signatures` left empty. This independence matters: an earlier version
    derived labels from a converted page's own surviving classification clusters, which meant a
    native word the classification chain's own filter steps wrongly dropped never became a label at
    all — silently excluded from ground truth rather than scored as a miss. `attach_vector_signatures
    (labels, page_index, vectors_pdf_path)` is the separate, heavier enrichment step (opt-in, not
    run by the benchmark's hot per-page loop): extracts vectors from a persisted
    `convert_page_text_only` render and assigns each to whichever line covers the most of its own
    bbox area (mirrors `pipelines/_steps.py::reassign_by_overlap`'s coverage-ratio pattern),
    populating `vector_signatures`.
  - **`raster_label.py`** (library) — `raster_geometry_for_page(pdf_path, page_index)`:
    `extract_vectors` on the *original, unconverted* page (CAD-vector text glyphs included — a
    raster/scanned pipeline has to trace all of it), flat-mapped through
    `geometry_annotations_for_vector`. `sync_text_from_vector_labels(raster_labels, vector_labels,
    page_indices)`: mutates `raster_labels.entries` in place, dropping then re-adding every
    `label_id=f"vecsync:{v.label_id}"` entry from the current `vector_labels` on those pages —
    reuses `vector_label`'s human-vetted text as raster ground truth (raster and vector share page
    coordinates) without ever touching a genuine hand-drawn raster entry. `ImageRegion` +
    `embedded_images_for_page(pdf_path, page_index)`: thin wrapper over `page.get_image_info
    (xrefs=True)` (same API `inspector/pdf_model.py::extract_image_items` uses), listing real
    embedded raster images for the manual tool's picker.
  - **`scripts/label/_common.py`** — shared Tk plumbing (`get_display_matrix`
    — `rotation_matrix * zoom`, **overlay-only**, since `get_pixmap()` bakes `/Rotate` itself and
    passing the full matrix would double-rotate the bitmap vs. the overlays on a rotated page, same
    split as `inspector/pdf_model.py`; `Tooltip`, ported from the former `debug_app.py`;
    `draw_vector`; `bezier_points`; zoom/color constants) imported by every tool below via the
    sibling-import trick (running a script directly puts its own directory first on `sys.path`, so
    `scripts/label/` needs no `__init__.py`).
  - **`scripts/label/native_label.py`** (`python scripts/label/native_label.py PDF --page N
    [--out]`) — thin CLI over the library: `native_label_pdf` + `convert_page_text_only` (writing a
    `<stem>_p<N>_native_vectors.pdf` sidecar) + `attach_vector_signatures`, merged onto any existing
    `--out` file by `label_id`.
  - **`scripts/label/vector_label.py`** (`VectorLabelApp`) — there is only ever **one flat pool of
    raw vectors** for the page (`extract_vectors(page)`); every vector is always individually
    clickable, no clustering. `separate_by_layer_color_width` (`rastervec.pipelines.current`)
    buckets them into one visibility checkbox per `(layer, color, width)` in a side panel — a
    filter, not a selection unit. Selecting unlabelled vectors (click/ctrl-click/drag) + Apply
    creates a new label; clicking a single **already-labelled** vector instead selects that whole
    label's vector set, pre-fills the inline label bar (text + a continuous `ttk.Scale` rotation,
    2.5° snap, with a live direction-arrow overlay), and enters "editing" it
    (`self._active_label_id`) so a further Apply overwrites the same entry in place. A rubber-band
    drag never enters edit mode by itself. Labelled vectors render green; "Hide labelled" hides them
    entirely. The text entry auto-focuses after *any* selection. `<`/`>` (or `PageUp`/`PageDown`)
    move between pages without relaunching — labels save on every page change and on close.
  - **`scripts/label/raster_label.py`** (`RasterLabelApp`) — scoped to content with **no vector
    backing at all** (a scanned inset, stamp, photo — the actual CLAUDE.md-scoped-out "Raster"
    pipeline concern); everything else already has ground truth via `native_label`/`vector_label`/
    the auto geometry step. On load it lists every embedded image across the whole document
    (`raster_label.embedded_images_for_page`) and lets you step through them one at a time,
    cropping into each via `page.get_pixmap(clip=bbox, ...)` — every saved
    `LabelEntry`/`GeometryAnnotation` still stores absolute page-space coordinates regardless. Three
    toolbar tools: **Text** (drag a bbox, same label bar as `vector_label`, `source="raster"`),
    **Line** (click-drag-release two points), **Curve** (click 4 points in sequence — start, 2
    controls, end; `Escape` cancels an in-progress one). Right-click deletes a text label or a
    line/curve. A PDF/page with no embedded images simply has nothing to show.
  - **`scripts/label/master_label.py`** — runs the whole workflow over one PDF into
    `outputs/labels/<stem>_label/` (`original.pdf`, `vectorised.pdf`, `rasterised.pdf`,
    `native_labels.json`, `vector_labels.json`, `raster_labels.json`, `manifest.json`): a **native**
    step (skippable per page via the manifest's `native_done`) assembling `vectorised.pdf` and
    `native_labels.json`; a **raster geometry** step (`rasterise_done`/`geometry_done`, independent
    flags — rasterising is the expensive part) assembling `rasterised.pdf` and replacing that page's
    `source="auto"` geometry entries; a **text re-sync** (`sync_text_from_vector_labels`, always
    runs, no skip, so editing vector labels later keeps raster labels in sync) — then opens
    `VectorLabelApp`, re-syncs again, then opens `RasterLabelApp`, then prints a summary. Re-running
    on the same PDF skips already-done pages for steps 1-2; the interactive steps always reopen for
    incremental labelling. Not unit-tested end to end (chains three real Tk event loops); its
    per-step assembly functions are plain file orchestration, smoke-tested by hand.
  - **`scripts/label/label_viewer.py`** (`LabelViewerApp`, read-only) — resolves a
    `master_label.py` folder from either the source PDF path or the folder itself
    (`resolve_label_folder`), and overlays all three label sets on one page view with four
    independent toggles: Native (bbox), Vector (bbox + its actual backing `Vector`s, resolved by
    `path_signature` against a fresh `extract_vectors(page)`), Raster geometry (lines/curves, dashed
    for `source="auto"` vs solid for hand-traced), Raster text (bbox, dashed for a
    `"vecsync:"`-prefixed `label_id` vs solid for genuine manual entries). No editing, no save —
    page nav + zoom + hover tooltips only.
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
  pred-vs-GT overlay. **`docs/EVAL_METRICS.md`** documents every metric's formula, both normalisation
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
- **`Evaluation/Evaluate/golden_schema.py` + `golden_regression.py`** *(implemented)*: a
  hand-curated golden/regression test suite, separate from the `LabelSet`/benchmark machinery
  above — one `GoldenCase` (pydantic, `golden_schema.py`, pure) is a single curated snapshot of
  one cluster's/segmentation's output at one of three stages (`"classification"`, `"fast"`,
  `"word_split"`), labelled `"positive"`/`"negative"` **freely by the curator** — the label is
  the human verdict ("this is text / the split is right" vs. not), assigned from the *combined*
  candidate pool, so a cluster the pipeline currently keeps/passes can be labelled `"negative"` (a
  known false positive) and one it drops labelled `"positive"` (a known miss); what the pipeline
  actually decided is stored separately in `role` and is what replay compares (`"word_split"` has
  no structural pass/drop signal, so its `role` is `None` and the label is pure visual judgement)
  — stored in one shared `GoldenCaseBank` (`add_case` — append, dedupe on
  `(stage, label, signature)` so a stage can hold several curated positives/negatives;
  `upsert_case` — the older one-per-`(stage, label)` variant, still used by tests; `drop_cases` —
  remove by index; `save_cases`/`load_cases`, JSON at `outputs/regression_cases/cases.json`, not
  one file per PDF, since a bank only ever holds a handful of self-describing cases rather than
  an exhaustive per-PDF labelling). `golden_regression.py` (the pipeline-facing half, alongside
  `adapters.py` the only other `Evaluation/Evaluate/` module importing `rastervec.pipelines`)
  builds each stage's candidate pool from the same `PipelineResult` fields the visualization
  notebook's own `render_*` functions read (`text_clusters`/`clustering`'s `role="dropped"`
  categories, `fast_passed`/`fast_dropped`, `regrouped_clusters` zipped with `segmentations`);
  `list_stage_candidates(res, stage, *, n, shuffle, seed)` returns one combined, thumbnailed,
  role-tagged, shuffled pool per stage (the older split `list_{classification,fast,word_split}_
  candidates` helpers are kept for tests), `capture_case` turns a picked `Candidate` into a
  `GoldenCase` (`label_schema.cluster_signature` for debug provenance only), `format_case_bank`
  renders the numbered bank listing, and `compare_case`/`run_regression` replay a bank against fresh
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
  browse-combined-pool / pick-indices-and-labels (`{candidate_index: "positive"|"negative"}`) /
  capture-and-save cell triplet (`capture_picks` → `add_case`), then **View all stored cases**
  (`format_case_bank`) + **Delete cases** (`DELETE_INDICES` → `drop_cases`) cells, then a shared
  **Replay** section (`run_regression` + `format_regression_report`) that re-checks every case in
  the bank (from any prior session, not just the current one) — the same cell to re-run later as a
  regression check.
- **`Evaluation/Evaluate/variants.py`** *(implemented)*: `PipelineVariant` (name, `engine`
  current/legacy, `p2`, `p3`, `enable_fast`) + the `VARIANTS` registry (`current` [default p2/p3],
  `legacy`, plus named presets for benchmark comparisons across P3 backends —
  `current_vectorclassification`, `current_legacyrecreation`, `current_junction`) +
  `DEFAULT_VARIANTS` + `resolve_variant`. `engine="current"` threads `p2`/`p3`/`enable_fast` into
  `rastervec.core.pipeline.run_pipeline` (the pluggable P1→P2_REGISTRY[p2]→P3_REGISTRY[p3]
  orchestrator, see the `core/` section below); `engine="legacy"` ignores `p2`/`p3` entirely.
  `scripts/generate_pipeline_report.py`'s `ReportConfig` doesn't go through this registry for the
  `current` engine — its own `p2`/`p3` fields build a `PipelineVariant` directly, so a report run
  gets exactly the config's own combo rather than a fixed named preset.
- **`Evaluation/Evaluate/legacy_adapter.py`** *(implemented)*: a thin `sys.path` + call-through
  that runs archive's `raster_parser.main_pipeline_extract.extract` unmodified and reshapes its
  output into `ClusterOcrResult`s for `metrics.evaluate_metrics` (the `legacy` variant).
  `_ensure_archive_importable` only prepends repo-root `archive/` to `sys.path` — **no
  compatibility shim** (`_paddle_compat.py` is gone): the repo now runs paddleocr 2.x, the API
  archive was written against. It logs a `WARNING` the first time `archive/` goes on the path, and
  `run_archive_pipeline` logs + **re-raises** any failure (never swallows it). `_run_legacy`
  (`benchmark_jobs.py`) no longer wraps its runs in `try/except` either — a legacy failure
  propagates to `run_page_task`'s outer boundary (logged, `PageResult.error` set), not a silent
  zero score. Nothing in `archive/` is touched. Archive's `raster_parser` still needs **LibreOffice
  on PATH** (`import raster_parser` launches a LibreOffice subprocess at import time), so `legacy`
  runs end-to-end only where LibreOffice is installed; elsewhere it warns and fails loudly.
- **`core/parallel/`** *(implemented, moved from the old `Reader/Parallel/`)*: two pools.
  **Pool 1** (page jobs) is `pool.py` — `worker_init` (pins `OMP`/`MKL`/`OPENBLAS` to 1 per
  worker), `default_worker_count`, `warmup` (builds the PaddleOCR rec + FAST caches in the
  *calling* process, so a spawn pool started next finds the models on disk and no worker races
  the first-run download — each P3 backend's own `PaddleRecBackend.warmup()` /
  `FastDetector.warmup()` classmethods force the existing lazy `_engine()`/`_model()` path), and
  `run_parallel(items, fn, *, workers, desc)` — an input-order map that is a plain serial loop
  when `workers <= 1` and a spawn `ProcessPoolExecutor` otherwise. Processes not threads: the
  PaddleOCR engine + FAST model module caches are unlocked shared singletons and PyMuPDF is not
  reentrant. **Pool 2** (compute) is a single `multiprocessing.Manager().Pool(processes=
  compute_workers)` built by the `pool.compute_pool(n)` context manager (`n <= 0` → yields
  `None`; warms model caches first; reusable outside the benchmark — a notebook running one page
  through `core.pipeline.run_pipeline(..., compute=...)` uses it too), shared by *every* Pool-1
  worker's page job — a complex page's many FAST/OCR jobs and simple pages' few jobs all queue
  into this one pool, so idle capacity is never stranded on a page that finished early. Pool 2
  never imports `fitz`/`pymupdf`; its jobs are top-level, picklable functions taking only plain
  data — each P3 backend's own `fast_detect._detect_job(weights_path, image_array)` and
  `paddle_engine._recognize_crops_job(crops, ocr_version, lang)` — each building/caching its own
  model/engine per Pool-2 worker process exactly like Pool 1's per-process caches.
  `benchmark_jobs.py` — the picklable per-page job: `PageTask` (pdf/page/manual entries/
  `variant`/reconstruct dir/…) → `run_page_task(task, compute=None)` (resolves `task.variant` via
  `variants.resolve_variant`, dispatches to `_run_current` — which now calls `core.pipeline.
  run_pipeline(..., p2=variant.p2, p3=variant.p3, ...)`, not the old `pipelines.current.
  run_pipeline` — or `_run_legacy`; `compute` is forwarded to the current engine only, never to
  legacy) → `PageResult` (`variant`, auto + manual `MetricSuiteResult`, stage durations,
  formatted report blocks, a few PNG-bytes `ShowcaseSample`s, `error`). **Each variant is run twice
  per page** on disjoint inputs — `convert_page_text_only` scored vs the `auto` labels,
  `convert_page_drawings_only` scored vs the `manual` labels (the manual run only when the page has
  manual labels) — so the two GT sources are scored against physically separate runs and can't
  contaminate each other's precision. Each `current`-engine run per page is wrapped in its own
  `try/except` (a failed run leaves that field `None` + a `report_blocks` line); the `legacy`
  runner is **not** wrapped — a legacy failure propagates and lands in `PageResult.error` (a hard,
  visible error, not a silent zero). `run_benchmark(tasks, *, workers, compute_workers=0, desc)` is
  the thin `run_parallel(tasks, run_page_task, …)` wrapper used by both `benchmark.py` and the
  notebook — `compute_workers > 0` builds Pool 2 (`pool.compute_pool`) and threads its proxy into
  every page job via `functools.partial(run_page_task, compute=compute)`, shutting it down after;
  `compute_workers=0` (default) is fully local, today's behavior. Per-variant reconstruct output goes to
  `RECONSTRUCT_DIR/<stem>_p<N>_<variant>/`.
- **`Evaluation/Evaluate/benchmark.py`** *(implemented)* — the CLI wiring Conversion → native_label →
  a real full pipeline run → `metrics.evaluate_metrics` together: `python -m
  rastervec.Evaluation.Evaluate.benchmark --pdf PATH [--pdf PATH2 ...] --pages 0,1,2
  [--iou-threshold 0.1] [--reconstruct-dir DIR] [--workers N] [--compute-workers N]
  [--variants current,legacy]`
  (`--iou-threshold` = `MetricConfig.iou_edge_min`; `--workers N>1` runs pages across Pool 1
  (`Reader/Parallel`'s spawn pool); `--compute-workers N>0` additionally runs FAST tile detection +
  OCR crop recognition on Pool 2, shared by every `--workers` process (see the `Reader/Parallel/`
  bullet above); `--variants` selects which `variants.VARIANTS` to run and compare;
  `--reconstruct-dir` defaults to `outputs/benchmark_cli/reconstructions/`).
  `run_one_page` is a thin wrapper over `Reader/Parallel/benchmark_jobs.run_page_task` (auto labels,
  returns that page's `MetricSuiteResult`); `main()` runs one `PageTask` product per selected variant
  and prints `format_aggregate_comparison` + `format_variant_timing_comparison` across them.
  `--reconstruct-dir` writes the per-page output PDFs (see the notebook bullet). `format_report` /
  `aggregate_results` / `format_aggregate` / `format_aggregate_comparison` /
  `format_variant_timing_comparison` / `summarize_stage_timings` are pure and unit-tested; `main()`'s
  actual OCR-backed path is a manual smoke test only (real PaddleOCR, first run downloads models —
  matches the existing `RASTERVEC_RUN_OCR_TESTS`-gated convention for OCR-dependent tests).
  `scripts/pipeline_report_benchmark.py` is the standalone folder-diff counterpart (replaced the old
  `benchmark_vector_classification.ipynb`): it takes **1 or 2** benchmark report folders (`--run`,
  `generate_pipeline_report.py` runs with `benchmark: true`), each embedding its single
  `<stem>/dump.json` + `ground_truth_auto.json` (+ `ground_truth_manual.json`), matches the folders
  on the inputs they **share by identity** (`pdf:<stem>` auto-only vs `labels:<stem>` auto+manual — a
  `B.pdf` run and a `B.json` run don't match on B), and per shared input scores every run with
  `metrics.evaluate_multiclass` → an `AUTO` table, a `MANUAL` table, the
  `{auto,manual}×{auto,manual,none}` GT-recall confusion matrix (`benchmark.format_confusion`), and
  `combined_candidate_precision`, one column per folder. Writes a timestamped
  `outputs/pipeline_report_benchmark/<ts>/` with `benchmark.txt`, `runs.json`, `viewer_commands.txt`
  (a `pipeline_report_viewer.py` line per shared input), and `charts/` (`Evaluation/Evaluate/
  charts.py` — matplotlib Agg; `metric_comparison_chart` grouped bars + `confusion_heatmap`, per
  page + per-key-aggregate + grand-aggregate, split per source). Miss-attribution metrics are absent
  (a dump has no classification state).
- **`Evaluation/Evaluate/label_overlays.py`** *(implemented, pure)*: `gt_bbox_overlay(graph)` and
  `gt_word_overlay(graph)` — GT-only visual-diff data (no rendering / pipeline import) reduced off a
  `metrics.OverlapGraph`, used by `generate_pipeline_report.py`'s benchmark mode for the
  `{auto,manual}_bbox.pdf` (green covered / red missed GT box) and `{auto,manual}_text.pdf` (GT text,
  per word green exact / yellow char edit distance 1-2 / red worse-or-unread — region bbox split into
  per-word slices along the `expected_rotation` reading axis).
- **`Evaluation/Evaluate/metrics.py::evaluate_multiclass`** *(implemented — see `docs/EVAL_METRICS.md`
  §5)*: for the report benchmark's **one combined `convert_page_to_vector_text` run** scored against
  auto GT + manual GT at once. Shares `_suite_for_source` with `evaluate_metrics` (which is now a thin
  `exclude_pred_idxs=frozenset()` wrapper, contract unchanged). For source S, predictions whose
  `pred_class` is the *other* source are dropped from the precision-family denominators only
  (`page_{char,word}_multiset_precision`, `pred_text_fully_contained_in_overlapping_gt_rate`).
  `classification_precision_candidate_is_text` becomes the combined
  `MulticlassResult.combined_candidate_precision`. Adds a `{auto,manual}×{auto,manual,none}` GT-recall
  confusion matrix (`detected_class`) + `aggregate_multiclass`.
- **`renderer/` — module-level functions, no `Renderer` class** *(rendering helpers, not a pipeline
  stage)*: a package split by output concern — `png.py` (rasterize vector paths for OCR / FAST
  input), `pdf.py` (`render_reconstructed_page`, `render_reconstructed_pdf`, `render_boxes_pdf`
  — colored rectangle outlines, each entry `(bbox, rgb)` or `(bbox, rgb, dashes)` — plus
  `render_text_pdf` / `render_vectors_pdf`, the two colour-callback stage-report primitives), `svg.py`
  (`render_page_svg`, a thin `get_svg_image()` wrapper), and `_shapes.py` (shared). Import straight
  from `rastervec.renderer` (`from rastervec.renderer import render_vector_cluster`, etc.).
  `stages.py` holds one `render_<stage>(res) -> bytes` per pipeline stage (one-page composite
  **stage PDF**, still used by tests / the notebook) plus `render_stage_layers(res, stage_key)
  -> [(label, hex, pdf_bytes)]` — the same visuals **split one single-purpose one-page PDF per
  visual element**, which is what `scripts/generate_pipeline_report.py` writes now
  (`<stage>__<layer>.pdf`) so the viewer toggles a layer by loading/not-loading its file (no
  colour-keying, no fringe). Both build on those three primitives + the local `_compose`
  helper. It also owns `STAGE_COLOR_LEGEND` / `STAGE_ARTIFACTS`. Every stage module keeps its one-line
  `from rastervec.renderer.stages import render_x` re-export. `notebook.py` is the leftover
  matplotlib display plumbing (`RenderResult`, `visualize`, `show_row`, ...) kept only for
  `golden_case_curation.ipynb`; deliberately **not** re-exported through `renderer/__init__.py`
  (it imports matplotlib, and this package is imported by the real pipeline). `stages.py` stays
  matplotlib-free and imports no `rastervec.pipelines` module at module level for the same reason.
  `_shapes.path_color_hex(path)` returns a path's real PDF stroke/fill color as hex (used by both the
  visualization notebook and OCR input rendering) — any B/W-style simplification stays purely
  internal to classification, never substituted into a rendered/displayed color.
  `_shapes.replay_drawing_paths(page, vectors, *, dx, dy)` is the accuracy-critical helper shared by
  png/pdf (takes a `fitz.Page`, not a pre-made `Shape` — it owns its own `new_shape()`/`commit()`
  now): it replays every item of each `Vector` (one whole `get_drawings()` drawing) into a shape,
  then calls `shape.finish()` **once per `Vector`** carrying that drawing's real `even_odd` /
  `line_join` / `line_cap` / stroke+fill opacity (ported from
  `archive/raster_parser/rendering/pdf_render/reconstruct.py`). Vectors are processed in consecutive
  `(blendmode, opacity)` runs — one `Shape`+`commit()` per run — and a run with a non-Normal blend
  mode or group opacity < 1 has its committed content stream wrapped in a `/BM`+`/ca`+`/CA`
  ExtGState (`_wrap_run_gstate`), since `Shape.finish()` has no blendmode param; without it a
  Multiply-blended line reconstructs fully opaque. `Vector.blendmode`/`opacity` are populated by
  `extract_vectors` reading `get_drawings(extended=True)` and folding in the enclosing
  transparency-group's blend/opacity (plain `get_drawings()` drops `/BM` entirely). This is why a multi-contour filled
  glyph (an "o", "e", "8", "A" — outer contour + inner counter, one drawing, `even_odd`) renders
  with its counter as a white hole instead of filled solid; drawing each `VectorPath` primitive on
  its own and calling `finish(closePath=True)` per primitive (the pre-split behaviour) filled every
  counter solid — a direct hit to OCR of vector text. `finish()` is per drawing; `commit()` is now
  internal to `replay_drawing_paths` (once per `(blendmode, opacity)` run). A drawing whose paths carry neither `stroke_color` nor
  `fill_color` is skipped outright: `finish()` emits a stroke operator whenever `fill` is `None`
  regardless of `color`, falling back to the default-black graphics state instead of staying
  invisible. `even_odd` / `line_cap` / `line_join` are drawing-level fields now copied onto every
  `VectorPath` of a drawing (like `fill_rule`), defaulted so existing constructions are unaffected;
  `vector.extract_records` normalises a tuple `lineCap` from `get_drawings()` to a plain int.
  `png.render_vector_cluster(paths, dpi)` *(implemented)*
  isolates a cluster onto a page of the shared per-process render document sized to **exactly**
  the cluster's own `union_bbox` — **no padding of any kind**. There is no `_cluster_frame` /
  `cluster_frame_size` helper any more, and no `MIN_CLUSTER_PADDING` / `OCR_*_PADDING_FRACTION`
  config: all padding in the pipeline is one explicit pixel-space step,
  `OCR/radon.py::pad_image` (see that bullet). The only margin logic left here is a
  `_MIN_CANVAS_SIDE = 1.0`pt floor on each side — a degeneracy guard so a flat cluster (a single
  horizontal rule, zero extent on one axis) doesn't build a 0-px pixmap, not padding. Don't
  reintroduce a border: `pixel_to_page_bbox`/`page_points_to_pixel` assume the bbox *is* the
  frame. `render_vector_cluster` then replays each drawing's items via `replay_drawing_paths`,
  rasterizes at `dpi` and returns a PIL `Image` — reusing PyMuPDF's own path/curve/fill
  rendering rather than reimplementing rasterization by hand.
  `png.pixel_to_page_bbox(paths, dpi, pixel_points)` inverts that same transform (pixel `(0,0)`
  *is* the bbox origin) to map pixel-space points back into PDF page space. A caller working in
  a *padded* copy of the render (`radon.segment_clusters`) must subtract `pad_image`'s returned
  pixel offset first.
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
  base14 `"helv"`). Two text helpers: **`_place_word`** (native words only) draws one word at its
  own extracted `font_size` and its own `origin`. **`_place_text(text, bbox, rotation, *, color)`**
  (OCR words — which carry no measured `font_size` — and `text_boxes`) derives the size from the
  box *height* via helv's own metrics (`fontsize = bbox_height / (ascender - descender)`,
  `baseline_y = bbox_top + ascender * fontsize`), then fills the box *width*: multiple words →
  widen the gaps between words (justified-text style, one `insert_text` per word, letterforms and
  intra-word spacing untouched); a single word → stretch it horizontally via a non-uniform scale
  in the `morph` matrix (one draw call, so `render_reconstructed_pdf`'s output stays word-
  searchable); a string too long even at natural spacing → shrink the font uniformly (via
  `fitz.Font.text_length` vs `bbox_width`) so it never spills past the box/page. Rotation is
  exact at any angle: since `insert_text`'s own `rotate` param only accepts multiples of 90, rotation
  is applied instead via its `morph=(fixpoint, matrix)` param — `(bbox_center, fitz.Matrix(1,
  1).prerotate(-angle))`, PyMuPDF's mechanism for arbitrary-angle text (a `cm` transform applied
  before drawing). The angle is **negated**: `Text.angle()` is in `get_text`'s `dir` convention
  (y down) and morph rotation turns the other way in that frame, so without the sign flip a word
  whose direction has a non-zero y component reconstructs mirrored about the x-axis. The fixpoint is the bbox's own center, not the baseline origin — using origin as
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
- **`rastervec/pipelines/`** — mostly **dead code** now, superseded by `core/pipeline.py` +
  `P3_Vector_Parsing/*/parse.py` (see the `core/` and `P3_Vector_Parsing/` bullets above):
  `current.py`, `_steps.py`, `_common.py`, `sub_pipelines/` are unused by anything live. The one
  live exception is **`pipelines/legacy.py`** — `run_pipeline(pdf_path, page_index, *,
  enable_raster_pass=False, verbose=False) -> PipelineResult` (the old, pre-split
  `pipelines/result.py::PipelineResult` shape — `page`, `texts`, `vectors=[]` always, since
  archive's own `TextDTO` has no geometry back-reference — plus that dataclass's many
  verbose-only Optional fields, all unused by the legacy path), a thin wrapper over
  `Evaluation/Evaluate/legacy_adapter.py::run_archive_pipeline`. This is what `pipeline: "legacy"`
  actually calls; `p2`/`p3` don't apply to it at all.
- **`scripts/generate_pipeline_report.py`** *(replaced `pipeline_stage_visualization.ipynb`)*:
  reads a JSON `ReportConfig` (pydantic; `pipeline: "current"|"legacy"` — the engine axis — plus
  `p2`/`p3` [only meaningful when `pipeline == "current"`, validated against
  `core.registry.P2_REGISTRY`/`P3_REGISTRY`], `enable_fast`, `final_stage` = one of `core.pipeline`'s
  short `phase1`/`phase2`/`phase3` step names [`"legacy"` ignores it entirely], `input_dir`/
  `input_files`, per-PDF `pages`, `vectorise` + `vectorise_mode` = an `Evaluation/conversion.py`
  mode), runs `core.pipeline.run_pipeline(..., p2=, p3=, verbose=True)` once per (pdf, page) for the
  `current` engine (`pipelines.legacy.run_pipeline` unchanged for `legacy` — always a full run;
  the new orchestrator has no `stop_after`, so `final_stage` only trims which report *artifacts*
  render, not how much of the pipeline executes), and writes
  `outputs/pipeline_report/<ts>__<config-stem>/<pdf-stem>/` — see the artifact breakdown below.
  **`benchmark: true`** (mutually exclusive with `vectorise`) additionally makes each `<pdf-stem>/` a
  scoring artifact for `pipeline_report_benchmark.py` — `input_files` may be `.pdf` or `.json` label
  sidecars; per page the pipeline runs **once** on `convert_page_to_vector_text` output (text-as-
  vectors over the untouched drawings) and that one run's OCR is scored against both label classes.
  Alongside the normal report artifacts (below) it writes `ground_truth_auto.json` (+
  `ground_truth_manual.json`), the split `{auto,manual}_{bbox,text}.pdf` overlays (`label_overlays.py`,
  registered in the manifest under stage `benchmark`), and a run-root `benchmark.json` marker
  (`entries[].key` = `pdf:<stem>` / `labels:<json-stem>`). Runs under `pipeline: legacy` too (archive's
  pipeline per converted page — `legacy_adapter.run_archive_pipeline` imports `torch` before the
  paddle-first archive import for the Windows `shm.dll` gotcha; legacy emits only `reconstructed`).
  Every mode writes: one multi-page PDF **per visual layer** (`<stage>__<layer>.pdf`), `dump.json`
  (`Evaluation/dump_io.py` — every `Text` + `Vector`, reloadable), `config_and_hyperparameters.txt`
  (config + every `rastervec.config` constant), `manifest.json` (its `layers` list — `{stage,
  layer, file, color}` — drives the viewer). For `pipeline: "current"`: the fixed
  `phase1__*.pdf`/`phase2__*.pdf`/`final__*.pdf`/`reconstructed__*.pdf` layers (see
  `commons/renderer/stages.py`), **plus every backend-specific debug layer** each active P2/P3
  backend's own `render_debug` produces (`_accumulate_debug_pdfs`, using
  `core.registry.P2_RENDER_DEBUG`/`P3_RENDER_DEBUG` — e.g. `VectorClassification` emits one
  kept/dropped layer per classification step plus fast/segment/ocr/drawing layers,
  `FastIntoPaddle` emits one layer per named step, `Junction` emits its own raster-stage layers —
  see the `P2_Raster_To_Vec/`/`P3_Vector_Parsing/` bullets above for what each backend renders).
  There is no per-stage `.txt` stats file for the `current` engine (the old engine's
  `Evaluation/Report/stage_stats.py` numeric-stats convention doesn't generalize across backends
  with genuinely different internals) — `dump.json` is the reloadable source of truth instead.
  `paddle_detect_images/`/`paddle_recog_images/`/`fast_tile_images/` (PNG debug crops — what
  PaddleOCR's detector/recognizer and FAST actually saw) are populated only by backends whose own
  verbose fields those old-engine-shaped helpers can read via `getattr(..., None)` — they no-op
  harmlessly for backends that don't have those fields. `legacy` still only ever emits the single
  `reconstructed` row. There is no `stop_after`/partial-run support for `pipeline: "current"` —
  `final_stage` (validated against `core.pipeline`'s short `phase1`/`phase2`/`phase3` names) only
  trims which of the fixed 4 phase-level artifacts render, not how much of the pipeline executes,
  and doesn't gate the per-backend debug layers at all (those always render in full).
  `scripts/pipeline_report_viewer.py` is the Tkinter counterpart: **1 or 2** per-PDF folders →
  toggleable source page + one side-by-side panel per folder, each a checkbox per layer PDF (grouped
  by stage, `all`/`none` per group, incl. the `benchmark` overlay group), each checked layer
  rasterized with a real alpha channel (`get_pixmap(alpha=True)`) and alpha-composited over the base
  in panel order — so a layer from folder A and a layer from folder B show together;
  zoom/pan/page-flip. `pipeline_report_benchmark.py` emits ready-to-paste invocations in
  `viewer_commands.txt`.

`scripts/rasterize_pdf.py` (outside `rastervec/`, a one-off utility not a pipeline stage): flattens
every page of a PDF to an image and rebuilds a pure-raster PDF from those images — not currently
consumed by anything in `rastervec/` (kept for possible future raster-image work).

`tests/rastervec/` mirrors `rastervec/`'s own folder layout (e.g. `tests/rastervec/
P1_Reading_Native/test_reader.py` for `rastervec/P1_Reading_Native/reader.py`,
`tests/rastervec/P3_Vector_Parsing/VectorClassification/` for that backend,
`tests/rastervec/core/` for `core/pipeline.py`/`registry.py`/etc, `tests/rastervec/renderer/
test_png.py` for `rastervec/commons/renderer/png.py`); modules that stay at
`rastervec/`'s top level (`output_types.py`) keep their tests at
`tests/rastervec/`'s top level too. `tests/conftest.py`'s `synthetic_pdf_factory` builds small
in-memory PDFs via `fitz.open()`/`insert_text`/`set_rotation` — preferred over `references/*.pdf`
for unit tests since those are gitignored and give no exact expected values to assert against.

### Adding a new P2/P3 backend

1. New folder under `P2_Raster_To_Vec/` or `P3_Vector_Parsing/`, implementing `Phase2Backend`/
   `Phase3Backend` (`core/interfaces.py`): `extract(images, page) -> (vectors, texts)` or
   `parse(vectors_p1, vectors_p2, page, **kwargs) -> (vectors, texts)`. **Fully self-contained**
   — duplicate whatever infra (FAST detector, PaddleOCR engine wrapper, layer/color separation,
   Radon deskew, ...) the backend needs internally rather than importing a sibling P2/P3 backend;
   `commons/` is the only shared layer. Split real logic into small private functions per
   sub-step, each independently testable.
2. One more line in `core/registry.py`'s `P2_REGISTRY`/`P3_REGISTRY`.
3. Optional: accept a `debug_out: dict | None = None` kwarg (mirrors the existing
   `radon.segment_clusters(debug_out=...)` convention) and, when given, stash your own
   intermediate step objects into it verbatim (no shape conversion — `core.pipeline` only
   forwards it when `verbose=True` and your signature declares it, storing the result in
   `PipelineResult.extra["p2_debug"]`/`["p3_debug"]`). Then define `render_debug(debug_out,
   page_meta) -> list[(stage, label, hex, pdf_bytes)]` in the same module, reading your own
   `debug_out` shape back and rendering it with the three shared primitives in `commons/renderer`
   (`render_boxes_pdf`/`render_text_pdf`/`render_vectors_pdf`) — no generic interpreter, no
   shared rendering abstraction; each backend renders its own data. Register it in
   `core/registry.py`'s `P2_RENDER_DEBUG`/`P3_RENDER_DEBUG`. A backend with nothing worth
   visualising (e.g. Stub) can skip this step entirely.

Also add tests under the matching `tests/rastervec/P2_Raster_To_Vec/`/`P3_Vector_Parsing/`
subfolder using the synthetic PDF fixtures, and new third-party dependencies to
`requirements.txt` only when the backend that needs them is actually implemented.
