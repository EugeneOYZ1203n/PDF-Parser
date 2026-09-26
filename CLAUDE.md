# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A raster-to-vector pipeline project for architectural/engineering shop drawings
(see `references/*.pdf`, gitignored sample PDFs). `rastervec/` is the only package here —
the standalone PDF-layer inspector tool that used to live at the repo root now lives inside it,
at `rastervec/Evaluation/inspector/` (see below).

`rastervec/` is the actual extraction pipeline, organized as **two pluggable phases sandwiched
between two always-the-same phases, behind one shared harness**, not a fixed step sequence:
**Phase 1** (`P1_Reading_Native/`, always the same) opens the PDF and extracts native text + raw
vectors + page/embedded images; **Phase 2** (`P2_Raster_To_Vec/`, pluggable — `Stub` no-op, or
`Junction`, a ported classical raster→vector pipeline) turns Phase 1's images into additional
vectors; **Phase 3** (`P3_Vector_Parsing/`, pluggable — `VectorClassification` (the default) or
`LegacyRecreation`) takes Phase 1's + Phase 2's vectors and produces the final vectors + OCR'd
text; **Phase 4** (`P4_Output_Organization/`, always the same) combines every phase's text/vector
output into the final `(texts, vectors)` pair and is a coordinate-space consistency backstop (logs
a warning if any item's bbox doesn't fit the page's own unrotated MediaBox — see "Coordinate
spaces" below). `core/` is the orchestrator + registry + the stable public API surface, and
`commons/` is the shared foundation (dataclasses, geometry/rendering/logging/paths helpers) every phase builds
on. See "`rastervec/` architecture" below for the full breakdown, including the hard rule that
sibling P2/P3 backends share **zero** code with each other — each is fully self-contained,
duplicating whatever infra it needs rather than importing a sibling's.

A separate, unrelated **`legacy`** engine (`pipelines/legacy.py` + `Evaluation/Evaluate/
legacy_adapter.py`) runs `archive/raster_parser`'s own pre-`rastervec` pipeline unmodified, kept
only as a benchmark comparison baseline — it's not one of the three phases and ignores `p2`/`p3`
entirely (see the `variants.py`/`legacy_adapter.py` bullets below). `P3_Vector_Parsing/
LegacyRecreation/` is a *different* thing: a genuine from-scratch port of that same old
algorithm onto `commons.models` types, selectable as a normal P3 backend.

Several top-level `rastervec/` folders/files from before this phase split are genuinely gone —
`rastervec/Reader/` (superseded by `P1_Reading_Native/reader.py` + `core/parallel/`) and
`rastervec/pipelines/_common.py`/`sub_pipelines/` (superseded by `core/pipeline.py` + the P3
backends' own `parse.py`/`steps.py`) no longer exist on disk. `rastervec/Vector_Similarity/` is
also gone — its one module moved to `P3_Vector_Parsing/FastIntoPaddle/similarity.py` at the phase
split, and has since been deleted outright, along with the `FastIntoPaddle` P3 backend itself (see
that bullet below) and the old pipeline's own similarity-grouping/reclassify step that was its
last real consumer (`pipelines/_steps.py::similarity_group`/`reclassify_by_similarity` — removed
from `pipelines/current.py`'s chain too, since neither `scripts/label/vector_label.py`/
`label_viewer.py` nor `Evaluation/Evaluate/benchmark.py` actually exercised it, only
`pipelines/current.py::run_pipeline` itself and its own tests did). A couple of comments/notebook
cells may still cite the old `Vector_Similarity/similarity.py` or `FastIntoPaddle/similarity.py`
paths; neither exists to import from any more.

**`rastervec/OCR/`, `rastervec/Vector/`, and `rastervec/pipelines/{current.py,_steps.py,
result.py,_cli.py}` are NOT dead — they're deprecated but still genuinely live**, an important
distinction from the folders above. `pipelines/current.py`+`_steps.py`+`result.py` form a
second, still-functioning pipeline implementation that predates `core/pipeline.py`, with its
own `PipelineResult` (`pipelines/result.py` — different shape from `core/result.py`'s: an
`engine` field and many per-stage Optional fields, no `extra` bucket) and its own
`run_pipeline`. Real current dependents, as of this writing: `scripts/label/vector_label.py`
and `scripts/label/label_viewer.py` (`pipelines._steps.extract_vectors`,
`pipelines.current.separate_by_layer_color_width`); `Evaluation/Evaluate/benchmark.py`
(`pipelines.current.STEP_NAMES`); `Evaluation/Evaluate/adapters.py`,
`commons/renderer/stages.py`, `commons/renderer/notebook.py` (all type-import
`pipelines.result.PipelineResult`); `core/parallel/pool.py::warmup()` (imports
`OCR/fast_detect.FastDetector` + `OCR/Paddle_OCR/ocr_backend.PaddleRecBackend` to warm model
caches); `P2_Raster_To_Vec/Junction/junction_test/pipeline.py` (imports
`OCR/Paddle_OCR/render_ocr.RenderOCR`, behind the `run_ocr=False`-by-default branch);
`P1_Reading_Native/vector_extract.py` (re-exports `Vector/layer_color_separation.py` "for
callers"); plus `pipelines/current.py`/`_steps.py` use both `OCR/` and `Vector/` directly. Its
own test suite: `tests/rastervec/pipelines/{test_current,test_steps,test_partial_run}.py`.
**New code should always call `core.pipeline.run_pipeline`, never `pipelines.current.
run_pipeline`** — `scripts/label/CAD_font_label.py` (a newer labelling tool) already treats
`rastervec.pipelines.current` as explicitly "off-limits" in its own docstring for exactly this
reason. A future migration to fully retire `OCR/`, `Vector/`, and the old `pipelines/` modules
is proposed (not yet executed) in `docs/old_pipeline_migration.md` — read that before assuming
any of the above is safe to delete.

`rastervec/pipelines/legacy.py` is unrelated to the current/core split above — it's the one
live file in `pipelines/` that isn't part of the old pipeline chain, and it's what
`pipeline: "legacy"` actually calls (see the `legacy` engine paragraph above).

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
.venv/Scripts/python.exe -m rastervec.core.pipeline --pdf PATH --page N [--p2 Stub] [--p3 VectorClassification]  # run the pluggable pipeline (P1 -> P2_REGISTRY[p2] -> P3_REGISTRY[p3])
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
.venv/Scripts/python.exe scripts/label/CAD_font_label.py PDF --page N               # CAD-font character + baseline labelling (GUI), standalone from master_label.py
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

**Package layout note:** most folders under `rastervec/` now carry a docstring-only
`__init__.py` (`rastervec/`, `pipelines/`, `core/` + `core/parallel/`, `commons/` +
`commons/models/`/`commons/renderer/`, `P1_Reading_Native/`, `P2_Raster_To_Vec/` + each of its
backend subfolders, `P3_Vector_Parsing/` + each of its backend subfolders,
`P4_Output_Organization/`) rather than being a bare PEP 420 namespace package — this list has
grown over time and isn't policed, so don't treat it as exhaustive or rely on any one folder's
current state; `rastervec/Evaluation/` is a notable holdout with no `__init__.py` anywhere in
it. `tests/` keeps its full `__init__.py` tree (pytest `importmode=prepend` + shared basenames).

**Pipelines (the old, deprecated-but-live implementation):** this paragraph describes
`rastervec/pipelines/current.py` — the older of the two pipeline systems described in the
top-of-file "not dead" note above, **not** `core/pipeline.py` (the current orchestrator new
code should use). It's defined as flat, readable block sequences (`native =
extract_native_text(page)` etc.) — see the `pipelines/` bullet below. `pipeline.py` (the old
`PipelineContext` + `STAGES` + `_run_stages` machinery) is gone; this system's own rule was
**new capability = one more named call in a `pipelines/` file**. Deskew + line/word
segmentation before OCR is a Radon transform (`skimage.transform.radon`) in `OCR/radon.py`
(an OCR-preprocessing concern, not a pipeline-orchestration `sub_pipelines/*.py` module),
replacing the old `OCR/Paddle_OCR/ink_segment.py`. There is one OCR backend in this old system
(`PaddleRecBackend`, recognition-only over Radon-segmented word crops — paddleocr 2.x's
`PaddleOCR(...).text_recognizer` batch call, PP-OCRv4); the old light/heavy split and
`PaddleOcrBackend` full-detection path are gone. Each P3 backend under `P3_Vector_Parsing/` now
has its own independent, duplicated copy of the FAST/Radon/OCR machinery this paragraph
describes (see that section) — this paragraph is about the original, shared `OCR/` copy only.

## `rastervec/Evaluation/inspector/` architecture

A standalone Tkinter + PyMuPDF desktop tool for visually inspecting what's inside a PDF (text,
images, annotations, vector drawings, as toggleable overlays) — predates `rastervec`'s own
extraction pipeline and shares no imports with it; it was built as step 0, to visually validate
what PyMuPDF extracts before writing real extraction logic elsewhere in `rastervec`. It now lives
inside `rastervec/` (under `Evaluation/`, alongside the not-yet-built benchmarking suite) since it
remains a useful dev-facing inspection tool, but its own five top-level modules' *responsibilities*
are otherwise unchanged, even though several are now internally split into smaller sibling files
(each re-exported through the original module's own path, so nothing importing e.g. `layers.X`
or `pdf_model.X` needs to change):

- **`layers.py`** — the extensibility core. `OverlayItem` is the normalized shape every extractor
  returns (bbox always in PDF page coordinates; `quad`/`points` optionally for non-axis-aligned
  geometry; `attrs` for machine-filterable values; `metadata` for human-readable hover info).
  `LayerSpec` (one top-level checkbox) and `SubFilterSpec` (a sub-checkbox group under a layer) are
  declarative — `build_layers(pdf_model)` wires the four current layers (text/images/annotations/
  drawings) to their extractor functions in `pdf_model.py`. `filter_items()` is the one shared
  filtering function all layers use (AND across sub-filter groups, OR within a group, empty
  selection = no restriction). Adding a new layer means adding one `LayerSpec` + one extractor
  function — nothing in `inspector.py`, `overlay_canvas.py`, or `control_panel.py` needs to change.
  `OverlayItem`/`LayerSpec`/`SubFilterSpec` themselves live in `layer_types.py`, and the seqno-rainbow
  color utilities in `layer_colors.py`; `layers.py` keeps `filter_items`/`summarize_selection`/
  `build_layers`.
- **`pdf_model.py`** — the only module that calls into `fitz` for extraction. `PdfDocument` wraps
  the open document; `extract_text_items`/`extract_image_items`/`extract_annot_items`/
  `extract_drawing_items` each return `list[OverlayItem]` for one page; `collect_drawing_colors`
  scans a page's `get_drawings()` once to populate the dynamic stroke/fill color sub-filters. Split
  into `pdf_model_core.py` (`PdfDocument` + shared geometry/matrix/color helpers),
  `pdf_model_text.py` (`extract_text_items`), `pdf_model_image.py` (`extract_image_items`),
  `pdf_model_drawing.py` (`extract_annot_items`/`extract_drawing_items`/`collect_drawing_colors`) —
  one file per independent fitz-extraction concern.
- **`overlay_canvas.py`** — `PageView`: the left-pane Tk `Canvas` showing the rendered page pixmap
  with overlay shapes drawn on top, plus page nav/zoom controls and a hover tooltip. The tooltip
  widget itself lives in `overlay_tooltip.py`, and the pure hover-metadata text formatting in
  `overlay_metadata_format.py`; `overlay_canvas.py` keeps `PageView`'s nav/draw/selection state,
  which are genuinely one cohesive widget.
- **`control_panel.py`** — `ControlPanel`: the right-pane checkbox tree built from the `LAYERS`
  registry, with collapsible sub-filter groups (checkboxes or color swatches).
- **`inspector.py`** (the package's entry point, `python -m rastervec.Evaluation.inspector.inspector
  [pdf]` — renamed from the original standalone tool's `app.py`) — `InspectorApp` wires the two
  panels together, owns `AppState` (current page/zoom, per-page extraction and color caches), and
  drives the redraw cycle. `REFERENCES_DIR` resolves to the repo-root `references/` folder (three
  levels above the `inspector/` package: `inspector` → `Evaluation` → `rastervec` → repo root).
  CLI arg-parsing/PDF-path-resolution (`parse_args`/`pick_initial_pdf`/`pick_pdf_with_dialog`/
  `resolve_pdf_path`/`main`) lives in `inspector_cli.py`, imported by `inspector.py`'s own
  `if __name__ == "__main__":` block only (avoids a module-load cycle, since `inspector_cli.py`
  itself imports `InspectorApp`/`REFERENCES_DIR` back from `inspector.py`).

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
terminology used throughout this section. `rastervec/` is organized into six buckets:
`commons/` (shared foundation), `core/` (orchestrator + registry + public API),
`P1_Reading_Native/` (the one, always-run extraction phase), `P2_Raster_To_Vec/` (pluggable
raster→vector backends), `P3_Vector_Parsing/` (pluggable vector-parsing/OCR backends),
`P4_Output_Organization/` (the one, always-run output-combination + coordinate-space-guard
phase) — plus `Evaluation/`, `notebooks/`, `weights/` alongside them (benchmarking/dev tooling,
not phase code). **Sibling P2 backends (`Stub`/`Junction`) and sibling P3 backends
(`VectorClassification`/`LegacyRecreation`) import nothing from each other** —
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
    p3="VectorClassification", enable_fast=True, verbose=False, compute=None, progress_counter=None,
    on_debug_layer=None) -> PipelineResult`. Body: `phase1 = P1.read_and_extract(...)` →
    `p2_vectors, p2_texts = P2_REGISTRY[p2](phase1.images, phase1.page)` → `p3_vectors, p3_texts =
    P3_REGISTRY[p3](phase1.vectors, p2_vectors, phase1.page, **forwarded_kwargs)` →
    `texts, vectors = P4.organize_outputs(phase1.texts, p2_texts, p3_texts, p3_vectors,
    phase1.page)` (forwarded kwargs — `enable_fast`/`verbose`/`compute`/`progress_counter`/
    `debug_out`/`on_debug_layer` — are only passed to a backend whose own signature declares that
    parameter, via `inspect.signature`; `on_debug_layer` is the streaming counterpart to
    `debug_out`/`render_debug` — see `registry.py`'s docstring). CLI:
    `python -m rastervec.core.pipeline --pdf PATH --page N [--p2 Stub] [--p3 VectorClassification]
    [--no-fast] [-v]`. No `stop_after`/partial-run support — always a full Phase1→P2→P3→P4 run.
  - **`registry.py`** — `P2_REGISTRY`/`P3_REGISTRY` (name → backend callable),
    `resolve_p2`/`resolve_p3` (`ValueError` listing valid names on a miss), `DEFAULT_P2="Stub"`,
    `DEFAULT_P3="VectorClassification"`. Also `P2_RENDER_DEBUG`/`P3_RENDER_DEBUG` — a *separate*,
    optional registry of each backend's own `render_debug(debug_out, page_meta) ->
    list[(stage, label, hex, pdf_bytes)]` function (a backend with nothing to render, e.g. Stub,
    simply isn't in these dicts) — the *batch* debug path: reads a fully-populated `debug_out`
    after the whole run finishes. Every P2/P3 backend with a `render_debug` also accepts an
    `on_debug_layer` kwarg on its own `extract`/`parse` — the *streaming* counterpart, a
    `(stage, label, hex, pdf_bytes) -> None` callback `run_pipeline` forwards whenever a caller
    passes one (independent of `verbose`/`debug_out`), which the backend calls immediately after
    rendering each of its own layers, interleaved with its normal computation, instead of only
    after the whole run — so a caller (`scripts/generate_pipeline_report.py`) never has to hold a
    backend's heavier step-local debug data (render crops, masks) any longer than that one step's
    own rendering needs it. Not itself a registry entry (it's a call-time callback). Adding a new
    backend = one folder implementing the `Phase2Backend`/`Phase3Backend` interface, plus one line
    in each applicable dict here.
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
    (`page.get_image_info(xrefs=True)`) — feeds Phase 2. The whole-page render composes
    `page.fitz_page.derotation_matrix` with the zoom matrix before calling `get_pixmap` —
    `get_pixmap()` always bakes the page's own `/Rotate` into what it renders regardless of the
    `matrix` passed, so without counter-rotating first, a 90°/270° page's raster comes back in
    rotated display space (dimensions swapped) while still tagged with the unrotated `Image.bbox`
    every other Phase-1 output uses, corrupting anything downstream (e.g. `Junction`'s `to_page`
    fraction-of-frame mapper) that trusts that bbox. Embedded images
    (`page.get_image_info`/raw XObject pixmaps) were never subject to `/Rotate` in the first
    place, so they need no such correction.
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
  - **`VectorClassification/`** — a reduced 2-step Vector Classification chain (down from the
    original 12-step chain restored from the repo's former `Vector_Classification/` package —
    every filter beyond the two below has since been removed; see `classify_vectors.py`'s own
    module docstring), followed by FAST filtering and a full-PaddleOCR-per-cluster stage
    matching `archive/raster_parser/scripts/type2_dump_extraction_pipeline.py::
    run_ocr_extraction`'s pattern, but with **no merge-across-buckets/re-grouping step in
    between** — a FAST-surviving classification cluster is itself the OCR unit, not re-clustered
    first. `parse.py` combines `vectors_p1 + vectors_p2` into one flat pool, then:
    `classify_vectors.py`'s per-`(layer,color)`-bucket `_classify_bucket` — `group_filters.py::
    combine_overlapping_seq` (seqno-overlap merge) then `cluster_filters.py::
    cluster_spatial_groups` (constrained single-linkage spatial clustering via
    `commons.helpers.clustering.cluster_spatial`); a `StepResult`/`CategoryResult` per step,
    `role="kept"` only now, since neither remaining step drops anything — → this folder's own
    `fast_filter.py` (`detect_text_fast`, a whole-page FAST mask scored per individual member
    vector of each surviving classification cluster, still per-`(layer,color)`-bucket cluster at
    this point; a cluster passes if any one member vector clears threshold) → per FAST-surviving
    cluster, in bounded chunks of `config.DETECT_RENDER_CHUNK_SIZE` clusters at a time (some
    clusters render to tens of MB even at the base dpi, so the whole page's clusters are never all
    held in memory at once): render (`commons.renderer.ocr_prep.render_cluster_with_dynamic_dpi`)
    → this folder's own `paddle_engine.py::PaddleDetectBackend.detect` (PaddleOCR's own text
    detector, bare `detect(bgr)` shape; dispatched per cluster across Pool-2 workers via
    `paddle_engine.py::_detect_job` when a caller passes `compute`) → crop + deskew each detected
    quad (`paddle_engine.py::hough_deskew` — an axis-aligned crop rotated by a raster-refined
    Hough-line + `cv2.minAreaRect` combined angle estimate, not a perspective warp). Every chunk's
    quads accumulate into one flat, page-wide pool, which is then recognized
    (`paddle_engine.py::PaddleRecBackend.recognize_crops`) in `config.OCR_BATCH_SIZE`-sized
    batches across the *whole page* at once (dispatched per batch via `paddle_engine.py::
    _recognize_crops_job` when `compute` is given) rather than one call per cluster — a page with
    hundreds of small text clusters makes a handful of batched PaddleOCR calls instead of hundreds
    — with a blank-recognition retry sweep (+90/180/270 via `recognize_crops_raw`/
    `_recognize_crops_raw_job`, same page-wide batching) before giving up on a detection. No Radon
    deskew and no whole-page similarity dedup (both removed — every surviving cluster still gets
    its own independent detect pass, just batched recognition). `parse.py::render_debug`
    (`P3_RENDER_DEBUG["VectorClassification"]`) renders one `kept bbox` layer per classification
    step whose kept boxes differ from the previous step's, plus fast/ocr/rotation/retry/drawing
    layers, from whatever `parse()` stashed into `debug_out` (`cluster_detections`'s raw
    per-cluster bgr arrays are only accumulated into `debug_out` at all when a caller actually
    passed one, for the same memory reason as the render chunking above).
  - **`LegacyRecreation/`** — a genuine from-scratch port (not a wrapper) of
    `archive/raster_parser`'s own Type-2 algorithm onto `commons.models` types, selectable as a
    normal P3 backend (distinct from the separate `legacy`/`legacy_adapter.py` engine axis, which
    shells out to the real unmodified archive codebase — see that bullet below). `parse.py`:
    `filters.py::filter_text_vectors` classifies filled vectors into glyph-ink candidates vs.
    everything else (drawing) → `wordgrouping.py::cluster_by_seqno` groups glyph candidates by
    content-stream draw-order adjacency into `WordGroup`s → each group is rendered once
    (`commons.renderer.render_vector_cluster`), padded, and run through this folder's own
    detect+recognize pair (`paddle_engine.py::PaddleDetectBackend.detect` — PaddleOCR's own
    text-detector — then `PaddleRecBackend.recognize_crops` on each detected quad's own
    perspective-cropped region, `_rotate_crop`), so a `WordGroup` yields zero, one, or several
    `Text`s, each positioned/rotated from its own detected quad mapped back to page space
    (`commons.renderer.pixel_to_page_bbox`) rather than from the group's own vector geometry.
    `parse.py::render_debug` (`P3_RENDER_DEBUG["LegacyRecreation"]`) renders
    `filter_fill`/`group_words`/`ocr`/`drawing` layers.

  **Clustering/filtering always operates within one `(layer, color)` bucket, never across
  buckets**, in every P3 backend that separates by layer/color at all — two vectors in different
  layers, or with different stroke/fill colors, are never spatially merged together regardless of
  page proximity.
- **`P4_Output_Organization/`** — the one, always-run output-organization phase (not pluggable,
  same style as `P1_Reading_Native/` — no reason for this to vary by backend):
  `organize.py::organize_outputs(texts_p1, texts_p2, texts_p3, vectors_p3, page) -> (texts,
  vectors)`. Two jobs: (1) combine every phase's text output (`phase1.texts + p2_texts +
  p3_texts`) with Phase 3's already-final vectors into the one `(texts, vectors)` pair
  `core.pipeline.run_pipeline` returns — moved out of `pipeline.py` itself into this named seam;
  (2) a coordinate-space consistency backstop — every `Text`/`Vector` is supposed to stay in
  unrotated MediaBox space end-to-end (see "Coordinate spaces" above), so this logs a warning
  (never silently drops or reprojects) for any item whose bbox doesn't fit the page's own
  unrotated `width`/`height`, the shape of bug that `P1_Reading_Native/image_extract.py`'s
  whole-page raster had before it was fixed to counter-rotate via `derotation_matrix` (see that
  module) — this phase exists to catch a future regression like that one at the seam instead of
  letting it silently reach final output.
- **`rastervec/OCR/`** (top-level: `fast_detect.py`, `radon.py`, `Paddle_OCR/ocr_backend.py` +
  `render_ocr.py`) — **deprecated, but not dead**: each P3 backend under `P3_Vector_Parsing/`
  now has its own duplicated copy (`P3_Vector_Parsing/VectorClassification/{fast_detect,
  paddle_engine}.py` — no Radon module, deskewing was dropped in favor of seqno clustering +
  PaddleOCR's own detector/angle-classifier — see the `P3_Vector_Parsing/` bullet above for what
  each backend actually does now), so no *new* P3 backend should import this folder. But it's
  still genuinely imported by `core/parallel/pool.py::warmup()`, `commons/renderer/stages.py`, the
  Junction P2 backend, and the old `pipelines/current.py`+`_steps.py` — see the top-of-file "not
  dead" note and
  `docs/old_pipeline_migration.md` for the full live-dependent list and the proposed cleanup.
  Don't build *new* code on it, but don't delete it either without finishing that migration.
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
  - **`scripts/label/CAD_font_label.py`** (`CadFontLabelApp`) — CAD-font character + baseline
    labelling, step 1 of `docs/cad_font_vector_recognition.md`; a separate, standalone workflow
    from the `master_label.py` chain above. Two independent Tk modes: **Baseline** (drag out a
    new baseline — a line with a direction, `Baseline(origin, direction)` — or click an existing
    one to make it active; a side panel lists every `cad_font` label on the page, checked iff
    assigned to the active baseline via `LabelEntry.baseline_id`); **Label** (identical flow to
    `vector_label.py` — flat vector pool, click/ctrl-click/drag-select, inline text+rotation
    label bar, click-a-labelled-vector edits in place — entries get `source="cad_font"`, never
    reads/writes `baseline_id`). Per this feature's "commons-only" import constraint,
    `extract_vectors` comes from `P1_Reading_Native.vector_extract` (not the deprecated
    `pipelines._steps`, which `vector_label.py` still uses), and there is no layer/color/width
    bucket-filter side panel since its backing `separate_by_layer_color_width` lives in the
    equally off-limits `pipelines.current` — `self.vectors` is always the full, unfiltered pool.
    Not unit-testable (a real Tk event loop); smoke-test manually via
    `.venv/Scripts/python.exe scripts/label/CAD_font_label.py path/to.pdf --page 0`.
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

  **Note:** this bullet's own function/type names (`evaluate_metrics`, `MetricSuiteResult`,
  `METRIC_GROUPS`, the `gt_miss_attributed_to_*` fields) have drifted from the current source,
  which exposes `evaluate_text_metrics`/`TextMetricSuiteResult` over 5 `TEXT_TYPES` instead —
  treat this whole bullet as describing the metric suite's *design*, not its exact current API,
  and read the source directly for real signatures. What's still accurate: `metrics.py` is split
  by concern across `metrics_core.py` (`OverlapGraph`/`Ratio`/`GtRegion`/`Prediction`/
  `build_overlap_graph(s)`), `metrics_text_overlap.py` (categories 1-3: label description, char/word
  overlap), `metrics_distributions.py` (font-size distribution, category 4 bbox accuracy, category 5
  rotation accuracy), and `metrics_suite.py` (category 6 funnel stats, the box-overlay data, and the
  top-level `evaluate_text_metrics`/`aggregate_text_metrics` orchestration) — `metrics.py` itself is
  now a re-export surface, since many callers import specific names straight from its path.
- **`Evaluation/Evaluate/vector_metrics.py`** *(implemented)*: the vector-provenance counterpart to
  `metrics.py`'s text metrics — vector pairing, vector count accuracy, endpoint accuracy, and
  per-property accuracy, scored over `GeometryEntry` (a `dataclass` unifying GT
  `GeometryAnnotation`s and pipeline `Vector` predictions into one comparable shape, with
  `VECTOR_TYPES = ("vector_to_raster", "original_raster")`). Pure — no pipeline/`label_schema`
  import; `adapters.py` builds `GeometryEntry` lists from real `Vector`/`GeometryAnnotation`
  objects via `geometry_entries_from_vector`/`geometry_entries_from_annotations` (duck-typed, not
  imported here).
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
- **`rastervec/pipelines/`** — superseded in *intent* by `core/pipeline.py` +
  `P3_Vector_Parsing/*/parse.py` (see the `core/` and `P3_Vector_Parsing/` bullets above), but
  **not dead in practice**: `_common.py` and `sub_pipelines/` no longer exist, but `current.py`,
  `_steps.py`, and `result.py` are still actively imported by several tools — see the
  top-of-file "not dead" note for the full list and `docs/old_pipeline_migration.md` for the
  proposed retirement plan. New code should use `core.pipeline.run_pipeline` instead.
  `pipelines/legacy.py` is a separate, unrelated live file in this folder —
  `run_pipeline(pdf_path, page_index, *,
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
  `outputs/pipeline_report/<ts>__<config-stem>/<pdf-stem>/` (the run's source config path is
  also recorded inside `config_and_hyperparameters.txt`) — see the artifact breakdown below.
  **`benchmark: true`** (mutually exclusive with `vectorise`) additionally makes each input's own
  scoring artifact for `pipeline_report_benchmark.py` — `input_files` may be `.pdf` or `.json` label
  sidecars; per page the pipeline runs **once** on `convert_page_to_vector_text` output (text-as-
  vectors over the untouched drawings) and that one run's OCR is scored against both label classes.
  In benchmark mode the per-input folder is named from `BenchInput.key` with its `pdf:`/`labels:`
  prefix stripped (`generate_pipeline_report.py::_bench_doc_name`), not the literal `<pdf-stem>` —
  a `scripts/label/master_label.py` folder's `pdf_path` is always that folder's own `original.pdf`,
  so keying off the PDF filename would collide (and silently overwrite) whenever a config
  benchmarks more than one labelled folder.
  Alongside the normal report artifacts (below) it writes `ground_truth_auto.json` (+
  `ground_truth_manual.json`), the split `{auto,manual}_{bbox,text}.pdf` overlays (`label_overlays.py`,
  registered in the manifest under stage `benchmark`), and a run-root `benchmark.json` marker
  (`entries[].key` = `pdf:<stem>` / `labels:<json-stem>`). Runs under `pipeline: legacy` too (archive's
  pipeline per converted page — `legacy_adapter.run_archive_pipeline` imports `torch` before the
  paddle-first archive import for the Windows `shm.dll` gotcha; legacy emits only `reconstructed`).
  Every mode writes: one multi-page PDF **per visual layer** (`<stage>__<layer>.pdf`), `dump.json`
  (`Evaluation/dump_io.py` — every `Text` + `Vector`, reloadable), `config_and_hyperparameters.txt`
  (config + every `rastervec.config` constant), `manifest.json` (its `layers` list — `{stage,
  layer, file, color}` — drives the viewer). Each layer file is written incrementally by a small
  `_LayerWriter` (one `fitz.Document` per layer filename, kept open across that PDF's page loop):
  every rendered page's bytes are inserted and dropped immediately rather than accumulated in a
  Python list and merged in one pass at the end; a layer blank on every page (no content
  stream — e.g. `phase2` under `p2: "Stub"`) is never written. For `pipeline: "current"`: the
  fixed `phase2__*.pdf`/`reconstructed__*.pdf` layers (no `phase1`/`final` — the inspector
  already shows native words + raw vectors, and each backend's own `drawing`/`ocr` layers show
  the final output; see
  `commons/renderer/stages.py` — these necessarily render post-hoc, from the whole finished
  `PipelineResult`), **plus every backend-specific debug layer** each active P2/P3 backend
  produces — streamed straight into the writer via `on_debug_layer` (`_debug_layer_sink`), passed
  into `run_pipeline` itself, so each layer reaches disk the moment that backend renders it rather
  than only after the whole page's pipeline run finishes (e.g. `VectorClassification` emits one
  `kept bbox` layer per classification step that changed the kept set plus
  fast/ocr/rotation/retry/drawing layers, `Junction` emits its own raster-stage layers —
  see the `P2_Raster_To_Vec/`/`P3_Vector_Parsing/` bullets above for what each backend renders,
  and `core/registry.py`'s docstring for the streaming/`on_debug_layer` convention itself).
  There is no per-stage `.txt` stats file for the `current` engine (the old engine's
  `Evaluation/Report/stage_stats.py` numeric-stats convention doesn't generalize across backends
  with genuinely different internals) — `dump.json` is the reloadable source of truth instead.
  `paddle_detect_images/`/`paddle_recog_images/`/`paddle_ocr_images/` (PNG debug crops — what
  PaddleOCR's detector/recognizer actually saw, per-P3-backend savers in
  `scripts/debug_image_savers.py`) are written unless the config sets `debug_images: false`
  (`ReportConfig`, default `true`), capped at `_DEBUG_IMAGE_CAP` (100) per folder per input
  document — an `_ImageReservoir` reservoir-samples uniformly across every page (seeded from the
  document name, so reruns pick the same crops; rejected crops are never encoded);
  `pipeline_report_benchmark.py` links the first 5 of each folder into `report.html` in place
  (never copied). Every P3 backend's debug layers render **lazily** — `_emit(lambda: ...)`
  renders nothing unless an `on_debug_layer` callback is attached, and that render + hand-off time
  is its own `debug_render` sub-step. Report-generation cost is timed too: each `PageDump` carries
  `debug_durations` (`conversion`/`stage_layers`/`debug_images`/`extra_predictions`) and
  `dump.json` a document-level `doc_durations` (`label_overlays`/`layer_save`);
  `Evaluation/Evaluate/timing.py` takes `debug_render` back out of `phase3` so `total` stays
  pipeline-only, and adds `debug.*`/`debug.total`/`total_incl_debug` rows (legacy engine
  included). In benchmark mode, an input with manual vector (`original_vector`) labels also gets
  `benchmark__extra_text.pdf` (OCR text touching no GT), `benchmark__extra_vectors.pdf` (vectors
  sent to OCR that no GT region covers) and `benchmark__missed_vectors.pdf` (drawing-output
  vectors inside a manual-label region) — `label_overlays.extra_predictions`, rendered by
  `report_artifacts._add_extra_prediction_layers`. `legacy` still only ever emits the single
  `reconstructed` row. There is no `stop_after`/partial-run support for `pipeline: "current"` —
  `final_stage` (validated against `core.pipeline`'s short `phase1`/`phase2`/`phase3` names) only
  trims which of the fixed phase-level artifacts render, not how much of the pipeline executes,
  and doesn't gate the per-backend debug layers at all (those always render in full).
  `scripts/pipeline_report_viewer.py` is the Tkinter counterpart: **1 or 2** per-PDF folders →
  toggleable source page + one side-by-side panel per folder, each a checkbox per layer PDF (grouped
  by stage, `all`/`none` per group, incl. the `benchmark` overlay group), each checked layer
  rasterized with a real alpha channel (`get_pixmap(alpha=True)`) and alpha-composited over the base
  in panel order — so a layer from folder A and a layer from folder B show together;
  zoom/pan/page-flip. `pipeline_report_benchmark.py` emits ready-to-paste invocations in
  `viewer_commands.txt`.

  Both scripts are split by concern across several sibling files (each with a thin
  re-export back through the main script's own path, so `from scripts.generate_pipeline_report
  import X` / `from scripts.pipeline_report_benchmark import X` keep working regardless of which
  file `X` actually lives in): `generate_pipeline_report.py` keeps only orchestration
  (`_process_pdf`/`_process_pdf_benchmark`/`build_arg_parser`/`main`) plus its own
  `_bench_ground_truth_by_type`/`_filter_valid_pages`/`_bench_doc_name`/`_image_dirs`;
  `scripts/report_config.py` holds its `ReportConfig`/`BenchInput` schema; `scripts/
  debug_image_savers.py` holds the seven per-P3-backend pre-OCR debug-image dumpers;
  `scripts/report_artifacts.py` holds the artifact-writing machinery (`_LayerWriter`/
  `_accumulate_page`/`_active_artifacts`/`_finalize_doc_dir`/`_write_label_overlays`/...).
  `pipeline_report_benchmark.py` keeps only its `main`/`build_arg_parser` orchestration;
  `scripts/benchmark_run_loading.py` holds `RunEntry`/`_load_run`/`_merge_gt`/`_score_text`/
  `_score_vectors`/`_load_timings`; `scripts/benchmark_report_sections.py` holds the HTML section
  builders (`_add_text_sections`/`_add_vector_sections`/`_add_timing_sections`). The benchmark
  report renders no example crops — individual errors are inspected with
  `pipeline_report_viewer.py` over the run folders (incl. the `benchmark__extra_*` layers).

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
   `PipelineResult.extra["p2_debug"]`/`["p3_debug"]`). Factor your actual rendering logic into one
   small `_render_<stage>_layers(...)` helper per pipeline step (built from the three shared
   primitives in `commons/renderer` — `render_boxes_pdf`/`render_text_pdf`/`render_vectors_pdf` —
   no generic interpreter, no shared rendering abstraction; each backend renders its own data),
   then two call sites reuse those same helpers: a `render_debug(debug_out, page_meta) ->
   list[(stage, label, hex, pdf_bytes)]` in the same module (the *batch* path, reading your
   `debug_out` shape back after the whole run finishes — register it in `core/registry.py`'s
   `P2_RENDER_DEBUG`/`P3_RENDER_DEBUG`), and an `on_debug_layer: Callable[[str, str, str, bytes],
   None] | None = None` kwarg on your `extract`/`parse` itself (the *streaming* path — call it
   with each stage's own layer(s) immediately after that stage computes them, interleaved with
   your normal computation, instead of only after the whole chain finishes; `core.pipeline`
   forwards it whenever a caller passes one, independent of `verbose`/`debug_out` — see every
   existing P2/P3 backend's `parse.py`/`adapter.py` for the pattern). This lets a caller that only
   wants the rendered layers (e.g. the report generator) never hold your heavier step-local debug
   data (render crops, masks) any longer than that one step's own rendering needs it. A backend
   with nothing worth visualising (e.g. Stub) can skip this step entirely.

Also add tests under the matching `tests/rastervec/P2_Raster_To_Vec/`/`P3_Vector_Parsing/`
subfolder using the synthetic PDF fixtures, and new third-party dependencies to
`requirements.txt` only when the backend that needs them is actually implemented.
