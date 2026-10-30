# Architecture: phases and backends

`extract()` runs one PDF page through four phases. Two of them are always the same; the other
two are **pluggable backends**, chosen by name. This page explains how the pieces fit together
and how to add a backend. For the public API (parameters, errors, wiring into a host app) see
`raster_parser/INTEGRATION.md`; for the output models see `docs/model_migration.md`.

## Layout

| Folder | What it holds |
|---|---|
| `models/` | Every data type: the shared models (`Text`, `Vector`, `Image`, `Page` / `PageMeta`, `ExtractResult`), `RunSettings`, and each phase's intermediate results. Nothing outside it declares a dataclass or enum. |
| `commons/` | The shared foundation: geometry and text-orientation helpers, rendering primitives, process pools and logging. `commons/v1_commons/` holds what the v1 backends (and `extract_vectors_from_png`) share. |
| `core/` | The orchestrator and the backend contract: `interfaces.py` (the backend protocols), `registry.py` (backend names → functions), `pipeline.py` (`run_pipeline`, `PHASE_NAMES`). |
| `phase_native_read/` | native_read, always the same. |
| `phase_raster_vectorize/<Name>/` | raster_vectorize backends: `stub`, `v1_raster_to_vec`, `junction`, `involution_stroke` (needs the `ml` extra); and the deep backends `deep_tech_vec`, `deep_vectoriser`, `implicit_sketch_vec` (`ml` extra + their own trained weights). |
| `phase_vector_classify/<Name>/` | vector_classify backends: `stub`, `v1_vector_class`, `v2_vector_class` (walked through in `docs/PIPELINE.md`, terms in `docs/Glossary.md`). |
| `phase_output_assemble/` | output_assemble, always the same. |
| `evaluation/` | Ground-truth labels, input conversion and the scoring behind the report tools (`docs/SCRIPTS.md`); `extract()` never uses it. |

## Flow

`extract()` (`main_pipeline_extract.py`) checks the backend names and the PDF path, builds a
`RunSettings` from its arguments and calls `core.pipeline.run_pipeline`:

```
               ┌── images ──▶ raster_vectorize (p2) ──▶ vectors, texts ──┐
native_read ───┤                                                         ├─▶ output_assemble ─▶ texts, vectors
  native texts ┼── vectors ─▶ vector_classify (p3) ──▶ vectors, texts ───┤
               └─────────────────────────────────────────────────────────┘
```

1. **native_read** (`phase_native_read/native_read.py::read_and_extract`) opens the PDF and returns the
   page plus everything native extraction gives without any classification: native words,
   raw vector paths (clip and group info folded into each path) and embedded images.
2. **raster_vectorize** (`p2`, default `v1_raster_to_vec`) gets **native_read's images only** and turns raster
   content into vectors and OCR text. `extract(enable_raster_pass=False)` runs `stub` instead.
3. **vector_classify** (`p3`, default `v2_vector_class`) gets **native_read's vectors only** and splits them
   into text (read with OCR) and drawing. raster_vectorize and vector_classify are independent branches: vector_classify
   never sees raster_vectorize's output, and raster_vectorize's vectors never go through vector_classify.
4. **output_assemble** (`phase_output_assemble/organize.py::organize_outputs`) combines everything into
   one `(texts, vectors)` pair:
   - texts: native words, then vector_classify text, then raster_vectorize text, with `block_no` renumbered into
     a separate range per phase;
   - vectors: vector_classify's, then raster_vectorize's, with raster_vectorize's `seqno` moved past vector_classify's largest, so
     raster traces paint on top and every seqno is unique;
   - it logs a warning for any item whose bbox falls outside the page (see
     [Coordinate spaces](#coordinate-spaces)). It never drops or moves an item.

`run_pipeline` returns the `ExtractResult(page, texts, vectors)` that `extract()` returns, with
per-phase timings in `step_durations` (keyed by `core.pipeline.PHASE_NAMES`; not part of its
JSON).

A backend may need more than its input list. A raster_vectorize backend can reopen the whole page from
`page.doc_path` / `page.meta.index`; `v1_raster_to_vec` does that (it clears the page with
LibreOffice and works on what is left) and ignores `images`.

## The backend contract

Every backend is a plain function matching a protocol in `core/interfaces.py`:

```python
# raster_vectorize
def extract(
    images: list[Image], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]: ...


# vector_classify
def parse(
    vectors: list[Vector], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]: ...
```

- The protocols are structural (`typing.Protocol`); nothing checks them at runtime.
- `settings` (`models/settings.py`, `RunSettings`) carries the run options every backend receives, whichever
  backend it is: `ocr_dpi`, `use_angle_cls`, `workers`, `dump_dir`, `debug`, plus the debug
  callbacks `on_debug_layer` / `on_debug_image` ([Debug output](#debug-output)). A backend reads
  the fields it needs and ignores the rest. A new option goes on `RunSettings`, not on one
  backend's signature.
- Both return `(vectors, texts)` in unrotated page space. A vector_classify backend returns the vectors
  it keeps as drawing; the ones it classified as text are left out.
- `core/registry.py` maps names to functions (`VECTORIZE_BACKENDS`, `CLASSIFY_BACKENDS`, with `DEFAULT_VECTORIZE` /
  `DEFAULT_CLASSIFY`). `resolve_vectorize` / `resolve_classify` look a name up and raise `ValueError` listing the
  valid names on a miss. `run_pipeline` and `extract()` resolve both names before doing any work.

## Backends don't import each other

A backend imports nothing from another backend in its own phase or in the other one. The only
imports allowed from outside its folder are:

- `raster_parser.models` (the data types) and `raster_parser.commons` (helpers, rendering, logging);
- `raster_parser.core.interfaces` (the contract).

Code two backends both need is not copied: it moves into `commons`. What every phase can use goes
in the general modules (`commons/helpers/`, `commons/renderer/`, ...); what only one family of
backends shares gets its own package, like `commons/v1_commons/` (autotrace, the original
reconstructor's replay, OCR crop normalisation and OCR word -> `Text`, used by `v1_raster_to_vec`,
`v1_vector_class` and `extract_vectors_from_png`). Logic only one backend needs stays in that backend.

`tests/raster_parser/test_module_layout.py` checks it: no backend imports a sibling backend,
and `commons` never imports a phase.

## Shared code map

Who uses what in `commons/` and `core/`. "Phases" means the backend folders under `phase_*`; a
module listed for only one family is a candidate to move into that family's own package.
`tests/test_architecture_doc.py` fails when a module is missing here.

| Module | What it is for | Used by |
|---|---|---|
| `commons/helpers/geometry.py` | Tuple geometry: bboxes, item points, quads, `bbox_iou`, `union_bbox`, axial angles (`wrap_axial`, `axial_distance`), colour rounding, `rgb01_to_hex`, `PDF_POINTS_PER_INCH` | models, `v1_commons`, `phase_native_read`, `v1_raster_to_vec`, `v1_vector_class`, `extract_vectors_from_png`, `v2_vector_class`, `scripts/inspector`, `commons/raster_text/`, `junction` |
| `commons/helpers/clustering.py` | `cluster_spatial` (bbox-gap single linkage on a hash grid), `UnionFind` | `v2_vector_class` |
| `commons/helpers/layer_separation.py` | Layer / colour / width buckets | `v2_vector_class` |
| `commons/helpers/pymupdf_geometry.py` | Live PyMuPDF objects <-> plain item tuples; hover metadata (angles, lengths, matrices) | models, `phase_native_read`, `scripts/inspector` |
| `commons/helpers/pixmap.py` | `pixmap_to_array`, `zoom_matrix(dpi)` | `phase_native_read`, `v1_raster_to_vec`, `extract_vectors_from_png`, `commons/raster_text/` |
| `commons/helpers/text_orientation.py` | Reading direction <-> the old `rotate` / glyph orientation | `v1_commons`; migration helpers for callers (`docs/model_migration.md`) |
| `commons/renderer/_shapes.py` | Exact `Vector` replay onto a page (`replay_drawing_paths`, `replay_item`), `path_color_hex` | `v1_commons`, `commons/renderer/draw.py`, `v2_vector_class`, `scripts/inspector` (tests) |
| `commons/renderer/draw.py` | The spec -> draw layer every PDF is built from (incl. bulk stroke / dot batches); `hex_to_rgb01` | `phase_output_assemble`, `v1_raster_to_vec`, `v1_vector_class`, `evaluation/report`, `evaluation/evaluate`, `commons/raster_text/`, `junction` |
| `commons/renderer/pdf.py` | One-layer debug PDFs (boxes, quads, text, vectors) over `draw.py`; `rasterize_pdf`, `layer_slug` | `phase_output_assemble`, `v1_raster_to_vec`, `v1_vector_class`, `evaluation/report`, `commons/raster_text/`, `junction` |
| `commons/renderer/png.py` | Monochrome cluster renders for OCR | `v2_vector_class` |
| `commons/renderer/ocr_prep.py` | Crop / pad / scale a render for PaddleOCR | `v2_vector_class` |
| `commons/v1_commons/autotrace.py` | pyautotrace PNG -> SVG -> PDF -> `Vector`s | `v1_raster_to_vec`, `extract_vectors_from_png` |
| `commons/v1_commons/ocr.py` | OCR crop normalisation, PaddleOCR 2.x result walking, OCR word -> `Text` | `v1_raster_to_vec`, `v1_vector_class` |
| `commons/v1_commons/render.py` | The original reconstructor's replay (its own `finish()` defaults) | `v1_raster_to_vec`, `v1_vector_class`, `extract_vectors_from_png` |
| `commons/paddle_engine.py` | The one PaddleOCR 2.x engine (`PaddleBackend`: detect, classify 0/180, recognise) | `v2_vector_class`, `commons/raster_text/` |
| `commons/raster_text/` | The raster prelude every raster_vectorize backend but `v1_raster_to_vec` runs first: tiled OCR, colour CLAHE, colour layers, text erasure (`prepare_images`); `strokes.py` (cubic-stroke geometry, layer masks), `debug_tools.py` (debug / sample-run helpers, debug rasters, diff codes), `training.py` (device / loader) | `junction`, `involution_stroke`, `deep_tech_vec`, `deep_vectoriser`, `implicit_sketch_vec` |
| `commons/workers.py` | Spawned process pools and per-worker BLAS thread pinning | `v1_raster_to_vec`, `v2_vector_class`, `extract_vectors_from_png` |
| `commons/logging_setup.py` | The `raster_parser` logger and `configure_logging` | core, `phase_native_read`, `phase_output_assemble`, `v1_raster_to_vec`, `evaluation`, `scripts`, `commons/raster_text/`, `junction`, `involution_stroke` |
| `commons/progress.py` | Stage lines and tqdm bars on stderr at DEBUG log level | core, `phase_native_read`, `v1_raster_to_vec`, `v1_vector_class`, `extract_vectors_from_png`, `scripts` |
| `commons/step_timing.py` | `StepClock`: named step durations | `core/pipeline.py`, `v2_vector_class` |
| `commons/paths.py` | Repo-relative output folders for the scripts | `scripts` (rasterise, report, benchmark) |
| `core/interfaces.py` | The backend protocols | `core/registry.py` |
| `core/registry.py` | Backend names -> functions, resolvers, defaults | `core/pipeline.py`, `main_pipeline_extract`, `evaluation/report` |
| `core/pipeline.py` | `run_pipeline`, `PHASE_NAMES`, `STEP_NAMES` (`stop_after`) | `main_pipeline_extract`, `evaluation/report`, `scripts/generate_pipeline_report` |

## Adding a backend

1. **Create the folder** `phase_raster_vectorize/<Name>/` or `phase_vector_classify/<Name>/`, with an
   `__init__.py` and a function matching `RasterVectorizer` / `VectorClassifier`. It imports
   no sibling backend (above). Split the work into small functions per step so each one can be
   tested on its own. The v1 backends name their steps `s1_*.py`, `s2_*.py`, ... in run order.
2. **Register it**: one line in `core/registry.py`'s `VECTORIZE_BACKENDS` or `CLASSIFY_BACKENDS`. It can then
   be selected with `extract(..., p2="<Name>")` / `p3="<Name>"`. Change `DEFAULT_VECTORIZE` /
   `DEFAULT_CLASSIFY` only if it should become the default, since that changes `extract()`'s output.
   `import raster_parser` imports the registry, so a backend whose modules load heavy libraries
   at import time (OpenCV, SciPy, PaddleOCR, ...) is registered through a small function that
   imports it on first call, as `_v2_vector_class` is.
3. **Write a `README.md`** in the folder, like the v1 ones: what it does step by step, its entry
   point, and what it needs installed. A backend that replaces or derives from an existing one
   lists every difference in output as a behaviour-change table.
4. **Add tests** under the matching `tests/raster_parser/phase_raster_vectorize/<Name>/` or
   `tests/raster_parser/phase_vector_classify/<Name>/` folder. Prefer the synthetic PDFs from
   `tests/conftest.py` (`synthetic_pdf_factory` and friends) over files in `references/`, which
   are not in the repository and give no exact values to assert against.
5. **Dependencies**: add new third-party packages to `pyproject.toml` only in the change that
   adds the backend that needs them.

6. **Debug output** (optional): render each step through the debug callbacks so the pipeline
   report can show it ([Debug output](#debug-output)).

## Debug output

`RunSettings` carries three optional callbacks for tools that want to see inside a run (the
pipeline report). `extract()` never sets them.

| Field | Called as | Meaning |
|---|---|---|
| `on_debug_layer` | `(stage, label, "#rrggbb", pdf_bytes)` | one rendered debug layer: a one-page PDF of the page, in unrotated page space |
| `on_debug_image` | `(folder, name, make_image)` | one debug image; `make_image()` builds the PIL image and is only called if the tool keeps it; `folder` may contain `/` |
| `on_step_time` | `(phase, name, seconds)` | one of the backend's own sub-steps, once per name when the page is done; the time is part of that phase's total (v2_vector_class uses `commons/step_timing.StepClock`) |

Rules for a backend:

- **Only when set.** Render and keep nothing extra when the callbacks are `None`. The output
  must be identical either way, so adding debug output is never a behaviour change.
- **Stream.** Emit each step's layer as soon as that step is done, instead of collecting debug
  data until the end. The tool writes it to disk straight away.
- **Use the standard renderer.** Build layers from `commons/renderer/draw.py` (a spec builder +
  a drawer per primitive, `render_specs_pdf`) or its one-layer wrappers in `pdf.py`
  (`render_boxes_pdf`, `render_quads_pdf`, `render_text_pdf`, `render_vectors_pdf`). Text is
  rotated to its angle, then scaled to fit its box, never sheared. Don't place geometry with
  PyMuPDF calls directly.
- **Name layers by step.** `stage` is the step (`ocr`, `word_grouping`, ...), `label` what the
  layer shows. A trailing per-page count such as `"word groups (25)"` is stripped by the report,
  so one layer keeps one file.
- Document the layers in the backend's `README.md` ("Debug output"). The v1 backends keep theirs
  in `debug_layers.py`.

## Progress output

`commons/progress.py` shows which stage is running, and when, on stderr:

- `stage(name)` -- a context manager writing `HH:MM:SS  ▶ name` before the block and
  `HH:MM:SS  ✔ name  1.23s` after it (`✖` when it raises);
- `bar(iterable, desc=..., total=...)` -- a tqdm bar over a long loop (tiles, batches, buckets,
  pages), including `as_completed` over a process pool;
- `note(line)` -- one timestamped line, e.g. a skipped stage.

All three are active while the `raster_parser.progress` logger is enabled for DEBUG. The
`raster_parser` logger starts at DEBUG (`commons/logging_setup.py`, unless a level was already
set), so progress shows by default; `setLevel(logging.INFO)` on it turns progress off.
`configure_logging()` defaults to DEBUG and writes log records through `tqdm.write`, so they
don't break an open bar. Only the calling process shows progress: spawn workers configure no
logging.

`run_pipeline` writes a stage per phase. A backend wraps each of its steps in `stage(...)`, named
`"p2 <Name> › <step>"` / `"p3 <Name> › <step>"`, and its long loops in `bar(...)`. Progress is
like the debug callbacks: it must never change the output.

`run_pipeline(..., stop_after="native_read" | "raster_vectorize" | "vector_classify")` stops after that phase: later
backends are not called. output_assemble still runs on what ran (with vector_classify skipped, native_read's raw
vectors are the output). `extract()` always runs every phase.

output_assemble's `render_output_pdf(page_meta, texts, vectors)` draws a run's final output as a page
(vectors in their real paint, text in its box); the report uses it for its `reconstructed`
layer. `extract()` does not call it.

## Coordinate spaces

Every `Text` and `Vector` stays in one coordinate space from native_read to the output: the page's
**unrotated** space, measured from the **CropBox**'s top-left corner. PyMuPDF's extraction calls
(`get_text`, `get_drawings`, `get_image_info`) all report coordinates this way, whatever the
page's `/Rotate`.

- `PageMeta.width` / `height` are the CropBox's unrotated size; the raw boxes are in
  `PageMeta.mediabox` / `cropbox`, and `PageMeta.rotation` says how the page is displayed.
- `/Rotate` is applied only when something is drawn for display. Rendered pixmaps and
  `page.rect` are already in rotated display space, so geometry measured on a render must be
  mapped back before it is returned.
- A raster_vectorize backend maps image pixels to the page through the image's placement:
  `Image.transform` maps the image's unit square into page space (an upright placement filling
  `Image.bbox` is assumed when it is `None`).
- A word's reading direction comes from `Text.direction`, never from the shape of its bbox. A
  bbox only equals the word's length × height when the word is horizontal.

output_assemble is the backstop: an item whose bbox does not fit inside `width × height` is logged as a
possible rotated-space leak from a raster_vectorize / vector_classify backend.

## Outside the phases: `extract_vectors_from_png`

`raster_parser/extract_vectors_from_png.py` (`extract_vectors_from_png`, `get_image_polys`) is a
standalone public utility that autotraces a PNG, or each embedded image of a page, into shared
`Vector`s. It is not a backend and not part of `extract()`'s flow; its autotrace is
`commons/v1_commons/autotrace.py`. An image's vectors are placed onto the page through the
image's placement matrix (`page.get_image_rects(xref, transform=True)`).
