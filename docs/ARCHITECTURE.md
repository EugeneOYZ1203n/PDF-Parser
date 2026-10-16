# Architecture: phases and backends

`extract()` runs one PDF page through four phases. Two of them are always the same; the other
two are **pluggable backends**, chosen by name. This page explains how the pieces fit together
and how to add a backend. For the public API (parameters, errors, wiring into a host app) see
`raster_parser/INTEGRATION.md`; for the output models see `docs/model_migration.md`.

## Layout

| Folder | What it holds |
|---|---|
| `commons/` | The shared foundation: the models (`Text`, `Vector`, `Image`, `Page` / `PageMeta`, `ExtractResult`), geometry and text-orientation helpers, rendering primitives, logging and paths. It stays thin: no phase-specific logic. |
| `core/` | The orchestrator and the backend contract: `interfaces.py` (the backend protocols), `registry.py` (backend names → functions), `settings.py` (`RunSettings`), `pipeline.py` (`run_pipeline`), `result.py` (`PipelineResult`). |
| `P1_Reading_Native/` | Phase 1, always the same. |
| `P2_Raster_To_Vec/<Name>/` | Phase 2 backends: `Stub`, `v1RasterToVec`. |
| `P3_Vector_Parsing/<Name>/` | Phase 3 backends: `Stub`, `v1VectorClass`, `v2VectorClass`. |
| `P4_Output_Organization/` | Phase 4, always the same. |

## Flow

`extract()` (`main_pipeline_extract.py`) checks the backend names and the PDF path, builds a
`RunSettings` from its arguments and calls `core.pipeline.run_pipeline`:

```
                     ┌── images ──▶ Phase 2 (p2) ──▶ vectors, texts ──┐
Phase 1 (read page) ─┤                                                ├─▶ Phase 4 ─▶ texts, vectors
    native texts ────┼── vectors ─▶ Phase 3 (p3) ──▶ vectors, texts ──┤
                     └────────────────────────────────────────────────┘
```

1. **Phase 1** (`P1_Reading_Native/phase1.py::read_and_extract`) opens the PDF and returns the
   page plus everything native extraction gives without any classification: native words,
   raw vector paths (clip and group info folded into each path) and embedded images.
2. **Phase 2** (`p2`, default `v1RasterToVec`) gets **Phase 1's images only** and turns raster
   content into vectors and OCR text. `extract(enable_raster_pass=False)` runs `Stub` instead.
3. **Phase 3** (`p3`, default `v1VectorClass`) gets **Phase 1's vectors only** and splits them
   into text (read with OCR) and drawing. Phase 2 and Phase 3 are independent branches: Phase 3
   never sees Phase 2's output, and Phase 2's vectors never go through Phase 3.
4. **Phase 4** (`P4_Output_Organization/organize.py::organize_outputs`) combines everything into
   one `(texts, vectors)` pair:
   - texts: native words, then Phase 3 text, then Phase 2 text, with `block_no` renumbered into
     a separate range per phase;
   - vectors: Phase 3's, then Phase 2's, with Phase 2's `seqno` moved past Phase 3's largest, so
     raster traces paint on top and every seqno is unique;
   - it logs a warning for any item whose bbox falls outside the page (see
     [Coordinate spaces](#coordinate-spaces)). It never drops or moves an item.

`run_pipeline` returns a `PipelineResult` (with per-phase timings in `step_durations`);
`extract()` turns it into the `ExtractResult(page, texts, vectors)` it returns.

A backend may need more than its input list. A Phase 2 backend can reopen the whole page from
`page.doc_path` / `page.meta.index`; `v1RasterToVec` does that (it clears the page with
LibreOffice and works on what is left) and ignores `images`.

## The backend contract

Every backend is a plain function matching a protocol in `core/interfaces.py`:

```python
# Phase 2
def extract(images: list[Image], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]: ...

# Phase 3
def parse(vectors: list[Vector], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]: ...
```

- The protocols are structural (`typing.Protocol`); nothing checks them at runtime.
- `settings` (`core/settings.py`) carries the run options every backend receives, whichever
  backend it is: `ocr_dpi`, `use_angle_cls`, `workers`, `dump_dir`, plus the debug callbacks
  `on_debug_layer` / `on_debug_image` ([Debug output](#debug-output)). A backend reads the
  fields it needs and ignores the rest. A new option goes on `RunSettings`, not on one
  backend's signature.
- Both return `(vectors, texts)` in unrotated page space. A Phase 3 backend returns the vectors
  it keeps as drawing; the ones it classified as text are left out.
- `core/registry.py` maps names to functions (`P2_REGISTRY`, `P3_REGISTRY`, with `DEFAULT_P2` /
  `DEFAULT_P3`). `resolve_p2` / `resolve_p3` look a name up and raise `ValueError` listing the
  valid names on a miss. `run_pipeline` and `extract()` resolve both names before doing any work.

## Backends are self-contained

**Sibling backends share no code.** A backend imports nothing from another backend in its own
phase or in the other one. If it needs infrastructure a sibling also has (a PaddleOCR wrapper,
colour separation, an autotrace step, ...), it keeps its own copy. The only imports allowed from
outside its folder are:

- `raster_parser.commons` (models, helpers, rendering, logging, paths);
- `raster_parser.core.interfaces` and `raster_parser.core.settings` (the contract).

The duplication is deliberate: any backend can be rewritten or removed without touching a
sibling's files, and a new backend can start as a copy of an existing one and change freely.
For the same reason, `commons/` only takes code that is genuinely phase-independent foundation;
logic that one backend needs stays in that backend.

The rule is enforced by review. All current backends follow it.

## Adding a backend

1. **Create the folder** `P2_Raster_To_Vec/<Name>/` or `P3_Vector_Parsing/<Name>/`, with an
   `__init__.py` and a function matching `Phase2Backend` / `Phase3Backend`. Keep it
   self-contained (above). Split the work into small functions per step so each one can be
   tested on its own. The v1 backends name their steps `s1_*.py`, `s2_*.py`, ... in run order.
2. **Register it**: one line in `core/registry.py`'s `P2_REGISTRY` or `P3_REGISTRY`. It can then
   be selected with `extract(..., p2="<Name>")` / `p3="<Name>"`. Change `DEFAULT_P2` /
   `DEFAULT_P3` only if it should become the default, since that changes `extract()`'s output.
   `import raster_parser` imports the registry, so a backend whose modules load heavy libraries
   at import time (OpenCV, SciPy, PaddleOCR, ...) is registered through a small function that
   imports it on first call, as `_v2_vector_class` is.
3. **Write a `README.md`** in the folder, like the v1 ones: what it does step by step, its entry
   point, and what it needs installed. A backend that replaces or derives from an existing one
   lists every difference in output as a behaviour-change table.
4. **Add tests** under the matching `tests/raster_parser/P2_Raster_To_Vec/<Name>/` or
   `tests/raster_parser/P3_Vector_Parsing/<Name>/` folder. Prefer the synthetic PDFs from
   `tests/conftest.py` (`synthetic_pdf_factory` and friends) over files in `references/`, which
   are not in the repository and give no exact values to assert against.
5. **Dependencies**: add new third-party packages to `pyproject.toml` only in the change that
   adds the backend that needs them.
6. **Debug output** (optional): render each step through the debug callbacks so the pipeline
   report can show it ([Debug output](#debug-output)).

## Debug output

`RunSettings` carries three optional callbacks for tools that want to see inside a run (the
pipeline report, `docs/SCRIPTS.md`). `extract()` never sets them.

| Field | Called as | Meaning |
|---|---|---|
| `on_debug_layer` | `(stage, label, "#rrggbb", pdf_bytes)` | one rendered debug layer: a one-page PDF of the page, in unrotated page space |
| `on_debug_image` | `(folder, name, make_image)` | one debug image; `make_image()` builds the PIL image and is only called if the tool keeps it; `folder` may contain `/` |
| `on_step_time` | `(phase, name, seconds)` | one of the backend's own sub-steps, once per name when the page is done; the time is part of that phase's total (v2VectorClass uses `commons/step_timing.StepClock`) |

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

`run_pipeline(..., stop_after="phase1" | "phase2" | "phase3")` stops after that phase: later
backends are not called. Phase 4 still runs on what ran (with Phase 3 skipped, Phase 1's raw
vectors are the output). `extract()` always runs every phase.

Phase 4's `render_output_pdf(page_meta, texts, vectors)` draws a run's final output as a page
(vectors in their real paint, text in its box); the report uses it for its `reconstructed`
layer. `extract()` does not call it.

## Coordinate spaces

Every `Text` and `Vector` stays in one coordinate space from Phase 1 to the output: the page's
**unrotated** space, measured from the **CropBox**'s top-left corner. PyMuPDF's extraction calls
(`get_text`, `get_drawings`, `get_image_info`) all report coordinates this way, whatever the
page's `/Rotate`.

- `PageMeta.width` / `height` are the CropBox's unrotated size; the raw boxes are in
  `PageMeta.mediabox` / `cropbox`, and `PageMeta.rotation` says how the page is displayed.
- `/Rotate` is applied only when something is drawn for display. Rendered pixmaps and
  `page.rect` are already in rotated display space, so geometry measured on a render must be
  mapped back before it is returned.
- A Phase 2 backend maps image pixels to the page through the image's placement:
  `Image.transform` maps the image's unit square into page space (an upright placement filling
  `Image.bbox` is assumed when it is `None`).
- A word's reading direction comes from `Text.direction`, never from the shape of its bbox. A
  bbox only equals the word's length × height when the word is horizontal.

Phase 4 is the backstop: an item whose bbox does not fit inside `width × height` is logged as a
possible rotated-space leak from a Phase 2/3 backend.

## Outside the phases: `png_vectors/`

`png_vectors/` (`extract_vectors_from_png`, `get_image_polys`, also re-exported from
`raster_parser.extract_vectors_from_png`) is a standalone public utility that autotraces a PNG,
or each embedded image of a page, into shared `Vector`s. It is not a backend, is not part of
`extract()`'s flow, and is not bound by the self-contained rule. How an image's vectors are
placed onto the page is described in `docs/UniqueCasesDuringExtractionOfVectors/`.
