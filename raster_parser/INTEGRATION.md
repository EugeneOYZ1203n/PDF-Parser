# Main application integration

How a host app calls the raster-parser library. `extract()` returns an `ExtractResult` of the
shared `Text` / `Vector` models (it used to return `NativePDFElements`: see `docs/model_migration.md`).

## Setup

```bash
pip install -e .
# or: uv pip install -e .
```

**System dependency (Type-4 / `enable_raster_pass=True`):** LibreOffice with the
`DeleteDrawings` macro registered (installed on the first LibreOffice call, via
`raster_parser/phase_raster_vectorize/v1_raster_to_vec/s1_libreoffice_macro.py`; importing the package does not need
LibreOffice).

## API

```python
from raster_parser import extract, ExtractResult

result = extract("/path/to/drawing.pdf", page_index=0)
result.page  # PageMeta: size, /Rotate, MediaBox, CropBox
result.texts  # list[Text] -- native words + vector_classify OCR + raster_vectorize (raster) OCR
result.vectors  # list[Vector] -- non-glyph drawings + raster_vectorize colour-fill autotrace
json_str = result.to_json()
again = ExtractResult.from_json(json_str)
```

`docs/model_migration.md` maps every old `TextDTO` / `VectorDTO` field to the new models.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_path` | (required) | PDF file path |
| `page_index` | `0` | Page to process (0-based) |
| `ocr_dpi` | `300` | Type-2 crop DPI and Type-4 render DPI |
| `use_angle_cls` | `True` | PaddleOCR angle classification (Type-2 and Type-4) |
| `enable_raster_pass` | `True` | After Type-2: LibreOffice clear → Type-4 if residual page |
| `workers` | `1` | Process pool size for Type-4 tile det / recog / colour autotrace |
| `dump_dir` | system temp dir | Parent for this call's LO + Type-4 dumps; each call works in its own new sub-folder |
| `debug` | `False` | Keep this call's dump sub-folder (its path is logged) and write the raster pass's debug images / QA files into it; otherwise only what the result needs is written and the sub-folder is removed after `extract()` |
| `p2` | `"v1_raster_to_vec"` | raster_vectorize (raster pass) backend — `raster_parser.core.registry.VECTORIZE_BACKENDS` |
| `p3` | `"v1_vector_class"` | vector_classify (vector parsing) backend — `raster_parser.core.registry.CLASSIFY_BACKENDS` |

An unknown `p2` / `p3` name raises `ValueError` listing the valid ones. `enable_raster_pass=False`
skips raster_vectorize whatever `p2` is (same as `p2="stub"`).

### Flow

`extract()` runs one page through four phases (`raster_parser.core.pipeline.run_pipeline`):

1. **native_read** (`phase_native_read`) — native words (`get_text`), vector paths
   (`get_drawings(extended=True)`, clip / group info folded into each path) and embedded images.
2. **raster_vectorize** (`p2`, default `v1_raster_to_vec`) — the raster pass: LibreOffice `DeleteDrawings`
   via `clear_page(..., dump_dir=...)` on a one-page copy (native text / vector shapes removed,
   embedded images kept); if anything is left, `extract_raster_page` runs on it — text/drawing
   split, tiled OCR detection + recognition, then **colour-fill autotrace** of the drawing layer.
3. **vector_classify** (`p3`, default `v1_vector_class`) — fill-glyph OCR on native_read's vectors: filled,
   non-white, non-box paths are grouped into words by content-stream order and read with
   PaddleOCR; every other vector is kept as drawing. vector_classify never sees raster_vectorize's output.
4. **output_assemble** (`phase_output_assemble`) — native words, then vector_classify text, then raster_vectorize text,
   with `block_no` renumbered per phase; vector_classify vectors, then raster_vectorize vectors with seqnos moved
   past vector_classify's (raster traces paint on top). Out-of-page items are logged.

No native-vs-OCR comparison. The original native page is not rasterized for vector_classify.

### Guarantees

- **Intermediates** — LO + Type-4 dumps go in a new sub-folder of `dump_dir` (default: the
  system temp dir) that `extract()` creates for that call and passes into `clear_page`; macro
  exports land beside each opened per-page PDF. With `debug=False` (default), `extract()`
  removes only that sub-folder after finishing, so files already in `dump_dir` and concurrent
  calls are safe; set `debug=True` to keep it (its path is logged).
  `delete_drawings_from_pdf(..., dump_dir=...)` still defaults to `./tmp`.
- **Lazy heavy imports** — Paddle / PyMuPDF / OpenCV load on first `extract()` call.
- **First call** may download PaddleOCR model weights to the local cache.

### Errors

- `FileNotFoundError` if `pdf_path` is missing.
- Paddle / PyMuPDF / LibreOffice errors propagate if OCR, PDF parsing, or clear fails.

## Wiring into a host app

```bash
pip install -e /path/to/2026_RasterToVec_Lib
```

Or in `pyproject.toml`:

```toml
dependencies = ["raster-parser"]

[tool.uv.sources]
raster-parser = { path = "../2026_RasterToVec_Lib", editable = true }
```

```python
from raster_parser import extract
```

## Dependencies

**Runtime (`extract()`):**

```bash
uv sync
pip install -e .
```

Also needs LibreOffice on the host when `enable_raster_pass=True`.

**Notebooks / stats / batch analysis** (not required for `extract()`):

```bash
uv sync --group dev --group notebook --group test
```

## Scripts (offline / QA — not the product API)

| Script | Role |
|--------|------|
| `scripts/inspector/` | PDF layer inspector GUI (`python -m scripts.inspector.inspector [pdf]`) — see its `README.md` |
| `scripts/golden_snapshot.py` | Write / compare the golden `extract()` snapshots in `tests/references/golden/` |
| `scripts/generate_pipeline_report.py` | Config-driven pipeline report: debug layers, debug images, `dump.json`; benchmark mode adds ground truth — see `docs/SCRIPTS.md` |
| `scripts/pipeline_report_viewer.py` | Tk viewer over 1–2 report folders, a checkbox per layer — see `docs/SCRIPTS.md` |
| `scripts/pipeline_report_benchmark.py` | Scores 1–2 benchmark report folders, writes `report.html` (needs `uv sync --group report`) — see `docs/SCRIPTS.md` |
| `scripts/rasterize_pdf.py` | Flatten a PDF to pure raster: `python scripts/rasterize_pdf.py SRC [DST] --dpi 300` (default DST `outputs/rasterize/<stem>_raster.pdf`) |

The Type-2 / Type-4 scripts, the native-vs-OCR / native-vs-raster evaluation helpers and the
DTO-replay script were written against the removed `models` / `parsing` / `rendering` modules
and are removed with them (see `docs/model_migration.md`).

## Tests

```bash
uv sync --group test && uv pip install -e .
uv run pytest tests/ -m "not slow"
uv run pytest tests/raster_parser/ -m "not slow"  # library logic (mirrors raster_parser/)
uv run pytest tests/scripts/                      # scripts/ helpers (mirrors scripts/)
uv run pytest tests/raster_parser/test_main_pipeline_extract.py -m slow
```

`tests/references/test_pdfs_*.pdf` are committed fixture PDFs.
