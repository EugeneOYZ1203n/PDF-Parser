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
`raster_parser/P2_Raster_To_Vec/v1RasterToVec/s1_libreoffice_macro.py`; importing the package does not need
LibreOffice).

## API

```python
from raster_parser import extract, ExtractResult

result = extract("/path/to/drawing.pdf", page_index=0)
result.page      # PageMeta: size, /Rotate, MediaBox, CropBox
result.texts     # list[Text] -- native words + Phase 3 OCR + Phase 2 (raster) OCR
result.vectors   # list[Vector] -- non-glyph drawings + Phase 2 colour-fill autotrace
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
| `use_angle_cls` | `True` | PaddleOCR angle classification (Type-2) |
| `enable_raster_pass` | `True` | After Type-2: LibreOffice clear → Type-4 if residual page |
| `workers` | `1` | Process pool size for Type-4 tile det / recog / colour autotrace |
| `dump_dir` | `./tmp` | LO + Type-4 dump root (passed to `clear_page`) |
| `debug` | `False` | Keep dumps on disk; otherwise removed after `extract()` |
| `p2` | `"v1RasterToVec"` | Phase 2 (raster pass) backend — `raster_parser.core.registry.P2_REGISTRY` |
| `p3` | `"v1VectorClass"` | Phase 3 (vector parsing) backend — `raster_parser.core.registry.P3_REGISTRY` |

An unknown `p2` / `p3` name raises `ValueError` listing the valid ones. `enable_raster_pass=False`
skips Phase 2 whatever `p2` is (same as `p2="Stub"`).

### Flow

`extract()` runs one page through four phases (`raster_parser.core.pipeline.run_pipeline`):

1. **Phase 1** (`P1_Reading_Native`) — native words (`get_text`), vector paths
   (`get_drawings(extended=True)`, clip / group info folded into each path) and embedded images.
2. **Phase 2** (`p2`, default `v1RasterToVec`) — the raster pass: LibreOffice `DeleteDrawings`
   via `clear_page(..., dump_dir=...)` on a one-page copy (native text / vector shapes removed,
   embedded images kept); if anything is left, `extract_raster_page` runs on it — text/drawing
   split, tiled OCR detection + recognition, then **colour-fill autotrace** of the drawing layer.
3. **Phase 3** (`p3`, default `v1VectorClass`) — fill-glyph OCR on Phase 1's vectors: filled,
   non-white, non-box paths are grouped into words by content-stream order and read with
   PaddleOCR; every other vector is kept as drawing. Phase 3 never sees Phase 2's output.
4. **Phase 4** (`P4_Output_Organization`) — native words, then Phase 3 text, then Phase 2 text,
   with `block_no` renumbered per phase; Phase 3 vectors, then Phase 2 vectors with seqnos moved
   past Phase 3's (raster traces paint on top). Out-of-page items are logged.

No native-vs-OCR comparison. The original native page is not rasterized for Phase 3.

How the phases and backends fit together, and how to add a backend: `docs/ARCHITECTURE.md`.

### Guarantees

- **Intermediates** — LO + Type-4 dumps under `dump_dir` (default: `./tmp`).
  `extract` passes that into `clear_page`; `delete_drawings_from_pdf(..., dump_dir=...)`
  uses the same default. Macro exports beside each opened per-page PDF. With
  `debug=False` (default), `extract()` removes its dump root after finishing; set
  `debug=True` to keep them.
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
| `scripts/type4_pipeline.py` | Type-4 CLI: text/drawing split + colour-fill + native-vs-OCR stats |
| `scripts/type2_nativetext_to_vector_pipeline.py` | Native text → vector PDF / SVG dump |
| `scripts/type2_dump_extraction_pipeline.py` | Type-2 dump + optional OCR |
| `scripts/type2_full_native_to_ocr_pipeline.py` | Native → vector → OCR → compare |
| `scripts/type2_native_wordgroup_png_pipeline.py` | Paddle-prep word-group PNGs from native text |
| `scripts/raster_drawing_autotrace.py` | Standalone PNG → autotrace (no soft/adaptive preprocess) |
| `scripts/raster_drawing_colour_fill_autotrace.py` | Standalone colour-fill autotrace on a PNG |
| `scripts/dto_reconstruct_pipeline.py` | Replay `NativePDFElements` JSON / PDF → drawings |
| `scripts/debug/ocr_test_script.py` | Staged Type-2 native-vs-OCR (single page) |
| `scripts/debug/batch_test_script.py` | Staged Type-2 batch eval |
| `scripts/inspector/` | PDF layer inspector GUI (`python -m scripts.inspector.inspector [pdf]`) — see its `README.md` |
| `scripts/label/` | Ground-truth labelling tools (native / vector / raster labels, `master_label.py` runs them all) — see its `README.md` |
| `scripts/rasterize_pdf.py` | Flatten a PDF to pure raster: `python scripts/rasterize_pdf.py SRC [DST] --dpi 300` (default DST `outputs/rasterize/<stem>_raster.pdf`) |

Helpers for batch eval live in `raster_parser.native_vs_ocr` (e.g. `text_to_vector`,
`page_stats`). That path writes `onlyfill.pdf`, `ocr.pdf`, `out_results.txt`, etc.,
and is separate from `extract()`.

## Tests

```bash
uv sync --group test && uv pip install -e .
uv run pytest tests/ -m "not slow"
uv run pytest tests/raster_parser/ -m "not slow"  # library logic (mirrors raster_parser/)
uv run pytest tests/scripts/                      # scripts/ helpers (mirrors scripts/)
uv run pytest tests/raster_parser/test_main_pipeline_extract.py -m slow
```

`tests/references/test_pdfs_*.pdf` are committed fixture PDFs.
