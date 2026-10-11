# Main application integration

How a host app calls the raster-parser library. Results serialize as
`NativePDFElements` (`TextDTO` / `VectorDTO`).

## Setup

```bash
pip install -e .
# or: uv pip install -e .
```

**System dependency (Type-4 / `enable_raster_pass=True`):** LibreOffice with the
`DeleteDrawings` macro registered (installed on the first LibreOffice call, via
`raster_parser/delete_natives/libreoffice_setup/`; importing the package does not need
LibreOffice).

## API

```python
from raster_parser import extract, NativePDFElements

result = extract("/path/to/drawing.pdf", page_index=0)
json_str = result.model_dump_json()
result.words     # list[TextDTO] — native + Type-2 OCR + Type-4 OCR
result.vectors   # list[VectorDTO] — non-fill drawings + Type-4 colour-fill autotrace
```

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

### Flow

1. Native `get_text` + `get_drawings`; Type-2 OCR on fill glyphs; keep non-fill vectors.
2. If `enable_raster_pass`: LibreOffice `DeleteDrawings` via `clear_page(..., dump_dir=...)`
   on a temp single-page copy (native text/vector shapes removed; embedded images remain).
3. If the cleared page is non-empty: Type-4 via
   `raster_parser.type4_methods.extract_raster_page` — text/drawing split, clean-raster
   re-detection, recognition, then **colour-fill autotrace** on `drawing_final`
   (no soft/adaptive preprocess; masks are already binary). Merge words/vectors.
4. No native-vs-OCR comparison. The original native page is not rasterized for Type-2.

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
