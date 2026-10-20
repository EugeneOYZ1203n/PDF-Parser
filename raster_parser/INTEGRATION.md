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
| `ocr_dpi` | `300` | Type-4 render DPI, and the crop DPI of `p3="v1VectorClass"` (v2VectorClass picks a DPI per cluster) |
| `use_angle_cls` | `True` | PaddleOCR angle classification for `p3="v1VectorClass"` (v2VectorClass always runs it) |
| `enable_raster_pass` | `True` | After Type-2: LibreOffice clear → Type-4 if residual page |
| `workers` | `1` | Process pool size for Type-4 tile det / recog / colour autotrace, and for `p3="v2VectorClass"`'s OCR detect / recognise |
| `dump_dir` | `./tmp` | LO + Type-4 dump root (passed to `clear_page`) |
| `debug` | `False` | Keep dumps on disk; otherwise removed after `extract()` |
| `p2` | `"v1RasterToVec"` | Phase 2 (raster pass) backend — `raster_parser.core.registry.P2_REGISTRY` |
| `p3` | `"v2VectorClass"` | Phase 3 (vector parsing) backend — `raster_parser.core.registry.P3_REGISTRY` |

With `workers > 1` the pools start their processes with `spawn`, which re-imports the calling
script: call `extract()` from under `if __name__ == "__main__":`.

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
3. **Phase 3** (`p3`, default `v2VectorClass`) — vector-text OCR on Phase 1's vectors, stroked
   or filled: drawing is removed by geometry per layer / colour / width bucket (oversize,
   dashed lines, repeated patterns, length outliers, crossed grids), the remaining clusters are
   rendered and read with PaddleOCR (detection, rotation from the detected quad, retries at
   90 / 180 / 270), and each read takes the vectors its quad owns; every other vector is kept as
   drawing (`docs/PIPELINE.md`). `p3="v1VectorClass"` is the original fill-glyph OCR: filled,
   non-white, non-box paths grouped into words by content-stream order and read with
   PaddleOCR. Phase 3 never sees Phase 2's output.
4. **Phase 4** (`P4_Output_Organization`) — native words, then Phase 3 text, then Phase 2 text,
   with `block_no` renumbered per phase; Phase 3 vectors, then Phase 2 vectors with seqnos moved
   past Phase 3's (raster traces paint on top). Out-of-page items are logged.

No native-vs-OCR comparison. The original native page is not rasterized for Phase 3.

How the phases and backends fit together, and how to add a backend: `docs/ARCHITECTURE.md`.
How `p3="v2VectorClass"` works step by step: `docs/PIPELINE.md` (terms: `docs/Glossary.md`).

### Guarantees

- **Intermediates** — LO + Type-4 dumps under `dump_dir` (default: `./tmp`).
  `extract` passes that into `clear_page`; `delete_drawings_from_pdf(..., dump_dir=...)`
  uses the same default. Macro exports beside each opened per-page PDF. With
  `debug=False` (default), `extract()` removes its dump root after finishing; set
  `debug=True` to keep them.
- **Lazy heavy imports** — Paddle / PyMuPDF / OpenCV load on first `extract()` call.
- **Progress output** — `extract()` writes a timestamped line as each stage starts and ends,
  plus tqdm bars over its long loops (OCR tiles / batches, buckets, autotrace colours), to
  stderr. It is on by default: the `raster_parser` logger starts at DEBUG
  (`commons/logging_setup.py`). Silence it with
  `logging.getLogger("raster_parser").setLevel(logging.INFO)`; a level set before the import is
  kept. The returned values are the same either way.
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

`p3="v2VectorClass"` scores OCR reads with pyenchant's `en_US` dictionary. The Windows and macOS
wheels bundle Enchant; on Linux install it and the dictionary:

```bash
sudo apt install libenchant-2-2 hunspell-en-us
```

**Notebooks / stats / batch analysis** (not required for `extract()`):

```bash
uv sync --group dev --group notebook --group test
```

## Scripts (offline / QA — not the product API)

| Script | Role |
|--------|------|
| `scripts/inspector/` | PDF layer inspector GUI (`python -m scripts.inspector.inspector [pdf]`) — see its `README.md` |
| `scripts/label/` | Ground-truth labelling tools (native / vector / raster labels, `master_label.py` runs them all) — see its `README.md` |
| `scripts/generate_pipeline_report.py` | Config-driven pipeline report: debug layers, debug images, `dump.json`; benchmark mode adds ground truth — see `docs/SCRIPTS.md` |
| `scripts/pipeline_report_viewer.py` | Tk viewer over 1–2 report folders, a checkbox per layer — see `docs/SCRIPTS.md` |
| `scripts/pipeline_report_benchmark.py` | Scores 1–2 benchmark report folders, writes `report.html` (needs `uv sync --group report`) — see `docs/SCRIPTS.md` |
| `scripts/rasterize_pdf.py` | Flatten a PDF to pure raster: `python scripts/rasterize_pdf.py SRC [DST] --dpi 300` (default DST `outputs/rasterize/<stem>_raster.pdf`) |

## Tests

```bash
uv sync --group test && uv pip install -e .
uv run pytest tests/ -m "not slow"
uv run pytest tests/raster_parser/ -m "not slow"  # library logic (mirrors raster_parser/)
uv run pytest tests/scripts/                      # scripts/ helpers (mirrors scripts/)
uv run pytest tests/raster_parser/test_main_pipeline_extract.py -m slow
RASTER_PARSER_RUN_OCR_TESTS=1 uv run pytest tests/raster_parser/P3_Vector_Parsing/v2VectorClass  # + real PaddleOCR tests
```

`tests/references/test_pdfs_*.pdf` are committed fixture PDFs.
