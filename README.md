# raster-parser

Text and vector extraction from engineering PDF drawings. One call returns, for a page, every piece of
text (native words, OCR'd glyph outlines, OCR'd raster text) and every vector (drawing paths, traced raster
lines) as the same two models, `Text` and `Vector`.

```python
from raster_parser import extract

result = extract("drawing.pdf", page_index=0)
result.texts  # list[Text]   -- words with bbox, direction and source
result.vectors  # list[Vector] -- non-glyph drawing paths, plus traced raster lines
result.to_json()
```

`Text` and `Vector` are pydantic models. Each carries a `source` (`Source.NATIVE_TEXT`,
`NATIVE_VECTOR` for drawing paths and text read from drawn glyphs, or `FROM_IMAGE` for anything traced
or read from an embedded image). The full API, with the options and what each returns, is in
[raster_parser/INTEGRATION.md](raster_parser/INTEGRATION.md).

## How it works

Four phases, two of them a swappable backend chosen by name:

```
PDF page
  1. native_read         always the same: native words, drawing paths, embedded images
  2. raster_vectorize    turns the page's raster images into Text + Vector      (p2=...)
  3. vector_classify     decides which vectors are glyph outlines and reads them  (p3=...)
  4. output_assemble     merges everything into one ExtractResult
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the layout, the backend contract and the map of shared
code. [docs/PIPELINE.md](docs/PIPELINE.md) walks one page through `v2_vector_class`.

### Backends

| Phase | `p2=` / `p3=` | What it is | Needs |
|---|---|---|---|
| raster_vectorize | `stub` | does nothing (use it to time or test classification alone) | none |
| raster_vectorize | `v1_raster_to_vec` | LibreOffice clear, tiled Paddle OCR, colour-fill autotrace (the default) | LibreOffice |
| raster_vectorize | `junction` | classical centreline tracing on colour layers | none |
| raster_vectorize | `involution_stroke` | learned strokes (the authors' pretrained weights); slow on CPU | `ml` extra |
| raster_vectorize | `deep_tech_vec` | learned line / curve primitives (the authors' pretrained weights) | `ml` extra |
| raster_vectorize | `deep_vectoriser` | learned strokes, implemented from the paper | `ml` extra + weights you train |
| raster_vectorize | `implicit_sketch_vec` | learned strokes via implicit surfaces, implemented from the paper | `ml` extra + weights you train |
| vector_classify | `stub` | passes vectors through | none |
| vector_classify | `v1_vector_class` | fill filter, word grouping, glyph render + OCR | none |
| vector_classify | `v2_vector_class` | clustering, filters, pattern lattice, OCR (the default) | none |

Per-backend notes: [v1_raster_to_vec](raster_parser/phase_raster_vectorize/v1_raster_to_vec/README.md),
[junction](raster_parser/phase_raster_vectorize/junction/README.md),
[involution_stroke](raster_parser/phase_raster_vectorize/involution_stroke/README.md),
[deep_tech_vec](raster_parser/phase_raster_vectorize/deep_tech_vec/README.md),
[deep_vectoriser training](raster_parser/phase_raster_vectorize/deep_vectoriser/TRAINING.md),
[implicit_sketch_vec training](raster_parser/phase_raster_vectorize/implicit_sketch_vec/TRAINING.md),
[v1_vector_class](raster_parser/phase_vector_classify/v1_vector_class/README.md),
[v2_vector_class](raster_parser/phase_vector_classify/v2_vector_class/README.md). Where the learned
backends differ from their papers: [docs/p2_paper_deviations.md](docs/p2_paper_deviations.md).

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # core
uv sync --extra ml           # + torch, for the learned raster backends
uv sync --group report       # + matplotlib, for the benchmark report
uv sync --group notebook     # + notebook tooling
```

Linux also needs `libenchant-2-2` and `hunspell-en-us` (English-word scoring); the viewer and the label
tool need Tk (`python3-tk`); `v1_raster_to_vec` needs LibreOffice. On Windows, run the Tk tools in the
project's venv; WSL works for everything else.

```bash
uv run pytest -m "not slow"  # fast suite
uv run pytest                # also the slow end-to-end tests
```

## Working with it

| I want to... | Do this | Read |
|---|---|---|
| look inside a PDF page | `python -m scripts.inspector.inspector file.pdf` | [scripts/inspector](scripts/inspector/README.md) |
| hand-label the original vector text | `python scripts/label/manual_vector_text_label.py file.pdf` | [scripts/label](scripts/label/README.md) |
| run the pipeline on PDFs and see every layer | `python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json` | [docs/SCRIPTS.md](docs/SCRIPTS.md) |
| score text and vectors against ground truth | `python scripts/pipeline_report_benchmark.py --run <report folder>` | [docs/SCRIPTS.md](docs/SCRIPTS.md), [docs/RasterToVectorEvaluation.md](docs/RasterToVectorEvaluation.md) |
| compare two backends | two report configs, one benchmark call with two `--run`s | [scripts/report_configs](scripts/report_configs/README.md) |
| explore classification filters, ground truth or timing | the notebooks | [notebooks](notebooks/README.md) |

## Where to find what

Everything is indexed in [docs/README.md](docs/README.md). Taking the project over:
[docs/HANDOVER.md](docs/HANDOVER.md). Pictures worth adding to this page are listed in
[docs/images/EXAMPLES.md](docs/images/EXAMPLES.md).
