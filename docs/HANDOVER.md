# Handover

What a new owner needs beyond the code: how the work is split into PRs, which tests guard what, what is
unfinished, and the limits you will hit.

## The PR stack

The port is a stack of branches, `final-prN-<topic>`, each built on the one before and each adding one
feature with its tests. Reviewing in order reads best.

| PR | Branch suffix | Adds |
|---|---|---|
| 2 | `ruff-format` | the starting code, formatted (`ruff`, line length 100), its unexplained numbers named and every `fitz` name changed to `pymupdf` |
| 3 | `native-read` | lazy LibreOffice macro registration, the shared geometry helpers, the shared models (`Text`, `Vector` -- pydantic models with a `Source` -- `Image`, `Page`), `phase_native_read` |
| 4 | `vector-classify` | `v1_vector_class` and the v1 OCR / render helpers |
| 5 | `raster-to-vec` | `v1_raster_to_vec` |
| 6 | `phase-pipeline` | `core/` (pipeline, registry, backend contract), `extract()` on the phases, golden snapshots, `ARCHITECTURE.md`, `model_migration.md` |
| 7 | `debug-layers` | the debug hooks, `stop_after`, the spec -> draw renderer, the v1 backends' debug layers |
| 8 | `report-tools` | label schema, native-text labels, input conversion, the report generator, viewer and benchmark |
| 9 | `progress` | stage lines and tqdm bars |
| 10 | `v2-vector-class` | `v2_vector_class`, now the default `p3` |
| 11 | `labelling` | the vector-text label tool: drafts from `v2_vector_class`, a label list, background page processing, multi-select delete |
| 12 | `raster-benchmark` | the Liu & Dori raster benchmark |
| 13 | `junction` | the shared raster prelude (`commons/raster_text`) and the `junction` backend |
| 14 | `involution-stroke` | `involution_stroke` and its vendored upstream code; the `ml` extra |
| 15 | `deep-p2` | `deep_tech_vec`, `deep_vectoriser`, `implicit_sketch_vec`, the stroke code they share with `involution_stroke`, and `visible_geometry` (ground truth cut to what the page shows) |
| 16 | `notebooks` | the notebooks and their tested helpers |
| 17 | `docs-handover` | this documentation set |

## Guardrails: run these before and after any change

| Test | Guards |
|---|---|
| `tests/raster_parser/test_golden_outputs.py` | `extract()` output on `tests/references/test_pdfs_1..5.pdf` against committed snapshots (`scripts/golden_snapshot.py` writes and compares them) |
| `tests/scripts/test_report_snapshot.py` | the report and benchmark HTML / JSON outputs, with times and paths normalised |
| `tests/raster_parser/phase_raster_vectorize/junction/test_source_parity.py` | Junction output equals what the original implementation produced (slow) |
| `tests/raster_parser/phase_raster_vectorize/involution_stroke/test_source_parity.py` | the same for InvolutionStroke; **opt-in** (`RUN_VERY_SLOW=1`, the downloaded weights; minutes on CPU) |
| `tests/raster_parser/test_module_layout.py` | `commons` never imports a `phase_*`; backends never import each other; every dataclass and enum lives in `raster_parser/models/` |
| `tests/test_architecture_doc.py` | every shared module appears in the shared-code map in `ARCHITECTURE.md` |
| `tests/test_no_duplicate_code.py` | the same function body does not live in two modules |
| `tests/test_docs_links.py` | every relative link in the docs resolves and every doc is reachable from the README |
| `tests/notebooks/test_notebooks.py` | the notebooks carry no outputs and their imports resolve (`RUN_NOTEBOOKS=1` also runs them) |

`uv run pytest -m "not slow"` is the quick suite. Keep comparing time and peak memory before and after
anything that touches the page loops (`/usr/bin/time -v` gives the peak).

## Unfinished or undecided

- **Native word order.** `phase_native_read` sorts native words into reading order (top to bottom, then
  left to right); the code this was ported from kept PyMuPDF's `get_text("words")` order. Everything
  else in `extract()`'s output matches the original on the test PDFs. Decide whether to keep the sort
  (and list it in `model_migration.md`) or drop it.
- **Weights for the learned backends.** `involution_stroke` and `deep_tech_vec` download the authors'
  checkpoints (`download_weights.py`). `deep_vectoriser` and `implicit_sketch_vec` have no released
  weights: train them first (their `TRAINING.md`).
- **Large modules.** `v2_vector_class/parse.py`, `v1_raster_to_vec/s5_colour_autotrace.py` and the
  inspector are long. They were left whole on purpose: each split is its own change and risks the golden
  outputs.
- **Notebooks dropped.** Not ported: `cad_font_matching_lab`, `radon_vector_deskew_lab`,
  `skew_method_comparison`, `vector_similarity_lab`, `golden_case_curation`. They depended on modules
  that are not part of this project. `pdf_dataset_stats` was replaced by `pdf_dataset_timing`.
- **Labels from before the rework.** Old label files still load; raster and CAD-font entries are dropped
  on load. The old master-label and raster-label tools are gone on purpose: the benchmark derives
  native-text and raster ground truth itself.

## Known limits

- **Memory on very large pages.** Classifying the largest page in `references/` (about 120k vectors)
  peaks near 7 GB. `notebooks/pdf_dataset_timing.ipynb` has a `MAX_VECTORS` cap for this and runs the
  report in a subprocess so one page cannot take the notebook down. The learned backends' tests and
  training smoke runs also need a few GB.
- **`involution_stroke` is slow on CPU** (tens of seconds per 1024 px tile) and has **no upstream
  licence**: internal research use only; ask the authors before redistributing the code or weights. See
  its [README](../raster_parser/phase_raster_vectorize/involution_stroke/README.md).
- **`deep_tech_vec`'s network is MPL-2.0** (derived from the authors' code); see its
  [README](../raster_parser/phase_raster_vectorize/deep_tech_vec/README.md).
- **PaddleOCR 2.x.** The project pins `paddleocr>=2.9,<3`. Torch must be imported before paddle in any
  process that uses both, and torch should be the CPU build.
- **LibreOffice** is needed by `v1_raster_to_vec` only; its macro registers itself on first use.

## How the code is kept shared

[ARCHITECTURE.md](ARCHITECTURE.md#shared-code-map) holds the rule (anything two backends need goes in
`commons/`; v1-only code goes in `commons/v1_commons/`) and a table of every shared module with who uses
it. Add a row when you add a shared module; the test fails if you forget.

## Pictures

[images/EXAMPLES.md](images/EXAMPLES.md) lists ten images to generate for the README and the docs, each
with the command that makes it.
