# Report, viewer and benchmark scripts

Three dev tools for looking at what the pipeline does and measuring how well it does it. They
are not part of `extract()`; they drive `core.pipeline.run_pipeline` directly with the debug
hooks switched on (see [Debug output](ARCHITECTURE.md#debug-output)).

| Script | What it does |
|---|---|
| `scripts/generate_pipeline_report.py` | Runs the pipeline on PDFs from a JSON config and writes one folder per input: a PDF per debug layer, debug images, `dump.json`, `manifest.json`. In benchmark mode, also the ground truth to score against. |
| `scripts/pipeline_report_viewer.py` | Tk viewer: 1 or 2 report folders side by side over the source page, a checkbox per layer. |
| `scripts/pipeline_report_benchmark.py` | Scores 1 or 2 benchmark report folders against their ground truth and writes `report.html` with tables and charts. |

The benchmark needs matplotlib: `uv sync --group report`. The viewer needs Tk (`python3-tk` on
Ubuntu).

The batch scripts (the report, the benchmark and `rasterize_pdf.py`) show on stderr which stage
is running and when, with tqdm bars over inputs, pages and the pipeline's long loops ([Progress
output](ARCHITECTURE.md#progress-output)).

## Typical workflow

1. **Label** (optional). `scripts/label/manual_vector_text_label.py PDF` writes
   `outputs/labels/<stem>.json` with hand-labelled vector text (format in
   `raster_parser/evaluation/labelling/label_schema.py`), starting each page from
   `v2_vector_class`'s drafts (`scripts/label/README.md`). Native-text labels are never made by hand: the benchmark derives them from the PDF, so a plain
   PDF works too.
2. **Report.** Write a config (`scripts/report_configs/`, field table in its `README.md`) and
   run

   ```
   python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json
   ```

   For scoring, set `"benchmark": true` and list PDFs or label JSON files in `input_files`. Run
   it once per backend combination you want to compare, e.g. `p3: "v1_vector_class"` and
   `p3: "v2_vector_class"` (`bench_v1.json` / `bench_v2.json`, worked through in the configs'
   `README.md`).
3. **Look.** Open any input folder of a run (two runs side by side for a comparison):

   ```
   python scripts/pipeline_report_viewer.py outputs/pipeline_report/<ts>__a/<doc> [outputs/pipeline_report/<ts>__b/<doc>]
   ```

4. **Score.**

   ```
   python scripts/pipeline_report_benchmark.py --run outputs/pipeline_report/<ts>__a [--run outputs/pipeline_report/<ts>__b]
   ```

   and open `outputs/pipeline_report_benchmark/<ts>/report.html`.

## Report folder

`outputs/pipeline_report/<ts>__<config-stem>/`:

| Path | Contents |
|---|---|
| `config_and_hyperparameters.txt` | the config, its path, and the `RunSettings` the backends got |
| `p2_work/` | raster_vectorize backends' own intermediate files (`RunSettings.dump_dir`) |
| `benchmark.json` | benchmark mode: the backends, the threshold and one entry per input (key, source PDF, labels, pages, ground-truth types) |
| `<doc>/manifest.json` | source PDF, pages, backends and the layer list; the viewer reads it |
| `<doc>/dump.json` | every page's final texts and vectors (the `ExtractResult` layout) plus timings: phase seconds, a backend's own sub-step seconds (`sub_durations`, from `RunSettings.on_step_time`) and report-generation seconds; the benchmark reads it |
| `<doc>/<stage>__<layer>.pdf` | one multi-page PDF per debug layer a backend emitted, page-aligned with `manifest.json`'s `pages` (blank where the layer had nothing); `reconstructed__page.pdf` is the final output as a page |
| `<doc>/<folder>/` | debug images a backend offered, at most `debug_image_cap` per folder, sampled uniformly across pages |
| `<doc>/converted_p<N>.pdf` | the converted input page (vectorise / benchmark modes) |
| `<doc>/ground_truth_<type>.json` | benchmark mode: the labels scored, per text type |
| `<doc>/<type>_bbox.pdf`, `<type>_text.pdf` | benchmark mode: label boxes green (reached) / red (missed); label text per word green / yellow / red by how well it was read |
| `<doc>/benchmark__extra_*.pdf`, `benchmark__missed_vectors.pdf` | benchmark mode, inputs with vector labels: OCR text outside every label, text-classified vectors outside every label, and drawing vectors inside a vector label |

`<doc>` is the PDF stem, or in benchmark mode the input key without its `pdf:` / `labels:`
prefix.

## What the benchmark scores

In benchmark mode each page is converted with `convert_page_to_vector_text`: the native text is
drawn as vector paths over the untouched drawings, so the text the pipeline has to find is known
exactly. The run's OCR text is scored against two kinds of ground truth:

| Type | Labels | From |
|---|---|---|
| `native_to_vector` | the page's native text, one label per line | derived from the PDF (`native_label.native_label_pdf`) |
| `original_vector` | hand-labelled vector text (text drawn as paths in the original PDF) | a label JSON (`scripts/label/manual_vector_text_label.py`) |

All metrics are reductions over one overlap graph between labels and OCR readings
(`raster_parser/evaluation/evaluate/metrics*.py`), micro-averaged across pages and inputs (sum
the numerators and denominators, then divide).

`report.html` sections, in order:

1. **Inputs**: each run folder (backends, final stage, config) and each file scored (source PDF,
   labels, pages, ground-truth types).
2. **Summary (all inputs)**: per text type, labels, char F1 / recall, word F1, bbox mean IoU and
   rotation-correct rate; candidate precision (the share of OCR readings that overlap any label);
   median pipeline seconds per page with and without report generation. One column per run.
3. Then the full tables for all inputs and for each input: labels, char and word overlap, bbox
   accuracy (plus spurious readings with no label), rotation, per-character accuracy (each
   label character detected / dropped / unreached / misclassified, with the top misreadings),
   extra predicted characters, and per-page timing with a median delta between runs (a
   backend's sub-steps as `vector_classify › <name>` rows under their phase, not added to `total`). Each
   input ends with the viewer command for it.

`benchmark.txt` has the same numbers as plain text, `viewer_commands.txt` one viewer command per
input, and `charts/` the PNGs the report links.
