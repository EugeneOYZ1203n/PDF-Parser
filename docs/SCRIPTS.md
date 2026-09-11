# Scripts & CLI entrypoints

Every command is run from the **repo root** with the venv's Python
(`.venv/Scripts/python.exe`). Paths in configs/args are resolved relative to
the repo root.

## Windows / PowerShell path gotcha

Do **not** end a quoted path argument with a backslash:

```powershell
# BROKEN - the \" is read as an escaped quote, a literal " ends up in the path
python scripts/pipeline_report_viewer.py '.\outputs\...\Thomson Height CC\'
#   -> OSError [Errno 22] Invalid argument: '...Height CC"\manifest.json'

# OK - no trailing separator
python scripts/pipeline_report_viewer.py '.\outputs\...\Thomson Height CC'
# OK - forward slashes
python scripts/pipeline_report_viewer.py 'outputs/.../Thomson Height CC'
```

`pipeline_report_viewer.py` now strips a stray trailing `"` / `\` / `/`
itself, but the other scripts do not - keep trailing separators off quoted
paths.

---

## `scripts/generate_pipeline_report.py`

Config-driven per-stage report. Runs the pipeline once per `(pdf, page)` and
writes a timestamped run folder with **one multi-page PDF per visual layer**
(`<stage>__<layer>.pdf`, e.g. `vector_classification__group_bbox.pdf` --
each a single element in one colour, so the viewer toggles it by
loading/not-loading the file), per-stage `.txt` stats, `dump.json`
(reloadable `Text`/`Vector`), `radon_images/` / `paddle_images/` (see
below), plus `manifest.json` (its `layers` list drives the viewer) and
`config_and_hyperparameters.txt`.

`radon_images/` and `paddle_images/` are *different inputs*, not the same
crop twice:

| folder | what | when |
|---|---|---|
| `radon_images/p<pg>_cluster_<i>.png` | the rendered cluster image Radon segments, one per FAST-surviving cluster, with the detected word/segment boxes drawn on it (red) | **before** Radon deskew + word split |
| `paddle_images/p<pg>_uniq_<i>__<text>.png` | the exact deskewed, white-padded word crop handed to PaddleOCR (one per elected unique segment), unpadded word region boxed (blue), recognised text in the filename | **before** PaddleOCR recognition |

```
.venv/Scripts/python.exe scripts/generate_pipeline_report.py \
    --config scripts/report_configs/full_current.json
```

`--config` is the only argument. Config schema (all fields optional except
one of `input_dir` / `input_files`):

| field | meaning |
|---|---|
| `pipeline` | `current` (default) / `current_nofast` / `legacy` |
| `final_stage` | a step name (`read native vectors classify fast segment similarity ocr restore drawing`); `null` = all. Later steps skipped, their layer PDFs not emitted. |
| `input_dir` | folder scanned for `*.pdf` |
| `input_files` | explicit PDF paths (merged with `input_dir`, deduped) |
| `label_files` | `{ "<pdf-stem>": "path/to/labels.json" }` - recorded in the manifest only |
| `pages` | `[0, 2]`, or `{ "<pdf-stem>": [0,1], "*": [0] }`, or `null` = page 0 |
| `vectorise` | run an `Evaluation/conversion.py` pre-step so the pipeline sees text-as-vector-paths |
| `vectorise_mode` | `to_vector_text` (default) / `text_only` / `drawings_only` |
| `benchmark` | benchmark mode (see below) - mutually exclusive with `vectorise` |
| `iou_edge_min` | `MetricConfig.iou_edge_min` for the benchmark overlays (default 0.1) |
| `dpi` | render dpi (default 300) |
| `output_root` | default `outputs/pipeline_report/` |

Output: `outputs/pipeline_report/<ts>__<config-stem>/<pdf-stem>/`.
Sample configs live in `scripts/report_configs/` (`full_current`,
`quick_classify`, `nofast_vectorised`, `directory_with_labels`, `benchmark`).

### Benchmark mode (`benchmark: true`)

A self-contained scoring artifact for `pipeline_report_benchmark.py`.
`input_files` entries may be `.pdf` **or** `.json` (a `LabelSet` sidecar - its
`pdf_path` names the source PDF, the file is the manual-label source); a bare
`.pdf` (or `input_dir` scan) gets auto ground truth only.

Per input, per page: the page is converted **once** with
`convert_page_to_vector_text` (native text as vectors on top of the untouched
drawings) and the pipeline runs **once**. `<pdf-stem>/` then gets the **full**
per-stage report (`manifest.json`, every `<stage>__<layer>.pdf`, `<stage>.txt`,
`radon_images/`, `paddle_images/`, `dump.json`, `converted_p*.pdf`) - legacy
engine only emits `reconstructed` - **plus**:

- `ground_truth_auto.json` (from `auto_label_pdf`), `ground_truth_manual.json`
  (the JSON's `source="manual"` entries; absent when there are none)
- `auto_bbox.pdf` / `manual_bbox.pdf` - GT boxes, green = covered by a
  prediction, red = missed
- `auto_text.pdf` / `manual_text.pdf` - GT text, per word green = exact /
  yellow = char edit distance 1-2 / red = worse or unread

The overlays are added to `manifest.json`'s `layers` under stage `benchmark`,
so the viewer toggles them like any other layer. The run root gets a
`benchmark.json` marker listing every input's key (`pdf:<stem>` /
`labels:<json-stem>`) and sources.

Works with `pipeline: legacy` too (runs `archive/`'s pipeline on each converted
page - needs LibreOffice on PATH). Run one `benchmark` config with
`pipeline: current` and one with `pipeline: legacy` (samples:
`report_configs/benchmark.json` / `benchmark_legacy.json`), then diff the two
folders with `pipeline_report_benchmark.py --run ... --run ...`.

`quick_classify.json` (`final_stage: "classify"`) is the fast smoke test - it
never touches OCR.

## `scripts/pipeline_report_viewer.py`

Tkinter viewer for **1 or 2** per-PDF report folders. Left pane = source page;
right side = one **panel per folder**, side by side. Each panel is one checkbox
per layer PDF, grouped by stage (including the `benchmark` group). A checked
layer is rasterized with a real alpha channel (`get_pixmap(alpha=True)`) and
alpha-composited over the base, in panel order - so you can show e.g. the OCR
output of pipeline A together with the clusters of pipeline B.

- **source page** checkbox: toggle the underlying page (off = layers composite
  over white).
- each stage header has `all` / `none` buttons.

```
.venv/Scripts/python.exe scripts/pipeline_report_viewer.py \
    "outputs/pipeline_report/<ts1>__benchmark/240118-..." \
    "outputs/pipeline_report/<ts2>__benchmark_legacy/240118-..."
```

Positional: 1 or 2 `<pdf-stem>` folders (each must contain `manifest.json`).
`pipeline_report_benchmark.py` writes ready-to-paste lines to
`viewer_commands.txt`.

## `scripts/pipeline_report_benchmark.py`

Compares **1 or 2 benchmark report folders** (`generate_pipeline_report.py`
runs with `benchmark: true`). No pipeline run - each folder embeds its single
`<pdf-stem>/dump.json` + `ground_truth_auto.json` (+ `ground_truth_manual.json`).
Scored with **`metrics.evaluate_multiclass`** (see `docs/EVAL_METRICS.md` §5):
one prediction set vs both label classes, so a prediction the *other* class
matched is not a false positive here. Folders are matched on the inputs they
**share, by input identity**: `pdf:<stem>` (auto only) vs `labels:<stem>`
(auto + manual) - a `B.pdf` run and a `B.json` run do not match on B.

```
.venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
    --run outputs/pipeline_report/<ts1>__benchmark \
    --run outputs/pipeline_report/<ts2>__benchmark_legacy
```

| arg | meaning |
|---|---|
| `--run DIR` | a benchmark run folder (repeatable; **1 or 2**) |
| `--iou-threshold F` | `MetricConfig.iou_edge_min` (default 0.1) |
| `--out-root DIR` | parent of the timestamped output folder (default `outputs/pipeline_report_benchmark/`) |

Writes `outputs/pipeline_report_benchmark/<ts>/`:

- `benchmark.txt` - per shared key: AUTO + MANUAL metric tables (one column per
  run), the `{auto,manual}x{auto,manual,none}` GT-recall confusion matrix, and
  combined candidate precision.
- `runs.json` - compared folders + threshold + shared keys.
- `viewer_commands.txt` - a `pipeline_report_viewer.py` line per shared input.
- `charts/` - `<key>__p<N>__{auto,manual,confusion}.png` per page,
  `<key>__aggregate__*.png`, and grand `aggregate__*.png`.

## `scripts/rasterize_pdf.py`

Flattens a PDF to a pure-raster PDF (every page rendered to an image, new PDF
rebuilt from the images - no vector/text survives). One-off input-prep
utility, not a pipeline stage.

```
.venv/Scripts/python.exe scripts/rasterize_pdf.py \
    "references/<stem>.pdf" --dpi 300
```

| arg | meaning |
|---|---|
| `src` | source PDF (positional, required) |
| `dst` | output path (positional, optional; default `outputs/rasterize/<src-stem>_raster.pdf`) |
| `--dpi N` | render resolution (default 300) |

## `scripts/generate_test_pdfs.py`

One-off generator for the `tests/references/test_pdfs_*.pdf` fixtures. Run
once, then `git add` the results. Each PDF uses its 1-based index as its
random seed, so regeneration is byte-identical. No arguments.

```
.venv/Scripts/python.exe scripts/generate_test_pdfs.py
```

---

## Module CLIs (`python -m ...`)

### `rastervec.pipelines.current`

Run the current 9-step pipeline on one page; logs per-step wall-clock.

```
.venv/Scripts/python.exe -m rastervec.pipelines.current \
    --pdf "references/<stem>.pdf" --page 0
```

| arg | meaning |
|---|---|
| `--pdf PATH` | input PDF (required) |
| `--page N` | 0-based page index (default 0) |
| `--no-fast` | FAST detection becomes a pass-through (speed testing) |
| `--stop-after STEP` | skip every step after `STEP` |
| `-v` / `-q` | DEBUG + keep intermediates / WARNING-only |

### `rastervec.pipelines.legacy`

Same CLI shape, runs `archive/`'s unmodified pipeline via
`legacy_adapter`. Needs `archive/` importable and **LibreOffice on PATH**
(`import raster_parser` spawns a LibreOffice subprocess). Only the
OCR-scored fields of the result are meaningful.

```
.venv/Scripts/python.exe -m rastervec.pipelines.legacy \
    --pdf "references/<stem>.pdf" --page 0
```

### `rastervec.Evaluation.Evaluate.benchmark`

Full Conversion -> auto_label -> real pipeline run -> `evaluate_metrics`,
per variant, with an aggregate + timing comparison.

```
.venv/Scripts/python.exe -m rastervec.Evaluation.Evaluate.benchmark \
    --pdf "references/<stem>.pdf" --pages 0,1,2 --variants current
```

| arg | meaning |
|---|---|
| `--pdf PATH` | a PDF (repeatable) |
| `--pages a,b,c` | 0-based page indices (default `0`) |
| `--iou-threshold F` | `MetricConfig.iou_edge_min` (default 0.1) |
| `--reconstruct-dir DIR` | per-page reconstruction / input / box-overlay PDFs (default `outputs/benchmark_cli/reconstructions/`) |
| `--workers N` | run pages across a spawn pool of size `N` (>1); default 1 serial |
| `--compute-workers N` | run FAST tiles + OCR crops on a shared pool of size `N` (>0); default 0 local |
| `--variants a,b` | `variants.VARIANTS` names to run/compare (default `current,legacy`) |

`--variants current,legacy` needs LibreOffice (legacy). `main()`'s real
OCR path is a manual smoke test; the pure formatting/aggregation helpers are
unit-tested.

### `rastervec.Evaluation.inspector.inspector`

Standalone Tkinter + PyMuPDF PDF layer inspector (text / images /
annotations / drawings as toggleable overlays). Predates the pipeline,
shares no imports with it.

```
.venv/Scripts/python.exe -m rastervec.Evaluation.inspector.inspector \
    "references/<stem>.pdf"
```

Optional positional `pdf`; with none it opens against the repo-root
`references/` folder.

### `rastervec.Evaluation.Labelling.manual_label`

GUI cluster-label editor - runs the real pipeline per page, lets a human
click clusters / paths, group / ungroup, and enter text + rotation. One
`LabelSet` spans every page; labels are saved on every page change and on
close.

```
.venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.manual_label \
    "references/<stem>.pdf" --page 0 --out outputs/labels/<stem>.json
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (positional, required) |
| `--page N` | 0-based page index |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

### `rastervec.Evaluation.Labelling.view_auto_labels`

Runs `auto_label_pdf`, merges onto any existing `--out` file, opens the same
editor window so you can view/adjust the auto labels.

```
.venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.view_auto_labels \
    "references/<stem>.pdf" --page 0
```

Same args as `manual_label`; `--out` default is
`outputs/labels/<stem>_p<N>_auto_labels.json`.

---

## Test suite

```
.venv/Scripts/python.exe -m pytest tests/ -q
```

OCR-dependent tests are gated behind `RASTERVEC_RUN_OCR_TESTS` and skipped by
default.
