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
| `dpi` | render dpi (default 300) |
| `output_root` | default `outputs/pipeline_report/` |

Output: `outputs/pipeline_report/<ts>__<config-stem>/<pdf-stem>/`.
Sample configs live in `scripts/report_configs/` (`full_current`,
`quick_classify`, `nofast_vectorised`, `directory_with_labels`).

`quick_classify.json` (`final_stage: "classify"`) is the fast smoke test - it
never touches OCR.

## `scripts/pipeline_report_viewer.py`

Tkinter viewer for one per-PDF report folder. Left pane = source page; right
sidebar = one checkbox per **layer PDF**, grouped by stage. A checked layer
is rasterized with a real alpha channel (`get_pixmap(alpha=True)` -- a stage
PDF page has no background, so it composites cleanly with no white-key halo
and no anti-alias fringe) and alpha-composited over the base.

- **source page** checkbox: toggle the underlying page on/off (off = layers
  composite over white).
- each stage header has `all` / `none` buttons to toggle its whole group.

```
.venv/Scripts/python.exe scripts/pipeline_report_viewer.py \
    "outputs/pipeline_report/20260910_232639__full_current/240118-Proposed OW Shopdrawings for Thomson Height CC"
```

One positional arg: a `<pdf-stem>` folder produced by
`generate_pipeline_report.py` (must contain `manifest.json`).

## `scripts/pipeline_report_benchmark.py`

Scores the OCR text inside one or more `dump.json` files against ground
truth. No pipeline run - reads the dumps only. Miss-attribution metrics are
absent (a dump has no classification state).

```
# auto ground truth from the source PDF's native text
.venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
    --dump outputs/pipeline_report/<ts>__full_current/<stem>/dump.json \
    --pdf "references/<stem>.pdf"

# ground truth from a label JSON (auto + manual scored separately)
.venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
    --dump run_a/<stem>/dump.json --dump run_b/<stem>/dump.json \
    --labels outputs/labels/<stem>.json
```

| arg | meaning |
|---|---|
| `--dump PATH` | a `dump.json` (repeatable) |
| `--pdf PATH` | target PDF for auto ground truth (mutually exclusive with `--labels`) |
| `--labels PATH` | label JSON, auto + manual scored separately |
| `--iou-threshold F` | `MetricConfig.iou_edge_min` (default 0.1) |
| `--out PATH` | default `outputs/pipeline_report/benchmark.txt` |

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
