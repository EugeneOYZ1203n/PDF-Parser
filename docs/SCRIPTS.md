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
(`<stage>__<layer>.pdf` -- each a single element in one colour, so the
viewer toggles it by loading/not-loading the file), `dump.json` (reloadable
`Text`/`Vector`), plus `manifest.json` (its `layers` list drives the
viewer) and `config_and_hyperparameters.txt`.

For `pipeline: "current"` (the default, pluggable `core.pipeline` engine):
the fixed `phase1__*.pdf` / `phase2__*.pdf` / `final__*.pdf` /
`reconstructed__*.pdf` layers, **plus every debug layer the active `p2`/`p3`
backend's own `render_debug` produces** (e.g. `p3: "VectorClassification"`
emits one kept/dropped layer per classification step plus
fast/segment/ocr/drawing; `p3: "FastIntoPaddle"` emits one layer per named
step; `p2: "Junction"` emits its own raster-stage layers) -- see
`CLAUDE.md`'s `P2_Raster_To_Vec/`/`P3_Vector_Parsing/` bullets for what each
backend actually renders. Each `p3` backend also gets its own distinct
pre-OCR debug image folder set, read from `res.extra["p3_debug"]`:
`p3: "FastIntoPaddle"` writes `paddle_detect_images/` + `paddle_recog_images/`
+ `fast_tile_images/` (what PaddleOCR's detector/recognizer and FAST tiles
actually saw); `p3: "VectorClassification"` (no detect stage) writes
`fast_tile_images/` (per-cluster crops of the whole-page FAST render, not a
literal tile grid) + `paddle_recog_images/` (post-dedup representative
segments); `p3: "LegacyRecreation"` (no detect, no FAST stage) writes a
single `paddle_ocr_images/` folder (one padded/DPI-boosted render per word
group). There are no per-stage `.txt` stats for the `current` engine --
`dump.json` is the reloadable source of truth. `pipeline: "legacy"` still
only ever emits the single `reconstructed` row.

```
.venv/Scripts/python.exe scripts/generate_pipeline_report.py \
    --config scripts/report_configs/full_current.json
```

`--config` is the only argument. Config schema (all fields optional except
one of `input_dir` / `input_files`):

| field | meaning |
|---|---|
| `pipeline` | `current` (default, the pluggable `core.pipeline` engine) / `legacy` (archive/raster_parser, unmodified) |
| `p2` | only meaningful when `pipeline: "current"` -- a `core.registry.P2_REGISTRY` name (`Stub` default, or `Junction`) |
| `p3` | only meaningful when `pipeline: "current"` -- a `core.registry.P3_REGISTRY` name (`FastIntoPaddle` default, `VectorClassification`, or `LegacyRecreation`) |
| `enable_fast` | forwarded to `p3` backends that accept it (default `true`) |
| `final_stage` | one of `core.pipeline`'s short step names (`phase1`/`phase2`/`phase3`); `null` = all. Only trims which of the 4 fixed phase-level artifacts render -- doesn't skip any actual pipeline work, and doesn't gate the per-backend debug layers (those always render in full). Ignored entirely for `pipeline: "legacy"`. |
| `input_dir` | folder scanned for `*.pdf` |
| `input_files` | explicit PDF paths (merged with `input_dir`, deduped) |
| `label_files` | `{ "<pdf-stem>": "path/to/labels.json" }` - recorded in the manifest only |
| `pages` | `[0, 2]`, or `{ "<pdf-stem>": [0,1], "*": [0] }`, or `null` = page 0. In `benchmark: true` mode, dict keys are each input's own name (see below), not the literal PDF filename. |
| `vectorise` | run an `Evaluation/conversion.py` pre-step so the pipeline sees text-as-vector-paths |
| `vectorise_mode` | `to_vector_text` (default) / `text_only` / `drawings_only` |
| `benchmark` | benchmark mode (see below) - mutually exclusive with `vectorise` |
| `iou_edge_min` | `MetricConfig.iou_edge_min` for the benchmark overlays (default 0.1) |
| `dpi` | render dpi (default 300) |
| `output_root` | default `outputs/pipeline_report/` |

Output: `outputs/pipeline_report/<ts>__<config-stem>/<pdf-stem>/`. The run's
source config path is also recorded in `config_and_hyperparameters.txt`.
Sample configs live in `scripts/report_configs/` (`full_current`,
`quick_classify`, `directory_with_labels`, `benchmark`).

### Benchmark mode (`benchmark: true`)

A self-contained scoring artifact for `pipeline_report_benchmark.py`.
`input_files` entries may be `.pdf` **or** `.json` (a `LabelSet` sidecar - its
`pdf_path` names the source PDF, the file is the manual-label source); a bare
`.pdf` (or `input_dir` scan) gets auto ground truth only.

Per input, per page: the page is converted **once** with
`convert_page_to_vector_text` (native text as vectors on top of the untouched
drawings) and the pipeline runs **once**. Each input gets its own
`<run_dir>/<input-name>/` folder, named from the input's own unique key
(`pdf:<stem>` / `labels:<stem>` with the prefix stripped) rather than the
literal source PDF filename -- a `scripts/label/master_label.py` folder's
`pdf_path` is always that folder's own `original.pdf`, so keying off the
PDF's filename stem would collide (and overwrite) whenever a config
benchmarks more than one labelled folder. That folder gets the **full**
per-stage report (`manifest.json`, every `<stage>__<layer>.pdf`, `dump.json`,
`converted_p*.pdf`, per-backend debug layers/PNG dirs for `pipeline:
"current"` -- see above) - `pipeline: "legacy"` only emits `reconstructed` -
**plus**:

- `ground_truth_auto.json` (from `native_label_pdf`), `ground_truth_manual.json`
  (the JSON's `source in ("vector", "raster")` entries; absent when there are none)
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

`quick_classify.json` (`final_stage: "phase2"`) is the fast smoke test - it
skips rendering the `final`/`reconstructed` artifacts (OCR still runs as
part of the pipeline itself -- `final_stage` doesn't skip pipeline work, see
above).

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

## `scripts/label/*.py` -- labelling tools

Ground-truth labelling, bundled per PDF into `outputs/labels/<stem>_label/`.
Each tool moved out of the `rastervec` package into a plain script (run
directly, not `python -m`); `label_schema.py` itself stayed in the package
(imported broadly elsewhere) at `rastervec.Evaluation.Labelling.label_schema`.

### `scripts/label/master_label.py`

Runs the whole workflow over one PDF: native step (auto text labels +
`vectorised.pdf`) -> raster geometry step (auto line/curve labels from the
original vectors + `rasterised.pdf`) -> text re-sync (pulls `vector_label`'s
human text labels into `raster_labels.json`) -> opens `vector_label` -> re-sync
-> opens `raster_label` -> prints a summary. Re-running skips already-done
pages for the native/raster-geometry steps (tracked in `manifest.json`); the
interactive steps always reopen.

```
.venv/Scripts/python.exe scripts/label/master_label.py "references/<stem>.pdf" --dpi 300
```

| arg | meaning |
|---|---|
| `pdf` | source PDF (positional, required) |
| `--dpi N` | raster step render resolution (default 300) |

Produces `outputs/labels/<stem>_label/`: `original.pdf`, `vectorised.pdf`,
`rasterised.pdf`, `native_labels.json`, `vector_labels.json`,
`raster_labels.json`, `manifest.json`.

### `scripts/label/native_label.py`

Auto-derives text ground truth from a page's own native text (no human, no
pipeline run) - `native_label_pdf` + a `convert_page_text_only` sidecar +
`attach_vector_signatures`.

```
.venv/Scripts/python.exe scripts/label/native_label.py "references/<stem>.pdf" --page 0
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (positional, required) |
| `--page N` | 0-based page index |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

### `scripts/label/vector_label.py`

GUI: flat pool of raw `Vector`s (no clustering) - click/drag-select
unlabelled vectors + Apply to label; click an already-labelled vector to
re-select its set and edit in place. Bucket checkboxes
(`layer`/`color`/`width`) filter visibility; "Hide labelled" hides labelled
vectors. Continuous rotation slider (2.5 degree snap) with a direction-arrow
overlay. One `LabelSet` spans every page; saved on page change and on close.

```
.venv/Scripts/python.exe scripts/label/vector_label.py "references/<stem>.pdf" --page 0
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (positional, required) |
| `--page N` | 0-based page index |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

### `scripts/label/raster_label.py`

GUI, scoped to content with **no vector backing** (scanned insets, stamps,
photos): lists every embedded raster image in the PDF
(`page.get_image_info`) and steps through them one at a time, cropped, for
hand-drawn text bbox / line / curve annotation. A PDF with no embedded
images has nothing to show.

```
.venv/Scripts/python.exe scripts/label/raster_label.py "references/<stem>.pdf"
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (positional, required) |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

### `scripts/label/view_native_labels.py`

Runs `native_label_pdf`, merges onto any existing `--out` file (by
`label_id`), then opens `vector_label`'s window on it so you can view/adjust
the native labels alongside the real vector pool.

```
.venv/Scripts/python.exe scripts/label/view_native_labels.py "references/<stem>.pdf" --page 0
```

Same args as `native_label.py`; `--out` default is
`outputs/labels/<stem>_p<N>_native_labels.json`.

### `scripts/label/label_viewer.py`

Read-only: overlays a `master_label.py` folder's `native_labels.json` /
`vector_labels.json` / `raster_labels.json` on one page view, one checkbox per
label set (plus the vector set's own backing `Vector`s). No editing, no save.

```
.venv/Scripts/python.exe scripts/label/label_viewer.py "references/<stem>.pdf"
```

Accepts either the source PDF path or the `<stem>_label` folder itself.

## `scripts/generate_test_pdfs.py`

One-off generator for the `tests/references/test_pdfs_*.pdf` fixtures. Run
once, then `git add` the results. Each PDF uses its 1-based index as its
random seed, so regeneration is byte-identical. No arguments.

```
.venv/Scripts/python.exe scripts/generate_test_pdfs.py
```

---

## Module CLIs (`python -m ...`)

### `rastervec.core.pipeline`

Run the pluggable Phase1 -> P2_REGISTRY[p2] -> P3_REGISTRY[p3] pipeline on
one page; prints texts/vectors counts + per-phase wall-clock
(`step_durations`). This is the primary CLI now -- new code should always
use it. The old `rastervec.pipelines.current` module (with its own `python
-m rastervec.pipelines.current` CLI) still exists and is genuinely called by
a few tools (the labelling GUIs, the benchmark CLI's step-name comparison --
see `CLAUDE.md`'s top-of-file note), so it isn't dead code, just deprecated;
don't build new scripts on it.

```
.venv/Scripts/python.exe -m rastervec.core.pipeline \
    --pdf "references/<stem>.pdf" --page 0 --p2 Stub --p3 FastIntoPaddle
```

| arg | meaning |
|---|---|
| `--pdf PATH` | input PDF (required) |
| `--page N` | 0-based page index (default 0) |
| `--p2 NAME` | `core.registry.P2_REGISTRY` name (default `Stub`) |
| `--p3 NAME` | `core.registry.P3_REGISTRY` name (default `FastIntoPaddle`) |
| `--no-fast` | `enable_fast=False`, forwarded to `p3` backends that accept it |
| `-v` / `--verbose` | DEBUG logging + populate `PipelineResult.extra` (phase1/phase2 intermediates + each backend's `debug_out`) |

No `--stop-after` -- always a full Phase1->P2->P3 run.

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

Full Conversion -> native_label -> real pipeline run -> `evaluate_metrics`,
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
| `--variants a,b` | `variants.VARIANTS` names to run/compare (default `current,legacy`) -- also `current_vectorclassification`, `current_legacyrecreation`, `current_junction` (named P2/P3 combo presets for benchmark comparisons) |

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

See `scripts/label/*.py` above for the labelling tools - they moved out of
`rastervec` into plain scripts, so they're no longer `python -m` module CLIs.

---

## Test suite

```
.venv/Scripts/python.exe -m pytest tests/ -q
```

OCR-dependent tests are gated behind `RASTERVEC_RUN_OCR_TESTS` and skipped by
default.
