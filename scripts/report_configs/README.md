# Report configs

JSON configs for `scripts/generate_pipeline_report.py` (`ReportConfig`,
`raster_parser/evaluation/report/report_config.py`). Every field is optional. Relative paths are
resolved from the directory the script is run in.

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json
```

| Field | Default | Meaning |
|---|---|---|
| `p2`, `p3` | `v1RasterToVec`, `v2VectorClass` | Backends by registry name (`core/registry.py`). An unknown name is rejected. |
| `final_stage` | `null` | Stop after `phase1`, `phase2` or `phase3` (`run_pipeline(stop_after=...)`). Later backends don't run and emit no layers. Phase 4 still runs. |
| `input_dir` | `null` | Every `*.pdf` in this folder. |
| `input_files` | `[]` | PDFs. In benchmark mode also label JSON files and `scripts/label/master_label.py` folders. |
| `label_files` | `{}` | Benchmark mode: `{pdf-stem: label JSON}`, to give a bare PDF hand-made labels. |
| `pages` | `null` (= `[0]`) | Page indices: one list for every input, or `{"<stem>": [...], "*": [...]}`. Out-of-range pages are skipped with a warning. |
| `vectorise` | `false` | Run on a converted copy of each page instead of the original. |
| `vectorise_mode` | `to_vector_text` | `text_only`, `drawings_only` or `to_vector_text` (`evaluation/conversion.py`). |
| `benchmark` | `false` | Write score-ready output for `pipeline_report_benchmark.py` (ground truth + overlays + `benchmark.json`). Runs on `to_vector_text` copies; can't be combined with `vectorise`. |
| `iou_edge_min` | `0.1` | Benchmark overlays: the IoU a prediction needs to count as reaching a label. |
| `output_root` | `null` | Run folder root; default `outputs/pipeline_report/`. |
| `debug_images` | `true` | Save the debug images the backends offer. |
| `debug_image_cap` | `100` | At most this many images per folder per document (uniform random across pages, same picks on a rerun). `null` = all. |
| `debug_layers` | `true` | Write the layer PDFs. Off: `dump.json`, ground truth and debug images are still written, which is all the benchmark needs. |
| `keep_converted_pdfs` | `true` | Keep each page's converted input (`converted_p<N>.pdf`). |
| `ocr_dpi`, `use_angle_cls`, `workers` | `300`, `true`, `1` | Passed to the backends in `RunSettings`. `workers > 1` runs v1RasterToVec's and v2VectorClass's OCR on a process pool (same output, faster). v2VectorClass doesn't use `ocr_dpi` / `use_angle_cls`. |

## Inputs

Keep inputs under `outputs/labels/` (git-ignored; `scripts/label/master_label.py` already
writes there). Three kinds can go in `input_files`:

| Input | Example | Scored against (benchmark mode) |
|---|---|---|
| a `master_label.py` folder | `outputs/labels/example_label` | its `native_labels.json` and hand-made `vector_labels.json` |
| a label JSON | `outputs/labels/example.json` | its labels; the PDF is the JSON's `pdf_path` |
| a bare PDF | `outputs/labels/plain.pdf` | labels derived from its own native text; with a `label_files` entry, that JSON's labels too |

The benchmark matches two runs on the inputs they share, by key (`labels:<folder or JSON
name>`, or `pdf:<stem>` for a bare PDF without `label_files`), so run both configs on the same
`input_files`.

## Worked examples

**Look at one backend's steps.** `example.json` (v1RasterToVec + v1VectorClass) and `example_v2.json`
(`p3: "v2VectorClass"`, no raster pass) report one page each; open a result in the viewer:

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/example_v2.json
python scripts/pipeline_report_viewer.py outputs/pipeline_report/<ts>__example_v2/example
```

**Compare two backends: v1VectorClass vs v2VectorClass.** `bench_v1.json` and `bench_v2.json`
differ only in `p3` (and `workers: 2` for v2). Both use `p2: "Stub"` so only Phase 3 is
measured, skip debug images (the benchmark doesn't need them), and score pages 0 and 1 of the
same three inputs:

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/bench_v1.json
python scripts/generate_pipeline_report.py --config scripts/report_configs/bench_v2.json
python scripts/pipeline_report_benchmark.py \
    --run outputs/pipeline_report/<ts>__bench_v1 --run outputs/pipeline_report/<ts>__bench_v2
python scripts/pipeline_report_viewer.py \
    outputs/pipeline_report/<ts>__bench_v1/example_label outputs/pipeline_report/<ts>__bench_v2/example_label
```

`outputs/pipeline_report_benchmark/<ts>/report.html` then has one column per run: char / word
F1, bbox IoU and rotation per ground-truth type, candidate precision, and seconds per page.
v2VectorClass also reports its own sub-steps (`classify_*`, `ocr_detect`, `ocr_recognize`, ...)
as `phase3 › <name>` rows under `phase3` in the timing table; they are part of `phase3`, not
added to `total`.

**Benchmark a backend pair faster.** Set `"debug_layers": false` as well: only `dump.json` and
the ground truth are written. Leave it on when you want to inspect errors in the viewer
(including the `benchmark__extra_*` layers).
