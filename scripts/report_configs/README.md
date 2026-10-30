# Report configs

JSON configs for `scripts/generate_pipeline_report.py` (`ReportConfig`,
`raster_parser/evaluation/report/report_config.py`). Every field is optional. Relative paths are
resolved from the directory the script is run in.

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json
```

| Field | Default | Meaning |
|---|---|---|
| `p2`, `p3` | `v1_raster_to_vec`, `v2_vector_class` | Backends by registry name (`core/registry.py`). An unknown name is rejected. |
| `final_stage` | `null` | Stop after `native_read`, `raster_vectorize` or `vector_classify` (`run_pipeline(stop_after=...)`). Later backends don't run and emit no layers. output_assemble still runs. |
| `input_dir` | `null` | Every `*.pdf` in this folder. |
| `input_files` | `[]` | PDFs. In benchmark mode also label JSON files (`raster_parser/evaluation/labelling/label_schema.py`). |
| `label_files` | `{}` | Benchmark mode: `{pdf-stem: label JSON}`, to give a bare PDF hand-made labels. |
| `pages` | `null` (= `[0]`) | Page indices: one list for every input, or `{"<stem>": [...], "*": [...]}`. Out-of-range pages are skipped with a warning. |
| `vectorise` | `false` | Run on a converted copy of each page instead of the original. |
| `vectorise_mode` | `to_vector_text` | `text_only`, `drawings_only` or `to_vector_text` (`evaluation/conversion.py`). |
| `benchmark` | `false` | Write score-ready output for `pipeline_report_benchmark.py` (ground truth + overlays + `benchmark.json`). Runs on `to_vector_text` copies; can't be combined with `vectorise`. |
| `iou_edge_min` | `0.1` | Benchmark overlays: the IoU a prediction needs to count as reaching a label. |
| `raster_benchmark` | `false` | Benchmark mode: also rasterise each page, run `p2` on the image and score its vectors against the page's own vectors (Liu & Dori; `docs/RasterToVectorEvaluation.md`). |
| `raster_dpi` | `300` | The rasterised page's resolution. |
| `output_root` | `null` | Run folder root; default `outputs/pipeline_report/`. |
| `debug_images` | `true` | Save the debug images the backends offer. |
| `debug_image_cap` | `100` | At most this many images per folder per document (uniform random across pages, same picks on a rerun). `null` = all. |
| `debug_layers` | `true` | Write the layer PDFs. Off: `dump.json`, ground truth and debug images are still written, which is all the benchmark needs. |
| `keep_converted_pdfs` | `true` | Keep each page's converted input (`converted_p<N>.pdf`). |
| `ocr_dpi`, `use_angle_cls`, `workers` | `300`, `true`, `1` | Passed to the backends in `RunSettings`. `workers > 1` runs v1_raster_to_vec's and v2_vector_class's OCR on a process pool (same output, faster). v2_vector_class doesn't use `ocr_dpi` / `use_angle_cls`. |

## Inputs

Keep inputs under `outputs/labels/` (git-ignored). Two kinds can
go in `input_files`:

| Input | Example | Scored against (benchmark mode) |
|---|---|---|
| a label JSON | `outputs/labels/example.json` | its hand-labelled vector text (`original_vector`) and labels derived from the PDF's native text (`native_to_vector`); the PDF is the JSON's `pdf_path` |
| a bare PDF | `outputs/labels/plain.pdf` | labels derived from its own native text; with a `label_files` entry, that JSON's labels too |

The benchmark matches two runs on the inputs they share, by key (`labels:<JSON name>`, or
`pdf:<stem>` for a bare PDF without `label_files`), so run both configs on the same
`input_files`.

## Worked examples

**Look at one backend's steps.** `example.json` (v1_raster_to_vec + v1_vector_class) and `example_v2.json`
(`p3: "v2_vector_class"`, no raster pass) report one page each; open a result in the viewer:

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/example_v2.json
python scripts/pipeline_report_viewer.py outputs/pipeline_report/<ts>__example_v2/example
```

**Compare two backends: v1_vector_class vs v2_vector_class.** `bench_v1.json` and `bench_v2.json`
differ only in `p3` (and `workers: 2` for v2). Both use `p2: "stub"` so only vector_classify is
measured, skip debug images (the benchmark doesn't need them), and score pages 0 and 1 of the
same three inputs:

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/bench_v1.json
python scripts/generate_pipeline_report.py --config scripts/report_configs/bench_v2.json
python scripts/pipeline_report_benchmark.py \
    --run outputs/pipeline_report/<ts>__bench_v1 --run outputs/pipeline_report/<ts>__bench_v2
python scripts/pipeline_report_viewer.py \
    outputs/pipeline_report/<ts>__bench_v1/example outputs/pipeline_report/<ts>__bench_v2/example
```

`outputs/pipeline_report_benchmark/<ts>/report.html` then has one column per run: char / word
F1, bbox IoU and rotation per ground-truth type, candidate precision, and seconds per page.
v2_vector_class also reports its own sub-steps (`classify_*`, `ocr_detect`, `ocr_recognize`, ...)
as `vector_classify › <name>` rows under `vector_classify` in the timing table; they are part of `vector_classify`, not
added to `total`.

**Benchmark a backend pair faster.** Set `"debug_layers": false` as well: only `dump.json` and
the ground truth are written. Leave it on when you want to inspect errors in the viewer
(including the `benchmark__extra_*` layers).

`benchmark_example.json` scores `v1_vector_class` (no raster pass) on two pages of a label JSON.
