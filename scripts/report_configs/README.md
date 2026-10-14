# Report configs

JSON configs for `scripts/generate_pipeline_report.py` (`ReportConfig`,
`raster_parser/evaluation/report/report_config.py`). Every field is optional. Relative paths are
resolved from the directory the script is run in.

```
python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json
```

| Field | Default | Meaning |
|---|---|---|
| `p2`, `p3` | `v1RasterToVec`, `v1VectorClass` | Backends by registry name (`core/registry.py`). An unknown name is rejected. |
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
| `ocr_dpi`, `use_angle_cls`, `workers` | `300`, `true`, `1` | Passed to the backends in `RunSettings`. |

`example.json` reports one page with the default backends. `benchmark_example.json` scores
`v1VectorClass` (no raster pass) on two pages of a labelled folder.
