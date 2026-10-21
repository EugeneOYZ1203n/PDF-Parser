# Sample `generate_pipeline_report.py` configs

```
.venv/Scripts/python.exe scripts/generate_pipeline_report.py --config scripts/report_configs/<name>.json
```

Every field is optional except that you need at least one of `input_dir` /
`input_files`. Paths are relative to the repo root (where you run the script).

| field | meaning |
|---|---|
| `pipeline` | `current` (default, the pluggable `core.pipeline` engine) / `legacy` |
| `p2` | only meaningful when `pipeline: "current"` — a `core.registry.P2_REGISTRY` name (`Stub` default, or `Junction`) |
| `p3` | only meaningful when `pipeline: "current"` — a `core.registry.P3_REGISTRY` name (`LatestVectorClassification` default, `OldVectorClassification` -- the frozen 2026-09-29 snapshot) |
| `enable_fast` | forwarded to `p3` backends that accept it (default `true`). Currently a no-op — no P3 backend declares it (LatestVectorClassification has no FAST; OldVectorClassification always runs its own FAST filter); kept so older configs still validate |
| `final_stage` | last phase to run: `null` (default) / `"phase1"` / `"phase2"` / `"phase3"` — see [`final_stage` values](#final_stage-values). Really stops the pipeline there (`run_pipeline(stop_after=...)`). Ignored for `pipeline: "legacy"` |
| `input_dir` | folder scanned for `*.pdf` |
| `input_files` | explicit list of PDF paths (merged with `input_dir`, deduped) |
| `label_files` | `{ "<pdf-stem>": "path/to/labels.json" }` — recorded in the manifest for the benchmark step |
| `pages` | `[0, 2]` (same pages for every PDF), or `{ "<pdf-stem>": [0,1], "*": [0] }`, or `null` = page 0 |
| `inputs` | which forms of each page the pipeline runs on — any non-empty combination of `"original"` (the page as-is), `"vectorised"` (`convert_page_to_vector_text`: native text as vector paths over the untouched drawings) and `"rasterised"` (the page flattened to one image — a `master_label.py` folder's own `rasterised.pdf` when there is one, else rendered on the fly at `dpi`). Default `["original"]`, or `["vectorised", "rasterised"]` in benchmark mode. See [Input forms](#input-forms) |
| `vectorise` | **deprecated** alias for `inputs: ["vectorised"]` (normal mode only); setting it together with `inputs` is an error |
| `vectorise_mode` | **deprecated**, only valid with `vectorise: true`: `to_vector_text` (default) / `text_only` / `drawings_only`. The `inputs` "vectorised" form is always `to_vector_text` |
| `benchmark` | benchmark mode (default `false`) — each input also becomes a scoring artifact for `scripts/pipeline_report_benchmark.py` (ground-truth JSONs + `benchmark.json`). `input_files` may then be `.pdf`s, `.json` label sidecars, or `scripts/label/master_label.py` folders. Select forms with `inputs`, not `vectorise` |
| `iou_edge_min` | `MetricConfig.iou_edge_min` for the benchmark overlays (default `0.1`) |
| `dpi` | resolution the "rasterised" form is rendered at when the input has no `rasterised.pdf` of its own (default 300) |
| `output_root` | default `outputs/pipeline_report/` |
| `debug_images` | write the `for_paddle_detect/` / `for_rotation_correction/` / `for_paddle_recog/` PNG crops (max `debug_image_cap` random per leaf folder per input, default `true`). `false` also stops backends keeping the full-size image arrays those crops come from |
| `debug_image_cap` | per-leaf-folder cap on `debug_images` PNGs (default `100`); `null` = uncapped |
| `debug_layers` | write the layer PDFs the viewer toggles (default `true`): every backend debug layer (`<stage>__<layer>.pdf`), `phase2`/`reconstructed`, `benchmark__extra_*`, and the benchmark `<type>_{bbox,text}.pdf` overlays. `false` skips all of them **and** the work done only to render them; `dump.json`, ground truth, stats and debug images are still written, so `pipeline_report_benchmark.py` still scores the run (the viewer then shows only the source page). With `p2: "Junction"`, also set `debug_images: false` for the fastest run |
| `keep_converted_pdfs` | keep the per-page pipeline-input PDFs the vectorised / rasterised forms write (`converted_p<N>.pdf`, `rasterised_p<N>.pdf`; default `true`). `false` deletes each right after its page's run — with `debug_layers`/`debug_images` also off, a run writes no PDFs at all |

## Input forms

| form | pipeline input | output folder (normal mode) | benchmark: scored on |
|---|---|---|---|
| `original` | the source page | `<stem>/` alone, `<stem>__original/` with other forms | native text (read by P1, reported as "native text"), `original_vector`, `original_raster` text — every output text is a prediction; `original_raster` geometry vs Phase 2's vectors |
| `vectorised` | `convert_page_to_vector_text` | `<stem>__vectorised/` | `native_to_vector`, `original_vector` (OCR output) |
| `rasterised` | `rasterised.pdf` page, else `rasterise_page` at `dpi` | `<stem>__rasterised/` | `vector_to_raster`, `original_raster`, `native_to_raster` text (OCR output); `vector_to_raster` + `original_raster` geometry vs Phase 2's vectors (`vector_to_raster` geometry is derived live from the original page when the input has none saved, so bare PDFs are scored too) |

In benchmark mode each input folder holds the ground truth plus one
sub-folder per form (`<input>/original/`, `<input>/vectorised/`,
`<input>/rasterised/`), each a full viewer folder with its own `dump.json`.
Every run records per-page counts (`dump.json` `counts`: texts by source,
vectors per phase, OCR blanks, Phase 2's per-colour-layer output), which the
benchmark report shows for every form. The rasterised form keeps the page's
`/Rotate` and unrotated coordinates, so its output lines up with the labels
on rotated pages too.

```json
{ "benchmark": true, "inputs": ["original", "vectorised", "rasterised"], "input_files": ["outputs/labels/X_label"], "pages": [0] }
```

## `final_stage` values

Passed to `core.pipeline.run_pipeline(stop_after=...)`, so later phases are
never called — they don't render debug layers, write debug images or take
time. Phase 4 (output combination) always runs on whatever was produced.

| value | phases run | layer PDFs written | `dump.json` texts / vectors |
|---|---|---|---|
| `null` (default) | P1 → P2 → P3 → P4 | all: P2 + P3 debug layers, `phase2`, `reconstructed` | final: native + P2 + P3 OCR text / P3's classified vectors |
| `"phase1"` | P1 → P4 | none | native text only / raw P1 vectors |
| `"phase2"` | P1 → P2 → P4 | P2 debug layers + `phase2` (written only if P2 produced anything, so nothing with `p2: "Stub"`) | native + P2 OCR text / raw P1 + P2 vectors |
| `"phase3"` | same as `null` | same as `null` | same as `null` |

Any other value is rejected when the config loads. With `benchmark: true`,
`"phase1"`/`"phase2"` mean no P3 OCR text, so text scores drop to near zero.

## Faster runs

For timing or benchmark-only runs, turn off both kinds of debug output:

```json
{
  "p3": "LatestVectorClassification",
  "input_files": ["references/some.pdf"],
  "pages": [0],
  "debug_layers": false,
  "debug_images": false,
  "keep_converted_pdfs": false
}
```

`native_to_vector_accuracy.json` is exactly that as a benchmark run: P3
`LatestVectorClassification`, no PDFs/PNGs written, OCR scored against the
native-text auto ground truth (`ground_truth_native_to_vector.json`). Score it
with `scripts/pipeline_report_benchmark.py --run <run dir>`; it is also the
template `rastervec/notebooks/pdf_dataset_stats.ipynb` uses.
