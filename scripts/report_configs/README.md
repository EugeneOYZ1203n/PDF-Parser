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
| `p3` | only meaningful when `pipeline: "current"` — a `core.registry.P3_REGISTRY` name (`LatestVectorClassification` default, `OldVectorClassification` -- the frozen 2026-09-29 snapshot -- or `LegacyRecreation`) |
| `enable_fast` | forwarded to `p3` backends that accept it (default `true`). Currently a no-op — no P3 backend declares it (LatestVectorClassification has no FAST; OldVectorClassification always runs its own FAST filter); kept so older configs still validate |
| `final_stage` | one of `core.pipeline`'s short step names (`phase1`/`phase2`/`phase3`); `null` = all. Trims which of the fixed phase-level artifacts render — doesn't skip any actual pipeline work or gate per-backend debug layers. Ignored for `pipeline: "legacy"`. |
| `input_dir` | folder scanned for `*.pdf` |
| `input_files` | explicit list of PDF paths (merged with `input_dir`, deduped) |
| `label_files` | `{ "<pdf-stem>": "path/to/labels.json" }` — recorded in the manifest for the benchmark step |
| `pages` | `[0, 2]` (same pages for every PDF), or `{ "<pdf-stem>": [0,1], "*": [0] }`, or `null` = page 0 |
| `vectorise` | run an `Evaluation/conversion.py` pre-step so the pipeline sees text-as-vector-paths |
| `vectorise_mode` | `to_vector_text` (default) / `text_only` / `drawings_only` |
| `benchmark` | benchmark mode (default `false`) — each input also becomes a scoring artifact for `scripts/pipeline_report_benchmark.py` (ground-truth JSONs + `benchmark.json`). `input_files` may then be `.pdf`s, `.json` label sidecars, or `scripts/label/master_label.py` folders. Mutually exclusive with `vectorise` |
| `iou_edge_min` | `MetricConfig.iou_edge_min` for the benchmark overlays (default `0.1`) |
| `dpi` | render dpi for the stage crops (default 300) |
| `output_root` | default `outputs/pipeline_report/` |
| `debug_images` | write the `for_paddle_detect/` / `for_rotation_correction/` / `for_paddle_recog/` / `paddle_ocr_images/` PNG crops (max `debug_image_cap` random per leaf folder per input, default `true`). `false` also stops backends keeping the full-size image arrays those crops come from |
| `debug_image_cap` | per-leaf-folder cap on `debug_images` PNGs (default `100`); `null` = uncapped |
| `debug_layers` | write the layer PDFs the viewer toggles (default `true`): every backend debug layer (`<stage>__<layer>.pdf`), `phase2`/`reconstructed`, `benchmark__extra_*`, and the benchmark `<type>_{bbox,text}.pdf` overlays. `false` skips all of them **and** the work done only to render them; `dump.json`, ground truth, stats and debug images are still written, so `pipeline_report_benchmark.py` still scores the run (the viewer then shows only the source page). With `p2: "Junction"`, also set `debug_images: false` for the fastest run |

## Faster runs

For timing or benchmark-only runs, turn off both kinds of debug output:

```json
{
  "p3": "LatestVectorClassification",
  "input_files": ["references/some.pdf"],
  "pages": [0],
  "debug_layers": false,
  "debug_images": false
}
```
