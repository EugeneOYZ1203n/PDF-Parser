# Sample `generate_pipeline_report.py` configs

```
.venv/Scripts/python.exe scripts/generate_pipeline_report.py --config scripts/report_configs/<name>.json
```

Every field is optional except that you need at least one of `input_dir` /
`input_files`. Paths are relative to the repo root (where you run the script).

| field | meaning |
|---|---|
| `pipeline` | `current` (default) / `current_nofast` / `legacy` |
| `final_stage` | a pipeline step name (`read native vectors classify fast segment similarity ocr restore drawing`); `null` = run all. Steps after it are skipped, so their stage PDFs are not emitted. |
| `input_dir` | folder scanned for `*.pdf` |
| `input_files` | explicit list of PDF paths (merged with `input_dir`, deduped) |
| `label_files` | `{ "<pdf-stem>": "path/to/labels.json" }` — recorded in the manifest for the benchmark step |
| `pages` | `[0, 2]` (same pages for every PDF), or `{ "<pdf-stem>": [0,1], "*": [0] }`, or `null` = page 0 |
| `vectorise` | run an `Evaluation/conversion.py` pre-step so the pipeline sees text-as-vector-paths |
| `vectorise_mode` | `to_vector_text` (default) / `text_only` / `drawings_only` |
| `dpi` | render dpi for the stage crops (default 300) |
| `output_root` | default `outputs/pipeline_report/` |
