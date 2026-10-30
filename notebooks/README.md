# Notebooks

Exploration and showcase notebooks. Each one picks the first PDF under `references/` unless `PDF_PATH`
is set in its config cell, and writes its pictures under `outputs/`. Notebooks are committed without
outputs; the logic lives in `notebooks/_helpers/`, which the tests in `tests/notebooks/` cover.

| Notebook | Helper | Shows |
|---|---|---|
| `filter_showcase` | `filter_showcase_helpers` | What four `v2_vector_class` filters (collinear, pattern lattice, length outliers, crossings) group, drop and keep on one page |
| `pattern_texture_lab` | `pattern_probe_helpers`, `vector_probe_helpers` | Hatch / pattern-texture detection on clusters |
| `dashed_line_collinear_lab` | `vector_probe_helpers` | Dashed lines and collinear grouping |
| `parallel_groups_lab` | `vector_probe_helpers` | Parallel groups per bucket |
| `vector_intersection_lab` | `vector_probe_helpers` | Crossings between segments |
| `pdf_dataset_timing` | `dataset_timing_helpers` | Vectorised vector counts over a folder of PDFs (histogram), then the time `scripts/report_configs/classify_timing_v2.json` (vector classification only) takes on an evenly sampled set |
| `liu_dori_showcase` | `liu_dori_showcase_helpers` | The Liu & Dori vector metrics on constructed cases (detection, false alarm, fragmentation) |
| `gt_fidelity_showcase` | `gt_fidelity_helpers` | How faithfully the vector -> raster ground truth (`visible_geometry`) reproduces the page: SSIM / PSNR of the original render vs a render of the GT, next to the paint-blind GT |
| `gt_problem_scan` | `gt_problem_scan_helpers` | Every place over a set of PDFs where that ground truth and the rendered page disagree, by category, with page overlays and crop galleries |

## Running

```
uv sync --group notebook
cd notebooks && jupyter lab
```

`RUN_NOTEBOOKS=1 uv run pytest tests/notebooks` executes every notebook with `nbconvert` (slow).

## Not shipped

- `cad_font_matching_lab`, `radon_vector_deskew_lab`, `skew_method_comparison`, `vector_similarity_lab`,
  `golden_case_curation`: depend on modules that are not part of this port.
