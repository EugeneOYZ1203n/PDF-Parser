# Example images to generate

Ten pictures that show the project better than prose. Generate each one, save it under `docs/images/`
with the file name given, and embed it where the last column says. Run commands from the repository root;
`references/test_pdfs_1.pdf` is a good default page (all five test PDFs are in `tests/references/`).

| # | Image (file name) | Command or steps | Embed in |
|---|---|---|---|
| 1 | Input page next to the `extract()` reconstruction (`01_input_vs_extract.png`) | Open the PDF page. Run the report below with `p2: v1_raster_to_vec`, `p3: v2_vector_class`; take the final-layer PDF from `outputs/pipeline_report/<ts>__<config>/<doc>/`. Put the two side by side. | `README.md`, top |
| 2 | Layers of one page in different colours: native, raster_vectorize, vector_classify (`02_phase_layers.png`) | `python scripts/pipeline_report_viewer.py outputs/pipeline_report/<ts>__<config>/<doc>`; tick one layer per phase; screenshot the window. | `README.md`, "How it works" |
| 3 | `v2_vector_class` debug layers: clusters, kept against dropped, text boxes (`03_v2_debug_layers.png`) | Same viewer, same run, ticking the `v2_vector_class` layers. Run the config with `"debug_layers": true`. | `docs/PIPELINE.md` |
| 4 | `v1_raster_to_vec` steps 1 to 5 on one raster image (`04_v1_raster_steps.png`) | Report config with `"p2": "v1_raster_to_vec"`, `"debug_images": true`; collage the step images from the run's `p2_work/` folder. | the `v1_raster_to_vec` README |
| 5 | Junction and InvolutionStroke on the same crop (`05_junction_vs_involution.png`) | Two configs, `"p2": "junction"` and `"p2": "involution_stroke"` (needs `uv sync --extra ml`; allow time), same input and page; open both folders in the viewer. | `docs/README.md` or the `junction` README |
| 6 | The labelling tool, a seeded draft before and after correction (`06_label_before_after.png`) | `python scripts/label/manual_vector_text_label.py "references/test_pdfs_1.pdf" --page 0`; screenshot on open, correct a few entries, screenshot again. | `scripts/label/README.md` |
| 7 | A benchmark report page: text metrics table and Liu & Dori chart (`07_benchmark_report.png`) | `python scripts/generate_pipeline_report.py --config scripts/report_configs/bench_raster_v1.json`, then `python scripts/pipeline_report_benchmark.py --run <that run folder>`; screenshot `report.html`. | `docs/SCRIPTS.md` |
| 8 | A Liu & Dori example: detection against false alarm (`08_liu_dori_example.png`) | Run `notebooks/liu_dori_showcase.ipynb`; save one of the case figures it writes under `outputs/`. | `docs/RasterToVectorEvaluation.md` |
| 9 | The inspector window on a page (`09_inspector.png`) | `python -m scripts.inspector.inspector references/test_pdfs_1.pdf`; screenshot with a few layers on and a tooltip open. | `scripts/inspector/README.md` |
| 10 | Classification time against vector count (`10_timing_scatter.png`) | Run `notebooks/pdf_dataset_timing.ipynb`; copy `time_vs_num_vectors_vectorised.png` from the run's `timing_plots/`. | `docs/HANDOVER.md`, "Known limits" |

Keep each image under about 1 MB (crop to the interesting area), use a light background, and add the
embed line (`![caption](images/<file>)`, with the path relative to the document) as you place each one.
