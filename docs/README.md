# Documentation index

Start with the [project README](../README.md) for what the project does and how to run it. This page
lists every other document, grouped by what you are trying to do.

## Understand the system

| Doc | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Folder layout, the four phases, the backend contract, the rule that backends never import each other, **the map of shared code** (`commons/`), coordinate spaces, debug and progress output |
| [PIPELINE.md](PIPELINE.md) | One page walked through `v2_vector_class`, step by step |
| [Glossary.md](Glossary.md) | Terms used by `v2_vector_class`: vector, bucket, collinear / parallel group, cluster |
| [model_migration.md](model_migration.md) | The shared `Text` / `Vector` / `ExtractResult` models and how they replaced the old element classes |
| [../raster_parser/INTEGRATION.md](../raster_parser/INTEGRATION.md) | Calling `extract()` from another application |

## Backends

| Backend | Doc |
|---|---|
| `v1_raster_to_vec` | [README](../raster_parser/phase_raster_vectorize/v1_raster_to_vec/README.md) |
| `junction` | [README](../raster_parser/phase_raster_vectorize/junction/README.md) |
| `involution_stroke` | [README](../raster_parser/phase_raster_vectorize/involution_stroke/README.md): weights, licence caveat, known behaviour |
| `deep_tech_vec` | [README](../raster_parser/phase_raster_vectorize/deep_tech_vec/README.md): weights, licence, refinement and merging |
| `deep_vectoriser` | [TRAINING](../raster_parser/phase_raster_vectorize/deep_vectoriser/TRAINING.md): building the dataset and training the model |
| `implicit_sketch_vec` | [TRAINING](../raster_parser/phase_raster_vectorize/implicit_sketch_vec/TRAINING.md): its three training stages |
| the learned backends | [p2_paper_deviations.md](p2_paper_deviations.md): where each implementation differs from its paper |
| `v1_vector_class` | [README](../raster_parser/phase_vector_classify/v1_vector_class/README.md) |
| `v2_vector_class` | [README](../raster_parser/phase_vector_classify/v2_vector_class/README.md) |

## Run, label and measure

| Doc | What it covers |
|---|---|
| [SCRIPTS.md](SCRIPTS.md) | The report, viewer and benchmark scripts, the typical workflow, the report folder layout |
| [../scripts/report_configs/README.md](../scripts/report_configs/README.md) | Every report-config field, with worked two-backend comparisons |
| [../scripts/label/README.md](../scripts/label/README.md) | The labelling tool: hand-label original vector text, seeded by `v2_vector_class`, with a label list and background page processing |
| [../scripts/inspector/README.md](../scripts/inspector/README.md) | The PDF inspector window |
| [RasterToVectorEvaluation.md](RasterToVectorEvaluation.md) | The Liu & Dori protocol, its deviations, the ground truth |
| [../notebooks/README.md](../notebooks/README.md) | The notebooks: classification filter showcases, pattern and line labs, Liu & Dori showcase, ground-truth fidelity and problem scan, dataset timing |

## Take over the project

| Doc | What it covers |
|---|---|
| [HANDOVER.md](HANDOVER.md) | The PR stack, guardrail tests, open decisions, known limits |
| [images/EXAMPLES.md](images/EXAMPLES.md) | Pictures to generate for the README, each with its command |
