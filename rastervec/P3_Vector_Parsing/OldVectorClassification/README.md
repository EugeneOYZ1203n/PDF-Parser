# OldVectorClassification — FROZEN

> **Do not edit anything in this folder, in any way.** Not for bug fixes, refactors,
> renames, formatting, or "small" cleanups. New P3 work goes in
> `LatestVectorClassification/`.

This is a recreation of the `VectorClassification` P3 backend exactly as it was on
**2026-09-29**, the state just before the first 2026-09-30 commit. The folder content
is identical to commits `5e195e3` and `0863b0e`. It's kept as a fixed comparison
baseline (`p3: "OldVectorClassification"`).

## What it does (as of 2026-09-29)
1. Separate vectors by `(layer, color)`, then run a 2-step classification: seqno-overlap merge, then spatial clustering.
2. Pre-OCR FAST page-heatmap filter on the clusters (`fast_filter.py`, `fast_detect.py`, FAST-Tiny weights at `rastervec/weights/`).
3. Per surviving cluster:
   - PaddleOCR detect.
   - Hough + minAreaRect deskew (`paddle_engine.hough_deskew`).
   - Page-wide batched recognition with the angle classifier.
   - A blank-only +90/180/270 retry.
4. Non-text vectors go to `drawing`.

## Isolation
Every non-rendering dependency it had on `rastervec/commons/` is vendored verbatim,
in its 2026-09-29 version, under `_vendored/`:

| vendored file | original |
|---|---|
| `_vendored/geometry.py` | `commons/helpers/geometry.py` |
| `_vendored/clustering.py` | `commons/helpers/clustering.py` |
| `_vendored/logging_setup.py` | `commons/logging_setup.py` |
| `_vendored/step_timing.py` | `commons/step_timing.py` |
| `_vendored/ocr_prep.py` | `commons/renderer/ocr_prep.py` |
| `_vendored/png.py` | `commons/renderer/png.py` |
| `_vendored/_shapes.py` | `commons/renderer/_shapes.py` |

The only shared imports still allowed are:
- `rastervec.commons.models`, the P3 interface types.
- `rastervec.commons.renderer.render_boxes_pdf`, `render_text_pdf` and `render_vectors_pdf`. These are debug-layer drawing only and never affect output.

`tests/rastervec/P3_Vector_Parsing/OldVectorClassification/test_isolation.py`
fails if any other `rastervec.*` import appears.

## One-time changes made at recreation (never repeat)
- Import paths were rewritten from `P3_Vector_Parsing.VectorClassification` / `commons.*` to this folder and `_vendored/`.
- `classification.cluster` had a dead lazy import of the deleted `pipelines.sub_pipelines.vector_classification`. It was repointed to this folder's own `classify_vectors._classify_bucket`.
- A `# FROZEN` header line was added to every `.py`.

Debug-image savers for this backend live outside the folder, in
`scripts/debug_image_savers.py` (`_save_oldvectorclassification_*`). Report tooling
may adapt to this backend's `debug_out` shape; this folder never adapts to the tooling.
