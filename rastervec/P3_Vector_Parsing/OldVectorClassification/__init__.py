# FROZEN -- see this docstring; do not edit.
"""OldVectorClassification -- a FROZEN recreation of the `VectorClassification`
P3 backend exactly as it stood on 2026-09-29 (before any 2026-09-30 commit;
folder content identical to commits `5e195e3`/`0863b0e`).

**DO NOT EDIT ANYTHING IN THIS FOLDER, IN ANY WAY.** It exists as a fixed
benchmark baseline. Every method here must stay isolated and untouched:
new work goes into `LatestVectorClassification/`, never here.

Isolation: every non-rendering dependency this backend had on `commons/`
at that date (geometry, clustering, logging, step timing, OCR-input render
prep, cluster/page rasterisation, drawing replay) is vendored verbatim, in
its 2026-09-29 version, under `_vendored/`, so later changes to `commons/`
can never change this backend's output. The only shared imports allowed
are:

- `rastervec.commons.models` -- the Phase-3 interface types themselves;
- `rastervec.commons.renderer.render_boxes_pdf` / `render_text_pdf` /
  `render_vectors_pdf` -- debug-layer drawing only, which never affects
  the returned `(vectors, texts)`.

`tests/rastervec/P3_Vector_Parsing/OldVectorClassification/test_isolation.py`
enforces this. The only changes ever made relative to the 2026-09-29 source
were one-time, at recreation: import paths rewritten to this folder /
`_vendored/`, the dead lazy import of the deleted
`pipelines.sub_pipelines.vector_classification` repointed to this folder's
own `classify_vectors._classify_bucket`, and these FROZEN header lines.
See README.md.
"""
