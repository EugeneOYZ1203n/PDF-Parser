# Retiring the old pipeline: a migration proposal

**Status: proposed, not started.** This document describes a follow-up piece of work, not
something that's been executed. It exists so the decision of *how* to retire the old pipeline
doesn't have to be re-derived from scratch later, and so nobody accidentally deletes
`rastervec/OCR/`, `rastervec/Vector/`, or `rastervec/pipelines/{current.py,_steps.py,result.py,
_cli.py}` before reading this.

## Why this exists

`rastervec/` currently has two coexisting pipeline implementations:

- **`core/pipeline.py::run_pipeline`** — the current, intended architecture: Phase 1 (always the
  same) → pluggable Phase 2 → pluggable Phase 3 → Phase 4 (always the same), with self-contained
  P2/P3 backends. This is what all new code should call.
- **`pipelines/current.py::run_pipeline`** — an older, single fixed pipeline (`extract_native_text
  → extract_vectors → similarity_group → filter_vectors_fast → reclassify_by_similarity →
  separate_by_layer_color_width → cluster_buckets → detect_text_paddle_per_cluster →
  reassign_by_overlap → rotate_paddle_detections → recognize_segments`) that predates the
  phase split. It has its own `PipelineResult` (`pipelines/result.py`) with a different shape
  from `core/result.py`'s (an `engine` field, many per-stage Optional fields, no `extra` bucket).

`CLAUDE.md` used to describe the old pipeline and its supporting folders (`rastervec/OCR/`,
`rastervec/Vector/`) as dead code. They aren't — see `CLAUDE.md`'s top-of-file note for the
current, accurate list of live dependents. That's a genuinely confusing state for anyone reading
the codebase: two `PipelineResult` classes, two `run_pipeline` entrypoints, and folders that
"shouldn't" be imported from but still are. `scripts/label/CAD_font_label.py` already treats
`rastervec.pipelines.current` as informally "off-limits" in its own docstring — this proposal is
about finishing that intent for real, everywhere, rather than leaving it as one script's private
convention.

## Goal state

`rastervec/OCR/`, `rastervec/Vector/`, and `rastervec/pipelines/{current.py,_steps.py,result.py,
_cli.py}` deleted for real. Every consumer runs on `core/pipeline.py` + the phase-folder
backends. One `PipelineResult` type. One `run_pipeline` entrypoint.

## Per-consumer migration steps

Each of these can likely be done as its own small, independently reviewable change; they don't
need to land in one big PR. Suggested order: infra first (lower risk, has test coverage),
GUI tools last (no automated tests, need manual verification).

### 1. `core/parallel/pool.py::warmup()`

Currently warms the old, shared `OCR/fast_detect.FastDetector` and
`OCR/Paddle_OCR/ocr_backend.PaddleRecBackend` unconditionally, regardless of which P3 backend a
run will actually use. Per CLAUDE.md's own registry docs, each P3 backend already exposes its
own `FastDetector`/`PaddleRecBackend` classmethods for exactly this purpose (e.g.
`P3_Vector_Parsing/FastIntoPaddle/fast_detect.py::FastDetector.warmup()`).

Two options:
- Thread the run's `p2`/`p3` backend names into `warmup()` so it warms only the backend(s) that
  will actually run.
- Simpler: have `warmup()` iterate every registered P3 backend's own warmup classmethod
  unconditionally. Slightly more startup cost, no need to plumb backend selection through the
  pool machinery.

Either way, once nothing calls `rastervec.OCR.fast_detect`/`rastervec.OCR.Paddle_OCR.ocr_backend`
from here, this consumer is migrated.

### 2. `commons/renderer/stages.py`

Imports `OCR/radon.py`'s `cluster_centre`, `rotate_pts`, `rotated_segments` for one of its
`render_<stage>` functions. These read as pure geometry (point/rotation math) with no
OCR-specific behavior — verify this at implementation time, then promote them into
`commons/helpers/geometry.py` (which already holds comparable pure-geometry helpers like
`point_angle`/`rect_gap`/`union_bbox`). Once moved, `stages.py` needs no import from
`rastervec.OCR` at all, and the promoted functions become properly shared foundation code
instead of borrowed from a deprecated module.

### 3. `P2_Raster_To_Vec/Junction/junction_test/pipeline.py`

Imports `OCR/Paddle_OCR/render_ocr.RenderOCR`, but only inside the branch gated by
`Params(run_ocr=True)` — and `run_ocr=False` is the default, per CLAUDE.md's own description of
this backend ("OCR text-box extraction inside `junction_test` stays off"). Two options:
- If `run_ocr=True` is never going to be turned on in practice, delete the branch and the import
  entirely.
- If it might still be exercised (check for any caller or test that sets `run_ocr=True` before
  deciding), vendor a minimal, Junction-owned copy of whatever `RenderOCR` actually provides, per
  the "self-contained backend" rule the rest of `P2_Raster_To_Vec/`/`P3_Vector_Parsing/` follows.

### 4. `P1_Reading_Native/vector_extract.py`

Re-exports `rastervec.Vector.layer_color_separation` "for callers" (`# noqa: F401`). Since P1 is
always-run, always-shared code, the cleanest fix is to move `layer_color_separation.py` itself
into `commons/helpers/` (it's generic layer/color/width separation over `Vector`s, fitting
`commons/`'s charter as pure shared foundation) rather than leaving it in a standalone top-level
`Vector/` folder. Update the real callers once moved: `pipelines/current.py`,
`scripts/label/vector_label.py`. After this, `Vector/` should have nothing left in it.

### 5. `scripts/label/vector_label.py` and `scripts/label/label_viewer.py`

The two labelling GUIs still built on the old pipeline (`pipelines._steps.extract_vectors`,
`pipelines.current.separate_by_layer_color_width`). `scripts/label/CAD_font_label.py` already
shows the intended replacement pattern: it imports `extract_vectors` from
`P1_Reading_Native.vector_extract` directly and — because it treats the old
`separate_by_layer_color_width` as off-limits — simply has no layer/color/width bucket-filter
side panel at all.

Migrating `vector_label.py`/`label_viewer.py` properly means restoring that side panel using a
new-architecture equivalent of `separate_by_layer_color_width` (which will exist in
`commons/helpers/` after step 4 above, once it's promoted there) rather than dropping the
feature the way `CAD_font_label.py` did. Since these are Tk GUIs with no automated tests, budget
time for manual smoke-testing per CLAUDE.md's existing testing convention for these tools.

### 6. `Evaluation/Evaluate/benchmark.py`

Imports `pipelines.current.STEP_NAMES` for a legacy/current step-name comparison (line ~390 as
of this writing). Once `pipelines.current` is gone, decide whether this comparison is still
meaningful (e.g. against `core.pipeline`'s own step names) or should simply be dropped.

### 7. Type-only `PipelineResult` imports

`Evaluation/Evaluate/adapters.py`, `commons/renderer/stages.py`, and
`commons/renderer/notebook.py` all import `pipelines.result.PipelineResult` purely for typing.
Once nothing constructs a `pipelines.result.PipelineResult` at runtime (i.e. after step 5 is
done and `pipelines/current.py` has no remaining callers), these can be dropped or repointed at
`core.result.PipelineResult` — check each usage site for which fields it actually reads, since
the two `PipelineResult` shapes differ (no `extra` bucket vs. `core.result.PipelineResult`'s
generic bucket, no `engine` field, etc.).

### 8. Final deletion

Once steps 1-7 are done and a repo-wide grep for `rastervec.OCR`, `rastervec.Vector\b`,
`pipelines.current`, `pipelines._steps`, `pipelines.result`, and `pipelines._cli` turns up
nothing outside those modules' own internals and their test suite
(`tests/rastervec/{OCR,Vector,pipelines}/`), delete `rastervec/OCR/`, `rastervec/Vector/`,
`rastervec/pipelines/current.py`, `_steps.py`, `result.py`, `_cli.py`, and their corresponding
test folders, then remove the now-unneeded "old pipeline" explainer from `CLAUDE.md` and this
document.

## Sequencing note

This touches core infra (pool warmup, the renderer) as well as three GUI tools with no
automated tests, so it should be done as its own reviewed piece of work, not folded into a
documentation/readability pass. Steps 1-4 are infra-only and covered by existing tests; steps
5-6 touch tools with no automated coverage and need manual verification; step 7 is a pure
type-import cleanup that should be safe once steps 1-6 land; step 8 is the actual deletion,
gated on a clean grep.
