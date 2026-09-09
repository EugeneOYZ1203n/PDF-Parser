# Glossary

Standardized terminology for `rastervec`'s vector-classification + OCR pipeline
(`rastervec/Vector/vector.py`, `rastervec/Vector_Classification/classification.py`,
`rastervec/OCR/radon.py`, `rastervec/pipelines/`). Every term below is scoped to
one page's run unless stated otherwise. See `docs/PIPELINE.md` for the full
step-by-step data flow this vocabulary describes.

## Group

The output of the seqno-overlap-merge step (`combine_overlapping_seq`,
`classification.cluster()`'s step 3), scoped to one `(layer, color)` bucket
(see `Vector.separate_by_layer`/`separate_by_color`). A group is the atomic
input unit handed to spatial clustering (step 6, `cluster_spatial_groups`)
-- the smallest thing the rest of the classification chain reasons about
as a single piece. A group is a `list[Vector]`.

## Cluster

The final output of the whole vector-classification chain for one
`(layer, color)` bucket -- what's left after spatial clustering (step 6)
and every filter step after it. A cluster is a `list[list[Vector]]`: real,
nested structure recording exactly which groups compose it (`cluster_spatial_
groups` keeps every merged cluster's member groups nested rather than
flattening them into a side `id()`-keyed lineage dict). Every cluster that
survives the whole chain becomes a *text candidate* (see below).

## Segment

One Radon-segmented word (`OCR/radon.py::segment_clusters`), at its real
page position -- the unit similarity grouping and FAST detection both
operate on. Runs directly on classification's kept clusters (flattened to
`list[Vector]` per cluster), *before* similarity grouping and FAST, so
dedup happens as early as possible. Carries the cluster's own Radon-
estimated skew `angle`, at full precision (never rounded to a quarter
turn -- see `docs/PIPELINE.md`'s angle-precision invariant).

## Similarity group

A whole-page grouping of *segments* judged geometrically equivalent --
same shapes, translation/rotation-exact (using each segment's own known
`angle` to normalize rotation directly, rather than searching for it) --
computed by `pipelines._steps.group_similar_segments`, after `segment` and
before `fast`. Segments in the same similarity group (e.g. repeated
instances of the same label or symbol at different positions/orientations
on the page) share one FAST verdict: a group passes only if *every*
member's own combined FAST score individually exceeds
`FAST_COMBINED_KEEP_THRESHOLD` -- one weak instance drops the whole group.

## Unique segment

The single representative of a passing similarity group that actually
gets rendered and OCR'd (`UniqueSegment`) -- its `vectors` are normalized
to a canonical frame (translated so the segment's own bbox origin is
`(0, 0)`, rotated by `-angle` so it's upright). Every other member of the
group keeps only a `SegmentMeta` (offset + rotation back to its own real
position), so the one OCR `Text` a `UniqueSegment` produces can be
duplicated and repositioned for every real occurrence without
re-rendering or re-recognizing it.

## Text candidate

A cluster that survived the entire vector-classification chain
(`classification.cluster()`'s final "kept" category) -- handed downstream
to Radon segmentation, similarity grouping, FAST, and OCR. There is no
separate drawing-vs-text heuristic inside the classification chain
itself; a text candidate is just "whatever wasn't filtered out." OCR
success/failure (via the FAST gate a segment's similarity group must
clear) is the real, final signal for whether it was actually text.

## Drawing vector

Anything that did *not* end up as real recognized text, folded into the
final flat `vectors` output by `pipelines._steps.build_drawing_output`.
This includes, without distinction: every `role="dropped"` category any
classification filter step produced, and every segment in a similarity
group that failed FAST's all-must-pass rule. Unlike the pre-refactor
design there is no separate "OCR failed" drop bucket -- FAST is the sole
page-content pass/fail decision; a blank OCR reading still becomes a
real, restored `Text` (with `text=""`), not a drop.
