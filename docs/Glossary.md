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

The same dataclass (`vectors`, `angle`, optionally `image`) shared by two
dedup passes at two granularities -- see `docs/PIPELINE.md`'s step
sequence and `models/segment.py`'s docstring:

- **Cluster-level (pre-Radon)**: one `Segment` per classification cluster,
  `angle` a cheap PCA principal-axis estimate over the cluster's own point
  cloud (`pipelines._steps._cluster_angle`) -- pure vector-geometry math,
  no rendering. The unit similarity grouping and FAST detection both
  operate on, run directly on classification's kept clusters, *before*
  Radon or OCR ever touch them.
- **Word-level (post-Radon, representatives only)**: one `Segment` per
  word, produced by `OCR/radon.py::segment_clusters` when run on just an
  elected representative's own canonical-frame vectors -- Radon's precise
  residual skew `angle`, at full precision (never rounded to a quarter
  turn -- see `docs/PIPELINE.md`'s angle-precision invariant), plus that
  word's own captured crop in `image`.

## Similarity group

A whole-page grouping of geometrically equivalent *clusters* --
same shapes, translation/rotation-tolerant (using each cluster's own PCA
angle estimate to normalize rotation) -- computed by
`pipelines._steps.group_similar_segments`, after `classify` and before
`fast`, i.e. *before* Radon ever runs. Clusters in the same similarity
group (e.g. repeated instances of the same label or symbol at different
positions/orientations on the page) share one FAST verdict: a group passes
only if *every* member's own combined FAST score individually exceeds
`FAST_COMBINED_KEEP_THRESHOLD` -- one weak instance drops the whole group.

## Unique segment

The single representative *cluster* of a passing similarity group --
its `vectors` are normalized to a canonical frame (translated so the
cluster's own bbox origin is `(0, 0)`, rotated by `-angle` so it's
upright). Every other member of the group keeps only a `SegmentMeta`
(offset + rotation back to its own real position). Only this
representative is Radon-segmented into words and OCR'd (`pipelines.
_steps.segment_unique_clusters` -> `OCR.Paddle_OCR.ocr_backend.
recognize_segments`); the resulting word-level `Text`s are then replayed
onto every other member's real position (`pipelines.sub_pipelines.ocr.
restore_cluster_texts`) without re-rendering or re-recognizing anything.

## Text candidate

A cluster that survived the entire vector-classification chain
(`classification.cluster()`'s final "kept" category) -- handed downstream
to similarity grouping, FAST, Radon segmentation, and OCR. There is no
separate drawing-vs-text heuristic inside the classification chain
itself; a text candidate is just "whatever wasn't filtered out." OCR
success/failure (via the FAST gate a cluster's similarity group must
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
