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

The dataclass (`vectors`, `angle`, optionally `image`) `OCR/radon.py::
segment_clusters` produces -- one `Segment` per word, at its real page
position, Radon's precise residual skew `angle` (full precision, never
rounded to a quarter turn -- see `docs/PIPELINE.md`'s worked example),
plus that word's own captured deskewed crop in `image`. This is the only
`Segment` lifecycle in the pipeline: FAST (`pipelines._steps.
detect_text_fast`) runs *before* Radon, directly on plain classification
clusters, and never needs a rotation estimate at all.

## Similarity group

A whole-page grouping of geometrically equivalent *word* `Segment`s --
same shapes, translation/rotation-tolerant (using each word's own
Radon-precise `angle` to normalize rotation) -- computed by
`pipelines._steps.group_similar_segments`, after `segment` (Radon) and
before `ocr`. Words in the same similarity group (e.g. repeated instances
of the same label at different positions/orientations on the page) share
one OCR result: `elect_unique_segments` (see "Unique segment" below) picks
one representative to actually recognize.

## Unique segment

The single elected representative word `Segment` of a passing similarity
group, canonicalized (`pipelines._steps.elect_unique_segments`) --
`vectors` normalized to a canonical frame (translated so the word's own
bbox origin is `(0, 0)`, rotated by `-angle` so it's upright, `angle` then
reset to `0.0`), `image` carried over unchanged from the real occurrence it
came from (Radon already captured it upright -- no re-render). Every other
member of the group keeps only a `SegmentMeta` (offset + rotation back to
its own real position). Only this representative is actually OCR'd
(`pipelines.sub_pipelines.ocr.recognize_unique_words` -> `OCR.Paddle_OCR.
ocr_backend.recognize_segments`); the resulting `Text` is then replayed
onto every other member's real position (`pipelines.sub_pipelines.ocr.
restore_word_texts`) without re-rendering or re-recognizing anything.

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
