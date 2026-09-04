# Glossary

Standardized terminology for `rastervec`'s vector-classification pipeline
(`rastervec/Vector/vector.py`, `rastervec/Vector_Classification/classification.py`,
`rastervec/pipeline.py`). Every term below is scoped to one page's run
unless stated otherwise.

## Group

The output of the seqno-overlap-merge step (`combine_overlapping_seq`,
`VectorClassifier.cluster()`'s step 3), scoped to one `(layer, color)` bucket
(see `Vector.separate_by_layer`/`separate_by_color`). A group is the atomic
input unit handed to spatial clustering (step 6, `cluster_spatial_groups`)
-- the smallest thing the rest of the classification chain reasons about
as a single piece.

## Cluster

The final output of the whole vector-classification chain for one
`(layer, color)` bucket -- what's left after spatial clustering (step 6)
and every filter step after it. A cluster is composed of one or more
groups; `StepResult.cluster_groups` (built at the end of
`VectorClassifier.cluster()`, keyed by `id(cluster)`) records exactly which
groups a given cluster is made of, via the `lineage` dict
`cluster_spatial_groups` builds internally. Every cluster that survives the
whole chain becomes a *text candidate* (see below).

## Similarity group

A whole-page grouping of *text-candidate clusters* judged geometrically
equivalent -- same shapes, translation/rotation-tolerant (see
`VectorClassifier.group_similar_clusters` /
`cluster_filters.group_similar_clusters`, `UNIQUE_CLUSTER_TOLERANCE`) --
computed by the `unique_clusters` pipeline stage, after `text_candidates`
and before `fast_text_detect`. Clusters in the same similarity group (e.g.
repeated instances of the same label or symbol at different
positions/orientations on the page) share one FAST verdict: the
`fast_text_detect` stage takes the min of each member's own combined FAST
score across the whole group, so if any one instance scored low, every
instance judged "the same" shares that low score.

## Text candidate

A cluster that survived the entire vector-classification chain
(`VectorClassifier.cluster()`'s final "kept" category) -- handed downstream
to `unique_clusters`/`fast_text_detect`/`ocr_compare`. There is no separate
drawing-vs-text heuristic inside the classification chain itself; a text
candidate is just "whatever wasn't filtered out."

## Drawing vector

Anything that did *not* end up as real recognized text, reassembled into
`DrawingVector`s by `_run_drawing_vectors` (`pipeline.py`). This includes,
without distinction: every `role="dropped"` category any classification
filter step produced, every cluster FAST found no text signal in
(`fast_dropped`), and every cluster whose OCR resolution failed
(`ocr_failed`, see `pipeline.py`'s `_run_ocr_compare`). OCR success/failure
is the real, final signal for whether a given cluster was actually text --
everything else is drawing content.
