# Glossary

Terms used by the Phase 3 backend v2VectorClass (`P3_Vector_Parsing/v2VectorClass/`) and its
documentation ([PIPELINE.md](PIPELINE.md)). Every term is scoped to one page's run.

## Vector

One `get_drawings()` path as Phase 1 extracts it (`commons/models/vector.py`), never split into
its items. Its `seqno` is its position in the page's content stream, i.e. its paint order. A
**straight** vector has only `"l"` items, all within `STRAIGHT_TOL_PT` of one line
(`line_geometry.straight_line`); only straight vectors take part in collinear and parallel
grouping.

## Bucket

The vectors sharing one layer, one colour (stroke + fill) and one stroke width
(`layer_color_separation`; fill-only vectors share one width key). Classification runs per
bucket and never merges vectors from different buckets.

## Collinear group / parallel group

Straight vectors of one bucket (or cluster) on the same infinite line
(`line_geometry.group_collinear`: angle within `ANGLE_TOL_DEG`, then perpendicular offset within
`COLLINEAR_OFFSET_TOL_PT`, anchored, no gap limit along the line), or merely at the same angle
(`group_parallel`). A parallel group needs at least `MIN_PARALLEL_GROUP_SIZE` members. Collinear
groups feed the collinear-drawing step and the global angles; parallel groups feed the
length-outlier step.

## Global angles

The page's likely text and drawing directions: the mean angle of every collinear group with at
least `GLOBAL_ANGLE_MIN_GROUP_SIZE` members, over every bucket, deduplicated within
`ANGLE_TOL_DEG` (`ClassificationResult.global_angles`). A detected quad's angle snaps to the
nearest one within `QUAD_ANGLE_SNAP_TOL_DEG`.

## Similarity bucket / lattice group

A **similarity bucket** holds the vectors of one bucket with the same item kinds in order and
the same per-item lengths (to `PATTERN_SIM_LENGTH_TOL_PT`), whatever their rotation
(`pattern_lattice.similarity_key`). A **lattice group** is a set of them placed on a regular
translation grid `origin + n1·v1 + n2·v2`, found by flood fill from a seed. A large lattice
group with little foreign content between its members is a pattern (stipple, hatching tiles,
symbol arrays) and goes to drawing.

## Group

The output of the seqno overlap merge (`group_filters.combine_overlapping_seq`): vectors
chained in `seqno` order while each stays within `SEQ_OVERLAP_TOLERANCE_PX` of the group's
bbox. A `list[Vector]`; the unit spatial clustering merges.

## Cluster

The output of spatial clustering (`cluster_filters.cluster_spatial_groups`): groups merged by
bbox gap (`SPATIAL_CLUSTER_THRESHOLD`) when their matching sides agree in size
(`SPATIAL_SIZE_TOLERANCE`). A `list[list[Vector]]`, keeping its member groups. The
length-outlier and crossings steps run per cluster, and each cluster is rendered and OCR'd on
its own.

## Dominant grid

Among a cluster's vectors that enough foreign pieces cross (`MIN_CROSSINGS`), the direction
(parallel or perpendicular, within `GRID_ANGLE_TOL_DEG`) holding more than `GRID_DOMINANCE` of
their line length (`line_geometry.dominant_grid`). Those vectors are a table or grid and go to
drawing; crossed vectors off that grid stay.

## Dropped / kept

Each classification step sorts its input into named categories with a role: **kept** (passed
on to the next step), **dropped** (sent to drawing, e.g. `oversize`, `drawing`, `pattern`,
`outliers`, `crossed`) or **info** (recorded for the debug layers only, e.g.
`pattern_rejected`, `crossed_off_grid`). See `classification.py`.

## Text candidate

A cluster that survived every classification step. It is not yet text: only an OCR read that
owns its vectors makes them text.

## Quad / detect quad

A text line found by PaddleOCR's detector in a cluster render: four corners of a rotated
rectangle, mapped back to page space. Its **long-edge angle**, snapped to a global angle, sets
the crop rotation. Each `Text` keeps it as `quad_points`, reordered so p0 → p1 runs along the
reading direction.

## Upright crop

The image region around a quad, rotated by the quad's angle and cut out
(`paddle_engine.upright_crop`): the image the recogniser reads.

## Score / selection score

A read's **score** is its confidence, halved for a single character, 0 when blank
(`paddle_engine.score`); a score below `RETRY_CONFIDENCE_THRESHOLD` triggers the 90 / 180 / 270
retries. The **selection score** multiplies the score by `ENGLISH_WORD_MULTIPLIER` per
dictionary English word (`english_words.selection_score`) and picks the winning attempt.

## Ownership

A non-blank quad **owns** a vector of its own cluster when the vector is inside it, or most of
its pieces touch it and most of its ink lies inside (`parse._quad_owns`). Owned vectors are
text; every other vector is drawing.

## Drawing

Everything that is not owned by a read: every vector a classification step dropped, plus every
unowned vector of the OCR'd clusters. Returned in paint order.
