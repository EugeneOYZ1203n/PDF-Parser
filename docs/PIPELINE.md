# The `current` pipeline, end to end

This document describes the *data flow* of `rastervec.pipelines.current` -- what goes in and out of
each step and why the step exists -- independent of `CLAUDE.md`'s file-by-file architecture notes.
Read `_common.py::run_current_pipeline` alongside this; every step named below is one named call
there.

## The two core models

Everything in this pipeline is either a `Vector` or a `Text` (`rastervec/models/`):

- **`Vector`** mirrors one whole `get_drawings()` drawing. `items` is stored as PyMuPDF's own raw
  item shape (`("l", p1, p2)` / `("re", rect)` / `("qu", quad)` / `("c", p1, p2, p3, p4)`, fitz
  objects converted to plain tuples) -- **a `Vector` is never decomposed into its own items anywhere
  in this pipeline.** Filters that need item-level detail (perimeter/density/constant-spacing
  scoring) inspect `v.items` internally but always keep or drop the whole `Vector`. This replaces the
  pre-refactor design, where every drawing was exploded into one `VectorPath` per item at extraction
  time and only reassembled for final output -- that reassembly step doesn't exist anymore because
  nothing was ever taken apart.
- **`Text`** mirrors `get_text()`'s field surface (`"words"` + the matching `"dict"` span) and
  doubles as the OCR result type (`source="native"` or `source="ocr"`). It stores `direction` (a unit
  vector) and computes `angle()`/`quad()` from it plus `bbox` on demand -- there is no stored `angle`,
  no stored `quad`, and no `rotation_used` field. An OCR `Text`'s `direction` is the single number
  that carries its whole orientation story: Radon's precise skew angle combined with PaddleOCR's
  0/180 flip correction, folded together once, before the `Text` is ever constructed.

## The step sequence

```
read -> native -> vectors -> classify -> segment -> similarity -> fast -> ocr -> restore -> drawing
```

1. **`read`** -- opens the PDF, hands back one `Page` (mediabox/rotation snapshot + a live
   `fitz.Page`).
2. **`native`** (`extract_native_text`) -- one `Text` per `get_text("words")` word, `source="native"`,
   font/direction metadata joined from the best-overlapping `get_text("dict")` span. These are
   final output as-is; nothing later in the pipeline touches them.
3. **`vectors`** (`extract_vectors`) -- one `Vector` per `get_drawings()` drawing, in page order. No
   per-kind item parsing happens here -- this step is deliberately thin, just drawing-level field
   copying plus stripping fitz objects out of `items`.
4. **`classify`** (`classify_vectors`) -- the fixed 12-step Vector Classification chain (see
   `Vector_Classification/classification.py`), run separately within each `(layer, color)` bucket.
   Every `Vector` any filter step drops is drawing content; everything that survives every step is a
   *text candidate cluster* -- there is no drawing-vs-text heuristic, OCR success/failure later is
   the real signal. Output: `text_clusters` (tiered `list[list[list[Vector]]]` -- clusters of their
   member groups of `Vector`s, real nested structure, no `id()`-keyed lineage side-channel) and
   `drawing_vectors` (flat, every dropped `Vector`).
5. **`segment`** (`segment_clusters`, `OCR/radon.py`) -- Radon-transform deskew, run directly on
   classification's kept clusters (flattened to `list[Vector]` per cluster), **before** any
   dedup/FAST/OCR. For each cluster: render (transient), estimate skew at full precision, deskew,
   split into line/word crops, then map each word's crop region back onto the cluster's own
   `Vector`s by bbox overlap. Output: a flat `list[Segment]`, one per word, each `Segment(vectors,
   angle)` at its **real page position**.
6. **`similarity`** (`group_similar_segments`, `pipelines/_steps.py`) -- groups segments by
   whole-page, translation+rotation-exact shape equivalence, using each segment's own known Radon
   `angle` to normalize rotation directly instead of searching for it (a PCA-based search was the
   pre-refactor approach, needed only because similarity grouping used to run on pre-Radon clusters
   with no known angle). Output: `list[list[int]]`, each inner list the indices of one similarity
   group.
7. **`fast`** (`detect_text_fast`, `pipelines/_steps.py`) -- scores every segment, at its real page
   position, against one whole-page FAST detection mask. **A group passes only if every one of its
   members individually exceeds `FAST_COMBINED_KEEP_THRESHOLD` (0.7)** -- not just the group's
   weakest member scraping by, every real occurrence has to look like text on its own. A passing
   group materializes its first member's canonical (translated-to-origin, rotated-to-upright)
   `vectors` as one `UniqueSegment`, plus one `SegmentMeta` per member (including that first member)
   recording how to get back to that member's real position. A failing group's real (un-normalized)
   vectors are flattened into drawing output.
8. **`ocr`** (`recognize`, `OCR/Paddle_OCR/ocr_backend.py::recognize_unique_segments`) -- **only
   `UniqueSegment`s are rendered and OCR'd** -- this is the whole point of steps 6-7. Each render is
   already one isolated, near-upright word. Recognition is a single pass per batch: PaddleOCR's own
   angle classifier (`use_angle_cls=True`) resolves the one remaining 0-vs-180-degree ambiguity Radon
   can't (a baseline is a line, not an arrow), flagged crops are rotated, then `text_recognizer` runs
   once over the batch -- no more double upright-and-flipped recognition calls. Output: one
   canonical-frame `Text` per input `UniqueSegment` (same order).
9. **`restore`** (`restore_segment_texts`, `pipelines/sub_pipelines/ocr.py`) -- duplicates each
   `UniqueSegment`'s one OCR `Text` across every `SegmentMeta` that points at it, transforming
   `bbox`/`origin`/`direction` by that meta's own `(offset, rotation)` -- the exact inverse of the
   normalization applied in step 6/7 -- to place it back onto that occurrence's real page position.
   Output: one restored, real-position `Text` per original segment occurrence (not per unique shape).
10. **`drawing`** (`build_drawing_output`) -- merges classification's drops (step 4) with FAST's drops
    (step 7) into one flat, `seqno`-ordered `list[Vector]`. Nothing is reassembled -- a `Vector` was
    never decomposed, so there's no per-drawing regrouping left to do.

Final, always-on `PipelineResult` fields: `texts` (native + restored OCR, flat) and `vectors`
(drawing content, flat). Everything else is verbose-only intermediate state (see below).

## Worked dedup example

Say a page has the same dimension label -- "12.5m" -- stamped five times at different positions and
one of them rotated 90 degrees.

- **Classify** keeps all five as one or more text-candidate clusters (spatial clustering only merges
  *nearby* geometry, so these five separate stamps stay as five separate clusters).
- **Segment** Radon-deskews each cluster independently and emits one `Segment` per stamp -- five
  `Segment`s, each with its own real `vectors` and its own measured `angle` (four near 0 degrees, one
  near 90).
- **Similarity** normalizes each `Segment` to its own canonical frame using its *own* `angle` first,
  so the four upright stamps and the one rotated stamp all land in the *same* shape after
  normalization -- one group of five indices.
- **FAST** scores all five segments at their real positions. All five must individually clear 0.7. If
  they do, the group's first member becomes one `UniqueSegment`, and five `SegmentMeta` entries are
  recorded (one per stamp, including the first), each carrying that stamp's own `(offset, rotation)`
  back to its real position.
- **OCR** renders and recognizes **that one `UniqueSegment` once** -- one render, one batch slot, one
  recognition call -- producing one canonical `Text("12.5m")`.
- **Restore** duplicates that one `Text` five times, each transformed by its own `SegmentMeta`, so the
  final output has five `Text("12.5m")` entries at five different real positions/rotations -- one of
  them rotated 90 degrees, matching the original page -- for the cost of one OCR call instead of
  five.

## The angle-precision invariant

**`Segment.angle` (and every value derived from it -- `SegmentMeta.rotation`, the offsets built from
it) is always the raw, full-precision value Radon's fine sweep produces (`RADON_ANGLE_STEP_DEG`
resolution, currently 0.25 degrees). It is never rounded or snapped to a multiple of 90 anywhere in
this pipeline.** This matters twice: the similarity check (step 6) normalizes rotation using this
exact value, and the final OCR `Text.direction` (step 8) combines it with PaddleOCR's 0/180
correction. Rounding it at either point would silently degrade every downstream angle to blocky
90-degree steps -- if you find yourself wanting to `round(angle / 90) * 90` anywhere in this pipeline,
that's a regression, not a simplification.

## `PipelineResult`'s verbose-only fields

Set only when `run_pipeline(..., verbose=True)`; each corresponds to one step above and is what the
visualization notebook reads via its own `render_<stage_name>` function:

| Field | Step | What it shows |
|---|---|---|
| `native_words` | native | The native `Text`s alone (same objects as in `texts`) |
| `vectors_raw` | vectors | Every extracted `Vector`, unclassified |
| `vectors_by_layer` / `vectors_by_layer_color` | classify | The bucketing classification runs within |
| `text_clusters` | classify | Tiered surviving clusters (`list[list[list[Vector]]]`) |
| `clustering` | classify | Per-bucket `ClusteringStageResult` (every step's kept/dropped categories) |
| `classification_dropped` | classify | Flat drawing-content `Vector`s from this step alone |
| `segments` | segment | Every `Segment` (real position, real angle) |
| `similarity_groups` | similarity | `list[list[int]]` indices into `segments` |
| `fast_result` | fast | Whole-page render/mask/per-group scores |
| `unique_segments` / `segment_metas` | fast | The dedup payoff itself |
| `fast_dropped_vectors` | fast | Real vectors from failing groups |
| `unique_texts` | ocr | One `Text` per `UniqueSegment`, canonical frame |
| `restored_texts` | restore | One `Text` per original segment occurrence, real position |
| `step_outputs` | (all) | Per-step `StepOutcome` (status/error/duration) |

## Simplifications, honestly stated

A few things below the level of this document's own claims were built with a "get the architecture
right, tune later" bias rather than fully optimized:

- **Word-to-Vector assignment** in `segment_clusters` assigns each cluster `Vector` to whichever
  Radon-detected word bbox it overlaps most (nearest-center as a fallback for no overlap at all) --
  a `Vector` spanning two words in practice would be assigned whole to one of them, not split.
- **Similarity tolerance** (`UNIQUE_CLUSTER_TOLERANCE`) is reused as-is from the pre-refactor
  per-page PCA-based check; it has not been independently re-tuned for the new Radon-angle-based
  normalization, though the two should behave similarly for text that Radon deskews well.
