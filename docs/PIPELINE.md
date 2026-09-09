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
read -> native -> vectors -> classify -> similarity -> fast -> segment -> ocr -> restore -> drawing
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
5. **`similarity`** (`build_cluster_candidates` + `group_similar_segments`, `pipelines/_steps.py`) --
   runs directly on classification's kept clusters (flattened to `list[Vector]` per cluster),
   **before** Radon or OCR ever touch them. `build_cluster_candidates` wraps each cluster as a
   `Segment(vectors, angle)`, `angle` a cheap PCA principal-axis estimate over the cluster's own
   point cloud (`_cluster_angle`) -- pure vector-geometry math, no rendering, unlike Radon's precise
   but render-dependent skew. `group_similar_segments` then groups these cluster-level `Segment`s by
   whole-page, translation+rotation-tolerant shape equivalence, normalizing rotation with that PCA
   estimate. Output: `list[list[int]]`, each inner list the indices of one similarity group of
   *clusters*.
6. **`fast`** (`detect_text_fast`, `pipelines/_steps.py`) -- scores every cluster candidate, at its
   real page position, against one whole-page FAST detection mask. **A group passes only if every
   one of its members individually exceeds `FAST_COMBINED_KEEP_THRESHOLD` (0.7)** -- not just the
   group's weakest member scraping by, every real occurrence has to look like text on its own. A
   passing group materializes its first member's canonical (translated-to-origin, PCA-rotated-to-
   upright) `vectors` -- a *whole representative cluster*, not yet split into words -- as one
   `UniqueSegment`, plus one `SegmentMeta` per member (including that first member) recording how to
   get back to that member's real position. A failing group's real (un-normalized) vectors are
   flattened into drawing output. This function is unchanged code from the pre-reorder pipeline --
   it only ever reads `Segment.vectors`/`.angle`, so calling it with cluster-level candidates instead
   of word-level ones just works.
7. **`segment`** (`segment_unique_clusters` + `segment_clusters`, `OCR/radon.py`) -- Radon-transform
   deskew, run **only on the small set of elected representative clusters** step 6 dedup down to (one
   `segment_clusters([u.vectors])` call per `UniqueSegment`) -- never on every surviving
   classification cluster, and never on a non-representative occurrence at all. For each
   representative: render, estimate skew at full precision, deskew, split into line/word crops, map
   each word's crop region back onto the representative's own `Vector`s by bbox overlap, **and
   capture that word's own deskewed pixel crop directly** (`Segment.image`) so OCR never has to
   re-render. Output: one `list[Segment]` per representative, each a word at its position *within
   that representative's own canonical frame* (not yet placed onto any real occurrence), `angle` now
   Radon's precise residual skew on top of the coarser PCA rotation already applied in step 5.
8. **`ocr`** (`recognize_unique_clusters`, `pipelines/sub_pipelines/ocr.py` ->
   `OCR/Paddle_OCR/ocr_backend.py::recognize_segments`) -- recognizes every representative's own word
   `Segment`s straight from their captured `.image` crops -- **no render happens in this step at
   all**. Recognition is a single pass per batch: PaddleOCR's own angle classifier
   (`use_angle_cls=True`) resolves the one remaining 0-vs-180-degree ambiguity Radon can't (a
   baseline is a line, not an arrow), flagged crops are rotated, then `text_recognizer` runs once
   over the batch. `direction` combines that word's own Radon residual angle with the classifier's
   flip. Output: one inner `list[Text]` per representative (that representative's own words, in its
   own canonical frame), same order/length as step 7's output.
9. **`restore`** (`restore_cluster_texts`, `pipelines/sub_pipelines/ocr.py`) -- for every
   `SegmentMeta` (one real cluster occurrence), replays *every one* of that occurrence's
   representative's word-level `Text`s through `transform_bbox`/`transform_direction`/
   `transform_point` by that meta's own `(offset, rotation)` -- the exact inverse of the PCA
   normalization applied in step 5/6 -- placing every word onto that occurrence's real page position.
   Output: one restored, real-position `Text` per (real cluster occurrence x representative word)
   pair.
10. **`drawing`** (`build_drawing_output`) -- merges classification's drops (step 4) with FAST's drops
    (step 6) into one flat, `seqno`-ordered `list[Vector]`. Nothing is reassembled -- a `Vector` was
    never decomposed, so there's no per-drawing regrouping left to do.

Final, always-on `PipelineResult` fields: `texts` (native + restored OCR, flat) and `vectors`
(drawing content, flat). Everything else is verbose-only intermediate state (see below).

## Worked dedup example

Say a page has the same dimension label -- "12.5m" -- stamped five times at different positions and
one of them rotated 90 degrees.

- **Classify** keeps all five as one or more text-candidate clusters (spatial clustering only merges
  *nearby* geometry, so these five separate stamps stay as five separate clusters).
- **Similarity** estimates each cluster's own PCA angle (cheap, no rendering) and normalizes each to
  its own canonical frame using that estimate, so the four upright stamps and the one rotated stamp
  all land in the *same* shape after normalization -- one group of five cluster indices.
- **FAST** scores all five cluster candidates at their real positions. All five must individually
  clear 0.7. If they do, the group's first member becomes one `UniqueSegment` (its whole canonical
  cluster, not yet split into words), and five `SegmentMeta` entries are recorded (one per stamp,
  including the first), each carrying that stamp's own `(offset, rotation)` back to its real
  position.
- **Segment** Radon-deskews **only that one representative cluster** -- the other four real stamps
  are never rendered or Radon-processed at all -- producing one word-level `Segment("12.5m", angle,
  image)` (Radon's precise residual angle on top of the coarser PCA rotation, plus the word's own
  captured crop).
- **OCR** recognizes that one word crop directly (no render) -- one batch slot, one recognition call
  -- producing one canonical `Text("12.5m")`.
- **Restore** replays that one `Text` across all five `SegmentMeta`s, each transformed by its own
  `(offset, rotation)`, so the final output has five `Text("12.5m")` entries at five different real
  positions/rotations -- one of them rotated 90 degrees, matching the original page -- for the cost
  of **one Radon segmentation and one OCR call instead of five**, not just one OCR call as before.

## The angle-precision invariant

There are now two distinct rotation estimates in play, used at different granularities:

- **Cluster-level (steps 5-6):** `_cluster_angle`'s PCA principal-axis estimate. Cheap (pure
  point-cloud math, no rendering) and good enough to canonicalize a whole cluster for
  similarity-grouping and FAST scoring purposes -- it does not need to be pixel-precise, since it's
  only ever used to decide *which clusters are the same shape* and to compute the exact affine
  transform back to each real occurrence (which is itself exact regardless of how the angle was
  estimated, since restore uses the same value both ways).
- **Word-level (step 7):** `Segment.angle` from Radon's fine sweep (`RADON_ANGLE_STEP_DEG`
  resolution, currently 0.25 degrees) -- **still never rounded or snapped to a multiple of 90**,
  exactly as before. This is the value the final OCR `Text.direction` (step 8) combines with
  PaddleOCR's 0/180 correction. Rounding it would silently degrade every downstream angle to blocky
  90-degree steps -- if you find yourself wanting to `round(angle / 90) * 90` anywhere in this
  pipeline, that's a regression, not a simplification.

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
| `cluster_segments` | similarity | Every cluster candidate `Segment` (PCA angle, no image) |
| `similarity_groups` | similarity | `list[list[int]]` indices into `cluster_segments` |
| `fast_result` | fast | Whole-page render/mask/per-group scores |
| `unique_segments` / `segment_metas` | fast | One `UniqueSegment` per representative *cluster*, one `SegmentMeta` per real cluster occurrence |
| `fast_dropped_vectors` | fast | Real vectors from failing groups |
| `word_segments` | segment | Radon's per-representative word-level `Segment`s (with `.image`) |
| `unique_texts` | ocr | One inner `list[Text]` per representative (that representative's own words) |
| `restored_texts` | restore | One `Text` per (real cluster occurrence x representative word), real position |
| `step_outputs` | (all) | Per-step `StepOutcome` (status/error/duration) |

## Simplifications, honestly stated

A few things below the level of this document's own claims were built with a "get the architecture
right, tune later" bias rather than fully optimized:

- **Word-to-Vector assignment** in `segment_clusters` assigns each cluster `Vector` to whichever
  Radon-detected word bbox it overlaps most (nearest-center as a fallback for no overlap at all) --
  a `Vector` spanning two words in practice would be assigned whole to one of them, not split.
- **Similarity tolerance** (`UNIQUE_CLUSTER_TOLERANCE`) is shared, as-is, between this PCA-based
  cluster-level check and (historically) a Radon-angle-based word-level check; it has not been
  independently re-tuned for the PCA estimate's coarser precision, though the two should behave
  similarly for well-separated, non-near-square cluster shapes.
- **No word-level dedup within one representative.** Every word Radon finds inside an elected
  representative cluster is recognized directly -- a representative containing several repeated
  words of its own (e.g. a table-like block of identical short labels) pays one OCR batch slot per
  word, not deduped further. Cluster-level dedup (steps 5-6) is the only dedup pass in this pipeline.
