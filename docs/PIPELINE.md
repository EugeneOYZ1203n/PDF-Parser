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
  scoring) inspect `v.items` internally but always keep or drop the whole `Vector`.
- **`Text`** mirrors `get_text()`'s field surface (`"words"` + the matching `"dict"` span) and
  doubles as the OCR result type (`source="native"` or `source="ocr"`). It stores `direction` (a unit
  vector) and computes `angle()`/`quad()` from it plus `bbox` on demand -- there is no stored `angle`,
  no stored `quad`, and no `rotation_used` field. An OCR `Text`'s `direction` is the single number
  that carries its whole orientation story: Radon's precise skew angle combined with PaddleOCR's
  0/180 flip correction, folded together once, before the `Text` is ever constructed.

## The step sequence

```
read -> native -> vectors -> classify -> fast -> segment -> similarity -> ocr -> restore -> drawing
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
5. **`fast`** (`detect_text_fast`, `pipelines/_steps.py`) -- runs directly on classification's kept
   clusters (flattened to plain `list[Vector]` per cluster -- no `Segment` wrapping, no rotation
   estimate of any kind), scoring each cluster (at its real page position) against one whole-page
   FAST detection mask. **Each cluster passes or fails entirely on its own score** -- there is no
   grouping yet at this point, so there is no "every occurrence of this shape must pass" check; a
   weak render of an otherwise-common shape is dropped on its own, independent of its stronger
   siblings. A passing cluster's real, unmodified `Vector`s flow on to Radon segmentation; a
   failing cluster's vectors are flattened into drawing output.
6. **`segment`** (`OCR/radon.py::segment_clusters`, called directly from `_common.py`) --
   Radon-transform deskew, run on **every FAST-surviving cluster** (not a deduped subset -- dedup
   happens after this step now). One call, `segment_clusters(fast.passed)`, over the whole list at
   once: for each cluster, render, estimate skew at full precision, deskew, split into line/word
   crops, map each word's crop region back onto the cluster's own `Vector`s by bbox overlap, and
   capture that word's own deskewed pixel crop directly (`Segment.image`) so OCR never has to
   re-render. Output: one flat `list[Segment]`, combined across every input cluster, each entry one
   word at its real page position with Radon's precise `angle`.
7. **`similarity`** (`group_similar_segments` + `elect_unique_segments`, `pipelines/_steps.py`) --
   groups the flat word `Segment` list from step 6 by whole-page, translation+rotation-tolerant
   shape equivalence, using each `Segment`'s own Radon-precise `angle` to normalize rotation -- this
   is the *only* rotation estimate anywhere in this pipeline now (there is no separate, coarser
   pre-Radon estimate). For each resulting group, `elect_unique_segments` canonicalizes `group[0]`
   into a zero-angle `Segment` (translation+rotation-normalized `vectors`, `image` carried over
   unchanged from the real occurrence -- already upright, no re-render), and builds one `SegmentMeta`
   per group member (including the representative's own occurrence) recording how to transform its
   eventual OCR result back onto its own real position. Output: `unique_segments` (one canonical
   `Segment` per group) and `segment_metas` (one per real word occurrence).
8. **`ocr`** (`recognize_unique_words`, `pipelines/sub_pipelines/ocr.py` ->
   `OCR/Paddle_OCR/ocr_backend.py::recognize_segments`) -- recognizes every elected representative's
   own captured `.image` crop directly -- **no render happens in this step at all**. Recognition is
   a single pass per batch: PaddleOCR's own angle classifier (`use_angle_cls=True`) resolves the one
   remaining 0-vs-180-degree ambiguity Radon can't (a baseline is a line, not an arrow), flagged
   crops are rotated, then `text_recognizer` runs once over the batch. `direction` combines that
   representative's own residual angle (0.0, since `elect_unique_segments` already canonicalized it)
   with the classifier's flip. Output: `unique_texts`, one canonical-frame `Text` per
   `unique_segments` entry, same order/length.
9. **`restore`** (`restore_word_texts`, `pipelines/sub_pipelines/ocr.py`) -- for every `SegmentMeta`
   (one real word occurrence), replays `unique_texts[meta.unique_index]` through
   `transform_bbox`/`transform_direction`/`transform_point` by that meta's own `(offset, rotation)`
   -- the exact inverse of the canonicalization applied in step 7 -- placing it onto that
   occurrence's real page position. Output: one restored, real-position `Text` per real word
   occurrence.
10. **`drawing`** (`build_drawing_output`) -- merges classification's drops (step 4) with FAST's
    drops (step 5) into one flat, `seqno`-ordered `list[Vector]`. Nothing is reassembled -- a
    `Vector` was never decomposed, so there's no per-drawing regrouping left to do.

Final, always-on `PipelineResult` fields: `texts` (native + restored OCR, flat) and `vectors`
(drawing content, flat). Everything else is verbose-only intermediate state (see below).

## Worked dedup example

Say a page has the same dimension label -- "12.5m" -- stamped five times at different positions and
one of them rotated 90 degrees.

- **Classify** keeps all five as one or more text-candidate clusters (spatial clustering only merges
  *nearby* geometry, so these five separate stamps stay as five separate clusters).
- **FAST** scores each of the five clusters independently against the page mask. Each stamp passes
  or fails purely on its own render quality -- if one of the five happens to render weakly (a stray
  overlap, a thin line width) it can be dropped on its own while the other four survive; there is no
  group to average across yet.
- **Segment (Radon)** deskews **every one of the surviving stamps individually** -- Radon runs once
  per surviving occurrence here, not once per shape, since dedup hasn't happened yet. Each produces
  its own word-level `Segment("12.5m", angle, image)` (Radon's precise per-occurrence skew, plus that
  occurrence's own captured crop).
- **Similarity** normalizes each of those word `Segment`s to its own canonical frame using its own
  Radon angle, so the (up to) four upright stamps and the one rotated stamp all land in the *same*
  shape after normalization -- one group of however many stamps survived FAST. `elect_unique_segments`
  picks the first as representative and records one `SegmentMeta` per surviving stamp.
- **OCR** recognizes only that one representative's captured crop directly (no render) -- one batch
  slot, one recognition call -- producing one canonical `Text("12.5m")`.
- **Restore** replays that one `Text` across every `SegmentMeta`, each transformed by its own
  `(offset, rotation)`, so the final output has one restored `Text("12.5m")` per stamp that survived
  FAST, at its real position/rotation -- for the cost of **one OCR call instead of many**. Unlike the
  previous ordering, dedup here only saves OCR calls: Radon still ran once per surviving occurrence,
  since grouping isn't known until after Radon produces a precise angle to group by.

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
| `fast_result` | fast | Whole-page render/mask/per-cluster scores |
| `fast_passed` | fast | FAST-surviving clusters, pre-Radon (`list[list[Vector]]`) |
| `fast_dropped_vectors` | fast | Real vectors from failing clusters |
| `word_segments` | segment | Every FAST-surviving cluster's own Radon word `Segment`s, flat |
| `similarity_groups` | similarity | `list[list[int]]` indices into `word_segments` |
| `unique_segments` / `segment_metas` | similarity | One canonical `Segment` per similarity group, one `SegmentMeta` per real word occurrence |
| `unique_texts` | ocr | One `Text` per `unique_segments` entry, canonical frame |
| `restored_texts` | restore | One `Text` per real word occurrence, real position |
| `step_outputs` | (all) | Per-step `StepOutcome` (status/error/duration) |

## Simplifications, honestly stated

A few things below the level of this document's own claims were built with a "get the architecture
right, tune later" bias rather than fully optimized:

- **Word-to-Vector assignment** in `segment_clusters` assigns each cluster `Vector` to whichever
  Radon-detected word bbox it overlaps most (nearest-center as a fallback for no overlap at all) --
  a `Vector` spanning two words in practice would be assigned whole to one of them, not split.
- **FAST filters per-occurrence, not per-shape.** Because FAST now runs before any dedup exists,
  identical stamps of the same shape are each scored independently; a page with many repeats of a
  weakly-rendered shape can end up keeping some occurrences and dropping others, where the previous
  (pre-reorder) group-min check would have dropped or kept them all together.
- **No word-level dedup within one representative's own multi-word neighbors.** Similarity dedup
  operates over the flat, whole-page word list produced by step 6 -- two occurrences of the same word
  in *different* clusters dedup together same as before, but this is the only dedup pass in the
  pipeline.
