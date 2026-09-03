# Evaluation metrics (`metrics.py`)

`metrics.py` scores a Vector-Classification + OCR pipeline run against a
`LabelSet`. Unlike the legacy `evaluate.py::evaluate_pipeline` (every metric
reduced out of one greedy 1:1 label↔prediction match), here **each metric is an
independent reduction over one shared overlap graph**, and the graph models
many-to-many overlap — the common case where one ground-truth line is covered by
several predicted OCR clusters.

This phase implements **20 metrics** (a user-selected subset of a larger
catalogue — see *Not yet implemented*). Adding more is cheap: write one function
that reduces the `OverlapGraph`, add its name to `_RATIO_FIELDS` + `METRIC_GROUPS`.

---

## 1. The overlap graph

`build_overlap_graph(gt_regions, predictions, cfg)` builds `OverlapGraph`.

- **Predictions considered**: non-blank only (`ocr_blank is False` and non-empty
  text). Blank OCR readings are dropped from the graph but their clusters still
  count as *text candidates* for the classification metrics.
- **Edge**: one `OverlapEdge` per `(gt i, prediction j)` pair with non-zero bbox
  intersection, carrying
  - `inter_area` — intersection area (PDF pt²)
  - `iou` — `inter / (area(gt) + area(pred) − inter)`
  - `gt_coverage` — `inter / area(gt)` ("how much of the GT this prediction covers")
  - `pred_coverage` — `inter / area(pred)` ("how much of the prediction lands on this GT")
- **`assigned_preds_by_gt[i]`** — the **N:1 assignment**. Every prediction with
  `pred_coverage ≥ coverage_tau` is assigned to GT `i`. If none qualifies but GT
  `i` has an edge whose `iou ≥ iou_edge_min`, its single best-IoU prediction is
  assigned as a fallback. Empty ⇒ GT `i` is **not localized** (a *miss*).
- **`overlapping_preds_by_gt[i]`** — *every* non-blank prediction with any
  intersection with GT `i` (looser than assignment; used by the coverage metrics).
- `localized_gt_idxs` / `missed_gt_idxs`, and `gt_has_overlap[i]` (any edge at all).

All bboxes are `(x0, y0, x1, y1)` in PDF **unrotated MediaBox** space, y-axis
down — the same space `LabelEntry.cluster_bbox` and `TextVectorResult.bbox` use,
so no coordinate transform is needed.

---

## 2. Normalization

Two independent kinds. Both matter; keep them straight.

### 2a. Text normalization — `text_metrics.normalize_text`

Applied to **every** ground-truth and predicted string before **any** character
or word comparison:

```
normalize_text(s) = " ".join(s.split()).upper()
```

i.e. **upper-case everything**, **strip both ends**, **collapse every internal
whitespace run (space, tab, newline) to a single space**. Casing and spacing
never count as errors.

| input | `normalize_text` |
|---|---|
| `"  Setback\tline "` | `"SETBACK LINE"` |
| `"SETBACK LINE"` | `"SETBACK LINE"` |
| `"Foo   Bar\nBaz"` | `"FOO BAR BAZ"` |
| `"5 mm"` | `"5 MM"` |
| `"   "` | `""` |

Then:
- **character metrics** compare `char_multiset(s)` — the `Counter` of the
  **non-space** characters of `normalize_text(s)` (word spacing must not leak
  into a *character* score).
- **word metrics** compare `word_tokens(s)` — `normalize_text(s)` split on the
  single spaces.

`normalize_for_cer` in `evaluate.py` is now a backward-compatible **alias** of
`normalize_text` (it used to upper-case only a hand-picked "confusable" letter
set and strip *all* whitespace — replaced by the full fold above).

### 2b. Edit distance — Levenshtein, never `difflib`

`text_metrics.levenshtein(a, b)` is a pure-Python two-row DP (no dependency),
working on strings (character distance) or token lists (word distance).
`char_error_rate` / `word_error_rate` = `levenshtein(ref, hyp) / max(len(ref), 1)`
after normalization (not clamped — a hypothesis much longer than the reference
can exceed 1.0).

`difflib.SequenceMatcher` is **not** used: it scores longest matching blocks, not
edits, so a transposition or a run of single-character substitutions diverges
sharply from the intuitive error count. (The frozen legacy `evaluate_pipeline`
still calls `difflib`; that path is not maintained.)

The two `region_concat_char_accuracy_*` metrics below call `levenshtein` (see
*Character accuracy — position-aware*). Every other metric is a multiset or
geometry reduction.

### 2c. Metric normalization — page = absolute counts, aggregate = micro-average

Every metric is a **`Ratio(numerator, denominator)`**. A **page** result stores
the **absolute counts**; the metric *value* is `numerator / denominator`.

`aggregate_suite(results)` **micro-averages**: for each metric,
`Ratio(Σ numerators, Σ denominators)` over the pages where that metric is
*applicable*. It is **not** the mean of per-page ratios.

> Worked example — `page_char_multiset_recall`:
> page 1 recovers 1 of 3 GT chars → `Ratio(1, 3)`; page 2 recovers 10 of 15 →
> `Ratio(10, 15)`. Aggregate = `Ratio(11, 18)` = **0.611**, *not*
> `(1/3 + 10/15) / 2 = 0.5`. A page with more text pulls the aggregate more.

The two `*_f1` fields are **derived**: per page from that page's own
recall/precision, and at aggregate from the **aggregated** recall/precision
(harmonic mean) — never averaged from per-page f1.

**"Not applicable" (`n/a`)**: when a denominator would be 0 (empty GT, no
predicted text, no candidates, zero misses) or a required input is absent
(`clustering=None` for the miss-attribution metrics), the page stores
`Ratio(0.0, nan)`. Its `.value` is `nan`, `format_report` prints `n/a`, and the
page is **excluded** from that metric's aggregate — it neither helps nor hurts.

---

## 3. `MetricConfig`

| field | default | effect |
|---|---|---|
| `iou_edge_min` | `0.10` | Minimum IoU for a localisation edge — the N:1 fallback assignment, the `attribute_miss` group match, and the IoU term in metrics 43/44. Lower ⇒ more GT counts as "reached"/localized. |
| `coverage_tau` | `0.50` | A prediction is *assigned* to a GT (N:1) when ≥ this fraction of the **prediction's** area is inside that GT. Also the coverage threshold for "GT reached an OCR candidate" (43) and "candidate is text" (44). |

The benchmark CLI's `--iou-threshold` maps onto `iou_edge_min`.

---

## 4. The 20 metrics

`G` = GT regions, `P` = non-blank predictions, `C` = text-candidate boxes,
`τ` = `coverage_tau`, `θ` = `iou_edge_min`.

### Character accuracy — position-independent

#### `page_char_multiset_recall`
- **Layer** bag (position-independent). **Inputs** all GT text, all `P` text.
- `Cg = char_multiset(join(g.text))`, `Cp = char_multiset(join(p.text))`.
- **numerator** `Σ (Cg ∩ Cp)` (multiset intersection total). **denominator** `Σ Cg`.
- **Measures** the fraction of ground-truth characters that were read *somewhere*
  on the page, regardless of position or grouping.
- **Removes the conflation**: the legacy `characters_found_pct` gives a
  correctly-read-but-fragmented line ~0 (only one predicted bbox can match it);
  this gives it full recall.
- **`n/a`** when the page has no GT characters.
- *Example* — GT `"foo bar baz"`, preds `"FOO"`,`"BAR"`,`"BAZ"` → `Ratio(9, 9)` = 1.0.

#### `page_char_multiset_precision`
- Same `Cg`, `Cp`. **numerator** `Σ (Cg ∩ Cp)`. **denominator** `Σ Cp`.
- **Measures** the fraction of predicted characters that belong somewhere in the
  ground truth — character-level over-reading (borders / drawing strokes OCR'd
  as glyphs).
- **`n/a`** when there is no predicted text.

#### `page_char_multiset_f1` *(derived)*
- Harmonic mean of `page_char_multiset_recall` and `page_char_multiset_precision`.
  Per page from that page's two ratios; at aggregate from the aggregated two.
- **`n/a`** when either side is `n/a` or both are 0.

### Character accuracy — position-aware

Both use `text_metrics.levenshtein` (unit-cost character edit distance) after
`normalize_text`. Per gt region `i`: `hyp_i` = the text of that region's
`overlapping_preds_by_gt` predictions, concatenated in reading order (top-to-
bottom then left-to-right); `correct_i = max(0, len(norm(gt_i)) −
levenshtein(norm(gt_i), norm(hyp_i)))` (i.e. `1 − CER` clamped at 0);
`total_i = len(norm(gt_i))`. Mirrors archive `native_vs_ocr._cer` /
`cer_percentage`.

#### `region_concat_char_accuracy_all_gt`
- Over **every** gt region (`hyp_i` empty for a gt no prediction overlaps → the
  whole region counts as wrong). **numerator** `Σ_i correct_i`. **denominator**
  `Σ_i total_i`.
- **Measures** end-to-end character accuracy in reading position — a global page
  CER built from per-region localised comparisons, honest about wholesale misses.
- **`n/a`** when the page has no gt characters.
- *Example* — gt `"HELLO"`, one overlapping pred `"HELPO"` → `Ratio(4, 5)`.

#### `region_concat_char_accuracy_overlapping`
- Same, but only over gt regions with ≥ 1 overlapping non-blank prediction
  (`gt_has_overlap`). **numerator** `Σ correct_i`. **denominator** `Σ total_i`
  over those regions.
- **Measures** read quality *given* the text was localised — isolates OCR /
  cluster-merge errors from classification misses. The gap between this and
  `_all_gt` is the wholesale-miss contribution.
- **`n/a`** when no gt region was overlapped.

### Word accuracy

#### `page_word_multiset_recall` — bag
- `Wg`, `Wp` = `Counter` of `word_tokens` over all GT / all `P`.
- **numerator** `Σ (Wg ∩ Wp)`. **denominator** `Σ Wg`.
- **Measures** the fraction of ground-truth word tokens read somewhere on the
  page. The most robust "did we get the words" signal — no position, no grouping
  penalty, repeated words counted with multiplicity.
- **`n/a`** when the page has no GT tokens.

#### `page_word_multiset_precision` — bag
- **numerator** `Σ (Wg ∩ Wp)`. **denominator** `Σ Wp`.
- **Measures** spurious predicted words anywhere on the page.
- **`n/a`** when there is no predicted text.

#### `page_word_multiset_f1` *(derived)* — harmonic mean of the two word ratios.

#### `pred_text_fully_contained_in_overlapping_gt_rate` — position-aware
- **Inputs** the graph (`edges_by_pred`), GT + `P` text.
- For each `p ∈ P`: `contained` iff **some** GT with an edge to `p` has a word
  multiset that contains every token of `p` (`gt_counter[tok] ≥ p_counter[tok]`
  for all tokens of `p`).
- **numerator** `#contained predictions`. **denominator** `|P|`.
- **Measures** whether each prediction's text is fully accounted for by a GT it
  actually sits on — catches hallucinated or bled-in predicted text that bbox
  overlap alone would pass.
- **`n/a`** when `|P| = 0`.
- *Example* — GT `"foo"`, pred `"FOO ZZZ"` over it → not contained → `Ratio(0, 1)`.

#### `gt_text_word_coverage_by_overlapping_preds` — position-aware
- For each GT `i`: `bag_i` = union `Counter` of `word_tokens` over
  `overlapping_preds_by_gt[i]`; `covered_i = Σ (Counter(word_tokens(gt_i)) ∩ bag_i)`;
  `total_i = |word_tokens(gt_i)|`.
- **numerator** `Σ_i covered_i`. **denominator** `Σ_i total_i`.
- **Measures** how much of each GT line's text actually shows up in the
  predictions sitting on top of it. Unlike `page_word_multiset_recall`, it gives
  **no** credit for the right word appearing somewhere unrelated on the page.
- **`n/a`** when `Σ_i total_i = 0`.

### Bbox accuracy — localization, text-agnostic

#### `per_gt_best_single_pred_iou_mean` — localization
- **numerator** `Σ_i max(edge.iou for edges of gt i, default 0)`. **denominator** `|G|`.
- **Measures** best-case per-GT localization — the tightest single prediction for
  each GT, averaged. A missed GT contributes 0 (honest, unlike the legacy
  `bbox_accuracy` which is a matched-only mean floored near the IoU threshold).
- **`n/a`** when `|G| = 0`.
- *Example* — GT `(0,0,30,10)`, three third-width preds → each IoU 1/3 → `Ratio(1/3, 1)`.

#### `per_gt_union_pred_iou_mean` — localization
- **numerator** `Σ_i IoU(gt_i, union_bbox(assigned preds))` (0 if none assigned).
  **denominator** `|G|`.
- **Measures** per-GT localization **after** over-segmentation is stitched back
  together. The gap `per_gt_union_pred_iou_mean − per_gt_best_single_pred_iou_mean`
  is the fragmentation severity.
- **`n/a`** when `|G| = 0`.
- *Example* — same N:1 scene → union bbox equals the GT → `Ratio(1, 1)`.

#### `undetected_gt_area_ratio` — localization *(lower is better)*
- **numerator** `Σ area(gt_i)` over GT with **no** overlapping non-blank
  prediction. **denominator** `Σ area(gt_i)` over all GT.
- **Measures** the area-weighted fraction of ground truth that no prediction
  touched at all — wholesale misses. Complements the two IoU means, which say
  nothing about *undetected* GT.
- **`n/a`** when total GT area is 0.

### Rotation accuracy

#### `rotation_accuracy_localized_gt`
- Per localized GT `i`: `vote_i` = majority `rotation` over
  `assigned_preds_by_gt[i]`, ties broken by the highest-IoU assigned prediction.
- **numerator** `#{i localized : vote_i == gt_i.expected_rotation}`.
  **denominator** `|localized_gt_idxs|`.
- **Measures** rotation correctness among the GT regions that were actually
  found — N:1-aware replacement for the legacy `rotation_accuracy`.
- **`n/a`** when no GT is localized.
- **Rotation signal** is `TextVectorResult.rotation_used ∈ {0, 90, 180, 270}`.
  (There is no rotation-correction stage in the current pipeline.)

### Vector-classification accuracy

*Text candidate* = a cluster that reached OCR (blank **or** not); box = the union
bbox of its member paths (`ctx.regrouped_clusters`).

#### `classification_recall_gt_reached_ocr`
- Per GT `i`: `reached` iff some `c ∈ C` has `bbox_coverage(gt_i, c) ≥ τ` **or**
  `bbox_iou(gt_i, c) ≥ θ`.
- **numerator** `#reached GT`. **denominator** `|G|`.
- **Measures** whether classification + FAST kept this text at all — separates
  *lost in classification* from *OCR failed / misread*. The legacy
  `classification_recall` folds a blank-OCR miss into the same number.
- **`n/a`** when `|G| = 0`.

#### `classification_precision_candidate_is_text`
- Per candidate `c`: `is_text` iff some GT has `bbox_coverage(c, gt) ≥ τ` **or**
  `bbox_coverage(gt, c) ≥ τ` **or** `bbox_iou(c, gt) ≥ θ`.
- **numerator** `#candidates that are text`. **denominator** `|C|`.
- **Measures** how much of what reached OCR was actually text — non-text
  candidates waste OCR and manufacture false positives.
- **`n/a`** when `|C| = 0`.

#### `gt_miss_attributed_to_classification_frac` *(lower is better)*
#### `gt_miss_attributed_to_fast_frac` *(lower is better)*
#### `gt_miss_attributed_to_ocr_blank_frac` *(lower is better)*
#### `gt_miss_attributed_to_not_found_frac` *(lower is better)*
- Over the **missed** GT (empty assignment). `attribute_miss(gt.bbox, clustering,
  fast_dropped, ocr_failed, cfg)` walks, in pipeline order, the `role="dropped"`
  classification categories (→ `"classification:<step label>"`, earliest match
  wins), then `fast_dropped` (→ `"fast_text_detect"`), then `ocr_failed`
  (→ `"ocr_blank"`), else `"not_found"`. A group matches when
  `bbox_iou(gt.bbox, union_bbox(group)) ≥ θ`.
- Each metric: **numerator** = count of missed GT with that reason,
  **denominator** = total missed GT.
- **Measures** the stage-attributed loss funnel: of the text we lost, where did
  it go.
- **`n/a`** for all four when `clustering is None` **or** there are no misses.

#### `per_stage_miss_counts` *(diagnostic dict)*
- `Counter` of every missed GT's reason, with distinct `"classification:<label>"`
  keys per step. `{}` when `clustering is None`. Merged (summed) across pages by
  `aggregate_suite`.

### `counts` *(diagnostic, every result)*
`n_gt`, `n_pred`, `n_pred_nonblank`, `n_text_candidates`, `n_gt_localized`,
`n_gt_missed`, `n_gt_with_overlap`. Summed across pages by `aggregate_suite`.

---

## 5. Not yet implemented (the rest of the catalogue)

Structured the same way — each a pure reduction over `OverlapGraph`. Add on
demand.

| name | dimension / layer | one-liner |
|---|---|---|
| `page_char_multiset_recall_normalized` | char / bag | recall after `normalize_text` folds confusable case — isolates real misreads |
| `region_concat_char_accuracy_localized` | char / position-aware | like the shipped `_overlapping` metric but over the N:1 *assigned* preds, not every overlapping pred |
| `page_reading_order_char_similarity` | char / position-aware | `1 − char_error_rate` of all-GT vs all-pred text, both in reading order |
| `page_word_count_ratio` | word / bag | predicted token count ÷ GT token count (over/under-read direction) |
| `page_vocabulary_jaccard` | word / bag | `|set(Wg) ∩ set(Wp)| / |union|` — vocab coverage without repetition |
| `region_word_recall_localized` | word / position-aware | per GT: GT tokens present in `concat(assigned preds)` ÷ GT tokens |
| `region_word_precision_localized` | word / position-aware | ÷ assigned-pred token count (neighbour bleed-in) |
| `region_word_f1_localized` | word / position-aware | harmonic mean of the two |
| `region_exact_text_match_rate` | word / position-aware | fraction of GT with `normalize_text(gt) == normalize_text(concat)` |
| `region_word_bag_match_rate` | word / position-aware | fraction of GT with `same_word_bag(gt, concat)` |
| `word_box_recall_iou50` | word / position-aware | GT word box ↔ pred word box, IoU ≥ 0.5 **and** equal text (needs per-word GT boxes — `LabelEntry.words`) |
| `word_box_precision_iou50` | word / position-aware | ÷ predicted word-box count |
| `word_box_f1_iou50` | word / position-aware | harmonic mean |
| `text_correct_word_box_iou_mean` | word / localization | mean IoU over word pairs whose text matches |
| `per_gt_area_covered_by_preds_mean` | bbox / localization | mean per-GT area fraction covered by overlapping preds |
| `per_pred_area_inside_any_gt_mean` | bbox / localization | mean per-pred area fraction landing inside some GT |
| `localization_recall_at_iou_050` (+`_030`,`_070`) | bbox / localization | fraction of GT with ≥1 pred at IoU ≥ t |
| `localization_precision_at_iou_050` (+`_030`,`_070`) | bbox / localization | fraction of preds with ≥1 GT at IoU ≥ t |
| `localization_f1_at_iou_050` (+`_030`,`_070`) | bbox / localization | harmonic mean |
| `primary_pair_center_offset_norm_mean` | bbox / localization | greedily-matched center distance ÷ √(GT area) — translational drift |
| `over_segmentation_ratio` | bbox / localization | mean preds-per-GT in mixed connected components |
| `under_segmentation_ratio` | bbox / localization | mean GT-per-pred in mixed connected components |
| `spurious_prediction_area_ratio` | bbox / localization | pred area with no GT overlap ÷ total pred area |
| `rotation_accuracy_text_correct_gt` | rotation | `rotation_accuracy_localized_gt` restricted to GT whose concat text matches ≥ 0.8 |
| `rotation_detection_recall_rotated_gt` | rotation | of non-zero-`expected_rotation` GT, fraction that got a pred at that rotation |
| `rotation_off_by_90_rate` | rotation | of wrongly-rotated GT, fraction off by ±90 vs 180 |
| `rotation_mean_absolute_error_degrees` | rotation | mean min(Δ, 360−Δ) over localized GT |
| `rotation_confusion_matrix` | rotation / dict | `{(expected, predicted): count}` |
| `classification_f1_candidate_vs_gt` | vector-classification | harmonic mean of recall/precision |
| `text_classified_as_drawing_rate` | vector-classification | fraction of GT covered ≥τ by the drawing-vector layer **and** not a candidate |
| `drawing_classified_as_text_rate` | vector-classification | fraction of candidates over no GT but over a drawing region |
| `text_candidate_area_recall_of_gt` | vector-classification | GT area covered by the candidate layer ÷ total GT area |
| `ocr_blank_rate_over_reached_gt` | vector-classification | of GT that reached OCR, fraction whose covering cluster came back blank |
