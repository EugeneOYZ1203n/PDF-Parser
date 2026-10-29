# v2_vector_class (vector_classify)

Vector-text recognition for any vector text, stroked or filled (`p3="v2_vector_class"`, the
default). It
classifies native_read's vectors into text candidates and drawing, OCRs the candidate clusters with
PaddleOCR, and decides which vectors each read owns:

1. **Classify** (`classify_vectors.py`): per `(layer, colour, stroke width)` bucket, drop to
   drawing the oversize vectors, long regular dashed lines (collinear), repeated patterns on a
   lattice (`pattern_lattice.py`), then merge into clusters (content-stream order + spatial
   gap) and drop length outliers and crossed grids per cluster. Also collects the page's global
   line angles.
2. **Detect** (`parse.py`, `paddle_engine.py`): render each cluster on its own and run
   PaddleOCR's text detector; each detected quad's long-edge angle, snapped to a global angle
   when close, gives the rotation; the quad is cut out upright (a long quad is split first).
3. **Recognise**: PaddleOCR's 0/180 angle classifier, then recognition, page-wide in batches; a
   low-confidence read is also tried at +90 / 180 / 270 and the best wins, with dictionary
   English words counting extra (`english_words.py`).
4. **Ownership**: a vector of an OCR'd cluster is text when a non-blank quad from its own
   cluster owns it (inside, or most of its ink inside); everything else is drawing.

The full walk-through, every threshold and the vocabulary: `docs/PIPELINE.md`,
`docs/Glossary.md`.

Entry point: `parse.parse(vectors, page, settings)`.

## Settings

| `RunSettings` field | Used |
|---|---|
| `workers` | `> 1`: detection and recognition run on a process pool (`ocr_pool.py`), inputs passed as `.npz` files under `dump_dir/v2_vector_class/`. Same output as `1`. |
| `dump_dir` | that pool's spill folder (removed afterwards) |
| `ocr_dpi`, `use_angle_cls` | not used: each cluster is rendered at its own DPI (`config.OCR_DPI`, raised for small clusters), and the angle classifier always runs on the first pass |
| `on_debug_layer`, `on_debug_image`, `on_step_time` | debug output, below |

Needs `scipy` and `pyenchant` (Linux: `libenchant-2-2` + `hunspell-en-us`); see
`INTEGRATION.md`.

## Port notes

Ported from rastervec's `LatestVectorClassification` backend. The classification, OCR and
ownership logic is unchanged; the differences are in how it is driven:

| rastervec | here |
|---|---|
| `parse(vectors, page, *, verbose, compute, debug_out, on_debug_layer, on_debug_image, step_durations, keep_debug_arrays)` | `parse(vectors, page, settings)`; everything comes from `RunSettings` |
| optional Pool-2 `compute` proxy for detect / recognise jobs | `settings.workers` and `ocr_pool.OcrPool` (one spawn pool per page, same batches, results in order) |
| batch debug path (`debug_out` + `render_debug`) | streaming only (`on_debug_layer` / `on_debug_image`); tests read decisions through the private `_trace` keyword |
| `step_durations` dict | `settings.on_step_time("vector_classify", name, seconds)` |
| `import torch` before PaddleOCR (Windows DLL clash) | dropped: raster_parser has no torch |
| output_assemble sorts the merged vectors by `seqno` | `parse` returns its drawing vectors sorted by `seqno` (raster_parser's output_assemble keeps each backend's order) |
| `commons` helpers | `cluster_spatial`, `StepClock`, `render_vector_cluster`, `render_cluster_with_dynamic_dpi` copied into raster_parser's `commons` |

## Files

| File | Role |
|---|---|
| `parse.py` | entry point: stages 2-4, debug layers |
| `classify_vectors.py` | stage 1, the classification steps per bucket |
| `group_filters.py`, `cluster_filters.py` (bucket split: `commons/helpers/layer_separation.py`) | seqno merge, spatial clustering |
| `pattern_lattice.py` | repeated-pattern (lattice) detection |
| `line_geometry.py` | straight-line fits, collinear / parallel groups, crossings, quad ink tests |
| `paddle_engine.py` | PaddleOCR detect / classify / recognise, quad geometry, upright crop |
| `ocr_pool.py` | the `workers > 1` process pool |
| `english_words.py` | English-word score for picking a retry's winner |
| `config.py` | every threshold |

## Debug output

Rendered only when the callbacks are set (the pipeline report sets them, `extract()` never
does); the output is the same either way.

| Stage | Layers |
|---|---|
| `classify_NN_<step>` | `kept bbox` when the kept boxes changed; `dropped <category>` per dropped category (pattern groups one hue each); `kept pattern, too much between (N groups)` |
| `intersection` | `dropped to drawing (N)`, `flagged, kept (off-grid) (N)` (crossed-grid step) |
| `remaining` | `NN after <step> (N)`: every vector still kept after each step that dropped something, in its real paint |
| `geometry` | collinear / parallel groups in the OCR'd clusters, one hue per group, plus singletons |
| `ocr` | `detect bbox`, `passed bbox`, `failed bbox` (the detector's quads), `passed text` |
| `rotation` | `raw quad angle`, `snapped angle (N)`, `final angle` arrows; `cls flipped (N)` quads |
| `retry` | `1 retry` / `2 retry` / `3 retry` (winning rotation), `retried, kept pass 1` |
| `ownership` | `text vectors`, `drawing vectors` |
| `drawing` | `drawing vectors` (the backend's vector output) |

Debug images (`DEBUG_IMAGE_FOLDERS` in `parse.py`), each the exact input of that model call:
`for_paddle_detect/` (cluster renders), `for_rotation_correction/quad_rotation/` and
`.../paddle_classifier/` (the region rotated about the quad, the upright crop),
`for_paddle_recog/{0_retry,1_retry,2_retry,3_retry,failed}/` (the crop the winning read used).

Sub-step seconds (`on_step_time`, phase `vector_classify`): `classify_separate`, `classify_oversize`,
`classify_collinear`, `classify_pattern`, `classify_seqno`, `classify_spatial`,
`classify_outliers`, `classify_crossings`, `classify_collect`, `ocr_render`, `ocr_detect`,
`ocr_crop`, `ocr_recognize`, `ocr_assemble`, `quad_ownership`, `drawing`, and `debug_render`
(time spent rendering debug output).
