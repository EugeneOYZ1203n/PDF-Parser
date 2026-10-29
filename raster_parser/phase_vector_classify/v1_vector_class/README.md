# v1_vector_class (vector_classify)

The original raster_parser "Type-2" vector parsing as a vector_classify backend (`p3="v1_vector_class"`,
the default). It finds text drawn as filled vector glyphs and reads it with OCR:

1. **Fill-glyph candidates** — filled (`"f"`) vectors that aren't white and aren't layout boxes
   or rules.
2. **Word grouping** — candidates grouped into words by content-stream order (`seqno`) and
   proximity; each group gets an orientation (horizontal / vertical).
3. **OCR** — each group is rendered on its own and read with PaddleOCR; a word from a vertical
   group reports `rotate` 90 (relative to the page rotation).
4. Every vector that isn't a glyph candidate is returned unchanged, as drawing.

Entry point: `parse.parse(vectors, page, settings)`.

## Behaviour changes vs the original raster_parser

The algorithm itself is unchanged. These are the differences in this backend's output, all
reviewed and approved during the migration.

| Change | Before | Now | Why |
|---|---|---|---|
| Clip / group entries | `get_drawings(extended=True)`'s `"clip"` and `"group"` entries were passed through as vectors of their own (no items). | Not emitted. Each path carries its enclosing clip rectangle in `scissor`, non-rectangular clips in `clips`, and the group's blend mode / opacity in `blendmode` / `opacity`. | `phase_native_read` folds them into the paths they apply to. They were never glyph candidates, so OCR results are unaffected. |
| Colours seen by the fill filter | PyMuPDF's raw floats. | Rounded to 3 decimals (as `phase_native_read` extracts them). | Shared `Vector` model. A colour within 0.0005 of white or of a listed text colour can now match the filter's exact-colour checks, so very rarely a different path becomes a glyph candidate. |
| OCR text model | Old `Text` (`rotate`, `glyph_orientation`). | Shared `Text` with `direction`, `source=Source.NATIVE_VECTOR`. | Shared model; `direction` is the old `rotate` as a unit vector. |

Pipeline-wide changes to `extract()`'s output (word order, `block_no` / `seqno` ranges, native
word orientation, the new `p2` / `p3` parameters) are listed in `docs/model_migration.md`.

## Files

| File | Step | Copied from (original raster_parser) |
|---|---|---|
| `parse.py` | entry point; builds the OCR words' shared `Text` | `main_pipeline_extract.py` (Type-2 block) |
| `s1_fill_filter.py` | fill-glyph candidates | `rendering/pdf_render/reconstruct.py` |
| `s2_word_grouping.py` | word groups + orientation | `ocr/wordgrouping.py` |
| `s3_glyph_render.py` | render one word group | `rendering/png_render/png_renderer.py` (replay: `commons/v1_commons/render.py`) |
| `s4_word_ocr.py` | crop prep + OCR per group | `parsing/parser.py` (`PdfRasterParser` methods, now functions) |
| `s4_paddle_ocr.py` | PaddleOCR wrapper | `ocr/paddle_ocr.py` |
| `s5_orientation.py` | each word's `rotate` | `rendering/pdf_render/reconstruct.py` |

Only code this module reaches was copied; the logic is unchanged. Crop normalisation and the
OCR word -> `Text` conversion are shared with `v1_raster_to_vec` in `commons/v1_commons/`.

## Debug output

When `RunSettings.on_debug_layer` / `on_debug_image` are set (the pipeline report does this;
`extract()` never does), `parse` also renders each step's result. Nothing is rendered or kept
otherwise, and the output is the same either way, so this is **not** a behaviour change.

| Stage | Layer | What it shows |
|---|---|---|
| `fill_filter` | `glyph candidates`, `drawing` | step 1's split, as recoloured strokes |
| `word_grouping` | `word groups` | one box per step-2 group, green horizontal / orange vertical / purple other |
| `ocr` | `read`, `blank`, `text` | groups step 4 read a word from / got nothing from, and the words placed in their boxes |
| `orientation` | `direction` | an arrow per word along its final direction (step 5) |

Debug images: `v1_ocr_input/` holds each group's normalised crop, the array step 4 hands to
PaddleOCR first (`s4_word_ocr.ocr_word_groups(on_crop=...)`).
