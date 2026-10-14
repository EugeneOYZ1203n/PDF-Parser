# v1VectorClass (Phase 3)

The original raster_parser "Type-2" vector parsing as a Phase 3 backend (`p3="v1VectorClass"`,
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

The algorithm itself is unchanged. These are the differences in what `extract()` returns for
this module's part of the output, all reviewed and approved during the migration.

**Specific to this module**

| Change | Before | Now | Why |
|---|---|---|---|
| Clip / group entries | `get_drawings(extended=True)`'s `"clip"` and `"group"` entries were passed through as vectors of their own (no items). | Not emitted. Each path carries its enclosing clip rectangle in `scissor`, non-rectangular clips in `clips`, and the group's blend mode / opacity in `blendmode` / `opacity`. | The new Phase 1 folds them into the paths they apply to. They were never glyph candidates, so OCR results are unaffected. |
| Colours seen by the fill filter | PyMuPDF's raw floats. | Rounded to 3 decimals (as Phase 1 extracts them). | Shared `Vector` model. A colour within 0.0005 of white or of a listed text colour can now match the filter's exact-colour checks, so very rarely a different path becomes a glyph candidate. |
| OCR text model | Old `Text` (`rotate`, `glyph_orientation`). | Shared `Text` with `direction`, `source="ocr"`. | Shared model. For OCR words `rotate` / `glyph_orientation` in the final output are the same as before, on every page rotation. |

**Pipeline-wide changes that affect this module's output** (field-level detail in
`docs/model_migration.md`)

| Change | Before | Now |
|---|---|---|
| Native words' orientation | Always `rotate=0`, `glyph_orientation="horizontal"`. | Derived from the word's direction: vertical native words report `rotate=90` (or 270) / `"vertical"`. |
| Native words on a rotated page | `glyph_orientation="horizontal"`. | Relative to the displayed page, like OCR words: text that runs across the PDF's own x-axis on a `/Rotate 90` page reads vertically, so it reports `"vertical"`; `rotate` stays relative to the page's own space (0). |
| Word order and `block_no` | OCR and native words sorted together by `block_no`; the two numberings collide (both start at 0). | Native words, then this module's OCR words, then raster-pass words; `block_no` renumbered into a disjoint range per phase. |
| Vector colours | Raw floats. | Rounded to 3 decimals. |
| `p3` parameter | — | New `extract(p3=...)`; an unknown name raises `ValueError`. |

## Files

| File | Step | Copied from (original raster_parser) |
|---|---|---|
| `parse.py` | entry point; converts the shared models in and out | `main_pipeline_extract.py` (Type-2 block) |
| `s1_fill_filter.py` | fill-glyph candidates | `rendering/pdf_render/reconstruct.py` |
| `s2_word_grouping.py` | word groups + orientation | `ocr/wordgrouping.py` |
| `s3_glyph_render.py` | render one word group | `rendering/png_render/png_renderer.py`, `reconstruct.py` |
| `s4_word_ocr.py` | crop prep + OCR per group | `parsing/parser.py` (`PdfRasterParser` methods, now functions) |
| `s4_paddle_ocr.py` | PaddleOCR wrapper | `ocr/paddle_ocr.py` |
| `s5_orientation.py` | each word's `rotate` | `rendering/pdf_render/reconstruct.py` |

Only code this module reaches was copied; the logic is unchanged.
