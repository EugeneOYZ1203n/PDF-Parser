# Migration guide: `NativePDFElements` → `ExtractResult`

`extract()` keeps its parameters but now returns an `ExtractResult` of the shared models
(`raster_parser.commons.models.Text` / `Vector`) instead of `NativePDFElements` (`TextDTO` /
`VectorDTO`). `extract_vectors_from_png` and `get_image_polys` return the shared `Vector` too. The
old classes are gone: `TextDTO`, `VectorDTO`, `NativePDFElements`, the old `Text` / `Vector` /
`GlyphOrientation`, and the `deserialize`, `deserialize_output`, `deserialize_text` and
`deserialize_vector` helpers.

## At a glance

| Before | Now |
|---|---|
| `from raster_parser import extract, NativePDFElements` | `from raster_parser import extract, ExtractResult, Text, Vector` |
| `result.words` (`list[TextDTO]`) | `result.texts` (`list[Text]`) |
| `result.vectors` (`list[VectorDTO]`) | `result.vectors` (`list[Vector]`) |
| — | `result.page` (`PageMeta`: size, `/Rotate`, MediaBox, CropBox) |
| `result.model_dump_json()` | `result.to_json()` |
| `NativePDFElements.model_validate_json(s)` | `ExtractResult.from_json(s)` |
| `deserialize_text(dto)` / `deserialize_vector(dto)` | not needed: the results are already the models (`Text.from_dict` / `Vector.from_dict` load JSON) |
| `extract_vectors_from_png(...) -> list[VectorDTO]` | `-> list[Vector]` (same for `get_image_polys`) |

Coordinates are unchanged: unrotated page space, relative to the CropBox's top-left corner.

## Text

| `TextDTO` | `Text` | Notes |
|---|---|---|
| `x0, y0, x1, y1` | `bbox` | One tuple: `x0, y0, x1, y1 = t.bbox`. |
| `word` | `text` | |
| `block_no`, `line_no`, `word_no` | same names | |
| `rotate` | `direction` | The unit vector the text runs along. |
| `glyph_orientation` | derived from `direction` + page rotation | See below. |
| — | `origin`, `font`, `font_size`, `color`, `flags`, `ascender`, `descender`, `wmode`, `source` (`"native"` / `"ocr"`), `confidence`, `quad_points`, … | New information. OCR words have no font or colour (`""`, `0.0`, `None`). |

Getting the old values back:

```python
from raster_parser.commons.helpers.text_orientation import direction_to_rotate, glyph_orientation_for

rotate = direction_to_rotate(t.direction)                               # 0 / 90 / 180 / 270
orientation = glyph_orientation_for(t.direction, result.page.rotation)  # "horizontal" / "vertical" / "other"
```

`rotate` is PyMuPDF's `insert_text(rotate=…)` angle in unrotated page space (counter-clockwise, y
down): `direction = (cos r, −sin r)`. `glyph_orientation` is how the text reads on the *displayed*
page: `vertical` at 90°/270°, `horizontal` at 0°/180°, and `other` when the direction is more than
10° off every axis.

**Behaviour changes in the values** (also listed in the v1 module READMEs):
- Native words used to be written as `rotate=0, glyph_orientation="horizontal"` whatever their
  real direction. Now a vertical native word reports `rotate=90` (or 270) and `vertical`.
- On a `/Rotate 90` page, native text that runs along the PDF's own x-axis reads vertically, so it
  reports `vertical`, like OCR words already did.

## Vector

| `VectorDTO` | `Vector` | Notes |
|---|---|---|
| `type` | `type` | `"s"`, `"f"`, `"fs"`. |
| `x0, y0, x1, y1` | `rect` | One tuple. |
| `items` (`{"op", "args"}` dicts) | `items` (plain tuples) | `("l", p1, p2)`, `("c", p1, p2, p3, p4)`, `("re", (x0, y0, x1, y1), …)`, `("qu", (ul, ur, lr, ll))`. |
| `color`, `fill` | `color`, `fill` | Rounded to 3 decimals (the old output kept PyMuPDF's raw floats). |
| `width`, `dashes`, `stroke_opacity` | same | `None` on fill-only paths (old: `0.0` / `""` / `1.0`). |
| `fill_opacity` | `fill_opacity` | `None` on stroke-only paths (old: `1.0`). |
| `line_cap` (3-tuple or `None`) | `lineCap` (3-tuple) | `Vector.line_cap` gives the single int `Shape.finish` takes. |
| `line_join` (float or `None`) | `lineJoin` (float) | |
| `close_path` | `closePath` | |
| `even_odd`, `seqno`, `layer`, `level`, `scissor`, `blendmode`, `opacity` | same | |
| `isolated`, `knockout` | same, `bool` | The DTO used `None` for "not a group". |
| — | `page_index`, `clips` | New: the page, and exact non-rectangular clip shapes. |

> **Warning: quad corner order.** A `"qu"` item's corners were `ul, ur, ll, lr` in the DTO; in
> `Vector.items` they are `ul, ur, lr, ll` (polygon order, so the corners can be drawn in
> sequence). Code that indexes quad corners must be updated, or it will silently swap two corners.

**Behaviour changes in the values:**
- `extract()` no longer outputs PyMuPDF's `"clip"` / `"group"` pseudo-entries (vectors with no
  items). Their information is on each path instead: the clip rectangle in `scissor`, exact
  non-rectangular clips in `clips`, and the group's blend mode and opacity in `blendmode` /
  `opacity`.
- Vectors from an embedded image (`get_image_polys`): a rectangle under a rotated or skewed
  placement becomes a `"qu"` with its true corners, and `scissor` is placed too. See
  `docs/UniqueCasesDuringExtractionOfVectors/`.

## JSON

`ExtractResult.to_json()` / `from_json()` (and `Text` / `Vector`'s `to_dict()` / `from_dict()`)
round-trip exactly; tuples become JSON lists and are restored on load. The layout is the models'
own field names (above), not the old DTO layout.

## Inside the library

| Code | Works on | Converted to the shared models by |
|---|---|---|
| `P3_Vector_Parsing/v1VectorClass/` | shared `Vector`; OCR words are `s4_word_ocr.OcrWord` records | `parse.py::_ocr_text` |
| `P2_Raster_To_Vec/v1RasterToVec/` | its own `s4_region_ocr.OcrWord` and `s5_colour_autotrace.TraceVector` records | `adapter.py::_ocr_text`, `s5_colour_autotrace.trace_vector_to_vector` |
| `png_vectors/` | its own `trace.TraceVector` | `trace.trace_vector_to_vector` |

`TraceVector` is the original raster_parser `Vector` (fitz item objects, unrounded colours), kept
as a private record because the autotrace steps are built around it. Converting it at the edge
keeps their output as before.

The tools that were still on the removed classes (`evaluation/native_vs_ocr/`,
`evaluation/native_vs_raster/` and the `scripts/type2_*`, `type4_pipeline.py`,
`dto_reconstruct_pipeline.py` and `raster_drawing_*` scripts) have been removed; the pipeline
report and benchmark (`docs/SCRIPTS.md`) replace them.
