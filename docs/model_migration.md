# Migration guide: `NativePDFElements` → `ExtractResult`

`extract()` keeps its parameters but now returns an `ExtractResult` of the shared models
(`raster_parser.models.Text` / `Vector`) instead of `NativePDFElements` (`TextDTO` /
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
| `dto.model_copy(update=...)` | the same: `Text` and `Vector` are pydantic models too |

Coordinates are unchanged: unrotated page space, relative to the CropBox's top-left corner.

## Text

| `TextDTO` | `Text` | Notes |
|---|---|---|
| `x0, y0, x1, y1` | `bbox` | One tuple: `x0, y0, x1, y1 = t.bbox`. |
| `word` | `text` | |
| `block_no`, `line_no`, `word_no` | same names | |
| `rotate` | `direction` | The unit vector the text runs along. |
| `glyph_orientation` | derived from `direction` + page rotation | See below. |
| — | `origin`, `font`, `font_size`, `color`, `flags`, `ascender`, `descender`, `wmode`, `source` (`Source.NATIVE_TEXT` / `NATIVE_VECTOR` for text read from drawn glyphs / `FROM_IMAGE` for text read from an image), `confidence`, `quad_points`, … | New information. OCR words have no font or colour (`""`, `0.0`, `None`). |

Getting the old values back:

```python
from raster_parser.commons.helpers.text_orientation import (
    direction_to_rotate,
    glyph_orientation_for,
)

rotate = direction_to_rotate(t.direction)  # 0 / 90 / 180 / 270
orientation = glyph_orientation_for(
    t.direction, result.page.rotation
)  # "horizontal" / "vertical" / "other"
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
| — | `source` | New: `Source.NATIVE_VECTOR` for a PDF drawing, `FROM_IMAGE` for a vector traced from an image. |

> **Warning: quad corner order.** A `"qu"` item's corners were `ul, ur, ll, lr` in the DTO; in
> `Vector.items` they are `ul, ur, lr, ll` (polygon order, so the corners can be drawn in
> sequence). Code that indexes quad corners must be updated, or it will silently swap two corners.

**Behaviour changes in the values:**
- `extract()` no longer outputs PyMuPDF's `"clip"` / `"group"` pseudo-entries (vectors with no
  items). Their information is on each path instead: the clip rectangle in `scissor`, exact
  non-rectangular clips in `clips`, and the group's blend mode and opacity in `blendmode` /
  `opacity`.
- Vectors from an embedded image (`get_image_polys`): a rectangle under a rotated or skewed
  placement becomes a `"qu"` with its true corners, and `scissor` is placed too.

## Order, numbering and parameters

| Change | Before | Now |
|---|---|---|
| Word order and `block_no` | OCR, native and raster words sorted together by `block_no`; the numberings collided (several started at 0), and raster words got numbers after the largest one. | Native words, then the vector-glyph OCR words (`p3`), then the raster-pass words (`p2`); `block_no` renumbered into a separate range per phase. |
| Vector order and `seqno` | Raster-pass vectors got `seqno` after the largest drawing `seqno` and were appended. | The same: drawings first, then raster-pass vectors with `seqno` past the drawings' largest, so every `seqno` is unique. |
| `p2` / `p3` parameters | — | New `extract(p2=..., p3=...)` pick the raster-pass and vector-glyph backends (defaults `v1_raster_to_vec` / `v1_vector_class`, the original behaviour). An unknown name raises `ValueError`, even with `enable_raster_pass=False`. |

## JSON

`ExtractResult.to_json()` / `from_json()` (and `Text` / `Vector`'s `to_dict()` / `from_dict()`)
round-trip exactly; tuples become JSON lists and are restored on load. The layout is the models'
own field names (above), not the old DTO layout.

## Inside the library

Every phase, backend and `extract_vectors_from_png` works on the shared `Text` / `Vector` end to
end. Code the v1 backends and `extract_vectors_from_png` share (autotrace, the original reconstructor's replay, OCR crop
normalisation, OCR word -> `Text`) is in `commons/v1_commons/`. A traced `Vector` keeps the old
conversion rules: colours rounded to 3 decimals, stroke fields only on stroked paths, fill opacity
only on filled ones (`v1_commons.autotrace.traced_vector`).

Removed with the old classes, because they were built on them: `evaluation/native_vs_ocr/`,
`evaluation/native_vs_raster/`, `native_vs_ocr/`, the old `type2_*` / `type4_*` /
`raster_drawing_*` pipeline scripts, and their tests. The pipeline
report and benchmark (`docs/SCRIPTS.md`) replace them on the shared models.
