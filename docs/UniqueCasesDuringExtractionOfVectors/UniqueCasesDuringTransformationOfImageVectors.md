# Unique cases when transforming an image's vectors onto the page

`get_image_polys` (`raster_parser/png_vectors/extract.py`) traces each embedded image once, in
the image's own frame, then places a copy of those vectors at every spot the image is drawn. Each
placement has a matrix (`page.get_image_rects(xref, transform=True)`) that maps the image's unit
square onto the page. `_transform_vectors_to_matrix` applies it.

A vector point `(x, y)` is in the traced PDF's points, so it first becomes the unit-square point
`(x / w_pt, y / h_pt)`, where `w_pt`, `h_pt` are the image size in points at the trace DPI. Then
it goes through the placement matrix.

## What each part of a vector becomes

| Part | Before (DTO version) | Now (shared `Vector`) |
|---|---|---|
| `"l"`, `"c"` points | each point mapped | same |
| `"qu"` corners | each corner mapped, kept in DTO order (ul, ur, ll, lr) | each corner mapped, kept in `Vector` order (ul, ur, lr, ll); mapping each corner is order-independent, so the shape is the same |
| `"re"` under scale / flip / 90° turns | the axis-aligned bounds of its 4 mapped corners | same; still an exact `"re"`, orientation flag kept |
| `"re"` under any other rotation or skew | **the axis-aligned bounds of its 4 mapped corners**: a rotated square became a larger upright square | **a `"qu"` with the 4 mapped corners** (ul, ur, lr, ll) |
| `rect` (the vector's own bbox) | bounds of its 4 mapped corners | same |
| `scissor` (clip rect) | **not transformed**: it stayed in the image's frame | **transformed**: bounds of its 4 mapped corners |
| anything else | passed through | same |

"Keeps the axes" means the matrix has `b = c = 0` (scale, flip) or `a = d = 0` (90°/270°
turns, with or without a flip).

## Why a rectangle can't stay a rotated `"re"`

A PDF `re` operator draws `x y w h` in the current coordinate system, so a rectangle is
axis-aligned by definition. PyMuPDF reports `"re"` items as a `Rect` (`x0, y0, x1, y1`) in page
space, and a `Rect` cannot hold a rotation or a skew. When a rectangle is drawn under a rotated
or skewed transform, `get_drawings()` itself reports it as `"qu"`, four free corners, which can
describe any parallelogram. Placing an image's vectors follows that rule:
- if the placement keeps the axes, the result is still an exact rectangle (`"re"`);
- if it doesn't, the four true corners are kept (`"qu"`).

The old behaviour (always the upright bounds) was only wrong in the second case.

## Why `scissor` is transformed too

`scissor` is the rectangle a path is clipped to, in the same space as its items. Leaving it in
the image's frame meant a placed vector could be clipped somewhere else on the page entirely. It
is now mapped like `rect`, as the bounds of its four mapped corners. A rotated clip therefore
becomes its upright bounds, the same approximation PyMuPDF itself uses for `scissor`.

## Verification

- `tests/raster_parser/png_vectors/test_transform.py` checks every case above: scale +
  translate, flip and quarter turn keep exact `"re"`; a 30° placement gives the exact `"qu"`
  corners; points, `rect` and `scissor` are mapped; the input vectors are not modified.
- Before the old DTO version was removed, both versions were run on the extract_vectors_from_png
  test's inputs plus a 30° rotated and a mirrored placement, and compared field by field. The only
  differences were fill colours rounded to 3 decimals (the shared `Vector` rounds colours, e.g.
  `0.49803921580314636 → 0.498`). The traced test image has only line items, so the `"re"` and
  `scissor` changes did not show there.
