# PDF layer inspector

A standalone Tkinter + PyMuPDF desktop tool for visually inspecting what is inside a PDF page —
native text, embedded images, annotations, vector drawings and clip masks — as toggleable,
filterable overlays on the rendered page. It shares no imports with `raster_parser`; it exists to
check what PyMuPDF actually extracts before (or while) writing extraction logic.

```bash
uv run python -m scripts.inspector.inspector [path/to.pdf]
```

With no argument it opens the first PDF in the repo-root `references/` folder (gitignored — drop
sample PDFs there), or a file dialog if that folder is empty.

## Modules

Several top-level modules are split into smaller sibling files and re-export them, so importing
`layers.X` / `pdf_model.X` keeps working regardless of which file `X` lives in.

- **`layers.py`** — the extensibility core. `OverlayItem` is the normalized shape every extractor
  returns (bbox always in PDF page coordinates; `quad`/`points` optionally for non-axis-aligned
  geometry; `attrs` for machine-filterable values; `metadata` for human-readable hover info).
  `LayerSpec` (one top-level checkbox) and `SubFilterSpec` (a sub-checkbox group under a layer)
  are declarative — `build_layers(pdf_model)` wires the five layers (text / images / annotations /
  drawings / clip masks) to their extractors. `filter_items()` is the one shared filter every
  layer uses (AND across sub-filter groups, OR within a group, empty selection = no restriction).
  The dataclasses live in `layer_types.py`, the seqno-rainbow colour utilities in
  `layer_colors.py`.
- **`pdf_model.py`** — the only module that calls into `pymupdf` for extraction:
  `pdf_model_core.py` (`PdfDocument` + shared geometry/matrix/colour helpers),
  `pdf_model_text.py`, `pdf_model_image.py`, `pdf_model_drawing.py` (annotations, drawings,
  `collect_drawing_colors` for the dynamic stroke/fill colour sub-filters) and
  `pdf_model_clip.py` (clip paths — only `get_drawings(extended=True)` returns them — plus
  `/SMask`, `/Mask` and stencil `/ImageMask` image masks).
- **`overlay_canvas.py`** — `PageView`: the left-pane canvas (page pixmap + overlay shapes, page
  nav, zoom, hover). Tooltip widget in `overlay_tooltip.py`, hover text formatting in
  `overlay_metadata_format.py`.
- **`control_panel.py`** — `ControlPanel`: the right-pane checkbox tree built from the layer
  registry, with collapsible sub-filter groups (checkboxes or colour swatches).
- **`inspector.py`** — entry point. `InspectorApp` wires both panels, owns the app state (page,
  zoom, per-page extraction and colour caches) and drives redraws. CLI parsing and PDF-path
  resolution live in `inspector_cli.py` (imported only from `inspector.py`'s `__main__` block to
  avoid an import cycle).

## Coordinate spaces

PyMuPDF's extraction APIs (`get_text`, `get_drawings`, `get_image_info`, `annots()`) return
coordinates in the page's **unrotated** space, whatever its `/Rotate`, while `page.get_pixmap()`
and `page.rect` are in **rotated display** space. `InspectorApp._get_display_matrix()` builds the
single transform (`page.rotation_matrix * zoom_matrix`) used for every overlay — never compute a
separate scale/rotation by hand, or overlays drift from the page image on rotated pages.

Unrotated page space is also **CropBox-relative**: coordinates are measured from the CropBox's
top-left, not the MediaBox's.

For rotated/vertical text, a word's axis-aligned bbox is not its along/normal extent;
`_make_oriented_quad` projects the bbox corners onto the span's `dir` vector to build a correctly
oriented quad — don't replace it with a bbox width/height shortcut.

## Adding a layer

1. Write `def extract_x_items(page: pymupdf.Page) -> list[OverlayItem]` (in a `pdf_model_*.py`
   file, re-exported from `pdf_model.py`).
2. Add `LayerSpec(key=..., extractor=pdf_model.extract_x_items, subfilters=[...])` to
   `layers.build_layers()`.
3. For a new filterable attribute, add a `SubFilterSpec` **and** register its `attr_getter` in
   `layers._GETTERS` — a `SubFilterSpec` missing from `_GETTERS` renders in the UI but silently
   filters nothing.

Nothing in `inspector.py`, `overlay_canvas.py` or `control_panel.py` needs to change.

## Tests

`tests/scripts_test/inspector/` (`uv run pytest tests/scripts_test/inspector`).
