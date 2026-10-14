# v1RasterToVec (Phase 2)

The original raster_parser "Type-4" raster pass as a Phase 2 backend (`p2="v1RasterToVec"`,
the default). It handles whatever is left on a page once native text and vectors are removed —
typically embedded scans or images:

1. **Clear** — LibreOffice's `DeleteDrawings` macro removes native text and vector shapes from a
   one-page copy; embedded images stay.
2. If the cleared page is not empty, it is rendered at `ocr_dpi` and:
   - **text / drawing split** — long straight runs and line work vs text-like ink;
   - **OCR** — tiled PaddleOCR detection (plus rotated passes), box refinement, recognition;
   - **colour-fill autotrace** — the drawing layer is split by colour and each colour mask is
     traced with autotrace into filled vector paths.

Entry point: `adapter.extract(images, page, settings)`. It works from the whole page
(`page.doc_path`), so Phase 1's `images` are not used. `enable_raster_pass=False` skips it.
Needs LibreOffice installed.

## Behaviour changes vs the original raster_parser

All reviewed and approved during the migration.

**Specific to this module**

| Change | Before | Now | Why |
|---|---|---|---|
| Macro registration | `import raster_parser` ran `libreoffice` and edited its config at import time. | Registered once per process, on the first LibreOffice call. | Importing the package no longer needs LibreOffice. |
| Macro path | Registered from `raster_parser/libreoffice_setup/` (doesn't exist), so the macro was never installed and every raster pass raised `FileNotFoundError`. | The real `.xba` location. | Bug fix: the raster pass runs. |
| Autotrace | `drawing_autotrace` called the `png_to_svg` / `svg_to_pdf` / `pdf_to_vectors` *modules*, so any page with a colour layer raised `TypeError: 'module' object is not callable`. | Calls the functions. | Bug fix: colour layers are traced. |
| Output model | Old `Text` / `Vector`. | Shared `Text` (`source="ocr"`) / `Vector`. | Shared model; see the pipeline-wide rows below. |

**Pipeline-wide changes that affect this module's output** (field-level detail in
`docs/model_migration.md`)

| Change | Before | Now |
|---|---|---|
| Native words on a rotated page | `glyph_orientation="horizontal"`. | Relative to the displayed page, like OCR words: text that runs across the PDF's own x-axis on a `/Rotate 90` page reads vertically, so it reports `"vertical"`; `rotate` stays relative to the page's own space (0). |
| Word order and `block_no` | Raster words got `block_no` after the largest native/OCR one, then all words were sorted by `block_no`. | Raster words come last, `block_no` renumbered after the Phase 3 range. |
| Vector order and `seqno` | Raster vectors got `seqno` after the largest drawing seqno and were appended. | Same idea: appended after Phase 3's vectors with `seqno` past Phase 3's largest. |
| Vector colours | Raw floats. | Rounded to 3 decimals. |
| `p2` parameter | — | New `extract(p2=...)`; an unknown name raises `ValueError` (even with `enable_raster_pass=False`). |

## Files

| File | Step | From the original raster_parser |
|---|---|---|
| `adapter.py` | entry point; converts the shared models out | `main_pipeline_extract.py` (Type-4 block) |
| `raster_page.py` | runs steps 2-5 on the cleared page, writes debug images | copied: `parsing/extract_raster_page.py` |
| `s1_clear_page.py` | LibreOffice clear + `page_is_nonempty` | moved: `delete_natives/delete_native.py` |
| `s1_libreoffice_macro.py`, `DeleteDrawings.xba` | macro registration | moved: `delete_natives/libreoffice_setup/` |
| `s2_render_page.py` | page -> BGR + pixel-to-point transform | copied: `parsing/extract_raster_page.py` |
| `s3_text_drawing_split.py` | tiled OCR detection, text / drawing split | copied: `parsing/text_drawing_parser.py` |
| `s4_region_ocr.py` | box refinement, recognition, missed-glyph recovery | copied: `ocr/region_ocr.py` (+ `normalise_crop_for_ocr` from `parsing/parser.py`) |
| `s5_colour_autotrace.py` | colour-fill autotrace of the drawing layer | copied: `autotrace/`, `conversion/png_to_vectors/`, `transforms/`, `vectors/`, `helpers/geometry.py`, parts of `serialisation/` and `rendering/pdf_render/` |
| `debug_layers.py` | debug layers / images for the pipeline report | new |

Copied files hold only the code this module reaches; the logic is unchanged. In
`s5_colour_autotrace.py`, `helpers/geometry.py`'s `_pt` / `_item_points` are renamed
`_geom_pt` / `_geom_item_points` (another `_item_points` lives in the same file).
`raster_parser.delete_drawings_from_pdf` (whole-document clear) now comes from
`s1_clear_page.py`; the raster pass itself doesn't use it.

## Debug output

When `RunSettings.on_debug_layer` / `on_debug_image` are set (the pipeline report does this;
`extract()` never does), the raster pass also renders its intermediate results as page-space
layers (pixels mapped through the render's own `to_points`). Nothing is rendered otherwise, the
files written under `dump_dir` are unchanged, and the output is the same either way, so this is
**not** a behaviour change.

| Stage | Layer | What it shows |
|---|---|---|
| `render` | `residual page` | the cleared page as rendered (step 2) |
| `split` | `text layer`, `drawing layer` | the final text and drawing rasters (inputs to recognition and to autotrace) |
| `detect` | `initial boxes`, `refined boxes` | step 3's detection on the whole page, and step 4's boxes on the text layer |
| `ocr` | `text` | the recognised words in their boxes and directions |
| `trace` | `vectors` | the autotraced vectors in their own colours |

Debug images: `v1_raster_boxes/` holds each refined box cut from the text layer (5 px padding,
before step 4 splits it into words).
