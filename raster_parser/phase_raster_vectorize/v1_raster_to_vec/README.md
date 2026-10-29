# v1_raster_to_vec (raster_vectorize)

The original raster_parser "Type-4" raster pass as a raster_vectorize backend (`p2="v1_raster_to_vec"`,
the default). It handles whatever is left on a page once native text and vectors are removed — typically embedded scans or images:

1. **Clear** — LibreOffice's `DeleteDrawings` macro removes native text and vector shapes from a
   one-page copy; embedded images stay.
2. If the cleared page is not empty, it is rendered at `ocr_dpi` and:
   - **text / drawing split** — long straight runs and line work vs text-like ink;
   - **OCR** — tiled PaddleOCR detection (plus rotated passes), box refinement, recognition;
   - **colour-fill autotrace** — the drawing layer is split by colour and each colour mask is
     traced with autotrace into filled vector paths.

Entry point: `adapter.extract(images, page, settings)`. It works from the whole page
(`page.doc_path`), so native_read's `images` are not used. `extract(enable_raster_pass=False)` skips it.
Needs LibreOffice installed.

## Behaviour changes vs the original raster_parser

| Change | Before | Now | Why |
|---|---|---|---|
| Macro path | Registered from `raster_parser/libreoffice_setup/` (doesn't exist), so the macro was never installed and every raster pass raised `FileNotFoundError`. | The `.xba` next to `s1_libreoffice_macro.py`. | Bug fix: the raster pass runs. |
| Output model | Old `Text` / `Vector`. | Shared `Text` (`source=Source.FROM_IMAGE`) / `Vector`. | Shared model; `direction` is the old `rotate` as a unit vector. |
| Traced vectors | Raw `get_drawings()` fields. | One rule (`commons/v1_commons/autotrace.traced_vector`): colours rounded to 3 decimals, stroke fields only on stroked paths, fill opacity only on filled ones. | Shared `Vector` model. |

Pipeline-wide changes to `extract()`'s output (word order, `block_no` / `seqno` ranges, native
word orientation, the new `p2` / `p3` parameters) are listed in `docs/model_migration.md`.

## Files

| File | Step | From the original raster_parser |
|---|---|---|
| `adapter.py` | entry point; sets the real page index on the steps' `Text` / `Vector`s | `main_pipeline_extract.py` (Type-4 block) |
| `raster_page.py` | runs steps 2-5 on the cleared page, writes debug images | copied: `parsing/extract_raster_page.py` |
| `s1_clear_page.py` | LibreOffice clear + `page_is_nonempty` | copied: `delete_natives/delete_native.py` |
| `s1_libreoffice_macro.py`, `delete_drawings.xba` | macro registration | copied: `delete_natives/libreoffice_setup/` |
| `s2_render_page.py` | page -> BGR + pixel-to-point transform | copied: `parsing/extract_raster_page.py` |
| `s3_text_drawing_split.py` | tiled OCR detection, text / drawing split | copied: `parsing/text_drawing_parser.py` |
| `s4_region_ocr.py` | box refinement, recognition, missed-glyph recovery | copied: `ocr/region_ocr.py` |
| `s5_colour_autotrace.py` | colour-fill autotrace of the drawing layer | copied: `autotrace/drawing_colour_fill_autotrace.py`, the wash-filter part of `helpers/geometry.py` |

Copied files hold only the code this module reaches; the logic is unchanged. The single-mask
autotrace (`conversion/png_to_vectors/`, `transforms/`, `vectors/`), the reconstructor's replay
and `normalise_crop_for_ocr` are shared with `v1_vector_class` in `commons/v1_commons/`.
