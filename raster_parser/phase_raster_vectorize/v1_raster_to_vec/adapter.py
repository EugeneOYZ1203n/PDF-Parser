"""v1_raster_to_vec -- the original raster_parser "Type-4" raster pass as a raster_vectorize backend.

1. LibreOffice's DeleteDrawings macro removes the page's native text and vector shapes from a
   one-page copy (`clear_page`); embedded images stay.
2. If anything is left (`page_is_nonempty`), `raster_page.extract_raster_page` renders the
   residual page at `settings.ocr_dpi` (s2), splits text from drawing (s3), recognises the text
   boxes (s4) and colour-fill autotraces the drawing layer (s5).

It works on the whole page (from `page.doc_path`), so it ignores native_read's `images`. The steps
return shared `Text` / `Vector`s for the one-page cleared copy (page index 0); they get the real
page index here. Intermediates go under `settings.dump_dir`.

With `settings.on_debug_layer` / `on_debug_image` set, the intermediate rasters, boxes and the
output are also rendered as debug layers (`debug_layers.py`); that never changes the output.
"""

from __future__ import annotations


from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.models.settings import RunSettings
from raster_parser.commons.progress import note, stage


def extract(
    images: list[Image], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]:
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.debug_layers import (
        RasterDebugLayers,
    )
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.raster_page import (
        extract_raster_page,
    )
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page import (
        clear_page,
        page_is_nonempty,
    )

    with stage("p2 v1_raster_to_vec › s1 clear page (LibreOffice)"):
        work_dir, cleared = clear_page(page.doc_path, page.meta.index, dump_dir=settings.dump_dir)
    if not page_is_nonempty(cleared, 0):
        note("- p2 v1_raster_to_vec › cleared page is empty, nothing to trace")
        return [], []
    debug = RasterDebugLayers.for_settings(settings, page.meta)
    words, traced = extract_raster_page(
        cleared,
        0,
        dpi=settings.ocr_dpi,
        workers=settings.workers,
        output_dump_dir=work_dir,
        use_angle_cls=settings.use_angle_cls,
        keep_intermediates=settings.debug,
        debug=debug,
    )
    index = page.meta.index
    vectors = [v.model_copy(update=dict(page_index=index)) for v in traced]
    texts = [t.model_copy(update=dict(page_index=index)) for t in words]
    if debug is not None:
        debug.output(texts, vectors)
    return vectors, texts
