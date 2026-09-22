"""`extract_image_items` -- embedded raster image placements."""
from __future__ import annotations

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem
from rastervec.Evaluation.inspector.pdf_model_core import (
    _format_matrix,
    _matrix_rotation,
    _matrix_scale,
)


def extract_image_items(
    page: "fitz.Page",
) -> list[OverlayItem]:
    """
    Extract actual image placements.

    get_image_info(xrefs=True) is preferable to get_images() here
    because it provides placement information, transformation data,
    resolution and mask information.
    """

    items = []

    try:
        image_infos = page.get_image_info(
            xrefs=True
        )
    except Exception:
        image_infos = []

    for index, info in enumerate(
        image_infos
    ):

        bbox = fitz.Rect(
            info.get(
                "bbox",
                (0, 0, 0, 0),
            )
        )

        width_px = info.get(
            "width",
            0,
        )

        height_px = info.get(
            "height",
            0,
        )

        xref = info.get(
            "xref",
            0,
        )

        smask = info.get(
            "smask",
            0,
        )

        transform = info.get(
            "transform",
            None,
        )

        metadata = {
            "image_index": index,

            "xref": xref,

            "width_px": width_px,
            "height_px": height_px,

            "display_width": round(
                bbox.width,
                3,
            ),

            "display_height": round(
                bbox.height,
                3,
            ),

            "display_area": round(
                bbox.get_area(),
                3,
            ),

            "pixel_count": (
                width_px * height_px
                if width_px and height_px
                else None
            ),

            "aspect_ratio": round(
                width_px / height_px,
                4,
            )
            if height_px
            else None,

            "colorspace": info.get(
                "cs-name",
                info.get(
                    "colorspace",
                    None,
                ),
            ),

            "bpc": info.get(
                "bpc",
                None,
            ),

            "xres": info.get(
                "xres",
                None,
            ),

            "yres": info.get(
                "yres",
                None,
            ),

            "has_mask": info.get(
                "has-mask",
                False,
            ),

            "smask": smask or None,

            "digest": info.get(
                "digest",
                None,
            ),
        }


        if transform is not None:

            # get_image_info() returns "transform" as a plain 6-tuple,
            # not a fitz.Matrix, so it must be wrapped before use.
            transform = fitz.Matrix(*transform)

            metadata[
                "transform"
            ] = _format_matrix(
                transform
            )

            metadata[
                "rotation"
            ] = round(
                _matrix_rotation(
                    transform
                ),
                3,
            )

            sx, sy = _matrix_scale(
                transform
            )

            metadata[
                "scale"
            ] = (
                round(sx, 5),
                round(sy, 5),
            )

        # --------------------------------------------------------------
        # Physical display DPI
        #
        # PDF points are 1/72 inch.
        # --------------------------------------------------------------

        if bbox.width > 0 and width_px:
            dpi_x = (
                width_px
                / bbox.width
                * 72.0
            )
        else:
            dpi_x = None

        if bbox.height > 0 and height_px:
            dpi_y = (
                height_px
                / bbox.height
                * 72.0
            )
        else:
            dpi_y = None

        metadata[
            "effective_dpi"
        ] = (
            round(dpi_x, 2)
            if dpi_x is not None
            else None,
            round(dpi_y, 2)
            if dpi_y is not None
            else None,
        )


        if xref:
            try:
                extracted = page.parent.extract_image(
                    xref
                )

                if extracted:
                    metadata[
                        "extension"
                    ] = extracted.get(
                        "ext"
                    )

                    metadata[
                        "compressed_size"
                    ] = extracted.get(
                        "size"
                    )

                    metadata[
                        "compressed_size_kb"
                    ] = round(
                        extracted.get(
                            "size",
                            0,
                        )
                        / 1024,
                        2,
                    )

                    metadata[
                        "image_colorspace"
                    ] = extracted.get(
                        "colorspace"
                    )

                    metadata[
                        "image_bpc"
                    ] = extracted.get(
                        "bpc"
                    )

            except Exception:
                pass

        attrs = {
            "xref": xref,
            "colorspace": metadata.get(
                "colorspace"
            ),
            "has_mask": metadata.get(
                "has_mask"
            ),
        }

        items.append(
            OverlayItem(
                bbox=bbox,
                kind="image",
                shape="rect",
                attrs=attrs,
                metadata=metadata,
                label=f"xref={xref}",
            )
        )

    return items
