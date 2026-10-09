"""`extract_clip_mask_items` -- everything on a page that is used only to clip
or mask other content, never painted itself:

- **vector clip paths** -- the `W n` paths plain `get_drawings()` drops; they
  only appear in `get_drawings(extended=True)` as `type == "clip"` entries
  (`items` = the clip path's ops, `scissor` = the effective clip rect,
  `level` = graphics-state nesting depth).
- **mask images** -- per image placement: its soft mask (`/SMask`), its
  explicit mask image (`/Mask <xref>`), or the image itself when it is a
  stencil mask (`/ImageMask true`). Each is shown at the placement's bbox,
  since that is where the mask applies.
"""
from __future__ import annotations

import pymupdf as fitz

from scripts.inspector.layers import OverlayItem
from scripts.inspector.pdf_model_core import _format_matrix
from scripts.inspector.pdf_model_drawing import _operation_item


def _rounded_rect(rect) -> tuple | None:

    if rect is None:
        return None

    rect = fitz.Rect(rect)

    return (
        round(rect.x0, 2),
        round(rect.y0, 2),
        round(rect.x1, 2),
        round(rect.y1, 2),
    )


def _extract_clip_path_items(
    page: "fitz.Page",
) -> list[OverlayItem]:

    items = []

    clip_index = 0

    for drawing in page.get_drawings(
        extended=True
    ):

        if drawing.get("type") != "clip":
            continue

        level = drawing.get(
            "level",
            None,
        )

        common_attrs = {
            "mask_source": "clip_path",
            "clip_index": clip_index,
            "level": level,
        }

        operations = drawing.get(
            "items",
            [],
        )

        common_metadata = {
            "mask_source": "clip_path",
            "clip_index": clip_index,
            "level": level,
            "scissor": _rounded_rect(
                drawing.get("scissor")
            ),
            "even_odd": drawing.get(
                "even_odd",
                None,
            ),
            "close_path": drawing.get(
                "closePath",
                None,
            ),
            "layer": drawing.get(
                "layer",
                None,
            ),
            "operation_count": len(
                operations
            ),
        }

        for item_index, it in enumerate(
            operations
        ):

            item = _operation_item(
                it,
                common_attrs,
                {
                    **common_metadata,
                    "item_index": item_index,
                    "operation": it[0],
                },
            )

            if item is not None:
                item.label = f"clip {clip_index}"
                items.append(item)

        clip_index += 1

    return items


def _xref_ref(
    doc: "fitz.Document",
    xref: int,
    key: str,
) -> int:
    """The xref a dict key points to (`/Mask 12 0 R` -> 12), else 0."""

    kind, value = doc.xref_get_key(
        xref,
        key,
    )

    if kind != "xref":
        return 0

    return int(value.split()[0])


def _mask_image_metadata(
    doc: "fitz.Document",
    mask_xref: int,
) -> dict:

    metadata = {}

    try:
        extracted = doc.extract_image(
            mask_xref
        )

        if extracted:
            metadata.update(
                {
                    "mask_width_px": extracted.get("width"),
                    "mask_height_px": extracted.get("height"),
                    "mask_bpc": extracted.get("bpc"),
                    "mask_colorspace": extracted.get("colorspace"),
                }
            )

    except Exception:
        pass

    return metadata


def _extract_mask_image_items(
    page: "fitz.Page",
) -> list[OverlayItem]:

    doc = page.parent

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

        xref = info.get(
            "xref",
            0,
        )

        bbox = fitz.Rect(
            info.get(
                "bbox",
                (0, 0, 0, 0),
            )
        )

        masks: list[tuple[str, int]] = []

        if xref:
            try:
                # get_image_info() has no "smask" key (PyMuPDF 1.28) --
                # read /SMask off the image dict like /Mask below.
                smask = _xref_ref(
                    doc,
                    xref,
                    "SMask",
                )

                if smask:
                    masks.append(("soft_mask", smask))

                explicit = _xref_ref(
                    doc,
                    xref,
                    "Mask",
                )

                if explicit:
                    masks.append(("explicit_mask", explicit))

                kind, value = doc.xref_get_key(
                    xref,
                    "ImageMask",
                )

                if kind == "bool" and value == "true":
                    masks.append(("stencil", xref))

            except Exception:
                pass

        for source, mask_xref in masks:

            metadata = {
                "mask_source": source,
                "image_index": index,
                "parent_xref": xref,
                "mask_xref": mask_xref,
                "display_width": round(bbox.width, 3),
                "display_height": round(bbox.height, 3),
                **_mask_image_metadata(
                    doc,
                    mask_xref,
                ),
            }

            transform = info.get(
                "transform",
                None,
            )

            if transform is not None:
                metadata["transform"] = _format_matrix(
                    fitz.Matrix(*transform)
                )

            items.append(
                OverlayItem(
                    bbox=bbox,
                    kind="mask_image",
                    shape="rect",
                    attrs={
                        "mask_source": source,
                        "xref": mask_xref,
                        "parent_xref": xref,
                    },
                    metadata=metadata,
                    label=f"{source} xref={mask_xref}",
                )
            )

    return items


def extract_clip_mask_items(
    page: "fitz.Page",
) -> list[OverlayItem]:

    return (
        _extract_clip_path_items(page)
        + _extract_mask_image_items(page)
    )
