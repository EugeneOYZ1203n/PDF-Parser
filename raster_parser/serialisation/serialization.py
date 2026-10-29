"""JSON-safe encoding for PyMuPDF drawing path items."""

from __future__ import annotations

from typing import Any

import pymupdf


def _quad_to_json(value: Any) -> list[list[float]]:
    """A ``pymupdf.Quad`` as its four corners: upper-left, upper-right, lower-left, lower-right."""
    return [
        [float(value.ul.x), float(value.ul.y)],
        [float(value.ur.x), float(value.ur.y)],
        [float(value.ll.x), float(value.ll.y)],
        [float(value.lr.x), float(value.lr.y)],
    ]


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if (
        hasattr(value, "ul")
        and hasattr(value, "ur")
        and hasattr(value, "ll")
        and hasattr(value, "lr")
    ):
        return _quad_to_json(value)
    if hasattr(value, "x0") and hasattr(value, "y1"):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    return str(value)


def drawing_item_to_dict(item: tuple | list) -> dict[str, Any]:
    """One ``get_drawings()`` path segment as a JSON-safe dict."""
    op = item[0]
    return {"op": op, "args": [_to_json_value(v) for v in item[1:]]}


def drawing_items_to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [drawing_item_to_dict(item) for item in items]


def _from_json_values(values: Any) -> Any:
    if len(values) == 2:
        return pymupdf.Point(values)
    if len(values) == 4:
        return pymupdf.Rect(values)
    raise ValueError(f"Cannot convert {values} to a pymupdf geometry object")


def vector_item_to_dict(item: dict[str, Any]) -> tuple[Any]:
    op = item["op"]
    return (op,) + tuple(_from_json_values(geom) for geom in item["args"])


def vector_items_to_dict(items: list[dict[str, Any]]) -> list[tuple]:
    return [vector_item_to_dict(item) for item in items]
