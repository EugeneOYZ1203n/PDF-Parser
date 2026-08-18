"""Generic layer / sub-filter data model shared by every extractor and UI widget.

Extend the app by adding a new LayerSpec to LAYERS (and its extractor in
pdf_model.py) -- nothing else needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pymupdf as fitz


@dataclass
class OverlayItem:
    bbox: "fitz.Rect"
    kind: str = ""
    shape: str = "rect"          # "rect" | "polygon" | "line"
    points: list | None = None   # page-space points for polygon/line
    attrs: dict = field(default_factory=dict)
    label: str = ""


@dataclass
class SubFilterSpec:
    key: str
    label: str
    attr_getter: Callable[[OverlayItem], Any]
    static_options: list[tuple[str, str]] | None = None
    dynamic: bool = False
    render_as: str = "checkbox"  # "checkbox" | "swatch"


@dataclass
class LayerSpec:
    key: str
    label: str
    color: str
    extractor: Callable[["fitz.Page"], list[OverlayItem]]
    subfilters: list[SubFilterSpec] = field(default_factory=list)
    enabled_default: bool = False


def rgb_to_hex(color) -> str:
    if not color:
        return "#000000"
    return "#%02x%02x%02x" % tuple(round(max(0.0, min(1.0, c)) * 255) for c in color)


NONE_OPTION = ("__none__", "(none)")

ITEM_KIND_OPTIONS = [
    ("l", "Line (l)"),
    ("re", "Rectangle (re)"),
    ("qu", "Quad (qu)"),
    ("c", "Curve (c)"),
]

PATH_TYPE_OPTIONS = [
    ("f", "Fill only (f)"),
    ("s", "Stroke only (s)"),
    ("fs", "Fill + Stroke (fs)"),
]

item_kind_filter = SubFilterSpec(
    key="item_kind",
    label="Item type",
    attr_getter=lambda it: it.attrs.get("kind"),
    static_options=ITEM_KIND_OPTIONS,
)

path_type_filter = SubFilterSpec(
    key="path_type",
    label="Path type",
    attr_getter=lambda it: it.attrs.get("path_type"),
    static_options=PATH_TYPE_OPTIONS,
)

stroke_color_filter = SubFilterSpec(
    key="stroke_color",
    label="Stroke color",
    attr_getter=lambda it: it.attrs.get("stroke_color") or NONE_OPTION[0],
    dynamic=True,
    render_as="swatch",
)

fill_color_filter = SubFilterSpec(
    key="fill_color",
    label="Fill color",
    attr_getter=lambda it: it.attrs.get("fill_color") or NONE_OPTION[0],
    dynamic=True,
    render_as="swatch",
)


def filter_items(items: list[OverlayItem], active_filters: dict[str, set[str]]) -> list[OverlayItem]:
    """active_filters: {subfilter_key: set(checked_option_values)}.

    A subfilter with an empty (or missing) checked set imposes no restriction.
    Across subfilters the result is ANDed; within a subfilter, checked options OR.
    """
    result = []
    for item in items:
        keep = True
        for key, checked in active_filters.items():
            if not checked:
                continue
            getter = _GETTERS.get(key)
            if getter is None:
                continue
            value = getter(item)
            if value not in checked:
                keep = False
                break
        if keep:
            result.append(item)
    return result


_GETTERS: dict[str, Callable[[OverlayItem], Any]] = {
    "item_kind": item_kind_filter.attr_getter,
    "path_type": path_type_filter.attr_getter,
    "stroke_color": stroke_color_filter.attr_getter,
    "fill_color": fill_color_filter.attr_getter,
}


def _register_getter(spec: SubFilterSpec) -> None:
    _GETTERS[spec.key] = spec.attr_getter


def build_layers(pdf_model) -> list[LayerSpec]:
    """pdf_model is the module holding the extractor functions (avoids a
    circular import since pdf_model.py imports OverlayItem from here)."""
    return [
        LayerSpec(
            key="text",
            label="Text",
            color="#2563eb",
            extractor=pdf_model.extract_text_items,
            enabled_default=True,
        ),
        LayerSpec(
            key="images",
            label="Images",
            color="#16a34a",
            extractor=pdf_model.extract_image_items,
        ),
        LayerSpec(
            key="annotations",
            label="Annotations",
            color="#dc2626",
            extractor=pdf_model.extract_annot_items,
        ),
        LayerSpec(
            key="drawings",
            label="Drawings",
            color="#9333ea",
            extractor=pdf_model.extract_drawing_items,
            subfilters=[
                item_kind_filter,
                path_type_filter,
                stroke_color_filter,
                fill_color_filter,
            ],
        ),
    ]
