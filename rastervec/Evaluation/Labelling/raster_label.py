"""Raster ground-truth generation, independent of any interactive tool.
Mirrors `native_label.py`'s split: fast pure functions here, the
interactive Tk app + CLI live under `scripts/label/raster_label.py`.

Two auto-derived pieces of raster ground truth:

- `raster_geometry_for_page` -- every `Vector` on the *original,
  unconverted* PDF page (CAD-vector text glyphs included -- a raster/
  scanned pipeline has to trace all of it regardless of what it "means"),
  decomposed via `label_schema.geometry_annotations_for_vector` into
  `GeometryAnnotation` line/curve entries carrying real paint attrs.
- `sync_text_from_vector_labels` -- reuses `vector_label`'s own
  human-vetted text labels as raster text ground truth, since raster and
  vector share the same page-space coordinates. Re-run on every
  `scripts/label/master_label.py` invocation (not gated by any "already
  done" flag) so editing vector labels later keeps raster labels in sync;
  only ever touches its own previously-synced entries (`label_id` prefixed
  `"vecsync:"`), never a genuine hand-drawn raster text entry.
- `sync_native_text_from_native_labels` -- same idea, from `native_label`'s
  auto-derived text instead, for the `native_to_raster` text type (native
  text scored against a rasterised-PDF pipeline run, not the vectorised
  one). Own prefix `"natsync:"`.

`embedded_images_for_page` enumerates the original PDF's own embedded
raster images (not a full-page render) -- the manual `raster_label` tool's
image picker: real scanned/rasterised content with no vector backing at
all is the actual CLAUDE.md-scoped-out "Raster" pipeline concern, so that's
what a human needs to hand-trace, not the whole (already vector-backed)
page.
"""
from __future__ import annotations

from dataclasses import dataclass

from rastervec.Evaluation.Labelling.label_schema import (
    GeometryAnnotation,
    LabelEntry,
    LabelSet,
    geometry_annotations_for_vector,
)
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Vector
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors

_LOG = get_logger("raster_label")

_VECSYNC_PREFIX = "vecsync:"
_NATSYNC_PREFIX = "natsync:"


def raster_geometry_for_page(pdf_path: str, page_index: int) -> list[GeometryAnnotation]:
    """`GeometryAnnotation`s for every `Vector` on `page_index` of the
    original, unconverted `pdf_path` -- flat-mapped through
    `geometry_annotations_for_vector`."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        vectors: list[Vector] = extract_vectors(page)
    out: list[GeometryAnnotation] = []
    for v in vectors:
        out.extend(geometry_annotations_for_vector(v))
    _LOG.debug(
        "raster_geometry_for_page: page %d, %d vector(s) -> %d annotation(s)",
        page_index, len(vectors), len(out),
    )
    return out


def sync_text_from_vector_labels(
    raster_labels: LabelSet, vector_labels: LabelSet, page_indices: list[int],
) -> None:
    """Mutates `raster_labels.entries` in place: drops every existing entry
    whose `label_id` starts with `"vecsync:"` on the given pages, then
    re-adds one fresh entry per `vector_labels` entry on those pages
    (`label_id=f"vecsync:{v.label_id}"`, `source="raster"`, text/
    cluster_bbox/expected_rotation/vector_signatures carried over
    verbatim). Never touches a genuine manually-added raster text entry
    (a real `uuid4()` label id)."""
    pages = set(page_indices)
    raster_labels.entries = [
        e for e in raster_labels.entries
        if not (e.page_index in pages and e.label_id.startswith(_VECSYNC_PREFIX))
    ]
    synced = [
        LabelEntry(
            page_index=v.page_index, cluster_bbox=v.cluster_bbox,
            cluster_signature=v.cluster_signature,
            label_id=f"{_VECSYNC_PREFIX}{v.label_id}",
            text=v.text, source="raster", expected_rotation=v.expected_rotation,
            vector_signatures=list(v.vector_signatures),
        )
        for v in vector_labels.entries if v.page_index in pages
    ]
    raster_labels.entries.extend(synced)


def sync_native_text_from_native_labels(
    raster_labels: LabelSet, native_labels: LabelSet, page_indices: list[int],
) -> None:
    """Mutates `raster_labels.entries` in place: drops every existing entry
    whose `label_id` starts with `"natsync:"` on the given pages, then
    re-adds one fresh entry per `native_labels` entry on those pages
    (`label_id=f"natsync:{n.label_id}"`, `source="native"`, text/
    cluster_bbox/expected_rotation carried over verbatim -- native labels
    never populate `vector_signatures`). Ground truth for the
    `native_to_raster` text type (native text scored against a rasterised-
    PDF pipeline run): the same native text region, just tracked as ground
    truth for the rasterised page instead of the vectorised one. Mirrors
    `sync_text_from_vector_labels`; never touches a genuine manually-added
    raster entry."""
    pages = set(page_indices)
    raster_labels.entries = [
        e for e in raster_labels.entries
        if not (e.page_index in pages and e.label_id.startswith(_NATSYNC_PREFIX))
    ]
    synced = [
        LabelEntry(
            page_index=n.page_index, cluster_bbox=n.cluster_bbox,
            cluster_signature=n.cluster_signature,
            label_id=f"{_NATSYNC_PREFIX}{n.label_id}",
            text=n.text, source="raster", expected_rotation=n.expected_rotation,
        )
        for n in native_labels.entries if n.page_index in pages
    ]
    raster_labels.entries.extend(synced)
    _LOG.debug(
        "sync_text_from_vector_labels: %d page(s), %d entry(ies) synced",
        len(pages), len(synced),
    )


@dataclass
class ImageRegion:
    page_index: int
    bbox: tuple[float, float, float, float]
    xref: int
    width: int
    height: int


def embedded_images_for_page(pdf_path: str, page_index: int) -> list[ImageRegion]:
    """Every embedded raster image placement on `page_index`, via
    `page.get_image_info(xrefs=True)` (page-space bbox, placement-aware --
    same API `Evaluation/inspector/pdf_model.py::extract_image_items` uses
    for the inspector tool's own image layer)."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        try:
            infos = page.fitz_page.get_image_info(xrefs=True)
        except Exception:  # noqa: BLE001 -- a malformed image stream shouldn't crash the picker
            infos = []
    regions = [
        ImageRegion(
            page_index=page_index,
            bbox=tuple(info.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            xref=info.get("xref", 0),
            width=info.get("width", 0),
            height=info.get("height", 0),
        )
        for info in infos
    ]
    _LOG.debug("embedded_images_for_page: page %d, %d image(s)", page_index, len(regions))
    return regions
