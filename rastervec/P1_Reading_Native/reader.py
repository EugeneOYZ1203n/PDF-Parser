"""Reader stage: opens a PDF and hands out individual pages."""
from __future__ import annotations

from typing import Iterator

import pymupdf as fitz

from rastervec.logging_setup import get_logger
from rastervec.models import Page, PageMeta

_LOG = get_logger("reader")


def _page_meta(fitz_page: "fitz.Page", index: int) -> PageMeta:
    """Build a PageMeta snapshot from a live fitz.Page.

    A free function (not a method) so it's testable directly against a
    synthetic fitz.Page without needing a full Reader/open document.
    `rotation` is normalised to [0, 360).
    """
    mediabox = fitz_page.mediabox
    return PageMeta(
        index=index,
        number=index + 1,
        mediabox=(mediabox.x0, mediabox.y0, mediabox.x1, mediabox.y1),
        rotation=int(fitz_page.rotation) % 360,
        width=mediabox.width,
        height=mediabox.height,
    )


class Reader:
    """Opens a PDF and yields `Page` objects one at a time. `page.meta.index`
    is always the source-PDF page index -- pass that same value back to
    `get_page` and you get the same page."""

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self._doc = fitz.open(path)
        except Exception as exc:  # noqa: BLE001 -- surface a clear message, not a raw fitz error
            raise ValueError(f"could not open PDF {path!r}: {exc}") from exc
        _LOG.info("opened %s (%d page(s))", path, self._doc.page_count)

    def page_count(self) -> int:
        return self._doc.page_count

    def get_page(self, index: int) -> Page:
        count = self.page_count()
        if not (0 <= index < count):
            raise IndexError(
                f"page index {index} out of range for document with {count} page(s)"
            )
        fitz_page = self._doc[index]
        meta = _page_meta(fitz_page, index)
        _LOG.debug(
            "page %d: rotation=%d mediabox=%s", index, meta.rotation, meta.mediabox,
        )
        return Page(doc_path=self.path, meta=meta, fitz_page=fitz_page)

    def iter_pages(self, indices: list[int] | None = None) -> Iterator[Page]:
        for index in range(self.page_count()) if indices is None else indices:
            yield self.get_page(index)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
