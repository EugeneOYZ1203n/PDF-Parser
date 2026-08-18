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
    """Consumes a PDF and splits it into pages.

    Pass `page_index` to restrict the opened document to just that one page
    (via `fitz.Document.select`, which drops every other page from the
    document's own page tree) -- useful for tools like the debug app that
    only ever need one page at a time and shouldn't pay to load the rest of
    a large PDF. `original_page_count` still reports the source PDF's full
    page count for display purposes even when restricted.
    """

    def __init__(self, path: str, page_index: int | None = None) -> None:
        self.path = path
        self._doc = fitz.open(path)
        self.original_page_count = self._doc.page_count
        self._page_offset = 0

        if page_index is not None:
            if not (0 <= page_index < self.original_page_count):
                raise IndexError(
                    f"page index {page_index} out of range for document "
                    f"with {self.original_page_count} page(s)"
                )
            self._doc.select([page_index])
            self._page_offset = page_index

        _LOG.info(
            "opened %s (%d/%d page(s) loaded)",
            path, self._doc.page_count, self.original_page_count,
        )

    def page_count(self) -> int:
        return self._doc.page_count

    def get_page(self, index: int) -> Page:
        count = self.page_count()
        if not (0 <= index < count):
            raise IndexError(
                f"page index {index} out of range for document with "
                f"{count} page(s)"
            )

        fitz_page = self._doc[index]
        meta = _page_meta(fitz_page, index + self._page_offset)
        _LOG.debug(
            "page %d: rotation=%d mediabox=%s",
            index,
            meta.rotation,
            meta.mediabox,
        )
        return Page(doc_path=self.path, meta=meta, fitz_page=fitz_page)

    def iter_pages(self, indices: list[int] | None = None) -> Iterator[Page]:
        if indices is None:
            indices = range(self.page_count())
        for index in indices:
            yield self.get_page(index)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
