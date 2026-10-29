"""`extract_vectors_from_png.get_image_polys`: scratch files and worker processes."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image as PILImage

from raster_parser import extract_vectors_from_png as png_extract
from raster_parser.extract_vectors_from_png import get_image_polys


@pytest.fixture
def image_pdf(tmp_path: Path) -> Path:
    arr = np.full((40, 40, 3), 255, np.uint8)
    arr[10:30, 10:30] = 0
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(pymupdf.Rect(20, 20, 100, 100), stream=buf.getvalue())
    out = tmp_path / "img.pdf"
    doc.save(out)
    doc.close()
    return out


def test_default_scratch_space_is_not_the_working_directory(
    image_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with pymupdf.open(image_pdf) as doc:
        polys = get_image_polys(doc, doc[0])
    assert polys and list(cwd.iterdir()) == []


def test_workers_are_spawned_not_forked(
    image_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, jobs):
            return map(fn, jobs)

    def fake_spawn_pool(max_workers, *args, **kwargs):
        seen.append(max_workers)
        return FakePool()

    monkeypatch.setattr(png_extract, "spawn_pool", fake_spawn_pool)
    with pymupdf.open(image_pdf) as doc:
        get_image_polys(doc, doc[0], tmp_dir=tmp_path / "work", workers=2)
    assert seen == [2]


def test_two_workers_give_the_same_vectors_as_one(image_pdf: Path, tmp_path: Path) -> None:
    with pymupdf.open(image_pdf) as doc:
        serial = get_image_polys(doc, doc[0], tmp_dir=tmp_path / "one", workers=1)
        pooled = get_image_polys(doc, doc[0], tmp_dir=tmp_path / "two", workers=2)
    assert pooled == serial
