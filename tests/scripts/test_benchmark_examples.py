"""benchmark_examples._crop_to_png: crops the correct region on rotated pages."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_examples.py"
_spec = importlib.util.spec_from_file_location("benchmark_examples", _MOD)
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)


def _make_pdf(path: Path, *, rotation: int) -> None:
    """A 200x100 page with a solid red square at unrotated-MediaBox (10,10,30,30)
    and nothing else nearby, then set to `rotation`."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(10, 10, 30, 30))
    shape.finish(color=(1, 0, 0), fill=(1, 0, 0))
    shape.commit()
    page.set_rotation(rotation)
    doc.save(str(path))
    doc.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_crop_to_png_lands_on_labelled_bbox_at_every_rotation(tmp_path, rotation):
    pdf_path = tmp_path / "page.pdf"
    _make_pdf(pdf_path, rotation=rotation)
    out_path = tmp_path / "crop.png"

    ok = be._crop_to_png(pdf_path, (10.0, 10.0, 30.0, 30.0), out_path, dpi=150.0)

    assert ok
    img = Image.open(out_path).convert("RGB")
    cx, cy = img.width // 2, img.height // 2
    r, g, b = img.getpixel((cx, cy))
    assert (r, g, b) == (255, 0, 0), (
        f"expected red crop center at rotation={rotation}, got {(r, g, b)}"
    )
