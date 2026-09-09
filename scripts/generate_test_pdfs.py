"""One-off generator for the `tests/references/test_pdfs_*.pdf` fixtures.

Not part of the test suite -- run once (`python scripts/generate_test_pdfs.py`)
and `git add` the resulting PDFs; tests read them as static fixtures.

Each generated page has 3 texts (varying angle/color/position) and 6 drawing
vectors covering all four `get_drawings()` item kinds (`l`, `c`, `qu`, `re` --
at least one of each) with varying stroke color/rotation/position, at least
one filled. Each PDF uses its own fixed random seed (its 1-based index) so
regenerating reproduces byte-identical geometry.

Operational note (verified empirically -- see the model round-trip scratch
check this script's design came from): PyMuPDF's `Shape` can decompose a
`draw_quad()` call into plain `"l"` items if it shares a `finish()` call with
other primitives (observed when a fill-type pass needs explicit line
segments to rasterize the fill). To guarantee each kind survives as its own
distinct item kind, every primitive here gets its own `shape = page.new_shape()`
+ one `draw_*` call + `shape.finish()` + `shape.commit()`.
"""
from __future__ import annotations

import random
from math import cos, radians, sin
from pathlib import Path

import pymupdf as fitz

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "references"
PAGE_W, PAGE_H = 400.0, 400.0
N_PDFS = 5


def _rand_color(rng: random.Random) -> tuple[float, float, float]:
    return (round(rng.uniform(0.0, 1.0), 3), round(rng.uniform(0.0, 1.0), 3), round(rng.uniform(0.0, 1.0), 3))


def _rand_point(rng: random.Random, margin: float = 40.0) -> tuple[float, float]:
    return (rng.uniform(margin, PAGE_W - margin), rng.uniform(margin, PAGE_H - margin))


def _rotate(p: tuple[float, float], center: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    theta = radians(angle_deg)
    x, y = p[0] - center[0], p[1] - center[1]
    return (
        center[0] + x * cos(theta) - y * sin(theta),
        center[1] + x * sin(theta) + y * cos(theta),
    )


def _add_texts(page: "fitz.Page", rng: random.Random) -> None:
    # Guarantee variety: one axis-aligned, one quarter-turn, one arbitrary angle.
    angles = [0.0, 90.0, rng.uniform(10.0, 80.0)]
    rng.shuffle(angles)
    words = ["Alpha", "Beta12", "Gamma-X"]
    for text, angle in zip(words, angles):
        origin = _rand_point(rng, margin=60.0)
        color = _rand_color(rng)
        fontsize = rng.uniform(10.0, 20.0)
        center = fitz.Point(origin)
        page.insert_text(
            origin, text, fontsize=fontsize, color=color, rotate=0,
            morph=(center, fitz.Matrix(1, 1).prerotate(angle)),
        )


def _finish_one(shape: "fitz.Shape", *, stroke, fill, width: float, closed: bool = False) -> None:
    kwargs: dict = {"width": width, "closePath": closed}
    if stroke is not None:
        kwargs["color"] = stroke
    if fill is not None:
        kwargs["fill"] = fill
    shape.finish(**kwargs)
    shape.commit()


def _add_vectors(page: "fitz.Page", rng: random.Random) -> None:
    # Mandatory kinds first (one each), then 2 more random extra kinds --
    # 6 total. The mandatory "re" is always filled (guarantees >= 1 filled
    # vector); every other vector has a coin-flip chance of also being
    # filled, for extra variety across the 5 generated files.
    kinds = ["l", "c", "qu", "re"] + [rng.choice(["l", "c", "qu", "re"]) for _ in range(2)]

    for i, kind in enumerate(kinds):
        shape = page.new_shape()
        stroke = _rand_color(rng)
        filled = (kind == "re" and i == kinds.index("re")) or rng.random() < 0.3
        fill = _rand_color(rng) if filled else None
        width = rng.uniform(1.0, 3.0)
        p0 = _rand_point(rng)
        angle = rng.uniform(0.0, 360.0)

        if kind == "l":
            p1 = _rotate((p0[0] + rng.uniform(20, 60), p0[1] + rng.uniform(20, 60)), p0, angle)
            shape.draw_line(p0, p1)
            _finish_one(shape, stroke=stroke, fill=None, width=width)
        elif kind == "re":
            w, h = rng.uniform(20, 50), rng.uniform(20, 50)
            rect = fitz.Rect(p0[0], p0[1], p0[0] + w, p0[1] + h)
            shape.draw_rect(rect)
            _finish_one(shape, stroke=stroke, fill=fill, width=width, closed=True)
        elif kind == "qu":
            w, h = rng.uniform(20, 50), rng.uniform(20, 50)
            corners = [(p0[0], p0[1]), (p0[0] + w, p0[1]), (p0[0], p0[1] + h), (p0[0] + w, p0[1] + h)]
            ul, ur, ll, lr = [_rotate(c, p0, angle) for c in corners]
            shape.draw_quad(fitz.Quad(ul, ur, ll, lr))
            _finish_one(shape, stroke=stroke, fill=fill, width=width, closed=True)
        elif kind == "c":
            p1 = (p0[0] + rng.uniform(10, 30), p0[1] + rng.uniform(30, 60))
            p2 = (p0[0] + rng.uniform(30, 60), p0[1] + rng.uniform(10, 30))
            p3 = (p0[0] + rng.uniform(40, 70), p0[1] + rng.uniform(40, 70))
            shape.draw_bezier(p0, p1, p2, p3)
            _finish_one(shape, stroke=stroke, fill=None, width=width)


def generate(seed: int) -> "fitz.Document":
    rng = random.Random(seed)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _add_texts(page, rng)
    _add_vectors(page, rng)
    return doc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, N_PDFS + 1):
        doc = generate(seed=i)
        try:
            out_path = OUT_DIR / f"test_pdfs_{i}.pdf"
            doc.save(str(out_path))
            page = doc[0]
            kinds = sorted({item[0] for d in page.get_drawings() for item in d["items"]})
            filled = any(d.get("fill") is not None for d in page.get_drawings())
            print(f"{out_path.name}: {len(page.get_drawings())} drawing(s), kinds={kinds}, filled={filled}")
        finally:
            doc.close()


if __name__ == "__main__":
    main()
