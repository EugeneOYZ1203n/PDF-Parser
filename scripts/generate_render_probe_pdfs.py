"""Eyeball check for text rotation / translation / scaling in rendering.

Builds a synthetic source PDF -- one page per case: `/Rotate` 0/90/180/270
and a CropBox offset inside its MediaBox -- each holding Helvetica words at
many angles (single and multi-word). Runs Phase 1 on it, then writes one
multi-page layer PDF per check plus a `manifest.json`, so
`scripts/pipeline_report_viewer.py` overlays them on the source page:

- `expected / oriented box` -- each native word's oriented box
  (`draw.oriented_box`) as a rotated quad;
- `expected / direction` -- an arrow along each word's angle;
- `p4 / reconstructed` -- Phase 4's final page (`render_output_pdf`).

The reconstructed text should sit exactly on the source text and inside
its box at every angle, on every page.

    .venv/Scripts/python.exe scripts/generate_render_probe_pdfs.py
    .venv/Scripts/python.exe scripts/pipeline_report_viewer.py outputs/render_probe/<ts>
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.commons.paths import output_dir  # noqa: E402
from rastervec.commons.renderer.draw import (  # noqa: E402
    arrow_spec,
    oriented_box,
    quad_spec,
    render_specs_pdf,
)
from rastervec.P1_Reading_Native.native_text import extract_native_text  # noqa: E402
from rastervec.P1_Reading_Native.reader import Reader  # noqa: E402
from rastervec.P1_Reading_Native.vector_extract import extract_vectors  # noqa: E402
from rastervec.P4_Output_Organization import render_output_pdf  # noqa: E402

ANGLES = (0, 15, 30, 45, 60, 90, 135, 180, 225, 270, 315)
WORDS = ("ABC", "Panel 12")
# (label, /Rotate, CropBox or None)
CASES = (
    ("rotate 0", 0, None),
    ("rotate 90", 90, None),
    ("rotate 180", 180, None),
    ("rotate 270", 270, None),
    ("cropbox offset", 0, (60.0, 40.0, 560.0, 760.0)),
)
PAGE_W, PAGE_H = 612.0, 792.0
_BOX = (0.15, 0.39, 0.92)
_ARROW = (0.86, 0.15, 0.15)


def _box_quad(cx: float, cy: float, w: float, h: float, angle: float):
    r = math.radians(angle)
    d = (math.cos(r), math.sin(r))
    n = (-d[1], d[0])
    return [
        (cx + a * d[0] + b * n[0], cy + a * d[1] + b * n[1])
        for a, b in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    ]


def build_source(path: Path) -> None:
    doc = fitz.open()
    for _label, rotation, crop in CASES:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        for i, angle in enumerate(ANGLES):
            for j, word in enumerate(WORDS):
                cx = 150.0 + 300.0 * j
                cy = 80.0 + 62.0 * i
                width = fitz.get_text_length(word, fontsize=14)
                page.insert_text(
                    fitz.Point(cx - width / 2, cy + 5), word, fontsize=14,
                    morph=(fitz.Point(cx, cy), fitz.Matrix(-angle)),
                )
        page.draw_rect(fitz.Rect(20, 20, PAGE_W - 20, PAGE_H - 20), color=(0.6, 0.6, 0.6), width=1)
        if crop is not None:
            page.set_cropbox(fitz.Rect(*crop))
        page.set_rotation(rotation)
    doc.save(str(path))
    doc.close()


def main() -> Path:
    out = output_dir("render_probe", datetime.now().strftime("%Y%m%d-%H%M%S"))
    source = out / "source.pdf"
    build_source(source)

    layers = {
        ("expected", "oriented box", "#2563eb"): fitz.open(),
        ("expected", "direction", "#dc2626"): fitz.open(),
        ("p4", "reconstructed", "#111827"): fitz.open(),
    }
    with Reader(str(source)) as reader:
        for index in range(reader.page_count()):
            page = reader.get_page(index)
            texts = extract_native_text(page)
            vectors = extract_vectors(page)
            boxes, arrows = [], []
            for t in texts:
                cx, cy, w, h = oriented_box(t.bbox, t.angle(), text=t.text)
                boxes.append(quad_spec(_box_quad(cx, cy, w, h, t.angle()), _BOX, width=0.75))
                arrows += arrow_spec((cx, cy), t.angle(), max(12.0, 0.6 * w), _ARROW, width=1.0)
            pages = (
                render_specs_pdf(page.meta, boxes),
                render_specs_pdf(page.meta, arrows),
                render_output_pdf(page.meta, texts, vectors),
            )
            for doc, data in zip(layers.values(), pages):
                src = fitz.open("pdf", data)
                doc.insert_pdf(src)
                src.close()

    manifest_layers = []
    for (stage, label, color), doc in layers.items():
        file = f"{stage}__{label.replace(' ', '_')}.pdf"
        doc.save(str(out / file))
        doc.close()
        manifest_layers.append({"stage": stage, "layer": label, "file": file, "color": color})
    (out / "manifest.json").write_text(json.dumps({
        "source_pdf": str(source.resolve()),
        "pages": list(range(len(CASES))),
        "engine": "render_probe",
        "variant": "render_probe",
        "layers": manifest_layers,
        "cases": [label for label, _r, _c in CASES],
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f'view: .venv/Scripts/python.exe scripts/pipeline_report_viewer.py "{out}"')
    return out


if __name__ == "__main__":
    main()
