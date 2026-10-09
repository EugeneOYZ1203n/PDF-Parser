import re
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import pymupdf as pymupdf

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.rendering.pdf_render.reconstruct import render_text_objects_to_pdf

XLINK_NS = "http://www.w3.org/1999/xlink"
XLINK_HREF = f"{{{XLINK_NS}}}href"

NativeWordVectorMapping = Tuple[List[Text], List[List[Vector]]]


def text_bbox(text: Text) -> Tuple[float, float, float, float]:
    return (text.x0, text.y0, text.x1, text.y1)


def extract_text_from_page(pdf_path: str, page_num: int, out_path: str) -> Path:
    """Render only native text from one page into a new PDF."""
    parser = PdfRasterParser(pdf_path, page_num)
    parser.extract_text()
    return render_text_objects_to_pdf(parser.page, parser.text, out_path)


class TextToVector:
    """Parse native-text SVG paths and group per-glyph vectors under PDF words."""

    def __init__(self, svg_path: str, text_words: List[Text] | None = None):
        root = ET.parse(svg_path).getroot()
        self._load(root, svg_path, text_words)

    @classmethod
    def from_svg_string(
        cls,
        svg: str,
        text_words: List[Text] | None = None,
    ) -> "TextToVector":
        return cls.__new__(cls)._load(ET.fromstring(svg), "<in-memory>", text_words)

    def _load(
        self,
        root: ET.Element,
        svg_path: str,
        text_words: List[Text] | None,
    ) -> "TextToVector":
        self.svg_path = svg_path
        tag = root.tag
        self.ns = tag.split("}")[0] + "}" if "}" in tag else ""
        self.vectors: List[Vector] = []
        self.mapping: Dict[int, Tuple[str, Vector]] = {}
        self.words: List[Text] = []
        self.word_vectors: List[List[Vector]] = []

        glyph_defs: Dict[str, str] = {}
        for defs in root.iter(f"{self.ns}defs"):
            for path in defs.iter(f"{self.ns}path"):
                pid = path.get("id", "")
                draw = path.get("d", "")
                if pid and draw:
                    glyph_defs[pid] = draw

        for seqno, use in enumerate(root.iter(f"{self.ns}use")):
            href = use.get(XLINK_HREF) or use.get("href", "")
            glyph_id = href.lstrip("#")
            if glyph_id not in glyph_defs:
                continue

            actual_char = use.get("data-text", "")
            d = glyph_defs[glyph_id]
            matrix = self._parse_transform(use.get("transform", ""))
            items = self._transform_path(d, matrix)
            rect = self._bounding_box(items)

            vector = Vector(
                items=items,
                type="f",
                rect=rect,
                fill=(0.0, 0.0, 0.0),
                color=None,
                width=0.0,
                even_odd=False,
                close_path=False,
                dashes="",
                fill_opacity=1.0,
                stroke_opacity=1.0,
                seqno=seqno,
                layer="",
            )
            self.vectors.append(vector)
            self.mapping[seqno] = (actual_char, vector)

        if text_words:
            self._build_word_mapping(text_words)
        return self

    def get_native_mapping(self) -> NativeWordVectorMapping:
        """Each native PDF word and its character vectors (same order as render)."""
        return self.words, self.word_vectors

    def get_vectors(self) -> List[Vector]:
        return self.vectors

    def get_mapping(self) -> Dict[int, Tuple[str, Vector]]:
        return self.mapping

    def _build_word_mapping(self, words: List[Text]) -> None:
        """Slice SVG glyphs in document order — one vector per rendered character."""
        idx = 0
        for word in words:
            text = (word.word or "").strip()
            if not text:
                continue

            n = len(text)
            chunk = self.vectors[idx : idx + n]
            if len(chunk) != n:
                raise ValueError(
                    f"glyph count mismatch at word {text!r}: "
                    f"expected {n} vectors, got {len(chunk)} "
                    f"(consumed {idx} of {len(self.vectors)} total)"
                )

            self.words.append(word)
            self.word_vectors.append(chunk)
            idx += n

        if idx != len(self.vectors):
            raise ValueError(
                f"unassigned glyphs: mapped {idx} vectors but SVG has {len(self.vectors)}"
            )

    def _parse_transform(self, transform: str) -> tuple:
        if not transform:
            return (1, 0, 0, 1, 0, 0)
        nums = transform.replace("matrix(", "").replace(")", "").replace(" ", "").split(",")
        return tuple(float(n) for n in nums)

    def _transform_path(self, d: str, matrix: tuple) -> list:
        a, b, c, dd, e, f = matrix
        if not d:
            return []

        def apply(x: float, y: float) -> pymupdf.Point:
            return pymupdf.Point(a * x + c * y + e, b * x + dd * y + f)

        items: list = []
        cx, cy = 0.0, 0.0
        sp_x, sp_y = 0.0, 0.0

        for cmd, args in self._parse_path(d):
            rel = cmd.islower()
            op = cmd.upper()

            if op == "Z":
                if (cx, cy) != (sp_x, sp_y):
                    items.append(("l", apply(cx, cy), apply(sp_x, sp_y)))
                cx, cy = sp_x, sp_y
                continue

            if op == "M":
                nx = cx + args[0] if rel else args[0]
                ny = cy + args[1] if rel else args[1]
                sp_x, sp_y = nx, ny
                cx, cy = nx, ny
                continue

            if op == "L":
                nx = cx + args[0] if rel else args[0]
                ny = cy + args[1] if rel else args[1]
                items.append(("l", apply(cx, cy), apply(nx, ny)))
                cx, cy = nx, ny
            elif op == "H":
                nx = cx + args[0] if rel else args[0]
                items.append(("l", apply(cx, cy), apply(nx, cy)))
                cx = nx
            elif op == "V":
                ny = cy + args[0] if rel else args[0]
                items.append(("l", apply(cx, cy), apply(cx, ny)))
                cy = ny
            elif op == "C":
                x1 = cx + args[0] if rel else args[0]
                y1 = cy + args[1] if rel else args[1]
                x2 = cx + args[2] if rel else args[2]
                y2 = cy + args[3] if rel else args[3]
                nx = cx + args[4] if rel else args[4]
                ny = cy + args[5] if rel else args[5]
                items.append(
                    ("c", apply(cx, cy), apply(x1, y1), apply(x2, y2), apply(nx, ny))
                )
                cx, cy = nx, ny

        return items

    def _bounding_box(self, items: list) -> Tuple[float, float, float, float]:
        xs, ys = [], []
        for item in items:
            op = item[0]
            if op == "l":
                pts = item[1:3]
            elif op == "c":
                pts = item[1:5]
            else:
                continue
            for p in pts:
                xs.append(p.x)
                ys.append(p.y)
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def _parse_path(self, d: str) -> List[Tuple[str, List[float]]]:
        if not d or not d.strip():
            return []

        tokens = re.findall(
            r"[MmLlHhVvCcSsQqTtAaZz]|"
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
            d,
        )
        counts = {
            "M": 2, "m": 2, "L": 2, "l": 2, "H": 1, "h": 1, "V": 1, "v": 1,
            "C": 6, "c": 6, "S": 4, "s": 4, "Q": 4, "q": 4, "T": 2, "t": 2,
            "A": 7, "a": 7, "Z": 0, "z": 0,
        }

        segments: List[Tuple[str, List[float]]] = []
        i = 0
        cmd: str | None = None

        while i < len(tokens):
            if re.fullmatch(r"[A-Za-z]", tokens[i]):
                cmd = tokens[i]
                i += 1
                if cmd in ("Z", "z"):
                    segments.append((cmd, []))
                    cmd = None
                continue

            if cmd is None:
                i += 1
                continue

            nums: List[float] = []
            while i < len(tokens) and not re.fullmatch(r"[A-Za-z]", tokens[i]):
                nums.append(float(tokens[i]))
                i += 1

            if cmd in ("M", "m"):
                if len(nums) >= 2:
                    segments.append((cmd, nums[:2]))
                    line_cmd = "L" if cmd == "M" else "l"
                    for k in range(2, len(nums), 2):
                        if k + 1 < len(nums):
                            segments.append((line_cmd, nums[k : k + 2]))
            else:
                n = counts[cmd]
                for k in range(0, len(nums), n):
                    chunk = nums[k : k + n]
                    if len(chunk) == n:
                        segments.append((cmd, chunk))

            cmd = None

        return segments
