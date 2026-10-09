"""
PDF parser — read text, vector metadata, export SVG.

Pipeline:

  1. ``extract_text()`` — word boxes from ``get_text("words")``.
  2. ``extract_vectors()`` — metadata from ``get_drawings()``.
  3. ``extract_svg()`` — runs (1) and (2), then ``get_svg_image()`` on the same page
     (native text and vectors are left intact).
"""

from dataclasses import dataclass, asdict
import numpy as np
import pymupdf as pymupdf
from typing import List, Dict, Any, Optional, Tuple
from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.models import GlyphOrientation
from raster_parser.ocr.paddle_ocr import PaddleOcr
from raster_parser.rendering.png_render.png_renderer import PngRenderer
import cv2


@dataclass(frozen=True)
class _PreparedPaddleGroup:
    index: int
    box: tuple[float, float, float, float]
    norm: np.ndarray
    crop: np.ndarray
    large_group: bool


@dataclass
class PageContent:
    """All vector + text content extracted from a single PDF page.

    Structured SVG markup is not included here; use ``PdfRasterParser.svg`` or
    the SVG string returned from :meth:`extract_svg`.
    """

    page_index: int
    width: float
    height: float
    text: List[Text]
    vectors: List[Vector]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PdfRasterParser:
    """One PDF page: extract text and vectors, export SVG.

    ``extract_svg()`` runs ``extract_text()`` and ``extract_vectors()`` before
    ``get_svg_image()``—the page is not modified for export.

    The file on disk is unchanged unless you ``doc.save(...)``.
    """

    def __init__(self, pdf_path: str, page: int):
        self.pdf_path = pdf_path
        self.doc = pymupdf.open(pdf_path)
        self.page_index = page
        self.page = self.doc[page]

        self.text_extracted = False
        self.vectors_extracted = False
        self.svg_extracted = False

        self.text: List[Text] = []
        self.vectors: List[Vector] = []
        self.svg: Optional[str] = None
        self._paddle_ocr: Optional[PaddleOcr] = None

    def close(self) -> None:
        """Release the underlying PyMuPDF document."""
        if self.doc is not None and not self.doc.is_closed:
            self.doc.close()

    def __enter__(self) -> "PdfRasterParser":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def extract(self) -> Dict[str, Any]:
        """Run ``extract_text``, ``extract_vectors``, then ``extract_svg``."""
        self.extract_text()
        self.extract_vectors()
        self.extract_svg()
        return PageContent(
            page_index=self.page_index,
            width=float(self.page.rect.width),
            height=float(self.page.rect.height),
            text=self.text,
            vectors=self.vectors,
        ).to_dict()

    def extract_text(self) -> List[Text]:
        """Collect every word on the page via `get_text("words")`.

        Each tuple from PyMuPDF has the form
        (x0, y0, x1, y1, word, block_no, line_no, word_no)
        """
        raw_words = self.page.get_text("words")
        self.text = [Text(*w) for w in raw_words]
        self.text_extracted = True
        return self.text

    def extract_vectors(self) -> List[Vector]:
        """Collect every vector drawing on the page via ``get_drawings(extended=True)``.

        Extended mode includes clip paths and transparency groups (``type`` clip/group,
        plus ``level`` / ``scissor`` / blend fields) which ``Vector.from_drawing`` maps.
        """
        raw_drawings = self.page.get_drawings(extended=True)
        self.vectors = [Vector.from_drawing(d) for d in raw_drawings]
        self.vectors_extracted = True
        return self.vectors

    def extract_svg(self) -> str:
        """Build ``self.svg`` via ``get_svg_image()``.

        Unless ``skip_drawings=True``, runs ``extract_vectors()`` first so ``self.vectors``
        is populated alongside ``extract_text()``.
        """
        if not self.text_extracted:
            self.extract_text()
        if not self.vectors_extracted:
            self.extract_vectors()

        self.svg = self.page.get_svg_image(text_as_path=False)
        self.svg_extracted = True
        return self.svg

    def _prepare_paddle_word_group(
        self,
        index: int,
        vectors: List[Vector],
        box: tuple[float, float, float, float],
        renderer: PngRenderer,
        orientation: GlyphOrientation = GlyphOrientation.HORIZONTAL,
    ) -> Tuple[_PreparedPaddleGroup | None, List[int]]:
        if not vectors:
            return None, []
        vector_indices = [v.seqno for v in vectors]
        x0, y0, x1, y1 = box
        crop = self.rotate_crop_for_ocr(
            np.array(renderer.render_word_group(vectors)),
            orientation,
        )
        if crop.size == 0:
            return None, []
        h, w = crop.shape[:2]
        norm = self.normalise_crop_for_ocr(crop)
        large_group = w > h and (w / max(h, 1) >= 8 or len(vectors) >= 20)
        return _PreparedPaddleGroup(
            index=index,
            box=(x0, y0, x1, y1),
            norm=norm,
            crop=crop,
            large_group=large_group,
        ), vector_indices


    def _paddle_ocr_prepared(
        self,
        prep: _PreparedPaddleGroup,
        vector_indices: List[int],
        *,
        min_confidence: float,
        use_angle_cls: bool,
    ) -> Tuple[Text | None, List[int]]:
        if self._paddle_ocr is None:
            raise RuntimeError("PaddleOcr not initialized")
        x0, y0, x1, y1 = prep.box
        hits = self._paddle_ocr.ocr_image(
            prep.norm, min_confidence=min_confidence, cls=use_angle_cls
        )
        if prep.large_group:
            hits_raw = self._paddle_ocr.ocr_image(
                prep.crop, min_confidence=min_confidence, cls=use_angle_cls
            )
            best_norm = max(hits, key=lambda h: h[2]) if hits else None
            best_raw = max(hits_raw, key=lambda h: h[2]) if hits_raw else None
            if best_raw and (not best_norm or best_raw[2] >= best_norm[2]):
                best_word = best_raw[1]
            elif best_norm:
                best_word = best_norm[1]
            else:
                return None, vector_indices
        elif hits:
            best_word = max(hits, key=lambda h: h[2])[1]
        else:
            return None, vector_indices
        return Text(x0, y0, x1, y1, best_word, prep.index, 0, 0), vector_indices

    def extract_text_from_paddle_word_groups(
        self,
        vector_groups: List[List[Vector]],
        group_boxes: List[tuple[float, float, float, float]],
        *,
        dpi: int = 300,
        min_confidence: float = 0.0,
        lang: str = "en",
        use_angle_cls: bool,
    ) -> List[Tuple[Text, List[int]]]:
        """OCR pre-grouped vector crops and map OCR boxes back to page coordinates."""
        if not self.page or not vector_groups:
            return []
        if len(vector_groups) != len(group_boxes):
            raise ValueError("vector_groups and group_boxes length mismatch")

        if self._paddle_ocr is None:
            self._paddle_ocr = PaddleOcr(
                use_angle_cls=use_angle_cls,
                lang=lang,
                drop_score=min_confidence,
            )

        renderer = PngRenderer(dpi=dpi, page=self.page)
        out: List[Tuple[Text, List[int]]] = []

        for i, (vectors, box) in enumerate(zip(vector_groups, group_boxes)):
            prepared, vector_indices = self._prepare_paddle_word_group(
                i, vectors, box, renderer
            )
            if prepared is None:
                continue
            text, vector_indices = self._paddle_ocr_prepared(
                prepared,
                vector_indices,
                min_confidence=min_confidence,
                use_angle_cls=use_angle_cls,
            )
            if text is not None:
                out.append((text, vector_indices))

        return out

    @staticmethod
    def rotate_crop_for_ocr(
        crop: np.ndarray,
        orientation: GlyphOrientation,
    ) -> np.ndarray:
        """Turn vertical word-group crops upright using deduced line orientation."""
        if crop.size == 0:
            return crop
        if orientation == GlyphOrientation.VERTICAL:
            return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop

    @staticmethod
    def normalise_crop_for_ocr(crop: np.ndarray, target_height: int = 48, target_width: int = 1024) -> np.ndarray:
        if crop.size == 0:
            return crop
        
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return crop

        # 1. FIX THE IMBALANCE:
        # We want VERY tight vertical padding so the letters fill the height,
        # but GENEROUS horizontal padding so the edges don't clip.
        pad_h = max(2, int(h * 0.05))   # Only 5% vertical padding (keeps text tall)
        pad_w = max(40, int(h * 0.30))  # 30% horizontal padding cushion based on font height
        
        padded_crop = cv2.copyMakeBorder(
            crop, 
            top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[255, 255, 255] if len(crop.shape) == 3 else 255
        )
        
        # Recalculate padded dimensions
        hp, wp = padded_crop.shape[:2]
        
        # 2. PROPORTIONAL ASPECT SCALE (Drop the rigid 320px width restriction!)
        # Let the width expand naturally to accommodate long strings like your image
        scale = target_height / hp
        new_w = max(1, int(wp * scale))
        
        # 3. CRITICAL OVERRIDE: Ensure it fits PaddleOCR's recognition shape ceiling
        # If the text is incredibly long, clamp it to your max target width parameter
        final_w = min(new_w, target_width)
        
        # 4. RESIZE PRESERVING SHAPE GEOMETRY
        if scale > 1.0:
            resized = cv2.resize(padded_crop, (final_w, target_height), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(padded_crop, (final_w, target_height), interpolation=cv2.INTER_AREA)
            
        return resized  


    