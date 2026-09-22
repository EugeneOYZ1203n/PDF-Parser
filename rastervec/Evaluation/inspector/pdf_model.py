"""The inspector's fitz-extraction surface -- re-exports `PdfDocument` and
every `extract_*`/`collect_drawing_colors` function from their own files
(`pdf_model_core.py`/`pdf_model_text.py`/`pdf_model_image.py`/
`pdf_model_drawing.py`), split by extraction concern. Callers (`inspector.py`,
`layers.py::build_layers`) use this module's own name, e.g.
`pdf_model.extract_text_items` -- keep every name below importable from here
even if the split above changes."""
from __future__ import annotations

from rastervec.Evaluation.inspector.pdf_model_core import (  # noqa: F401
    PdfDocument,
    _format_matrix,
    _line_length,
    _matrix_rotation,
    _matrix_scale,
    _point_angle,
    _quad_angle,
    _quad_metadata,
    _rect_metadata,
    _round_color,
)
from rastervec.Evaluation.inspector.pdf_model_text import (  # noqa: F401
    _find_best_span,
    _make_oriented_quad,
    extract_text_items,
)
from rastervec.Evaluation.inspector.pdf_model_image import (  # noqa: F401
    extract_image_items,
)
from rastervec.Evaluation.inspector.pdf_model_drawing import (  # noqa: F401
    _drawing_common_metadata,
    _path_length,
    collect_drawing_colors,
    extract_annot_items,
    extract_drawing_items,
)
