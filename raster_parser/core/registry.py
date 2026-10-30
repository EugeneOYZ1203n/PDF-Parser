"""The `p2` / `p3` names `extract()` and `run_pipeline` accept, mapped to backend callables.

Adding a backend: a new folder under `phase_raster_vectorize/` or `phase_vector_classify/` exposing a
function that matches `core.interfaces.RasterVectorizer` / `VectorClassifier`, plus one line here.
"""

from __future__ import annotations

from raster_parser.core.interfaces import RasterVectorizer, VectorClassifier
from raster_parser.phase_raster_vectorize.stub.stub import extract as _p2_stub
from raster_parser.phase_vector_classify.stub.stub import parse as _p3_stub


def _v1_raster_to_vec(images, page, settings):
    """`v1_raster_to_vec.adapter.extract`, imported on first use (OpenCV, PaddleOCR)."""
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.adapter import extract

    return extract(images, page, settings)


def _junction(images, page, settings):
    """`junction.adapter.extract`, imported on first use (OpenCV, scikit-image, PaddleOCR)."""
    from raster_parser.phase_raster_vectorize.junction.adapter import extract

    return extract(images, page, settings)


def _involution_stroke(images, page, settings):
    """`involution_stroke.adapter.extract`, imported on first use (torch: the `ml` extra)."""
    from raster_parser.phase_raster_vectorize.involution_stroke.adapter import extract

    return extract(images, page, settings)


def _deep_tech_vec(images, page, settings):
    """`deep_tech_vec.adapter.extract`, imported on first use (torch: the `ml` extra)."""
    from raster_parser.phase_raster_vectorize.deep_tech_vec.adapter import extract

    return extract(images, page, settings)


def _deep_vectoriser(images, page, settings):
    """`deep_vectoriser.adapter.extract`, imported on first use (torch: the `ml` extra)."""
    from raster_parser.phase_raster_vectorize.deep_vectoriser.adapter import extract

    return extract(images, page, settings)


def _implicit_sketch_vec(images, page, settings):
    """`implicit_sketch_vec.adapter.extract`, imported on first use (torch: the `ml` extra)."""
    from raster_parser.phase_raster_vectorize.implicit_sketch_vec.adapter import extract

    return extract(images, page, settings)


def _v1_vector_class(vectors, page, settings):
    """`v1_vector_class.parse.parse`, imported on first use (PaddleOCR, PIL)."""
    from raster_parser.phase_vector_classify.v1_vector_class.parse import parse

    return parse(vectors, page, settings)


def _v2_vector_class(vectors, page, settings):
    """`v2_vector_class.parse`, imported on first use: its modules load OpenCV, SciPy and
    pyenchant at import time, and `import raster_parser` must stay light."""
    from raster_parser.phase_vector_classify.v2_vector_class.parse import parse

    return parse(vectors, page, settings)


VECTORIZE_BACKENDS: dict[str, RasterVectorizer] = {
    "stub": _p2_stub,
    "v1_raster_to_vec": _v1_raster_to_vec,
    "junction": _junction,
    "involution_stroke": _involution_stroke,
    # optional deep backends (pr10b): need the `ml` extra and their own trained weights
    "deep_tech_vec": _deep_tech_vec,
    "deep_vectoriser": _deep_vectoriser,
    "implicit_sketch_vec": _implicit_sketch_vec,
}

CLASSIFY_BACKENDS: dict[str, VectorClassifier] = {
    "stub": _p3_stub,
    "v1_vector_class": _v1_vector_class,
    "v2_vector_class": _v2_vector_class,
}

DEFAULT_VECTORIZE = "v1_raster_to_vec"
DEFAULT_CLASSIFY = "v2_vector_class"


def _resolve(registry: dict, name: str, phase: str):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"unknown {phase} backend {name!r}; choose from {sorted(registry)}"
        ) from None


def resolve_vectorize(name: str) -> RasterVectorizer:
    return _resolve(VECTORIZE_BACKENDS, name, "p2")


def resolve_classify(name: str) -> VectorClassifier:
    return _resolve(CLASSIFY_BACKENDS, name, "p3")
