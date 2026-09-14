"""Named registries of interchangeable P2/P3 backends -- the one place a
run config's `p2`/`p3` string gets resolved to a real callable. Adding a
new backend = one more folder under P2_Raster_To_Vec/ or
P3_Vector_Parsing/ implementing the Phase2Backend/Phase3Backend interface
(see `core/interfaces.py`), plus one more line here.

`P2_RENDER_DEBUG`/`P3_RENDER_DEBUG` are a SEPARATE, optional registry: each
backend may define its own `render_debug(debug_out, page_meta) ->
list[(stage, label, hex, pdf_bytes)]` next to its `extract`/`parse`
function, reading back whatever it stashed into the `debug_out` dict
`core.pipeline.run_pipeline` forwards it (only when `verbose=True` and the
backend's own signature declares `debug_out` -- see that module). There is
no shared interpretation of `debug_out`'s contents here or in
`core/pipeline.py` -- each backend renders its own internal data structures
with the three generic primitives in `commons/renderer`
(`render_boxes_pdf`/`render_text_pdf`/`render_vectors_pdf`). A backend that
defines neither `debug_out` nor `render_debug` (e.g. Stub) simply isn't in
these dicts."""
from __future__ import annotations

from rastervec.core.interfaces import Phase2Backend, Phase3Backend
from rastervec.P2_Raster_To_Vec.Junction.adapter import extract as _junction_extract
from rastervec.P2_Raster_To_Vec.Junction.adapter import render_debug as _junction_render_debug
from rastervec.P2_Raster_To_Vec.Stub.stub import extract as _stub_extract
from rastervec.P3_Vector_Parsing.FastIntoPaddle.parse import parse as _fast_into_paddle_parse
from rastervec.P3_Vector_Parsing.FastIntoPaddle.parse import render_debug as _fast_into_paddle_render_debug
from rastervec.P3_Vector_Parsing.LegacyRecreation.parse import parse as _legacy_recreation_parse
from rastervec.P3_Vector_Parsing.LegacyRecreation.parse import render_debug as _legacy_recreation_render_debug
from rastervec.P3_Vector_Parsing.VectorClassification.parse import parse as _vector_classification_parse
from rastervec.P3_Vector_Parsing.VectorClassification.parse import render_debug as _vector_classification_render_debug

P2_REGISTRY: dict[str, Phase2Backend] = {
    "Stub": _stub_extract,
    "Junction": _junction_extract,
}

P3_REGISTRY: dict[str, Phase3Backend] = {
    "VectorClassification": _vector_classification_parse,
    "FastIntoPaddle": _fast_into_paddle_parse,
    "LegacyRecreation": _legacy_recreation_parse,
}

P2_RENDER_DEBUG: dict[str, object] = {
    "Junction": _junction_render_debug,
}

P3_RENDER_DEBUG: dict[str, object] = {
    "VectorClassification": _vector_classification_render_debug,
    "FastIntoPaddle": _fast_into_paddle_render_debug,
    "LegacyRecreation": _legacy_recreation_render_debug,
}

DEFAULT_P2 = "Stub"
DEFAULT_P3 = "FastIntoPaddle"


def resolve_p2(name: str) -> Phase2Backend:
    try:
        return P2_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown P2 backend {name!r}; valid: {sorted(P2_REGISTRY)}") from None


def resolve_p3(name: str) -> Phase3Backend:
    try:
        return P3_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown P3 backend {name!r}; valid: {sorted(P3_REGISTRY)}") from None
