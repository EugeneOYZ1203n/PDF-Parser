"""Named registries of interchangeable P2/P3 backends -- the one place a
run config's `p2`/`p3` string gets resolved to a real callable. Adding a
new backend = one more folder under P2_Raster_To_Vec/ or
P3_Vector_Parsing/ implementing the Phase2Backend/Phase3Backend interface
(see `core/interfaces.py`), plus one more line here."""
from __future__ import annotations

from rastervec.core.interfaces import Phase2Backend, Phase3Backend
from rastervec.P2_Raster_To_Vec.Junction.adapter import extract as _junction_extract
from rastervec.P2_Raster_To_Vec.Stub.stub import extract as _stub_extract
from rastervec.P3_Vector_Parsing.FastIntoPaddle.parse import parse as _fast_into_paddle_parse
from rastervec.P3_Vector_Parsing.LegacyRecreation.parse import parse as _legacy_recreation_parse
from rastervec.P3_Vector_Parsing.VectorClassification.parse import parse as _vector_classification_parse

P2_REGISTRY: dict[str, Phase2Backend] = {
    "Stub": _stub_extract,
    "Junction": _junction_extract,
}

P3_REGISTRY: dict[str, Phase3Backend] = {
    "VectorClassification": _vector_classification_parse,
    "FastIntoPaddle": _fast_into_paddle_parse,
    "LegacyRecreation": _legacy_recreation_parse,
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
