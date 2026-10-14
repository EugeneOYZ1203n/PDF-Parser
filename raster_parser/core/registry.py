"""The `p2` / `p3` names `extract()` and `run_pipeline` accept, mapped to backend callables.

Adding a backend: a new folder under `P2_Raster_To_Vec/` or `P3_Vector_Parsing/` exposing a
function that matches `core.interfaces.Phase2Backend` / `Phase3Backend`, plus one line here.
See `docs/ARCHITECTURE.md`.
"""
from __future__ import annotations

from raster_parser.core.interfaces import Phase2Backend, Phase3Backend
from raster_parser.P2_Raster_To_Vec.Stub.stub import extract as _p2_stub
from raster_parser.P2_Raster_To_Vec.v1RasterToVec.adapter import extract as _v1_raster_to_vec
from raster_parser.P3_Vector_Parsing.Stub.stub import parse as _p3_stub
from raster_parser.P3_Vector_Parsing.v1VectorClass.parse import parse as _v1_vector_class

P2_REGISTRY: dict[str, Phase2Backend] = {
    "Stub": _p2_stub,
    "v1RasterToVec": _v1_raster_to_vec,
}

P3_REGISTRY: dict[str, Phase3Backend] = {
    "Stub": _p3_stub,
    "v1VectorClass": _v1_vector_class,
}

DEFAULT_P2 = "v1RasterToVec"
DEFAULT_P3 = "v1VectorClass"


def _resolve(registry: dict, name: str, phase: str):
    try:
        return registry[name]
    except KeyError:
        raise ValueError(f"unknown {phase} backend {name!r}; choose from {sorted(registry)}") from None


def resolve_p2(name: str) -> Phase2Backend:
    return _resolve(P2_REGISTRY, name, "p2")


def resolve_p3(name: str) -> Phase3Backend:
    return _resolve(P3_REGISTRY, name, "p3")
