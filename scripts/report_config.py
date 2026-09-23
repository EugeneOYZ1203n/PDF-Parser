"""`generate_pipeline_report.py`'s config schema: `ReportConfig`/`BenchInput`,
the `NEW_STEP_NAMES` step list `ReportConfig.final_stage` validates against,
and the `_CONVERT`/`_layer_slug` small helpers the schema itself depends on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, field_validator, model_validator

from rastervec.Evaluation import conversion
from rastervec.Evaluation.Evaluate import metrics
from rastervec.Evaluation.Labelling.label_schema import (
    load_labels,
    load_labels_from_master_folder,
)
from rastervec.core.registry import P2_REGISTRY, P3_REGISTRY, DEFAULT_P2, DEFAULT_P3

# The new `core.pipeline` engine's short, generic step list (`phase1` ->
# `phase2` -> `phase3`) -- what `ReportConfig.final_stage` validates against
# regardless of `pipeline`, and what `report_artifacts.py::_reached` gates
# `_NEW_ARTIFACTS` rows on.
NEW_STEP_NAMES = ["phase1", "phase2", "phase3"]

_CONVERT = {
    "text_only": conversion.convert_page_text_only,
    "drawings_only": conversion.convert_page_drawings_only,
    "to_vector_text": conversion.convert_page_to_vector_text,
}


def _layer_slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "layer"


class ReportConfig(BaseModel):
    # `pipeline` is the top-level engine axis: "current" (the new pluggable
    # core.pipeline, P1 -> p2 -> p3) or "legacy" (archive/raster_parser,
    # unchanged, p2/p3 ignored). `p2`/`p3` only apply when
    # `pipeline == "current"` -- they select a `core.registry.P2_REGISTRY`/
    # `P3_REGISTRY` backend by name.
    pipeline: Literal["current", "legacy"] = "current"
    p2: str = DEFAULT_P2
    p3: str = DEFAULT_P3
    enable_fast: bool = True
    final_stage: str | None = None
    input_dir: Path | None = None
    input_files: list[Path] = Field(default_factory=list)
    label_files: dict[str, Path] = Field(default_factory=dict)
    pages: list[int] | dict[str, list[int]] | None = None
    vectorise: bool = False
    vectorise_mode: str = "to_vector_text"
    benchmark: bool = False
    iou_edge_min: float = metrics.MetricConfig().iou_edge_min
    dpi: int = 300
    output_root: Path | None = None
    # Per-page PNG dumps of what PaddleOCR's detector/recognizer saw
    # (`paddle_detect_images/`, `paddle_recog_images/`, `paddle_ocr_images/`)
    # -- one file per cluster / word crop, so the bulk of a report's files.
    debug_images: bool = True

    @field_validator("p2")
    @classmethod
    def _known_p2(cls, v: str) -> str:
        if v not in P2_REGISTRY:
            raise ValueError(f"p2 must be one of {sorted(P2_REGISTRY)}")
        return v

    @field_validator("p3")
    @classmethod
    def _known_p3(cls, v: str) -> str:
        if v not in P3_REGISTRY:
            raise ValueError(f"p3 must be one of {sorted(P3_REGISTRY)}")
        return v

    @field_validator("final_stage")
    @classmethod
    def _known_stage(cls, v: str | None) -> str | None:
        # Validated against the new engine's short step list regardless of
        # `pipeline` (legacy's `_active_artifacts` never consults
        # `final_stage` at all, so it's a harmless no-op there too).
        if v is not None and v not in NEW_STEP_NAMES:
            raise ValueError(f"final_stage must be one of {NEW_STEP_NAMES}")
        return v

    @field_validator("vectorise_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in _CONVERT:
            raise ValueError(f"vectorise_mode must be one of {list(_CONVERT)}")
        return v

    @model_validator(mode="after")
    def _benchmark_excludes_vectorise(self) -> "ReportConfig":
        if self.benchmark and self.vectorise:
            raise ValueError(
                "benchmark mode runs its own text-only / drawings-only conversions; "
                "set vectorise=false"
            )
        return self

    def resolved_pdfs(self) -> list[Path]:
        pdfs: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            pdfs += sorted(Path(self.input_dir).glob("*.pdf"))
        seen, out = set(), []
        for p in pdfs:
            p = Path(p).resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def pages_for(self, stem: str) -> list[int]:
        if self.pages is None:
            return [0]
        if isinstance(self.pages, dict):
            return list(self.pages.get(stem, self.pages.get("*", [0])))
        return list(self.pages)

    def benchmark_inputs(self) -> list["BenchInput"]:
        """One `BenchInput` per config input, benchmark mode only. A `.json`
        input (or a pdf with a `label_files` entry) carries ground truth and
        is keyed `labels:<json-stem>`; a directory (a `scripts/label/
        master_label.py` output folder) is keyed `labels:<folder-stem>`,
        `pdf_path` from the folder's own `original.pdf`, `labels_path` the
        directory itself; a bare `.pdf` is auto-only and keyed
        `pdf:<pdf-stem>`. The key is what the benchmark script matches
        shared inputs on."""
        out: list[BenchInput] = []
        seen: set[str] = set()

        def _add(key: str, pdf: Path, labels: Path | None, rasterised: Path | None = None) -> None:
            if key not in seen:
                seen.add(key)
                out.append(BenchInput(
                    key=key, pdf_path=pdf.resolve(), labels_path=labels,
                    rasterised_pdf_path=rasterised.resolve() if rasterised else None,
                ))

        raw: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            raw += sorted(Path(self.input_dir).glob("*.pdf"))
        for item in raw:
            item = Path(item)
            if item.is_dir():
                folder = item.resolve()
                labels = load_labels_from_master_folder(folder)
                pdf = Path(labels.pdf_path)
                rasterised = folder / "rasterised.pdf"
                _add(f"labels:{folder.stem}", pdf, folder, rasterised if rasterised.is_file() else None)
                continue
            if item.suffix.lower() == ".json":
                labels = item.resolve()
                pdf = Path(load_labels(str(labels)).pdf_path)
                _add(f"labels:{labels.stem}", pdf, labels)
                continue
            paired = self.label_files.get(item.stem)
            if paired is not None:
                paired = Path(paired).resolve()
                _add(f"labels:{paired.stem}", item, paired)
            else:
                _add(f"pdf:{item.stem}", item, None)
        return out


class BenchInput(NamedTuple):
    key: str
    pdf_path: Path
    labels_path: Path | None
    # A master_label.py folder's own `rasterised.pdf` (per-page flattened-
    # to-image renders, 1:1 page-index-aligned with `original.pdf`) -- when
    # present, `_process_pdf_benchmark` runs the pipeline a SECOND time per
    # page directly on this file, and vector_to_raster/original_raster/
    # native_to_raster are scored from that run's own OCR output instead of
    # the vectorised run's. `None` for a bare `.pdf`/`.json` input (no
    # rasterised counterpart known) -- those 3 types stay GT-only.
    rasterised_pdf_path: Path | None = None


ReportConfig.model_rebuild()
