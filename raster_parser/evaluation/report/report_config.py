"""`scripts/generate_pipeline_report.py`'s config schema: `ReportConfig` (read from a JSON file)
and `BenchInput` (one benchmark input). Field-by-field description: `scripts/report_configs/README.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field, field_validator, model_validator

from raster_parser.core.pipeline import STEP_NAMES
from raster_parser.core.registry import (
    DEFAULT_CLASSIFY,
    DEFAULT_VECTORIZE,
    resolve_classify,
    resolve_vectorize,
)
from raster_parser.models.settings import RunSettings
from raster_parser.evaluation import conversion
from raster_parser.evaluation.evaluate.metrics import MetricConfig
from raster_parser.evaluation.labelling.label_schema import load_labels

CONVERT = {
    "text_only": conversion.convert_page_text_only,
    "drawings_only": conversion.convert_page_drawings_only,
    "to_vector_text": conversion.convert_page_to_vector_text,
}


class ReportConfig(BaseModel):
    # Backends, by registry name.
    p2: str = DEFAULT_VECTORIZE
    p3: str = DEFAULT_CLASSIFY
    # Stop after this phase (`run_pipeline(stop_after=...)`); None = run everything.
    final_stage: str | None = None
    # Inputs: PDFs (or, in benchmark mode, label JSON files).
    input_dir: Path | None = None
    input_files: list[Path] = Field(default_factory=list)
    # Benchmark mode: a ground-truth label JSON per PDF stem.
    label_files: dict[str, Path] = Field(default_factory=dict)
    # Page indices: one list for every input, or {stem: [...], "*": [...]}; None = [0].
    pages: list[int] | dict[str, list[int]] | None = None
    # Run on a converted copy of each page (`evaluation/conversion.py`) instead of the original.
    vectorise: bool = False
    vectorise_mode: str = "to_vector_text"
    # Score-ready output for `pipeline_report_benchmark.py` (runs on `to_vector_text` copies).
    benchmark: bool = False
    iou_edge_min: float = MetricConfig().iou_edge_min
    # Run folder root; None = `outputs/pipeline_report/`.
    output_root: Path | None = None
    # Debug images offered by the backends (`RunSettings.on_debug_image`), capped per folder
    # (uniform random sample across every page; None = uncapped).
    debug_images: bool = True
    debug_image_cap: int | None = 100
    # Every layer PDF (backend debug layers, the reconstructed page, benchmark overlays).
    # Off: dump.json / ground truth / debug images are still written.
    debug_layers: bool = True
    # Keep each page's converted input PDF (`converted_p<N>.pdf`).
    keep_converted_pdfs: bool = True
    # `RunSettings` fields passed to the backends.
    ocr_dpi: int = RunSettings.ocr_dpi
    use_angle_cls: bool = RunSettings.use_angle_cls
    workers: int = RunSettings.workers

    @field_validator("p2")
    @classmethod
    def _known_p2(cls, v: str) -> str:
        resolve_vectorize(v)  # raises ValueError naming the valid backends
        return v

    @field_validator("p3")
    @classmethod
    def _known_p3(cls, v: str) -> str:
        resolve_classify(v)
        return v

    @field_validator("final_stage")
    @classmethod
    def _known_stage(cls, v: str | None) -> str | None:
        if v is not None and v not in STEP_NAMES:
            raise ValueError(f"final_stage must be one of {list(STEP_NAMES)}")
        return v

    @field_validator("vectorise_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in CONVERT:
            raise ValueError(f"vectorise_mode must be one of {list(CONVERT)}")
        return v

    @model_validator(mode="after")
    def _benchmark_excludes_vectorise(self) -> "ReportConfig":
        if self.benchmark and self.vectorise:
            raise ValueError(
                "benchmark mode runs its own to_vector_text conversion; set vectorise=false"
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
        """One `BenchInput` per input. A `.json` label file (or a PDF with a `label_files`
        entry) is keyed `labels:<json-stem>`; a bare PDF is keyed `pdf:<stem>` and only gets
        native-derived ground truth. The benchmark script matches runs on these keys."""
        out: list[BenchInput] = []
        seen: set[str] = set()

        def _add(key: str, pdf: Path, labels: Path | None) -> None:
            if key not in seen:
                seen.add(key)
                out.append(BenchInput(key=key, pdf_path=Path(pdf).resolve(), labels_path=labels))

        raw: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            raw += sorted(Path(self.input_dir).glob("*.pdf"))
        for item in raw:
            item = Path(item)
            if item.suffix.lower() == ".json":
                labels = item.resolve()
                _add(f"labels:{labels.stem}", Path(load_labels(str(labels)).pdf_path), labels)
            elif (paired := self.label_files.get(item.stem)) is not None:
                paired = Path(paired).resolve()
                _add(f"labels:{paired.stem}", item, paired)
            else:
                _add(f"pdf:{item.stem}", item, None)
        return out


class BenchInput(NamedTuple):
    key: str
    pdf_path: Path
    labels_path: Path | None  # a label JSON file; None = native only
