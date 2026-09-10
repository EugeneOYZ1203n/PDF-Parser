"""Config-driven pipeline stage report generator.

Reads a JSON config, runs the pipeline once per (pdf, page), and writes a
timestamped run folder:

    outputs/pipeline_report/<ts>__<config-stem>/
        config_and_hyperparameters.txt
        <pdf-stem>/
            manifest.json
            native_text.pdf  vector_extraction.pdf  separation.pdf
            vector_classification.pdf  fast_heatmap.pdf  segmentation.pdf
            similarity.pdf  paddle_ocr.pdf  drawing_vectors.pdf
            reconstructed.pdf                         (one multi-page PDF per stage)
            native_text.txt ...                       (one stats file per stage)
            dump.json                                 (every Text + Vector, reloadable)
            radon_images/    paddle_images/

Replaces `rastervec/notebooks/pipeline_stage_visualization.ipynb`.

    .venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rastervec.config as rvconfig
from rastervec.Evaluation import conversion, dump_io
from rastervec.Evaluation.Evaluate.variants import resolve_variant
from rastervec.Evaluation.Report import stage_stats
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir
from rastervec.pipelines._common import STEP_NAMES
from rastervec.renderer import stages

_LOG = get_logger("generate_pipeline_report")

_CONVERT = {
    "text_only": conversion.convert_page_text_only,
    "drawings_only": conversion.convert_page_drawings_only,
    "to_vector_text": conversion.convert_page_to_vector_text,
}

# filename -> (renderer, stats-key or None, gate STEP_NAME)
_ARTIFACTS: list[tuple[str, str, str | None, str]] = [
    ("native_text.pdf", "render_native", "native", "native"),
    ("vector_extraction.pdf", "render_vectors", "vectors", "vectors"),
    ("separation.pdf", "render_layer_color_buckets", "separation", "classify"),
    ("vector_classification.pdf", "render_clustering_steps", "classify", "classify"),
    ("fast_heatmap.pdf", "render_fast", "fast", "fast"),
    ("segmentation.pdf", "render_radon", "segment", "segment"),
    ("similarity.pdf", "render_similarity", "similarity", "similarity"),
    ("paddle_ocr.pdf", "render_ocr_results", "ocr", "restore"),
    ("drawing_vectors.pdf", "render_drawing", None, "drawing"),
    ("reconstructed.pdf", "render_reconstructed", None, "drawing"),
]


class ReportConfig(BaseModel):
    pipeline: str = "current"
    final_stage: str | None = None
    input_dir: Path | None = None
    input_files: list[Path] = Field(default_factory=list)
    label_files: dict[str, Path] = Field(default_factory=dict)
    pages: list[int] | dict[str, list[int]] | None = None
    vectorise: bool = False
    vectorise_mode: str = "to_vector_text"
    dpi: int = 300
    output_root: Path | None = None

    @field_validator("pipeline")
    @classmethod
    def _known_pipeline(cls, v: str) -> str:
        resolve_variant(v)  # raises ValueError listing valid names
        return v

    @field_validator("final_stage")
    @classmethod
    def _known_stage(cls, v: str | None) -> str | None:
        if v is not None and v not in STEP_NAMES:
            raise ValueError(f"final_stage must be one of {STEP_NAMES}")
        return v

    @field_validator("vectorise_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in _CONVERT:
            raise ValueError(f"vectorise_mode must be one of {list(_CONVERT)}")
        return v

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


ReportConfig.model_rebuild()


def _reached(step: str, final_stage: str | None) -> bool:
    if final_stage is None:
        return True
    return STEP_NAMES.index(step) <= STEP_NAMES.index(final_stage)


def _save_images(arrays, folder: Path, prefix: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, arr in enumerate(arrays or []):
        if arr is None:
            continue
        Image.fromarray(arr).save(folder / f"{prefix}_{i:03d}.png")
        n += 1
    return n


def _merge_pdfs(page_bytes: list[bytes], out_path: Path) -> None:
    out = fitz.open()
    try:
        for b in page_bytes:
            src = fitz.open("pdf", b)
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        out.save(str(out_path))
    finally:
        out.close()


def _write_hyperparams(path: Path, config: ReportConfig, variant) -> None:
    lines = ["# Run config\n", config.model_dump_json(indent=2), "\n\n# Variant\n"]
    lines.append(json.dumps(
        {"name": variant.name, "engine": variant.engine, "enable_fast": variant.enable_fast},
        indent=2,
    ))
    lines.append("\n\n# rastervec.config constants\n")
    for name in sorted(vars(rvconfig)):
        if name.isupper():
            lines.append(f"{name} = {getattr(rvconfig, name)!r}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _process_pdf(pdf_path: Path, config: ReportConfig, variant, run_dir: Path) -> None:
    from rastervec.pipelines.current import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    doc_dir = run_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    radon_dir = doc_dir / "radon_images"
    paddle_dir = doc_dir / "paddle_images"

    pages = config.pages_for(pdf_path.stem)
    if variant.engine == "legacy":
        active = [row for row in _ARTIFACTS if row[0] == "reconstructed.pdf"]
    else:
        active = [row for row in _ARTIFACTS if _reached(row[3], config.final_stage)]

    stage_pages: dict[str, list[bytes]] = {row[0]: [] for row in active}
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for page_index in pages:
        run_input = str(pdf_path)
        run_page = page_index
        if config.vectorise:
            conv_path = doc_dir / f"converted_p{page_index}.pdf"
            _CONVERT[config.vectorise_mode](str(pdf_path), page_index, str(conv_path))
            run_input, run_page = str(conv_path), 0

        _LOG.info("running %s page %d (%s)", pdf_path.name, page_index, variant.name)
        if variant.engine == "legacy":
            res = run_legacy(run_input, run_page, verbose=True)
        else:
            res = run_current(
                run_input, run_page, enable_fast=variant.enable_fast, verbose=True,
                stop_after=config.final_stage,
            )

        for filename, render_name, stats_key, _gate in active:
            try:
                pdf_bytes = getattr(stages, render_name)(res)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("%s render failed for page %d: %s", filename, page_index, exc)
                pdf_bytes = fitz.open().tobytes()
            stage_pages[filename].append(pdf_bytes)
            if stats_key is not None:
                stats_pages[filename].append(
                    (page_index, stage_stats.stats_for_stage(res, stats_key))
                )

        _save_images(getattr(res, "word_segments", None) and
                     [s.image for s in res.word_segments], radon_dir, f"p{page_index}_seg")
        _save_images(getattr(res, "unique_segments", None) and
                     [s.image for s in res.unique_segments], paddle_dir, f"p{page_index}_uniq")

        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta,
            texts=list(res.texts or []),
            vectors=list(res.vectors or []),
            engine=res.engine,
            step_durations=dict(res.step_durations or {}),
        ))

    for filename, _rn, stats_key, _gate in active:
        _merge_pdfs(stage_pages[filename], doc_dir / filename)
        if stats_key is not None:
            body = "".join(
                f"\n## page {pi}\n{stage_stats.format_stats(stats_key, data)}"
                for pi, data in stats_pages[filename]
            )
            (doc_dir / filename.replace(".pdf", ".txt")).write_text(
                f"# {stats_key} stats for {pdf_path.name}\n{body}", encoding="utf-8"
            )

    dump_io.write_dump(doc_dir / "dump.json", str(pdf_path), dumps)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": str(pdf_path),
        "pages": pages,
        "engine": variant.engine,
        "variant": variant.name,
        "final_stage": config.final_stage,
        "vectorised": config.vectorise,
        "stages": [row[0] for row in active],
        "color_legend": stages.STAGE_COLOR_LEGEND,
    }, indent=2), encoding="utf-8")
    _LOG.info("wrote %s", doc_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="Path to the run config JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    config = ReportConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    variant = resolve_variant(config.pipeline)

    root = config.output_root or output_dir("pipeline_report")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}__{Path(args.config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hyperparams(run_dir / "config_and_hyperparameters.txt", config, variant)

    pdfs = config.resolved_pdfs()
    if not pdfs:
        _LOG.error("no input PDFs (set input_dir and/or input_files)")
        return 1
    for pdf_path in pdfs:
        _process_pdf(pdf_path, config, variant, run_dir)

    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
