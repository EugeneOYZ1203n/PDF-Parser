"""The raster benchmark's two halves (`ReportConfig.raster_benchmark`).

`write_raster_benchmark` (called by `generate_pipeline_report.py` per benchmark input) writes
`<doc>/raster/`:

    rasterised_p<N>.pdf                     the page flattened to one image (`conversion.rasterise_page`)
    dump.json                               the pipeline run on those pages (the raster pass's vectors)
    ground_truth_vector_to_raster.json      per page: the original page's vectors + the text label boxes

`score_raster_doc` (called by `pipeline_report_benchmark.py`) scores that folder with
`raster_benchmark.score_vectors`, and `add_raster_sections` / `format_raster_table` put the
scores into `report.html` / `benchmark.txt`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from raster_parser.models.vector import Vector
from raster_parser.commons.progress import stage
from raster_parser.core.pipeline import run_pipeline
from raster_parser.evaluation import conversion, dump_io
from raster_parser.evaluation.evaluate import raster_benchmark
from raster_parser.phase_native_read.native_read import read_and_extract

if TYPE_CHECKING:
    from raster_parser.models.settings import RunSettings
    from raster_parser.evaluation.evaluate.html_report import ReportBuilder
    from raster_parser.evaluation.evaluate.vector_metrics import VectorMetricSuiteResult
    from raster_parser.evaluation.labelling.label_schema import LabelSet
    from raster_parser.evaluation.report.report_config import ReportConfig

RASTER_DIR = "raster"
GT_FILE = f"ground_truth_{raster_benchmark.VECTOR_TYPE}.json"


def write_raster_benchmark(
    pdf_path: Path,
    pages: list[int],
    doc_dir: Path,
    config: "ReportConfig",
    settings_for: "Callable[[int], RunSettings]",
    gt_by_type: "dict[str, LabelSet]",
) -> None:
    """Rasterise each page, run the pipeline (`config.p2`; nothing for vector_classify to do on
    an image-only page) and write the folder described in the module docstring."""
    from raster_parser.evaluation.report.report_artifacts import restamp_page

    out = doc_dir / RASTER_DIR
    out.mkdir(parents=True, exist_ok=True)
    dumps: list[dump_io.PageDump] = []
    gt_pages: list[dict] = []
    for p in pages:
        raster_pdf = out / f"rasterised_p{p}.pdf"
        with stage(f"report › rasterise {doc_dir.name} p{p} ({config.raster_dpi} dpi)"):
            conversion.rasterise_page(str(pdf_path), p, str(raster_pdf), dpi=config.raster_dpi)
        res = run_pipeline(str(raster_pdf), 0, p2=config.p2, p3="stub", settings=settings_for(p))
        restamp_page(res, p)
        dumps.append(dump_io.PageDump(result=res, p2=config.p2, p3="stub"))
        gt_pages.append(
            {
                "page_index": p,
                "vectors": [v.to_dict() for v in read_and_extract(str(pdf_path), p).vectors],
                "text_boxes": [
                    list(e.bbox)
                    for labels in gt_by_type.values()
                    for e in labels.entries
                    if e.page_index == p
                ],
            }
        )
        if not config.keep_converted_pdfs:
            raster_pdf.unlink(missing_ok=True)
    dump_io.write_dump(out / "dump.json", str(pdf_path), dumps)
    (out / GT_FILE).write_text(
        json.dumps({"dpi": config.raster_dpi, "pages": gt_pages}), encoding="utf-8"
    )


def score_raster_doc(doc_dir: Path) -> "VectorMetricSuiteResult | None":
    """The aggregate raster score of one benchmark input's folder; `None` without a raster run."""
    raster = doc_dir / RASTER_DIR
    if not (raster / GT_FILE).is_file():
        return None
    gt = json.loads((raster / GT_FILE).read_text(encoding="utf-8"))
    gt_by_page = {g["page_index"]: g for g in gt["pages"]}
    results = []
    for page in dump_io.load_dump(raster / "dump.json").pages:
        g = gt_by_page[page.result.page.index]
        results.append(
            raster_benchmark.score_vectors(
                list(page.result.vectors),
                [Vector.from_dict(v) for v in g["vectors"]],
                page.result.page,
                dpi=float(gt["dpi"]),
                text_boxes=[tuple(b) for b in g["text_boxes"]],
            )
        )
    return raster_benchmark.aggregate(results)


def _fmt(x: float) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.3f}"


_ROWS: "tuple[tuple[str, Callable[[VectorMetricSuiteResult], float]], ...]" = (
    ("D_v (detection)", lambda r: r.liu_dori.d_v),
    ("F_v (false alarm)", lambda r: r.liu_dori.f_v),
    ("VRI", lambda r: r.liu_dori.vri),
    ("D_v text lines", lambda r: r.liu_dori.by_text["text"].d_v.value),
    ("D_v non-text lines", lambda r: r.liu_dori.by_text["non-text"].d_v.value),
    ("D_p (pixel detection)", lambda r: r.pixel.d_p.value),
    ("F_p (pixel false alarm)", lambda r: r.pixel.f_p.value),
    ("PRI", lambda r: r.pixel.pri),
    ("length ratio pred/GT", lambda r: r.length.ratio.value),
    ("direction match", lambda r: r.direction.intersection),
    ("SSIM", lambda r: r.image.ssim),
    ("PSNR (dB)", lambda r: r.image.psnr),
)


def raster_rows(by_run: "dict[str, VectorMetricSuiteResult | None]") -> list[list[str]]:
    return [
        [name] + ["n/a" if r is None else _fmt(value(r)) for r in by_run.values()]
        for name, value in _ROWS
    ]


def add_raster_sections(
    builder: "ReportBuilder", by_run: "dict[str, VectorMetricSuiteResult | None]"
) -> None:
    """One "Raster vectorization (Liu & Dori)" table, a column per run."""
    builder.add_group_header("Raster vectorization (Liu & Dori)")
    builder.add_subsection(
        "Line detection, pixels, length, direction, image", ["metric", *by_run], raster_rows(by_run)
    )


def format_raster_table(by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str) -> str:
    rows = raster_rows(by_run)
    width = max(len(r[0]) for r in rows)
    lines = [title, "  " + " " * width + "  " + "  ".join(f"{name:>12}" for name in by_run)]
    lines += ["  " + r[0].ljust(width) + "  " + "  ".join(f"{c:>12}" for c in r[1:]) for r in rows]
    return "\n".join(lines)
