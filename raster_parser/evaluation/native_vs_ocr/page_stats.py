"""Per-page and per-run statistics containers for the OCR pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.evaluation.native_vs_ocr import Stats, _char_outcomes
from raster_parser.evaluation.native_vs_ocr import cer_percentage as page_cer_percentage
from raster_parser.evaluation.native_vs_ocr import (
    character_match_percentage as page_character_match_percentage,
)
from raster_parser.evaluation.native_vs_ocr import (
    native_text_char_count as page_native_text_char_count,
)


@dataclass
class GlyphSummary:
    total_native_characters: int = 0
    pass_characters: int = 0
    failure_characters: int = 0
    failed_ocr_characters: int = 0
    exact_match_rate: float = 0.0
    ocr_word_rows: int = 0
    character_match_percentage: float = 0.0
    cer_percentage: float = 0.0
    native_text_characters: int = 0

    @classmethod
    def from_comparison(
        cls,
        stats: List[Stats],
        not_ocred: List[Tuple[int, str]],
    ) -> GlyphSummary:
        outcomes = _char_outcomes(stats, not_ocred)
        pass_n = sum(1 for v in outcomes.values() if v == "pass")
        fail_n = sum(1 for v in outcomes.values() if v == "fail")
        failed_n = sum(1 for v in outcomes.values() if v == "failed_ocr")
        total = len(outcomes)
        rate = (100.0 * pass_n / total) if total else 0.0
        char_match_pct = page_character_match_percentage(stats, not_ocred, total)
        text_chars = page_native_text_char_count(stats, not_ocred)
        cer_pct = page_cer_percentage(stats, not_ocred, text_chars)
        return cls(
            total_native_characters=total,
            pass_characters=pass_n,
            failure_characters=fail_n,
            failed_ocr_characters=failed_n,
            exact_match_rate=rate,
            ocr_word_rows=len(stats),
            character_match_percentage=char_match_pct,
            cer_percentage=cer_pct,
            native_text_characters=text_chars,
        )

    def add(self, other: GlyphSummary) -> GlyphSummary:
        total = self.total_native_characters + other.total_native_characters
        pass_n = self.pass_characters + other.pass_characters
        rate = (100.0 * pass_n / total) if total else 0.0
        char_match_pct = (
            (
                (
                    self.character_match_percentage * self.total_native_characters
                    + other.character_match_percentage * other.total_native_characters
                )
                / total
            )
            if total
            else 0.0
        )
        text_chars = self.native_text_characters + other.native_text_characters
        cer_weight = text_chars or total
        cer_pct = (
            (
                (
                    self.cer_percentage
                    * (self.native_text_characters or self.total_native_characters)
                    + other.cer_percentage
                    * (other.native_text_characters or other.total_native_characters)
                )
                / cer_weight
            )
            if cer_weight
            else 0.0
        )
        return GlyphSummary(
            total_native_characters=total,
            pass_characters=pass_n,
            failure_characters=self.failure_characters + other.failure_characters,
            failed_ocr_characters=self.failed_ocr_characters + other.failed_ocr_characters,
            exact_match_rate=rate,
            ocr_word_rows=self.ocr_word_rows + other.ocr_word_rows,
            character_match_percentage=char_match_pct,
            cer_percentage=cer_pct,
            native_text_characters=text_chars,
        )


@dataclass
class PageStats:
    """One PDF page: on-disk outputs plus a small in-memory summary.

    Heavy comparison payloads (``native_mapping``, ``stats``, etc.) are written
    to disk during the pipeline and are **not** retained in memory after step 3.
    Use :func:`load_stats_from_out_results` / :func:`load_native_mapping_for_page`
    when viz or analysis needs the full data.
    """

    pdf_path: Path
    page_index: int
    onlyfill_pdf: Path
    ocr_pdf: Path
    out_results: Path
    out_ocr_txt: Path
    native_mapping_path: Path | None = None
    native_mapping: Dict[int, Tuple[str, Vector]] = field(default_factory=dict)
    ocr_txt_with_mapping: List[Tuple[Text, List[int]]] = field(default_factory=list)
    stats: List[Stats] = field(default_factory=list)
    not_ocred: List[Tuple[int, str]] = field(default_factory=list)
    summary: GlyphSummary = field(default_factory=GlyphSummary)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def drop_heavy_payloads(self) -> None:
        """Release large in-memory fields after outputs are on disk."""
        self.native_mapping = {}
        self.ocr_txt_with_mapping = []
        self.stats = []
        self.not_ocred = []


@dataclass
class RunStats:
    """One directory batch: all pages and document-level summary."""

    run_dir: Path
    pdf_dir: Path
    pages: List[PageStats] = field(default_factory=list)
    total_statistics: Path | None = None
    summary: GlyphSummary = field(default_factory=GlyphSummary)

    @property
    def pages_ok(self) -> int:
        return sum(1 for p in self.pages if p.ok)

    @property
    def pages_failed(self) -> int:
        return sum(1 for p in self.pages if not p.ok)


def load_native_mapping_for_page(page: PageStats) -> Dict[int, Tuple[str, Vector]]:
    """Load ``native_mapping.pkl`` for a page (for viz / offline analysis)."""
    import pickle

    path = page.native_mapping_path
    if path is None:
        path = page.out_results.parent / "native_mapping.pkl"
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def load_stats_from_out_results(path: Path) -> tuple[List[Stats], List[Tuple[int, str]]]:
    """Parse ``out_results.txt`` back into comparison rows (for stats_viz)."""
    import ast
    import re

    from raster_parser.models import GlyphOrientation

    text = Path(path).read_text(encoding="utf-8")
    stats: List[Stats] = []
    not_ocred: List[Tuple[int, str]] = []

    ocr_line = re.compile(
        r"^ocr\[(\d+)\] failed_ocr=(\w+) orientation=(\w+) rotate=(-?\d+) "
        r"match=(\w+) IOU=([\d.]+) CER=([\d.]+) character_match_rate=([\d.]+) "
        r"native_glyphs=(\[[^\]]*\]) native=(.+?) ocr=(.+)$"
    )
    missed_line = re.compile(
        r"^native\[(\d+)\] failed_ocr=True orientation=n/a rotate=n/a "
        r"native=(.+?) ocr=''$"
    )

    for raw in text.splitlines():
        line = raw.strip()
        m = ocr_line.match(line)
        if m:
            stats.append(
                Stats(
                    match=m.group(5) == "True",
                    ocr_num=int(m.group(1)),
                    native_nums=ast.literal_eval(m.group(9)),
                    native_char=ast.literal_eval(m.group(10)),
                    ocr_char=ast.literal_eval(m.group(11)),
                    IOU=float(m.group(6)),
                    CER=float(m.group(7)),
                    failed_ocr=m.group(2) == "True",
                    glyph_orientation=GlyphOrientation[m.group(3)],
                    native_rotate=int(m.group(4)),
                    character_match_rate=float(m.group(8)),
                )
            )
            continue
        m = missed_line.match(line)
        if m:
            not_ocred.append((int(m.group(1)), ast.literal_eval(m.group(2))))

    return stats, not_ocred


def _page_index_from_dir(page_dir: Path) -> int:
    m = re.match(r"page_(\d+)$", page_dir.name)
    return int(m.group(1)) if m else 0


def _pdf_name_from_page_dir(run_dir: Path, page_dir: Path) -> str:
    label = page_dir.parent.name if re.match(r"page_\d+$", page_dir.name) else page_dir.name
    if label == run_dir.name:
        return "unknown.pdf"
    return f"{label}.pdf"


def parse_total_statistics(path: Path) -> tuple[Path, GlyphSummary, dict[tuple[str, int], dict]]:
    """Parse ``total_statistics.txt`` into run metadata and per-page rows."""
    text = Path(path).read_text(encoding="utf-8")
    pdf_dir = Path(".")
    summary = GlyphSummary()
    page_rows: dict[tuple[str, int], dict] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("pdf_dir:"):
            pdf_dir = Path(line.split(":", 1)[1].strip())
        elif line.startswith("total_native_characters:"):
            summary.total_native_characters = int(line.split(":", 1)[1].strip())
        elif line.startswith("pass_characters:"):
            summary.pass_characters = int(line.split(":", 1)[1].strip())
        elif line.startswith("failure_characters:"):
            summary.failure_characters = int(line.split(":", 1)[1].strip())
        elif line.startswith("failed_ocr_characters:"):
            summary.failed_ocr_characters = int(line.split(":", 1)[1].strip())
        elif line.startswith("exact_match_rate:"):
            summary.exact_match_rate = float(line.split(":", 1)[1].strip().rstrip("%"))
        elif line.startswith("character_match_rate:"):
            summary.character_match_percentage = float(line.split(":", 1)[1].strip().rstrip("%"))
        elif line.startswith("cer:"):
            summary.cer_percentage = float(line.split(":", 1)[1].strip().rstrip("%"))
        elif line.startswith("ocr_word_rows:"):
            summary.ocr_word_rows = int(line.split(":", 1)[1].strip())
        elif " | " in line and not line.startswith("pdf |"):
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) >= 10 and parts[1].isdigit():
                pdf_name = parts[0]
                page_index = int(parts[1])
                page_rows[(pdf_name, page_index)] = {
                    "total": int(parts[2]),
                    "pass": int(parts[3]),
                    "fail": int(parts[4]),
                    "failed_ocr": int(parts[5]),
                    "exact_match_rate": float(parts[6]),
                    "ocr_rows": int(parts[7]),
                    "status": parts[8],
                    "character_match_percentage": float(parts[9]),
                    "cer_percentage": float(parts[10]) if len(parts) >= 11 else 0.0,
                }

    return pdf_dir, summary, page_rows


def load_run_from_disk(run_dir: str | Path) -> RunStats:
    """Rebuild :class:`RunStats` from a completed run folder on disk.

    Expects ``total_statistics.txt`` at the run root and per-page
    ``out_results.txt`` / ``native_mapping.pkl`` under ``<pdf_label>/page_XXXX/``.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")

    total_path = run_dir / "total_statistics.txt"
    pdf_dir = Path(".")
    summary = GlyphSummary()
    page_rows: dict[tuple[str, int], dict] = {}
    if total_path.is_file():
        pdf_dir, summary, page_rows = parse_total_statistics(total_path)

    pages: List[PageStats] = []
    for out_results in sorted(run_dir.rglob("out_results.txt")):
        page_dir = out_results.parent
        page_index = _page_index_from_dir(page_dir)
        pdf_name = _pdf_name_from_page_dir(run_dir, page_dir)
        row = page_rows.get((pdf_name, page_index))

        error: str | None = None
        page_summary = GlyphSummary()
        if row:
            status = row["status"]
            if status != "ok":
                error = status.removeprefix("error: ").strip() or status
            page_summary = GlyphSummary(
                total_native_characters=row["total"],
                pass_characters=row["pass"],
                failure_characters=row["fail"],
                failed_ocr_characters=row["failed_ocr"],
                exact_match_rate=row["exact_match_rate"],
                ocr_word_rows=row["ocr_rows"],
                character_match_percentage=row["character_match_percentage"],
                cer_percentage=row["cer_percentage"],
            )
            if page_summary.cer_percentage == 0.0 and out_results.is_file():
                stats, not_ocred = load_stats_from_out_results(out_results)
                text_chars = page_native_text_char_count(stats, not_ocred)
                page_summary.native_text_characters = text_chars
                page_summary.cer_percentage = page_cer_percentage(stats, not_ocred, text_chars)
        elif out_results.is_file():
            stats, not_ocred = load_stats_from_out_results(out_results)
            page_summary = GlyphSummary.from_comparison(stats, not_ocred)

        pages.append(
            PageStats(
                pdf_path=(pdf_dir / pdf_name).resolve() if pdf_dir != Path(".") else Path(pdf_name),
                page_index=page_index,
                onlyfill_pdf=page_dir / "onlyfill.pdf",
                ocr_pdf=page_dir / "ocr.pdf",
                out_results=out_results,
                out_ocr_txt=page_dir / "out_ocr.txt",
                native_mapping_path=page_dir / "native_mapping.pkl",
                summary=page_summary,
                error=error,
            )
        )

    pages.sort(key=lambda p: (p.pdf_path.name, p.page_index))
    if not summary.total_native_characters and pages:
        summary = GlyphSummary()
        for page in pages:
            if page.ok:
                summary = summary.add(page.summary)
    elif pages and summary.cer_percentage == 0.0:
        ok_summaries = [p.summary for p in pages if p.ok]
        if ok_summaries:
            summary.cer_percentage = run_cer_percentage(ok_summaries)

    return RunStats(
        run_dir=run_dir,
        pdf_dir=pdf_dir,
        pages=pages,
        total_statistics=total_path if total_path.is_file() else None,
        summary=summary,
    )


def list_run_dirs(runs_root: str | Path) -> List[Path]:
    """Return completed run folders (those with ``total_statistics.txt``), newest first."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    runs = [p for p in runs_root.iterdir() if p.is_dir() and (p / "total_statistics.txt").is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def flatten_page_statistics(run: RunStats) -> List[GlyphSummary]:
    return [p.summary for p in run.pages]


def run_character_match_percentage(summaries: List[GlyphSummary]) -> float:
    total_chars = sum(s.total_native_characters for s in summaries)
    return (
        (
            sum(s.character_match_percentage * s.total_native_characters for s in summaries)
            / total_chars
        )
        if total_chars
        else 0.0
    )


def run_cer_percentage(summaries: List[GlyphSummary]) -> float:
    def _weight(s: GlyphSummary) -> int:
        return s.native_text_characters or s.total_native_characters

    total_chars = sum(_weight(s) for s in summaries)
    return (
        (sum(s.cer_percentage * _weight(s) for s in summaries) / total_chars)
        if total_chars
        else 0.0
    )


def format_total_statistics(run: RunStats) -> str:
    """Summary-only report for all pages in a run (glyph OCR stats)."""
    summaries = flatten_page_statistics(run)
    char_match_pct = run_character_match_percentage(summaries)
    cer_pct = run_cer_percentage(summaries)
    lines = [
        "=== OCR pipeline — run total (all PDFs, all pages) ===",
        "",
        f"pdf_dir: {run.pdf_dir}",
        f"run_dir: {run.run_dir}",
        f"pages_total: {len(run.pages)}",
        f"pages_ok: {run.pages_ok}",
        f"pages_failed: {run.pages_failed}",
        "",
        "=== Document totals (by native glyph) ===",
        f"total_native_characters: {run.summary.total_native_characters}",
        f"pass_characters: {run.summary.pass_characters}",
        f"failure_characters: {run.summary.failure_characters}",
        f"failed_ocr_characters: {run.summary.failed_ocr_characters}",
        f"exact_match_characters: {run.summary.pass_characters}",
        f"exact_match_rate: {run.summary.exact_match_rate:.2f}%",
        f"character_match_rate: {char_match_pct:.2f}%",
        f"cer: {cer_pct:.2f}%",
        f"ocr_word_rows: {run.summary.ocr_word_rows}",
        "",
        "=== Per page ===",
        "pdf | page | total | pass | fail | failed_ocr | rate% | ocr_rows | status | character_match_percentage% | cer%",
    ]
    for p in run.pages:
        s = p.summary
        status = "ok" if p.ok else f"error: {p.error}"
        lines.append(
            f"{p.pdf_path.name} | {p.page_index} | {s.total_native_characters} | "
            f"{s.pass_characters} | {s.failure_characters} | {s.failed_ocr_characters} | "
            f"{s.exact_match_rate:.2f} | {s.ocr_word_rows} | {status} | "
            f"{s.character_match_percentage:.2f} | {s.cer_percentage:.2f}",
        )
    lines.append("")
    return "\n".join(lines)
