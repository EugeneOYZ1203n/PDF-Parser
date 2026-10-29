"""Native textbox ↔ word-group flags: correct (1 group), split (>1), joint (shared group)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.ocr.wordgrouping import WordGroup


@dataclass
class TextboxGroupingResult:
    total_native_text: int
    correct_total: int
    split_total: int
    joint_total: int
    split_score: float
    joint_score: float
    f1_score: float

    @classmethod
    def from_counts(
        cls,
        total: int,
        correct: int,
        split: int,
        joint: int,
    ) -> TextboxGroupingResult:
        split_score = split / (split + correct) if (split + correct) else 0.0
        joint_score = joint / (joint + correct) if (joint + correct) else 0.0
        denom = split_score + joint_score
        f1 = (2 * split_score * joint_score / denom) if denom else 0.0
        return cls(total, correct, split, joint, split_score, joint_score, f1)

    @classmethod
    def rollup(cls, results: List[TextboxGroupingResult]) -> TextboxGroupingResult:
        return cls.from_counts(
            sum(r.total_native_text for r in results),
            sum(r.correct_total for r in results),
            sum(r.split_total for r in results),
            sum(r.joint_total for r in results),
        )


def analyze_page_textbox_stats(
    native_words: List[Text],
    native_word_vectors: List[List[Vector]],
    word_groups: List[WordGroup],
) -> TextboxGroupingResult:
    native_to_groups: Dict[int, Set[int]] = defaultdict(set)
    group_to_natives: Dict[int, Set[int]] = defaultdict(set)

    for i, vectors in enumerate(native_word_vectors):
        seqnos = {v.seqno for v in vectors}
        for gi, wg in enumerate(word_groups):
            if seqnos & {g.element_id for g in wg.words}:
                native_to_groups[i].add(gi)
                group_to_natives[gi].add(i)

    correct = split = joint = 0
    for i in range(len(native_words)):
        groups = native_to_groups.get(i, set())
        if len(groups) == 1:
            correct += 1
        if len(groups) > 1:
            split += 1
        if any(len(group_to_natives.get(g, set())) > 1 for g in groups):
            joint += 1

    return TextboxGroupingResult.from_counts(len(native_words), correct, split, joint)


def format_textbox_statistics(
    run: object,
    textbox_by_page: dict[tuple[str, int], TextboxGroupingResult],
    *,
    pdf_dir: Path,
) -> str:
    from raster_parser.evaluation.page_stats import RunStats

    if not isinstance(run, RunStats):
        raise TypeError("run must be RunStats")

    rows = [
        textbox_by_page[(p.pdf_path.name, p.page_index)]
        for p in run.pages
        if p.ok and (p.pdf_path.name, p.page_index) in textbox_by_page
    ]
    total = TextboxGroupingResult.rollup(rows) if rows else None

    lines = [
        "=== Textbox grouping (native text vs word groups) ===",
        "",
        f"pdf_dir: {pdf_dir}",
        f"run_dir: {run.run_dir}",
        f"pages_with_stats: {len(rows)}",
        f"pages_total: {len(run.pages)}",
        "",
    ]
    if total:
        lines.extend(
            [
                "=== Run totals ===",
                f"native_text: {total.total_native_text}",
                f"correct: {total.correct_total}",
                f"split: {total.split_total}",
                f"joint: {total.joint_total}",
                f"split_score: {total.split_score:.4f}",
                f"joint_score: {total.joint_score:.4f}",
                f"f1: {total.f1_score:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "=== Per page ===",
            "pdf | page | native_text | correct | split | joint | split_score | joint_score | f1",
        ]
    )
    for p in run.pages:
        tb = textbox_by_page.get((p.pdf_path.name, p.page_index))
        if tb is None:
            lines.append(
                f"{p.pdf_path.name} | {p.page_index} | n/a | n/a | n/a | n/a | n/a | n/a | n/a"
            )
        else:
            lines.append(
                f"{p.pdf_path.name} | {p.page_index} | {tb.total_native_text} | "
                f"{tb.correct_total} | {tb.split_total} | {tb.joint_total} | "
                f"{tb.split_score:.4f} | {tb.joint_score:.4f} | {tb.f1_score:.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def compute_textbox_grouping_for_page(
    page: object,
    *,
    run_dir: Path | None = None,
    pdf_dir: Path | None = None,
) -> TextboxGroupingResult | None:
    from raster_parser.evaluation.page_stats import PageStats, load_native_mapping_for_page
    from raster_parser.parsing.parser import PdfRasterParser
    from raster_parser.rendering.pdf_render.reconstruct import filter_text_vectors
    from raster_parser.ocr.wordgrouping import cluster_by_seqno, convert_vectors_to_glyphs

    if not isinstance(page, PageStats):
        raise TypeError("page must be PageStats")
    if not Path(page.onlyfill_pdf).is_file():
        return None

    mapping = load_native_mapping_for_page(page)
    if not mapping:
        return None

    pdf = _resolve_source_pdf(page, run_dir=run_dir, pdf_dir=pdf_dir)
    if pdf is None:
        return None

    source = PdfRasterParser(str(pdf), page.page_index)
    source.extract_text()
    ordered = [mapping[k][1] for k in sorted(mapping)]
    words, vecs = _word_vectors(source.text, ordered)

    of = PdfRasterParser(str(page.onlyfill_pdf), page=0)
    of.extract_vectors()
    fills = filter_text_vectors(of.vectors)
    groups = cluster_by_seqno(convert_vectors_to_glyphs(fills), int(of.page.rotation or 0))
    return analyze_page_textbox_stats(words, vecs, groups)


def _word_vectors(
    words: List[Text], ordered: List[Vector]
) -> Tuple[List[Text], List[List[Vector]]]:
    out_w: List[Text] = []
    out_v: List[List[Vector]] = []
    idx = 0
    for w in words:
        t = (w.word or "").strip()
        if not t:
            continue
        chunk = ordered[idx : idx + len(t)]
        if len(chunk) != len(t):
            raise ValueError(f"glyph mismatch at {t!r}")
        out_w.append(w)
        out_v.append(chunk)
        idx += len(t)
    return out_w, out_v


def _resolve_source_pdf(
    page: object,
    *,
    run_dir: Path | None = None,
    pdf_dir: Path | None = None,
) -> Path | None:
    from raster_parser.evaluation.page_stats import PageStats

    if not isinstance(page, PageStats):
        return None

    page_dir = Path(page.out_results).parent
    label = page_dir.parent.name if page_dir.name.startswith("page_") else page_dir.stem
    pdf_name = label if label.endswith(".pdf") else f"{label}.pdf"
    rel = label.replace("__", "/")
    rel_pdf = rel if rel.endswith(".pdf") else f"{rel}.pdf"

    candidates: List[Path] = []
    if page.pdf_path and Path(page.pdf_path).is_file():
        candidates.append(Path(page.pdf_path))
    if pdf_dir is not None:
        root = pdf_dir.resolve()
        candidates += [
            root / rel_pdf,
            root / pdf_name,
            root / page.pdf_path.name,
        ]
    if run_dir is not None:
        candidates += [
            run_dir / pdf_name,
            run_dir / label / pdf_name,
        ]

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c.resolve()

    if pdf_dir is not None:
        for match in pdf_dir.resolve().rglob(pdf_name):
            if match.is_file():
                return match.resolve()
    return None
