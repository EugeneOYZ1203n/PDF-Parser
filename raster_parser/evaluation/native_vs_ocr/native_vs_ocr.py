from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.models import GlyphOrientation


@dataclass
class Stats:
    match: bool
    ocr_num: int
    native_nums: List[int]
    native_char: str
    ocr_char: str
    IOU: float
    CER: float
    failed_ocr: bool = False
    glyph_orientation: GlyphOrientation = GlyphOrientation.HORIZONTAL
    native_rotate: int = 0
    character_match_rate: float = 0.0


def _rect_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _union_rect(rects: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = zip(*rects)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]



def _text_for_compare(text: str) -> str:
    """Strip whitespace; uppercase ambiguous letters zxcvmskwuop (case-insensitive)."""
    target_chars = set("zxcvmskwuop")
    return "".join(
        c.upper() if c.lower() in target_chars else c
        for c in "".join(text.split())
    )



def _cer(native: str, ocr: str) -> float:
    native = _text_for_compare(native)
    ocr = _text_for_compare(ocr)
    if not native:
        return 0.0 if not ocr else 1.0
    return _edit_distance(native, ocr) / len(native)


def _character_match_rate(native_string: str, ocr_word: str) -> float:
    """Fraction of native chars with a positional match in OCR (whitespace stripped)."""
    native = _text_for_compare(native_string)
    if not native:
        return 0.0
    return _positional_match_count(native_string, ocr_word) / len(native)


def _positional_match_count(native_string: str, ocr_word: str) -> int:
    native = _text_for_compare(native_string)
    ocr = _text_for_compare(ocr_word)
    return sum(1 for i in range(len(native)) if i < len(ocr) and native[i] == ocr[i])


def _native_indices(
    vector_seqnos: List[int],
    native_text: Dict[int, Tuple[str, Vector]],
) -> List[int]:
    return [s for s in vector_seqnos if s in native_text]


class NativeVsOcr:
    """Compare SVG native glyphs to Paddle OCR rows.

    Each OCR row is ``(Text, vector_seqnos)`` from the word-group OCR step;
    ``vector_seqnos`` are ``Vector.seqno`` values clustered into that group.
    """

    def __init__(
        self,
        native_text: Dict[int, Tuple[str, Vector]],
        ocr_txt_with_mapping: List[Tuple[Text, List[int]]],
    ):
        self.native_text = native_text
        self.ocr_txt_with_mapping = ocr_txt_with_mapping

    def _native_string(self, native_nums: List[int]) -> str:
        ordered = sorted(native_nums, key=lambda i: self.native_text[i][1].seqno)
        return "".join(self.native_text[i][0] for i in ordered)

    def native_glyphs_not_ocred(self) -> List[Tuple[int, str]]:
        """Native glyph seqnos not present in any OCR word-group mapping."""
        assigned: set[int] = set()
        for _, vector_seqnos in self.ocr_txt_with_mapping:
            assigned.update(_native_indices(vector_seqnos, self.native_text))
        return [
            (idx, self.native_text[idx][0])
            for idx in sorted(self.native_text)
            if idx not in assigned
        ]

    def _bbox_iou(self, text: Text, native_nums: List[int]) -> float:
        if not native_nums:
            return 0.0
        ocr_rect = (text.x0, text.y0, text.x1, text.y1)
        glyph_rects = [self.native_text[i][1].rect for i in native_nums]
        return _rect_iou(ocr_rect, _union_rect(glyph_rects))

    def compare_words(self) -> List[Stats]:
        results: List[Stats] = []
        for ocr_num, (text, vector_seqnos) in enumerate(self.ocr_txt_with_mapping):
            native_nums = _native_indices(vector_seqnos, self.native_text)
            native_string = self._native_string(native_nums)
            results.append(
                self.compare_words_stats(ocr_num, text, native_nums, native_string, text.word)
            )
        return results

    def compare_words_stats(
        self,
        ocr_num: int,
        text: Text,
        native_nums: List[int],
        native_string: str,
        ocr_word: str,
    ) -> Stats:  
        iou = self._bbox_iou(text, native_nums)
        cer = _cer(native_string, ocr_word)
        match = _text_for_compare(native_string) == _text_for_compare(ocr_word)
        char_match = _character_match_rate(native_string, ocr_word)
        return Stats(
            match=match,
            ocr_num=ocr_num,
            native_nums=native_nums,
            native_char=native_string,
            ocr_char=ocr_word,
            IOU=iou,
            CER=cer,
            failed_ocr=False,
            glyph_orientation=text.glyph_orientation,
            native_rotate=int(text.rotate or 0),
            character_match_rate=char_match,
        )


def _char_outcomes(
    stats: List[Stats],
    not_ocred: List[Tuple[int, str]],
) -> Dict[int, str]:
    """Map each native glyph index to pass | fail | failed_ocr."""
    outcome: Dict[int, str] = {}
    for s in stats:
        label = "pass" if s.match else "fail"
        for idx in s.native_nums:
            if outcome.get(idx) != "pass":
                outcome[idx] = label
    for idx, _ in not_ocred:
        outcome[idx] = "failed_ocr"
    return outcome


def _format_ocr_line(s: Stats) -> str:
    ori = s.glyph_orientation.name
    return (
        f"ocr[{s.ocr_num}] failed_ocr={s.failed_ocr} orientation={ori} "
        f"rotate={s.native_rotate} match={s.match} IOU={s.IOU:.4f} CER={s.CER:.4f} "
        f"character_match_rate={s.character_match_rate:.4f} "
        f"native_glyphs={s.native_nums} native={s.native_char!r} ocr={s.ocr_char!r}"
    )


def format_stats_report(
    stats: List[Stats],
    not_ocred: List[Tuple[int, str]] | None = None,
    *,
    total_native_characters: int | None = None,
) -> str:
    """Format per-word stats and summary for ``out_results.txt``."""
    lines = [
        "=== Native vs OCR comparison ===",
        "",
        "Ground truth: SVG data-text glyphs by vector seqno (OCR word-group mapping).",
        "OCR text: one Paddle string per vector word-group (same box as onlyfill).",
        "match/CER/character_match_rate: whitespace ignored (glyph runs have no spaces).",
        "character_match_rate: native[i]==ocr[i] at each index / len(native); shorter OCR → rest fail.",
        "orientation/rotate: deduced word-group orientation and PDF draw degrees (ocr.pdf).",
        "exact_match_rate: exact_match_characters / total_native_characters.",
        "character_match_rate (page): positional matches / total_native_characters (failed_ocr = 0).",
        "cer (page): sum(edit_distance) / native text length × 100 (failed_ocr = full miss).",
        "",
    ]
    not_ocred = not_ocred or []
    total = total_native_characters if total_native_characters is not None else (
        len({idx for s in stats for idx in s.native_nums}) + len(not_ocred)
    )
    if total == 0 and not stats and not not_ocred:
        lines.append("(no native glyphs to compare)")
        return "\n".join(lines) + "\n"

    outcomes = _char_outcomes(stats, not_ocred)
    pass_chars = sum(1 for v in outcomes.values() if v == "pass")
    fail_chars = sum(1 for v in outcomes.values() if v == "fail")
    failed_ocr_chars = sum(1 for v in outcomes.values() if v == "failed_ocr")
    rate = (100.0 * pass_chars / total) if total else 0.0
    char_match_pct = character_match_percentage(stats, not_ocred, total)
    cer_pct = cer_percentage(stats, not_ocred, total)

    pass_rows = [s for s in stats if s.match and s.native_char]
    fail_rows = [s for s in stats if not s.match and s.native_char]
    empty_ocr_rows = [s for s in stats if not s.native_char]

    lines.extend([
        "=== Summary (by native character) ===",
        f"total_native_characters: {total}",
        f"pass_characters: {pass_chars}",
        f"failure_characters: {fail_chars}",
        f"failed_ocr_characters: {failed_ocr_chars}",
        f"exact_match_characters: {pass_chars}",
        f"exact_match_rate: {rate:.2f}%",
        f"character_match_rate: {char_match_pct:.2f}%",
        f"cer: {cer_pct:.2f}%",
        "",
        f"ocr_word_rows: {len(stats)}",
        f"ocr_rows_pass: {len(pass_rows)}",
        f"ocr_rows_failure: {len(fail_rows)}",
        f"ocr_rows_no_native_in_box: {len(empty_ocr_rows)}",
        "",
    ])

    if pass_rows:
        lines.append(f"=== Pass ({pass_chars} characters, {len(pass_rows)} OCR rows) ===")
        for s in pass_rows:
            lines.append(_format_ocr_line(s))
        lines.append("")

    if fail_rows:
        lines.append(
            f"=== Failure ({fail_chars} characters, {len(fail_rows)} OCR rows) ==="
        )
        for s in fail_rows:
            lines.append(_format_ocr_line(s))
        lines.append("")

    if empty_ocr_rows:
        lines.append(
            f"=== Failure — OCR box with no native glyphs ({len(empty_ocr_rows)} rows) ==="
        )
        for s in empty_ocr_rows:
            lines.append(_format_ocr_line(s))
        lines.append("")

    if not_ocred:
        lines.append(
            f"=== Failed OCR ({failed_ocr_chars} characters, no OCR word box) ==="
        )
        for idx, char in not_ocred:
            lines.append(
                f"native[{idx}] failed_ocr=True orientation=n/a rotate=n/a "
                f"native={char!r} ocr=''"
            )
        lines.append("")

    return "\n".join(lines) + "\n"

def character_match_percentage(
    stats: List[Stats],
    not_ocred: List[Tuple[int, str]],
    total: int | None = None,
) -> float:
    if total is None:
        total = len({idx for s in stats for idx in s.native_nums}) + len(not_ocred)
    if total == 0:
        return 0.0
    matched = sum(_positional_match_count(s.native_char, s.ocr_char) for s in stats)
    return 100.0 * matched / total


def native_text_char_count(
    stats: List[Stats],
    not_ocred: List[Tuple[int, str]],
) -> int:
    return sum(len(_text_for_compare(s.native_char)) for s in stats) + sum(
        len(_text_for_compare(c)) for _, c in not_ocred
    )


def cer_percentage(
    stats: List[Stats],
    not_ocred: List[Tuple[int, str]],
    total: int | None = None,
) -> float:
    """Page-level CER as a percentage (0–100), summed edit distance / native text length."""
    char_total = total if total is not None else native_text_char_count(stats, not_ocred)
    if char_total == 0:
        return 0.0
    edit_sum = sum(
        _edit_distance(_text_for_compare(s.native_char), _text_for_compare(s.ocr_char))
        for s in stats
    )
    for _, char in not_ocred:
        edit_sum += _edit_distance(_text_for_compare(char), "")
    return 100.0 * edit_sum / char_total


