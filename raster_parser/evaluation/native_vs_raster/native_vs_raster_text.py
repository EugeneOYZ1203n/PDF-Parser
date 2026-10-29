"""Compare native (PDF-extracted) words against raster/OCR words on the same page.

Both sides are ``Text`` objects with an axis-aligned bbox in PDF *points*. The goal is to
answer, per word: did OCR find the native word, and did it read the same string?

The hard part is TOKENIZATION: native extraction and OCR rarely split text the same way. OCR
often returns a whole line/word-group as one box ("MATCHLINE AREA") while native has two words
("MATCHLINE", "AREA"), or vice-versa ("MATCH" + "LINE" vs "MATCHLINE"). A 1:1 box match would
score these as mismatches even though the text is identical.

Algorithm
---------
1. CLUSTER -- build a bipartite overlap graph: connect a native box and an OCR box whenever
   their bboxes overlap enough (IoU >= thresh, or centers within a point tolerance). Union-find
   groups everything that transitively overlaps into one cluster, so a big OCR box that covers
   two native words pulls all three into the same cluster.
2. STITCH -- within a cluster, concatenate each side's words in reading order (top-to-bottom,
   left-to-right) with ALL whitespace removed, so differing splits/spacing collapse away.
3. COMPARE -- the cluster "agrees" if either:
     (a) the two stitched strings are exactly equal (case-folded; punctuation optionally
         dropped), or the reverse token order matches for vertically-stacked clusters; or
     (b) every native word appears verbatim in the OCR stitch (any order) and OCR has no
         leftover characters -- so ``linesetbackbuilding5m`` vs ``5mbuildingsetbackline``
         agrees when the words are the same bag (reading-order / grouping mismatch only).
   No fuzzy character similarity beyond that. Vertically-stacked clusters additionally
   accept reverse token order for the exact-string path.
4. LEFTOVERS -- native words in a cluster with no OCR box were missed by OCR; OCR words in a
   cluster with no native box are extra (hallucinated / artefacts past the confidence gate).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from raster_parser.models import Text

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def _norm(word: str, *, drop_punct: bool) -> str:
    """Canonical form for exact comparison: case-folded, ALL whitespace removed."""
    w = _WS.sub("", word).casefold()
    if drop_punct:
        w = _PUNCT.sub("", w)
    return w


def _iou(a: Text, b: Text) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x1 - a.x0) * max(0.0, a.y1 - a.y0)
    area_b = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_stacked_texts(texts: list[Text], *, iou_thresh: float = 0.9) -> list[Text]:
    """Drop CAD/PDF overlays of the same string painted on (nearly) the same box.

    AEC sheets often embed the same label 2–4× with bit-identical (or IoU≈1) bboxes
    from nested XObjects. Visually one word; ``get_text("words")`` returns all copies,
    which then stitch to ``wordwordword`` vs a single OCR read and tank exact-match accuracy.
    Keeps the first occurrence in list order.
    """
    kept: list[Text] = []
    for t in texts:
        dup = False
        for k in kept:
            if t.word != k.word:
                continue
            if _iou(t, k) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def _center_dist(a: Text, b: Text) -> float:
    acx, acy = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2
    bcx, bcy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _containment(a: Text, b: Text) -> float:
    """Overlap relative to the SMALLER box: intersection / min(area_a, area_b).

    IoU underlinks across granularities -- a small native word inside a wide OCR line-box
    has intersection == the word's area, so IoU == word_area / box_area, which is tiny and
    fails an IoU gate even though the word is fully inside the box. Containment is ~1.0 in
    that case, so a word contained in a line/word-group box still links to it.
    """
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x1 - a.x0) * max(0.0, a.y1 - a.y0)
    area_b = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _ordered_norm_tokens(tokens: list[Text], *, drop_punct: bool) -> list[str]:
    """Cluster's normalized words in reading order (top-to-bottom, then left-to-right).

    Rows are quantized by half the median token height so words on the same visual line sort
    together before falling to the next row.
    """
    if not tokens:
        return []
    heights = sorted(max(1.0, t.y1 - t.y0) for t in tokens)
    tol = max(1.0, heights[len(heights) // 2] * 0.5)
    ordered = sorted(tokens, key=lambda t: (round(((t.y0 + t.y1) / 2) / tol), (t.x0 + t.x1) / 2))
    return [_norm(t.word, drop_punct=drop_punct) for t in ordered]


def _stitch(tokens: list[Text], *, drop_punct: bool) -> str:
    """Reading-order concatenation of a cluster's words, whitespace removed."""
    return "".join(_ordered_norm_tokens(tokens, drop_punct=drop_punct))


def _stitch_reversed(tokens: list[Text], *, drop_punct: bool) -> str:
    """Reading-order concatenation with the token order REVERSED (see _is_vertical_stack)."""
    return "".join(reversed(_ordered_norm_tokens(tokens, drop_punct=drop_punct)))


def _is_vertical_stack(tokens: list[Text]) -> bool:
    """True when a cluster's tokens are stacked vertically (spread more in y than in x).

    A vertical schedule tag like ``PC-160 / /2HR / /STC50`` stacks its words along y, and the
    reading direction is ambiguous: native words emerge geometrically top-to-bottom while the
    recognizer often reads the rotated label bottom-to-top, so the two stitched strings come
    out token-reversed even though the content is identical. Only such clusters get reversal
    tolerance; a normal left-to-right line (spread mostly in x) never does.
    """
    if len(tokens) < 2:
        return False
    xs = [(t.x0 + t.x1) / 2 for t in tokens]
    ys = [(t.y0 + t.y1) / 2 for t in tokens]
    return (max(ys) - min(ys)) > (max(xs) - min(xs))


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _positional_matches(a: str, b: str) -> int:
    """Count of positions where a[i] == b[i] (shorter b => the rest of a is unmatched)."""
    return sum(1 for i in range(len(a)) if i < len(b) and a[i] == b[i])


def _greedy_word_hits(native_norms: list[str], ocr_str: str) -> int:
    """Granularity-robust per-word recall: how many native words appear verbatim in OCR.

    Matches in **any order** (longest word first) so a correct bag of words read in a
    different sequence -- e.g. native ``LINE SETBACK BUILDING 5M`` vs OCR
    ``5M BUILDING SETBACK LINE`` -- still scores full hits. Spans are not reused.
    """
    hits, _ = _cover_words_any_order(native_norms, ocr_str)
    return hits


def _cover_words_any_order(native_norms: list[str], ocr_str: str) -> tuple[int, int]:
    """Find native words as non-overlapping substrings of ``ocr_str`` in any order.

    Returns ``(hits, leftover_chars)``. Longest words are claimed first so shorter tokens
    do not steal spans inside longer ones (``set`` inside ``setback``).
    """
    if not ocr_str:
        return (0, 0)
    remaining = list(ocr_str)
    hits = 0
    for w in sorted((x for x in native_norms if x), key=len, reverse=True):
        s = "".join(remaining)
        idx = s.find(w)
        if idx < 0:
            continue
        hits += 1
        for i in range(idx, idx + len(w)):
            remaining[i] = "\0"
    leftover = sum(1 for c in remaining if c != "\0")
    return hits, leftover


def _same_word_bag(native_norms: list[str], ocr_str: str) -> bool:
    """True when OCR is exactly the native words concatenated in some order (no extras)."""
    words = [w for w in native_norms if w]
    if not words or not ocr_str:
        return False
    if sum(len(w) for w in words) != len(ocr_str):
        return False
    hits, leftover = _cover_words_any_order(words, ocr_str)
    return hits == len(words) and leftover == 0


@dataclass
class GroupMatch:
    native: list[Text]  # native words stitched into this cluster
    raster: list[Text]  # OCR words stitched into this cluster
    native_str: str  # stitched native text (normalized, forward reading order)
    raster_str: str  # stitched OCR text (normalized)
    same: bool  # exact string equality after stitching
    n_chars: int = 0  # native character count (comparison basis)
    match_chars: int = 0  # positional char matches (best of forward/reverse)
    edit_chars: int = 0  # edit distance native<->ocr (best of forward/reverse)
    word_hits: int = 0  # granularity-robust: native words found verbatim in OCR stitch


@dataclass
class ComparisonResult:
    groups: list[GroupMatch] = field(default_factory=list)  # clusters with both native + OCR
    missed: list[Text] = field(default_factory=list)  # native, no overlapping OCR
    extra: list[Text] = field(default_factory=list)  # OCR, no overlapping native

    @property
    def agree(self) -> list[GroupMatch]:
        return [g for g in self.groups if g.same]

    @property
    def disagree(self) -> list[GroupMatch]:
        return [g for g in self.groups if not g.same]

    def summary(self) -> dict[str, float | int]:
        native_words = sum(len(g.native) for g in self.groups) + len(self.missed)
        raster_words = sum(len(g.raster) for g in self.groups) + len(self.extra)
        agree = sum(len(g.native) for g in self.groups if g.same)
        disagree = sum(len(g.native) for g in self.groups if not g.same)

        # Character-level metrics. Missed native words count as a full miss (0 matches, edit
        # distance == their length) so recall is reflected, matching ocr_test_pipeline.
        missed_chars = sum(len(_norm(t.word, drop_punct=False)) for t in self.missed)
        native_chars = sum(g.n_chars for g in self.groups) + missed_chars
        match_chars = sum(g.match_chars for g in self.groups)
        edit_chars = sum(g.edit_chars for g in self.groups) + missed_chars
        # Granularity-robust per-word recall: native words found verbatim in their OCR cluster
        # (missed words contribute 0). Not penalized by detector split/merge differences, so it
        # is the closest apples-to-apples with ocr_test_pipeline's per-word (seqno) scoring.
        word_hits = sum(g.word_hits for g in self.groups)
        return {
            "native_words": native_words,
            "raster_words": raster_words,
            "matched": agree + disagree,  # native words that landed in a both-sided cluster
            "agree": agree,  # ... whose stitched text matched exactly
            "disagree": disagree,
            "missed": len(self.missed),
            "extra": len(self.extra),
            # exact stitched-cluster recall (sensitive to detector split/merge)
            "accuracy": (agree / native_words) if native_words else 0.0,
            # granularity-robust per-word recall (detector split/merge insensitive)
            "word_hits": word_hits,
            "word_accuracy": (word_hits / native_words) if native_words else 0.0,
            # character-level (near-miss aware)
            "native_chars": native_chars,
            "match_chars": match_chars,
            "edit_chars": edit_chars,
            "char_match_rate": (match_chars / native_chars) if native_chars else 0.0,
            "cer": (edit_chars / native_chars) if native_chars else 0.0,
        }


class NativeVsRasterText:
    def __init__(self, native_text: list[Text], raster_text: list[Text]):
        self.native_text = native_text
        self.raster_text = raster_text

    def compare(
        self,
        *,
        iou_thresh: float = 0.3,
        contain_thresh: float = 0.5,
        center_tol: float = 0.0,
        drop_punct: bool = False,
    ) -> ComparisonResult:
        """Cluster overlapping boxes, stitch each side's words, then compare exactly.

        - ``iou_thresh``    : min IoU for a native and OCR box to be linked into a cluster.
        - ``contain_thresh``: min containment (intersection / smaller-box area) to link. This is
                              what lets a small native word link to the wide OCR line/word-group
                              box that contains it, which plain IoU misses. Set to >1 to disable.
        - ``center_tol``    : if > 0, also link a pair whose bbox centers are within this many
                              points (rescues links where box sizes differ but are well-centered).
        - ``drop_punct``    : ignore punctuation when comparing stitched strings.
        """
        n, m = len(self.native_text), len(self.raster_text)
        # Union-find over native nodes [0..n) and raster nodes [n..n+m).
        parent = list(range(n + m))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, nt in enumerate(self.native_text):
            for j, rt in enumerate(self.raster_text):
                iou = _iou(nt, rt)
                linked = iou >= iou_thresh or _containment(nt, rt) >= contain_thresh
                if not linked and center_tol > 0 and iou > 0:
                    linked = _center_dist(nt, rt) <= center_tol
                if linked:
                    union(i, n + j)

        # Bucket every node by its cluster root.
        clusters: dict[int, tuple[list[Text], list[Text]]] = {}
        for i in range(n):
            clusters.setdefault(find(i), ([], []))[0].append(self.native_text[i])
        for j in range(m):
            clusters.setdefault(find(n + j), ([], []))[1].append(self.raster_text[j])

        result = ComparisonResult()
        for natives, rasters in clusters.values():
            if natives and rasters:
                ns = _stitch(natives, drop_punct=drop_punct)
                rs = _stitch(rasters, drop_punct=drop_punct)
                # Vertical/stacked labels have an ambiguous reading direction (top-to-bottom vs
                # bottom-to-top), so the native and OCR stitches can be token-reversed while the
                # content is identical. For a vertically-stacked cluster, also try the native
                # stitch in reverse token order and keep the better of the two orientations.
                native_norms = _ordered_norm_tokens(natives, drop_punct=drop_punct)
                rs_rev = "".join(reversed(_ordered_norm_tokens(rasters, drop_punct=drop_punct)))
                bag_same = _same_word_bag(native_norms, rs)
                if _is_vertical_stack(natives) or _is_vertical_stack(rasters):
                    ns_rev = _stitch_reversed(natives, drop_punct=drop_punct)
                    same = ns == rs or ns_rev == rs or bag_same
                    match_chars = max(_positional_matches(ns, rs), _positional_matches(ns_rev, rs))
                    edit_chars = min(_edit_distance(ns, rs), _edit_distance(ns_rev, rs))
                    # any-order cover already; still take max against reversed OCR stitch
                    # in case OCR split into multiple tokens whose stitch order is odd.
                    word_hits = max(
                        _greedy_word_hits(native_norms, rs), _greedy_word_hits(native_norms, rs_rev)
                    )
                else:
                    same = ns == rs or bag_same
                    match_chars = _positional_matches(ns, rs)
                    edit_chars = _edit_distance(ns, rs)
                    word_hits = _greedy_word_hits(native_norms, rs)
                # When we only agreed via word-bag (order mismatch), char metrics still
                # reflect the stitch difference; word_hits should be full cover.
                if bag_same and ns != rs:
                    word_hits = max(word_hits, sum(1 for w in native_norms if w))
                    edit_chars = 0
                    match_chars = max(match_chars, len(ns))
                result.groups.append(
                    GroupMatch(
                        native=natives,
                        raster=rasters,
                        native_str=ns,
                        raster_str=rs,
                        same=same,
                        n_chars=len(ns),
                        match_chars=match_chars,
                        edit_chars=edit_chars,
                        word_hits=word_hits,
                    )
                )
            elif natives:
                result.missed.extend(natives)
            else:
                result.extra.extend(rasters)
        return result


def format_report(result: ComparisonResult, *, page_label: str = "") -> str:
    """Render a detailed, file-dependent stats report for one page.

    Mirrors the ocr_test_pipeline ``out_results.txt`` style: an explanatory header, a summary
    block, then per-cluster detail sections (agree / disagree) and the leftover native (missed)
    and OCR (extra) words.
    """
    s = result.summary()
    lines = [
        f"=== Native vs OCR comparison (raster){f'  [{page_label}]' if page_label else ''} ===",
        "",
        "Ground truth: native PDF words (page.get_text), de-duped when the same string is",
        "stacked on (nearly) the same bbox (CAD overlays). OCR text: recognized clean-raster boxes.",
        "Words are clustered by bbox IoU/containment, stitched per side in reading order, then",
        "compared exactly (case-folded, all whitespace removed), or as the same bag of",
        "native words covering the OCR string with no leftover chars (order-tolerant).",
        "matched: native words that landed in a both-sided cluster.",
        "agree:   native words in a cluster whose stitch matched or word-bag matched.",
        "missed:  native words with no overlapping OCR box.  extra: OCR words with no native.",
        "word_accuracy: native words found verbatim in their OCR cluster / native_words",
        "  (granularity-robust: not penalized by detector split/merge; ~ ocr_test_pipeline).",
        "char_match_rate: positional char matches / native chars (near-miss aware).",
        "cer: edit distance / native chars (missed = full miss).",
        "",
        "=== Summary (by native word) ===",
        f"native_words:    {s['native_words']}",
        f"ocr_words:       {s['raster_words']}",
        f"matched:         {s['matched']}",
        f"agree:           {s['agree']}",
        f"disagree:        {s['disagree']}",
        f"missed:          {s['missed']}",
        f"extra:           {s['extra']}",
        f"word_accuracy:   {s['word_accuracy']:.2%}  ({s['word_hits']} words)",
        "",
        "=== Summary (by character) ===",
        f"native_chars:    {s['native_chars']}",
        f"char_match_rate: {s['char_match_rate']:.2%}",
        f"cer:             {s['cer']:.2%}",
        "",
        f"groups: {len(result.groups)} (both-sided clusters)",
        "",
    ]

    agree, disagree = result.agree, result.disagree
    if agree:
        lines.append(
            f"=== Agree ({sum(len(g.native) for g in agree)} native words, {len(agree)} groups) ==="
        )
        for g in agree:
            lines.append(f"[n={len(g.native)}/{len(g.raster)}] {g.native_str!r}")
        lines.append("")

    if disagree:
        lines.append(
            f"=== Disagree ({sum(len(g.native) for g in disagree)} native words, {len(disagree)} groups) ==="
        )
        for g in disagree:
            lines.append(
                f"[n={len(g.native)}/{len(g.raster)}] native={g.native_str!r} ocr={g.raster_str!r}"
            )
        lines.append("")

    if result.missed:
        lines.append(f"=== Missed native words ({len(result.missed)}) ===")
        lines.append(", ".join(repr(t.word) for t in result.missed))
        lines.append("")

    if result.extra:
        lines.append(f"=== Extra OCR words ({len(result.extra)}) ===")
        lines.append(", ".join(repr(t.word) for t in result.extra))
        lines.append("")

    return "\n".join(lines) + "\n"
