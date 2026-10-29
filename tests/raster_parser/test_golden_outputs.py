"""Regression guardrail: `extract()` on the committed fixture PDFs still gives the stored output.

Snapshots live in `tests/references/golden/<p2>_<p3>/` and are written by
`scripts/golden_snapshot.py write` (compact: text, bbox, direction, source per text; type, rect,
colours, width, item count, seqno per vector). Coordinates compare within 1e-3, everything else
exactly. Real LibreOffice + PaddleOCR, so `slow`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "references" / "golden"
sys.path.insert(0, str(REPO / "scripts"))
from golden_snapshot import diff, snapshot  # noqa: E402

# golden folder -> (p2, p3) backend names
COMBOS = {
    "v1_v1": ("v1_raster_to_vec", "v1_vector_class"),
    "v1_v2": ("v1_raster_to_vec", "v2_vector_class"),
}

CASES = [(combo, path) for combo in COMBOS for path in sorted((GOLDEN / combo).glob("*_p*.json"))]


@pytest.mark.slow
@pytest.mark.parametrize("combo,path", CASES, ids=[f"{c}/{p.stem}" for c, p in CASES])
def test_extract_matches_golden(combo: str, path: Path, tmp_path: Path) -> None:
    pytest.importorskip("paddleocr")
    from raster_parser import extract

    stem, _, page = path.stem.rpartition("_p")
    p2, p3 = COMBOS[combo]
    result = extract(
        REPO / "tests" / "references" / f"{stem}.pdf",
        int(page),
        p2=p2,
        p3=p3,
        dump_dir=tmp_path / "dump",
    )
    problems = diff(json.loads(path.read_text(encoding="utf-8")), snapshot(result))
    assert not problems, "\n".join(problems)
