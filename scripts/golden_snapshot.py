"""Snapshot `extract()` output as compact, diffable JSON -- the regression guardrail for refactors.

    python scripts/golden_snapshot.py write  OUT_DIR --p2 NAME --p3 NAME [PDF[:PAGE] ...]
    python scripts/golden_snapshot.py compare OUT_DIR --p2 NAME --p3 NAME [PDF[:PAGE] ...]

PDF paths are relative to the working directory. With no PDFs given it uses `tests/references/test_pdfs_1..5.pdf` page 0. One JSON per
(pdf, page): texts (text, bbox, direction, source) and vectors (type, rect, colours, width,
item count, seqno), each in output order. `compare` re-runs and reports differences against what
`write` stored; it exits non-zero on any. `tests/raster_parser/test_golden_outputs.py` uses the
same functions on the committed snapshots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUTS = [f"tests/references/test_pdfs_{i}.pdf:0" for i in range(1, 6)]
TOL = 1e-3


def _r(values, nd: int = 4):
    return None if values is None else [round(float(v), nd) for v in values]


def snapshot(result) -> dict:
    """The compact form of one `ExtractResult`."""
    return {
        "texts": [
            {"text": t.text, "bbox": _r(t.bbox), "direction": _r(t.direction), "source": t.source}
            for t in result.texts
        ],
        "vectors": [
            {
                "type": v.type,
                "rect": _r(v.rect),
                "color": _r(v.color),
                "fill": _r(v.fill),
                "width": None if v.width is None else round(float(v.width), 4),
                "n_items": len(v.items),
                "seqno": v.seqno,
            }
            for v in result.vectors
        ],
    }


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return len(a) == len(b) and all(abs(x - y) <= TOL for x, y in zip(a, b))


def diff(expected: dict, actual: dict, limit: int = 10) -> list[str]:
    """Human-readable differences; empty when `actual` matches `expected` within `TOL`."""
    out: list[str] = []
    for kind, numeric in (("texts", ("bbox", "direction")), ("vectors", ("rect", "color", "fill"))):
        exp, act = expected[kind], actual[kind]
        if len(exp) != len(act):
            out.append(f"{kind}: count {len(exp)} -> {len(act)}")
        for i, (e, a) in enumerate(zip(exp, act)):
            bad = [
                k
                for k in e
                if (k in numeric and not _close(e[k], a.get(k)))
                or (k not in numeric and k != "width" and e[k] != a.get(k))
                or (
                    k == "width"
                    and not _close(
                        None if e[k] is None else [e[k]], None if a.get(k) is None else [a[k]]
                    )
                )
            ]
            if bad:
                out.append(
                    f"{kind}[{i}] {', '.join(f'{k}: {e[k]!r} -> {a.get(k)!r}' for k in bad)}"
                )
            if len(out) >= limit:
                return out + ["..."]
    return out


def snapshot_name(pdf: str, page: int) -> str:
    return f"{Path(pdf).stem}_p{page}.json"


def run_one(pdf: str, page: int, p2: str, p3: str) -> dict:
    from raster_parser import extract

    return snapshot(extract(Path(pdf), page, p2=p2, p3=p3))


def _parse_inputs(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items or DEFAULT_INPUTS:
        pdf, _, page = (
            item.rpartition(":") if item.rsplit(":", 1)[-1].isdigit() else (item, "", "0")
        )
        out.append((pdf, int(page)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["write", "compare"])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--p3", required=True)
    ap.add_argument("inputs", nargs="*")
    args = ap.parse_intermixed_args(argv)

    failures = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pdf, page in _parse_inputs(args.inputs):
        path = args.out_dir / snapshot_name(pdf, page)
        snap = run_one(pdf, page, args.p2, args.p3)
        if args.mode == "write":
            path.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {path} ({len(snap['texts'])} texts, {len(snap['vectors'])} vectors)")
            continue
        problems = diff(json.loads(path.read_text(encoding="utf-8")), snap)
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {pdf}:{page}")
        for line in problems:
            print(f"     {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
