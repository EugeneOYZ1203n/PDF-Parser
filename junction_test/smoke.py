"""Fast, notebook-free sanity check.

    .venv/Scripts/python.exe -m junction_test.smoke
"""
from __future__ import annotations

import sys

from .metrics import evaluate, summarize
from .pipeline import Params, run
from .synthetic import generate


def main() -> int:
    rows = []
    for seed in range(5):
        img, gt = generate(seed=seed, size=512, noise=3.0)
        res = run(img, Params())
        m = evaluate(res, gt)
        rows.append(m)
        print(
            f"seed {seed}: IoU={m['mask_iou']:.3f} cover={m['coverage_pct']:.3f} "
            f"junc P/R={m['junction_precision']:.2f}/{m['junction_recall']:.2f} "
            f"segs={m['n_segments']} arcs={m['n_arcs']} "
            f"dashedR={m['dashed_recall']:.2f} "
            f"stairR={m['staircase_recall']:.2f}({m['n_staircases']}) "
            f"doorR={m['symbol_door_recall']:.2f} winR={m['symbol_window_recall']:.2f} "
            f"t={sum(res.timings.values()):.2f}s"
        )
    s = summarize(rows)
    print("\nmean:", {k: round(v, 3) for k, v in s.items()})

    # dashed-line recovery, the staircase texture heuristic, and symbol
    # recognition are all known-weak / not fully general (Dosch 2000 itself
    # relies on interactive correction for dashed lines and calls its own
    # staircase filter "crude and simplistic"; symbol recognition here covers
    # 2/7 families, validated only against synthetic data) -- none are part
    # of the pass gate.
    ok = s["coverage_pct"] > 0.90 and s["junction_recall"] > 0.80 and s["mask_iou"] > 0.60
    print("\nSMOKE", "PASS" if ok else "FAIL",
          f"(dashed recall {s['dashed_recall']:.2f}, staircase recall {s['staircase_recall']:.2f}, "
          f"door recall {s['symbol_door_recall']:.2f}, window recall {s['symbol_window_recall']:.2f} "
          f"-- informational)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
