"""Ablation harness for JUNCTION_ABLATION.md.

Swap one pipeline stage at a time from the baseline, hold the rest fixed, and
score every (method, sample) over a bucketed synthetic benchmark.

    .venv/Scripts/python.exe -m junction_test.ablation           # -> ablation_results.csv
    .venv/Scripts/python.exe -m junction_test.ablation --quick   # fewer buckets

The notebook (`explore.ipynb` Sec 7) calls `run_ablation()` / `bucket_table()`
directly for the interactive view.
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

from . import metrics, synthetic
from .pipeline import Params, run

# --------------------------------------------------------------- method presets
# Each entry is a set of Params overrides on top of the baseline. Only the named
# stage differs; everything else stays at baseline (JUNCTION_ABLATION.md Sec 6.2).
METHODS: dict[str, dict] = {
    "baseline":          {},                                        # S1 skeletonize + degree map
    "S2_medial_axis":    {"skeleton_method": "medial_axis"},
    "S3_junction_repair": {"junction_repair": True},
    "S8_lsd":            {"centerline_method": "lsd"},
    "S9_hough":          {"centerline_method": "hough"},
    # a couple of F-stage swaps that need no new code (already in Params)
    "F1_rdp":            {"approx_method": "rdp"},
}

# --------------------------------------------------------------- sample buckets
# archetype knobs x overlap density x clean/noisy (Sec 3, condensed).
_OVERLAP = {"low": 0, "med": 3, "high": 7}

_ARCHETYPES = {
    "floor_plan":   dict(weight_ladder=True),
    "curved":       dict(curved_walls=True, weight_ladder=True),
    "mep_overlay":  dict(color_layers=True, dash_styles=True, weight_ladder=True),
    "dimension":    dict(dash_styles=True, residue_level=0.4),
    "residue":      dict(residue_level=0.8, weight_ladder=True),
}

_KEY_METRICS = (
    "mask_iou", "coverage_pct", "junction_f1", "junction_type_accuracy",
    "x_passthrough_accuracy", "coincident_unrelated_false_merge",
    "width_mae", "dash_style_accuracy", "arc_radius_rel_err",
    "curve_misfit_count", "reconstruction_ssim", "segment_count_ratio", "runtime_s",
)


def _samples(quick: bool):
    archs = list(_ARCHETYPES.items())
    overlaps = list(_OVERLAP.items())
    if quick:
        archs = archs[:3]
        overlaps = [overlaps[0], overlaps[2]]        # low + high only
    seeds = range(1) if quick else range(4)
    for arch, kn in archs:
        for oname, ocount in overlaps:
            for noisy in ((False,) if quick else (False, True)):
                for seed in seeds:
                    yield dict(
                        archetype=arch, overlap=oname, noisy=noisy, seed=seed,
                        knobs={**kn, "forced_crossings": ocount,
                               "coincident_unrelated": 1 if ocount else 0,
                               "noise": 6.0 if noisy else 2.0},
                    )


def run_ablation(quick: bool = False, methods: list[str] | None = None) -> list[dict]:
    methods = methods or list(METHODS)
    rows: list[dict] = []
    samples = list(_samples(quick))
    for si, sm in enumerate(samples):
        kn = dict(sm["knobs"])
        noise = kn.pop("noise")
        img, gt = synthetic.generate(seed=sm["seed"], size=512, noise=noise, **kn)
        for mname in methods:
            p = dataclasses.replace(Params(), **METHODS[mname])
            try:
                res = run(img, p)
                m = metrics.evaluate(res, gt)
                err = ""
            except Exception as exc:                       # keep the row, note the failure
                m = {k: float("nan") for k in _KEY_METRICS}
                err = f"{type(exc).__name__}: {exc}"
            rows.append({
                "method": mname, "archetype": sm["archetype"], "overlap": sm["overlap"],
                "noisy": sm["noisy"], "seed": sm["seed"], "error": err,
                **{k: m.get(k, float("nan")) for k in _KEY_METRICS},
            })
        print(f"  [{si + 1}/{len(samples)}] {sm['archetype']}/{sm['overlap']}"
              f"/{'noisy' if sm['noisy'] else 'clean'} seed {sm['seed']}", flush=True)
    return rows


def bucket_table(rows: list[dict], by: str = "archetype", metric: str = "junction_f1") -> dict:
    """mean(metric) per (method, bucket) -> {method: {bucket: value}}."""
    import numpy as np
    out: dict = {}
    buckets = sorted({r[by] for r in rows}, key=str)
    for mname in sorted({r["method"] for r in rows}):
        out[mname] = {}
        for b in buckets:
            vals = [r[metric] for r in rows
                    if r["method"] == mname and r[by] == b
                    and isinstance(r[metric], (int, float)) and not np.isnan(r[metric])]
            out[mname][b] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_csv(rows: list[dict], path: str | Path = "ablation_results.csv") -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _print_table(t: dict, title: str) -> None:
    buckets = list(next(iter(t.values())).keys())
    print(f"\n{title}")
    print("  " + "method".ljust(20) + "".join(b.rjust(12) for b in buckets))
    for mname, row in t.items():
        print("  " + mname.ljust(20) + "".join(f"{row[b]:12.3f}" for b in buckets))


def main() -> int:
    quick = "--quick" in sys.argv
    rows = run_ablation(quick=quick)
    out = write_csv(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")
    for metric in ("junction_f1", "x_passthrough_accuracy", "reconstruction_ssim", "runtime_s"):
        _print_table(bucket_table(rows, "archetype", metric), f"{metric} by archetype")
        _print_table(bucket_table(rows, "overlap", metric), f"{metric} by overlap density")
    return 0


if __name__ == "__main__":
    sys.exit(main())
