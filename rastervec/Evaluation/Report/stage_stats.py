"""Per-stage numeric stats for `scripts/generate_pipeline_report.py`.

One `stats_<stage>(res)` per pipeline stage, each returning a plain nested
dict (JSON-friendly); `format_stats` renders one to text for the
`<stage>.txt` sidecar files. Reuses `benchmark.distribution_stats` for every
min/q1/median/mean/q3/max block so the report and the benchmark agree on
what a "distribution" is.
"""
from __future__ import annotations

import json
from collections import Counter

from rastervec.Evaluation.Evaluate.benchmark import distribution_stats
from rastervec.commons.helpers.geometry import max_dimension, union_bbox


def _dist(values) -> dict:
    return distribution_stats([float(v) for v in values])


# ---------------------------------------------------------------------------
def stats_native(res) -> dict:
    words = res.native_words or []
    lines = {(w.block_no, w.line_no) for w in words}
    all_text = "".join(w.text for w in words)
    return {
        "lines": len(lines),
        "words": len(words),
        "chars": len(all_text),
        "spans_with_matched_span": sum(1 for w in words if w.raw_span is not None),
        "line_direction_angle_counts": dict(
            Counter(round(w.angle()) for w in words)
        ),
        "char_counts": dict(Counter(all_text)),
        "orientation_source_counts": dict(Counter(w.orientation_source for w in words)),
    }


def stats_vectors(res) -> dict:
    vectors = res.vectors_raw or []
    by_type: dict[str, list] = {}
    for v in vectors:
        by_type.setdefault(v.type, []).append(v)
    return {
        "count": len(vectors),
        "count_by_type": {k: len(vs) for k, vs in by_type.items()},
        "size_all": _dist(max_dimension(v.bbox) for v in vectors),
        "size_by_type": {
            k: _dist(max_dimension(v.bbox) for v in vs) for k, vs in by_type.items()
        },
    }


def stats_similarity(res) -> dict:
    groups = res.similarity_groups or []
    sizes = [len(g.members) for g in groups]
    return {
        "group_count": len(groups),
        "vector_count": sum(sizes),
        "members_per_group": _dist(sizes),
        "group_sizes_by_count_desc": sorted(sizes, reverse=True)[:50],
    }


def stats_reclassify(res) -> dict:
    rc = res.reclassify_result
    if rc is None:
        return {}
    return {
        "fail_reclassified_pass": rc.fail_reclassified_pass,
        "pass": rc.pass_count,
        "fail": rc.fail_count,
    }


def stats_fast(res) -> dict:
    fr = res.fast_result
    passed = res.fast_passed or []
    dropped = res.fast_dropped_vectors or []
    out = {
        "dropped_vectors": len(dropped),
        "passed_vectors": len(passed),
        "detect_seconds": getattr(fr, "detect_seconds", None) if fr else None,
    }
    if fr is not None:
        out["tile_count"] = fr.tile_count
        out["skipped_tiles"] = len(fr.skipped_tiles or [])
        out["per_tile_seconds"] = _dist(fr.tile_seconds or [])
        out["vector_score"] = _dist((fr.scores or {}).values())
    return out


def stats_separation(res) -> dict:
    buckets = res.separation_buckets or []
    return {
        "bucket_count": len(buckets),
        "vectors_per_bucket": _dist(len(b) for b in buckets),
        "total_vectors": sum(len(b) for b in buckets),
    }


def stats_clusters(res) -> dict:
    clusters = res.spatial_clusters or []
    return {
        "cluster_count": len(clusters),
        "vectors_per_cluster": _dist(len(c) for c in clusters),
        "cluster_size": _dist(max_dimension(union_bbox([v.bbox for v in c])) for c in clusters if c),
    }


def stats_paddle_detect(res) -> dict:
    cds = [cd for cd in (res.cluster_detections or []) if cd is not None]
    detections_per_cluster = [len(cd.detections) for cd in cds]
    rotations = [d.rotation_deg for cd in cds for d in cd.detections]
    return {
        "clusters_with_detections": sum(1 for n in detections_per_cluster if n > 0),
        "cluster_count": len(res.cluster_detections or []),
        "detection_count": sum(detections_per_cluster),
        "detections_per_cluster": _dist(detections_per_cluster),
        "rotation_deg": _dist(rotations),
    }


def stats_assignment(res) -> dict:
    text_assigned = res.reassigned_text or []
    drawing = res.reassigned_drawing or []
    return {
        "assigned_to_text": len(text_assigned),
        "reassigned_to_drawing": len(drawing),
    }


def stats_rotate(res) -> dict:
    segs = res.rotated_segments or []
    dbg = res.rotation_debug or []
    deltas = [d["resolved_theta"] - d["source_theta"] for d in dbg]
    return {
        "segment_count": len(segs),
        "detection_count": len(dbg),
        "resolved_theta": _dist(d["resolved_theta"] for d in dbg),
        "refinement_delta_deg": _dist(deltas),
    }


def stats_ocr(res) -> dict:
    restored = res.restored_texts or []
    passed = [t for t in restored if t.text.strip()]
    return {
        "restored_texts": len(restored),
        "passed": len(passed),
        "failed": len(restored) - len(passed),
        "rotation_counts": dict(
            Counter(int(round(t.angle() / 90.0) * 90) % 360 for t in restored)
        ),
    }


_STATS = {
    "native": stats_native,
    "vectors": stats_vectors,
    "similarity": stats_similarity,
    "fast": stats_fast,
    "reclassify": stats_reclassify,
    "separation": stats_separation,
    "clusters": stats_clusters,
    "paddle_detect": stats_paddle_detect,
    "assignment": stats_assignment,
    "rotate": stats_rotate,
    "ocr": stats_ocr,
}


def stats_for_stage(res, stage_key: str) -> dict:
    fn = _STATS.get(stage_key)
    if fn is None:
        return {}
    try:
        return fn(res)
    except Exception as exc:  # noqa: BLE001 -- a stats sidecar; never fail the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def format_stats(stage_key: str, data: dict) -> str:
    return f"# {stage_key} stats\n\n" + json.dumps(data, indent=2, default=str) + "\n"
