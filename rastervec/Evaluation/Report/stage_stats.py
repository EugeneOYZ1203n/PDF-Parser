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
from rastervec.helpers.geometry import max_dimension, union_bbox


def _dist(values) -> dict:
    return distribution_stats([float(v) for v in values])


def _entry_vectors(entry: list) -> list:
    if entry and isinstance(entry[0], list):
        return [v for g in entry for v in g]
    return entry


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


def stats_separation(res) -> dict:
    vbl = res.vectors_by_layer or {}
    vblc = res.vectors_by_layer_color or {}
    layer_counts = {str(k): len(v) for k, v in vbl.items()}
    color_counts: Counter = Counter()
    bucket_counts: dict[str, int] = {}
    for layer, by_color in vblc.items():
        for color, vs in by_color.items():
            color_counts[str(color)] += len(vs)
            bucket_counts[f"{layer!r} / {color!r}"] = len(vs)
    return {
        "layer_counts": layer_counts,
        "color_counts": dict(color_counts),
        "bucket_counts": bucket_counts,
    }


def stats_classify(res) -> dict:
    clustering = res.clustering or {}
    per_step: dict[str, dict] = {}
    for stage in clustering.values():
        for step in stage.steps:
            slot = per_step.setdefault(
                step.label, {"dropped_entries": 0, "dropped_vectors": 0, "kept_entries": 0}
            )
            for name, cat in step.categories.items():
                if cat.role == "dropped":
                    slot["dropped_entries"] += len(cat.groups)
                    slot["dropped_vectors"] += sum(
                        len(_entry_vectors(e)) for e in cat.groups
                    )
                elif cat.role == "kept":
                    slot["kept_entries"] = len(cat.groups)

    final_clusters = res.text_clusters or []
    vecs_per_cluster = [len(_entry_vectors(c)) for c in final_clusters]
    groups_per_cluster = [len(c) for c in final_clusters]  # tiered: groups per cluster
    return {
        "per_step": per_step,
        "final_cluster_count": len(final_clusters),
        "final_vector_count": sum(vecs_per_cluster),
        "vectors_per_cluster": _dist(vecs_per_cluster),
        "groups_per_cluster": _dist(groups_per_cluster),
    }


def stats_fast(res) -> dict:
    fr = res.fast_result
    passed = res.fast_passed or []
    dropped = res.fast_dropped_vectors or []
    out = {
        "dropped_vectors": len(dropped),
        "passed_clusters": len(passed),
        "detect_seconds": getattr(fr, "detect_seconds", None) if fr else None,
    }
    if fr is not None:
        out["tile_count"] = fr.tile_count
        out["skipped_tiles"] = len(fr.skipped_tiles or [])
        out["per_tile_seconds"] = _dist(fr.tile_seconds or [])
        out["cluster_score"] = _dist((fr.scores or {}).values())
    return out


def stats_segment(res) -> dict:
    segs = res.word_segments or []
    dbg = res.segmentation_debug or []
    return {
        "segment_count": len(segs),
        "cluster_count": len(dbg),
        "segments_per_cluster": _dist(len(d["segment_bboxes"]) for d in dbg),
        "dropped_segments": sum(len(d.get("dropped_segment_bboxes", [])) for d in dbg),
        "dropped_vectors": sum(len(d.get("dropped_vector_bboxes", [])) for d in dbg),
        "skew_angle": _dist(s.angle for s in segs),
    }


def stats_similarity(res) -> dict:
    groups = res.similarity_groups or []
    return {
        "group_count": len(groups),
        "word_count": len(res.word_segments or []),
        "group_size": _dist(len(g) for g in groups),
        "groups_with_multiple_members": sum(1 for g in groups if len(g) > 1),
    }


def stats_ocr(res) -> dict:
    restored = res.restored_texts or []
    uniques = res.unique_texts or []
    passed = [t for t in restored if t.text.strip()]
    return {
        "unique_words_ocr_d": len(uniques),
        "restored_texts": len(restored),
        "passed": len(passed),
        "failed": len(restored) - len(passed),
        "rotation_counts": dict(
            Counter(int(round(t.angle() / 90.0) * 90) % 360 for t in restored)
        ),
    }


def stats_paddle_detect(res) -> dict:
    boxes = res.paddle_boxes or []
    return {
        "box_count": len(boxes),
        "box_size": _dist(max_dimension(b) for b in boxes),
    }


_STATS = {
    "native": stats_native,
    "vectors": stats_vectors,
    "separation": stats_separation,
    "classify": stats_classify,
    "fast": stats_fast,
    "segment": stats_segment,
    "similarity": stats_similarity,
    "ocr": stats_ocr,
    "paddle_detect": stats_paddle_detect,
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
