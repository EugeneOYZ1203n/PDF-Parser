"""Loading a `generate_pipeline_report.py --benchmark` run folder
(`RunEntry`/`_merge_gt`/`_load_run`), scoring one shared input's text
(`_score_text`) / vector (`_score_vectors`) metrics on one input form,
and reading its per-form counts (`_load_counts`), P3 stats and per-page
wall-clock timings (`_load_timings`).

A run folder is either the current layout -- `<input>/` holding the ground
truth and one sub-folder per form (`original/`, `vectorised/`,
`rasterised/`), each with its own `dump.json` (`benchmark.json` entries
carry `"forms": {form: "<input>/<form>"}`) -- or the pre-`inputs` legacy
layout: one `<input>/dump.json` whose `texts`/`vectors` are the vectorised
run and `raster_*` fields the rasterised run. `_form_pages` hides the
difference. Which types each form is scored on: `rastervec/Evaluation/
forms.py`."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import NamedTuple

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters, timing, vector_metrics, vector_render
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    OverlapGraph,
    TextMetricSuiteResult,
    aggregate_text_metrics,
    build_overlap_graphs_by_type,
    evaluate_text_metrics,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VectorMetricConfig,
    VectorMetricSuiteResult,
    aggregate_vector_metrics,
)
from rastervec.Evaluation.forms import (
    FORM_TEXT_TYPES,
    FORM_VECTOR_GT,
    FORMS,
    all_texts_are_predictions,
)
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels


class RunEntry(NamedTuple):
    run_name: str
    run_dir: Path
    key: str
    pdf_stem: str
    doc_dir: Path
    dump_path: Path  # legacy layout's single dump (unused when `forms` is set)
    gt: LabelSet  # merged native/vector/raster labels for this input
    # form -> that form's run folder (current layout); None = legacy layout.
    forms: "dict[str, Path] | None" = None


def _merge_gt(doc: Path) -> LabelSet:
    """Merges whichever `ground_truth_*.json` files this run wrote (every
    name in `TEXT_TYPES`, falling back to the pre-rework `auto`/`manual`
    names for an older report folder) into one `LabelSet`."""
    entries = []
    geometry_entries = []
    pdf_path = ""
    for name in (*TEXT_TYPES, "auto", "manual"):
        p = doc / f"ground_truth_{name}.json"
        if p.is_file():
            ls = load_labels(str(p))
            pdf_path = pdf_path or ls.pdf_path
            entries.extend(ls.entries)
            geometry_entries.extend(ls.geometry_entries)
    return LabelSet(pdf_path=pdf_path, entries=entries, geometry_entries=geometry_entries)


def _load_run(run_dir: Path, parser) -> dict[str, RunEntry]:
    marker = run_dir / "benchmark.json"
    if not marker.is_file():
        parser.error(f"{run_dir} has no benchmark.json (not a benchmark run)")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if not meta.get("benchmark"):
        parser.error(f"{run_dir}: benchmark.json does not mark this as a benchmark run")

    entries: dict[str, RunEntry] = {}
    for e in meta.get("entries", []):
        doc = run_dir / e["dir"]
        forms = e.get("forms")
        entries[e["key"]] = RunEntry(
            run_name=run_dir.name,
            run_dir=run_dir.resolve(),
            key=e["key"],
            pdf_stem=e["pdf_stem"],
            doc_dir=doc,
            dump_path=doc / "dump.json",
            gt=_merge_gt(doc),
            forms=None if forms is None else {f: run_dir / rel for f, rel in forms.items()},
        )
    return entries


def entry_forms(entry: RunEntry) -> list[str]:
    """The forms this entry has a run of, in `FORMS` order."""
    if entry.forms is not None:
        return [f for f in FORMS if f in entry.forms]
    return ["vectorised", "rasterised"]


def form_dir(entry: RunEntry, form: str) -> Path:
    """The viewer folder of `form`'s run (legacy: the input folder)."""
    return entry.forms[form] if entry.forms is not None else entry.doc_dir


@functools.lru_cache(maxsize=2)
def _cached_dump(path: str, mtime_ns: int) -> "dump_io.Dump":
    return dump_io.load_dump(path)


def _load_dump(path: Path) -> "dump_io.Dump":
    """`dump.json` parsed once per file instead of once per scorer
    (`_score_text`, `_score_vectors`, the counts / timing / FAST / retry
    loaders all read the same file). Two entries: the benchmark compares at
    most two runs per shared input. Read-only -- callers never mutate it."""
    p = Path(path)
    return _cached_dump(str(p), p.stat().st_mtime_ns)


class FormPage(NamedTuple):
    """One page of one form's run, whatever layout it was stored in."""
    page_meta: object
    texts: list
    vectors: list
    p2_vectors: "list | None"
    dpi: "float | None"
    step_durations: dict
    substep_durations: dict
    debug_durations: dict
    counts: "dict | None"
    fast_cluster_stats: "dict | None"
    retry_stats: "dict | None"


def _form_pages(entry: RunEntry, form: str) -> "tuple[list[FormPage], dict] | None":
    """`(pages, doc_durations)` of `form`'s run, `None` when this entry has
    no such run."""
    if form not in entry_forms(entry):
        return None
    if entry.forms is not None:
        path = entry.forms[form] / "dump.json"
        if not path.is_file():
            return None
        dump = _load_dump(path)
        return [
            FormPage(p.page_meta, p.texts, p.vectors, p.p2_vectors, p.raster_dpi, p.step_durations,
                     p.substep_durations, p.debug_durations, p.counts, p.fast_cluster_stats, p.retry_stats)
            for p in dump.pages
        ], dump.doc_durations
    dump = _load_dump(entry.dump_path)
    if form == "vectorised":
        return [
            FormPage(p.page_meta, p.texts, p.vectors, None, None, p.step_durations, p.substep_durations,
                     p.debug_durations, p.counts, p.fast_cluster_stats, p.retry_stats)
            for p in dump.pages
        ], dump.doc_durations
    return [
        FormPage(p.page_meta, p.raster_texts, p.raster_vectors, None, p.raster_dpi, p.raster_step_durations,
                 p.raster_substep_durations, {}, None, None, None)
        for p in dump.pages
    ], {}


def _score_text(
    entry: RunEntry, form: str, cfg: MetricConfig,
) -> "tuple[list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]], TextMetricSuiteResult | None] | None":
    """Text metrics of `form`'s run against the text types scored on that
    form (`forms.FORM_TEXT_TYPES`; other types' GT left empty). Predictions:
    every text on the original form (native words + OCR), the OCR output
    elsewhere. `None` when the entry has no run of `form`."""
    loaded = _form_pages(entry, form)
    if loaded is None:
        return None
    pages, _doc = loaded
    types = FORM_TEXT_TYPES[form]
    every_text = all_texts_are_predictions(form)
    gt_by_type_all = adapters.gt_regions_by_text_type(entry.gt)
    entries_by_type_all = adapters.entries_by_text_type(entry.gt)
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]" = []
    for page in pages:
        pi = page.page_meta.index
        gt = {t: ([g for g in gt_by_type_all[t] if g.page_index == pi] if t in types else []) for t in TEXT_TYPES}
        entries = {t: (entries_by_type_all[t] if t in types else []) for t in TEXT_TYPES}
        preds = adapters.predictions_from_texts(
            [t for t in page.texts if every_text or t.source == "ocr"]
        )
        res = evaluate_text_metrics(
            gt, entries, preds, cfg=cfg, page_area=page.page_meta.width * page.page_meta.height,
        )
        per_page.append((pi, res, build_overlap_graphs_by_type(gt, preds, cfg)))
    return per_page, aggregate_text_metrics([r for _pi, r, _g in per_page])


def _score_vectors(
    entry: RunEntry, form: str, cfg: VectorMetricConfig,
) -> "tuple[list[tuple[int, VectorMetricSuiteResult]], VectorMetricSuiteResult | None] | None":
    """Vector metrics of `form`'s Phase 2 vectors against the GT geometry
    types scored on that form (`forms.FORM_VECTOR_GT`: rasterised -> both
    raster types, original -> `original_raster` only) -- Liu-Dori line
    detection, ink masks for D_p/F_p/PRI (`vector_render.render_ink_pair`)
    and grayscale renders for SSIM/PSNR (`vector_render.render_pair`).
    `None` for the vectorised form (no P2 geometry GT) or a missing run.
    The dpi floor comes from the dump (`PageDump.raster_dpi`), else (legacy
    rasterised dumps) the label folder's `rasterised.pdf`, else
    `cfg.default_dpi`."""
    gt_types = FORM_VECTOR_GT[form]
    if not gt_types:
        return None
    loaded = _form_pages(entry, form)
    if loaded is None:
        return None
    pages, _doc = loaded
    per_page: "list[tuple[int, VectorMetricSuiteResult]]" = []
    rasterised = (
        Path(entry.gt.pdf_path).parent / "rasterised.pdf"
        if form == "rasterised" and entry.gt.pdf_path else None
    )
    for page in pages:
        pi = page.page_meta.index
        page_gt = LabelSet(
            pdf_path=entry.gt.pdf_path,
            entries=[e for e in entry.gt.entries if e.page_index == pi],
            geometry_entries=[g for g in entry.gt.geometry_entries if g.page_index == pi],
        )
        inputs = adapters.build_vector_eval_inputs(
            page_gt, page.p2_vectors if page.p2_vectors is not None else page.vectors,
        )
        gt_by_type = {t: (v if t in gt_types else []) for t, v in inputs.gt_by_type.items()}
        label_counts = {t: (n if t in gt_types else 0) for t, n in inputs.label_counts.items()}
        dpi = page.dpi
        if dpi is None and rasterised is not None and rasterised.is_file():
            dpi = adapters.raster_dpi_for_page(str(rasterised), pi)
        dpi = dpi or cfg.default_dpi
        images = vector_render.render_pair(gt_by_type, inputs.preds, page.page_meta, cfg)
        ink = vector_render.render_ink_pair(gt_by_type, inputs.preds, page.page_meta, dpi, cfg)
        res = vector_metrics.evaluate_vector_metrics(
            gt_by_type, inputs.preds, label_counts, images=images, ink=ink,
            dpi=dpi, text_boxes=inputs.text_boxes, cfg=cfg,
        )
        del images, ink
        per_page.append((pi, res))
    return per_page, aggregate_vector_metrics([r for _pi, r in per_page])


def _sum_counts(acc: dict, counts: dict) -> None:
    for k, v in counts.items():
        if k == "p2_layers":
            layers = acc.setdefault("p2_layers", {})
            for colour, c in (v or {}).items():
                tgt = layers.setdefault(colour, {"vectors": 0, "items": 0})
                tgt["vectors"] += c.get("vectors", 0)
                tgt["items"] += c.get("items", 0)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0) + v


def _load_counts(entry: RunEntry, form: str) -> "dict | None":
    """`form`'s per-page `PageDump.counts` summed (P2 colour layers merged
    by colour) plus `pages`. `None` when no page recorded counts."""
    loaded = _form_pages(entry, form)
    if loaded is None:
        return None
    pages, _doc = loaded
    acc: dict = {}
    seen = False
    for page in pages:
        if page.counts is None:
            continue
        seen = True
        _sum_counts(acc, page.counts)
    if not seen:
        return None
    acc["pages"] = len(pages)
    return acc


def _label_counts(entry: RunEntry, form: str) -> "dict[str, int]":
    """GT items per type scored on `form`: text regions per text type, plus
    geometry annotations per vector GT type."""
    by_text = adapters.entries_by_text_type(entry.gt)
    out = {f"{t} (text)": len(by_text[t]) for t in FORM_TEXT_TYPES[form] if by_text[t]}
    geometry = adapters.gt_geometry_by_vector_type(entry.gt)
    out.update({f"{t} (geometry)": len(geometry[t]) for t in FORM_VECTOR_GT[form] if geometry[t]})
    return out


def _load_fast_cluster_stats(entry: RunEntry, form: str) -> "dict | None":
    """`{"total": N, "passed": P}` summed across every page of `form`'s run
    that recorded a `PageDump.fast_cluster_stats` (the
    OldVectorClassification P3 backend's FAST-filter cluster counts, see
    `generate_pipeline_report.py::_fast_cluster_stats`). `None` if no page
    recorded this (a different P3 backend, the legacy engine, or an older
    dump without the field)."""
    loaded = _form_pages(entry, form)
    if loaded is None:
        return None
    total = passed = 0
    seen = False
    for page in loaded[0]:
        stats = page.fast_cluster_stats
        if stats is None:
            continue
        seen = True
        total += stats.get("total", 0)
        passed += stats.get("passed", 0)
    return {"total": total, "passed": passed} if seen else None


_RETRY_BUCKETS = ("0", "1", "2", "3", "failed")


def _load_retry_stats(entry: RunEntry, form: str) -> "dict | None":
    """`{"0": N, "1": N, "2": N, "3": N, "failed": N}` blank-recognition
    retry counts summed across every page of `form`'s run that recorded a
    `PageDump.retry_stats` (the Latest/OldVectorClassification P3 backend's
    rotation retry sweep, see `generate_pipeline_report.py::_retry_stats`).
    `None` if no page recorded this. Mirrors `_load_fast_cluster_stats`."""
    loaded = _form_pages(entry, form)
    if loaded is None:
        return None
    totals = {k: 0 for k in _RETRY_BUCKETS}
    seen = False
    for page in loaded[0]:
        stats = page.retry_stats
        if stats is None:
            continue
        seen = True
        for k in _RETRY_BUCKETS:
            totals[k] += stats.get(k, 0)
    return totals if seen else None


def _load_timings(entry: RunEntry) -> "dict[str, list[dict]]":
    """`{form: [flattened per-page timing row]}` for every form this entry
    ran (`timing.flatten_page_timing`, incl. the page's report-generation
    `debug_durations`), plus `timing.DOC_KIND` -> one `doc_durations` row
    per form run (`timing.flatten_doc_timing`). A page that recorded
    nothing contributes no row."""
    out: "dict[str, list[dict]]" = {kind: [] for kind in (*timing.RUN_KINDS, timing.DOC_KIND)}
    for form in entry_forms(entry):
        loaded = _form_pages(entry, form)
        if loaded is None:
            continue
        pages, doc_durations = loaded
        for page in pages:
            row = timing.flatten_page_timing(page.step_durations, page.substep_durations, page.debug_durations)
            if row:
                out[form].append(row)
        doc_row = timing.flatten_doc_timing(doc_durations)
        if doc_row:
            out[timing.DOC_KIND].append(doc_row)
    return out
