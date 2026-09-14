"""Compare 1-2 benchmark report folders.

Each folder is a `generate_pipeline_report.py` run with `benchmark: true`.
Such a run scores **one** combined pipeline run per page
(`convert_page_to_vector_text`) against ground truth for up to 4 text types
and 3 vector types, and embeds `<pdf-stem>/dump.json` +
`ground_truth_<text_type>.json` (only the non-empty types) + a
`benchmark.json` marker.

This script matches the folders on the inputs they **share** (by input
identity: `pdf:<stem>` = native_to_vector-only, `labels:<stem>` = whatever
label sources the run had), scores each shared input's text metrics
(`metrics.evaluate_text_metrics`) and vector metrics
(`vector_metrics.evaluate_vector_metrics`, built from `dump.json`'s own
`vectors` -- a weaker approximation of `original_vector` predictions than a
live pipeline run since a dump has no `reassigned_text`), and writes one
self-contained `report.html` -- one section per shared key, broken into
Text (Char/Word/Font size/Rotation/Bbox/Vector classification/Reading
order/Confusion) and Vector (Count/Endpoint/Property) subsections, each
with linked chart PNGs, illustrated error-example galleries (extra
predictions / missed GT / confusion misreads, cropped from the run's own
`converted_p<N>.pdf`), the run's own diagnostic image galleries
(`*_images/` folders), and a ready-to-paste `pipeline_report_viewer.py`
command. Output (timestamped folder under `outputs/pipeline_report_benchmark/`):

    report.html             the primary deliverable (see above)
    benchmark.txt           supplementary plain-text dump (grep/diff-friendly)
    runs.json               the compared folders + threshold
    viewer_commands.txt     a viewer CLI line per shared input
    charts/                 <key>__aggregate__*.png, aggregate__*.png
    examples/                <key>__<run>__<type>__<category>__<n>.png crops

    .venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
        --run outputs/pipeline_report/<ts1>__benchmark \
        --run outputs/pipeline_report/<ts2>__benchmark_legacy
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf as fitz

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters, charts, vector_metrics
from rastervec.Evaluation.Evaluate.benchmark import (
    _fmt_ratio,
    format_aggregate_comparison,
    format_confusion_table,
    format_vector_aggregate_comparison,
)
from rastervec.Evaluation.Evaluate.confusion_metrics import align_chars, closest_pred_word
from rastervec.Evaluation.Evaluate.html_report import ExampleCard, ReportBuilder
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    OverlapGraph,
    Prediction,
    TextMetricSuiteResult,
    aggregate_text_metrics,
    build_overlap_graphs_by_type,
    evaluate_text_metrics,
)
from rastervec.Evaluation.Evaluate.text_metrics import word_tokens
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VECTOR_TYPES,
    VectorMetricConfig,
    VectorMetricSuiteResult,
    aggregate_vector_metrics,
)
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir
from rastervec.Reader.reader import Reader
from rastervec.Vector.vector import extract_vectors

_LOG = get_logger("pipeline_report_benchmark")

_VENV_PY = ".venv/Scripts/python.exe"
_EXAMPLE_CAP = 5


class RunEntry(NamedTuple):
    run_name: str
    run_dir: Path
    key: str
    pdf_stem: str
    doc_dir: Path
    dump_path: Path
    gt: LabelSet  # merged native/vector/raster labels for this input


def _merge_gt(doc: Path) -> LabelSet:
    """Merges whichever `ground_truth_*.json` files this run wrote (the
    4-way `native_to_vector`/`original_vector`/`vector_to_raster`/
    `original_raster` names, falling back to the pre-rework `auto`/`manual`
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


def _load_run(run_dir: Path, parser: argparse.ArgumentParser) -> dict[str, RunEntry]:
    marker = run_dir / "benchmark.json"
    if not marker.is_file():
        parser.error(f"{run_dir} has no benchmark.json (not a benchmark run)")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if not meta.get("benchmark"):
        parser.error(f"{run_dir}: benchmark.json does not mark this as a benchmark run")

    entries: dict[str, RunEntry] = {}
    for e in meta.get("entries", []):
        doc = run_dir / e["dir"]
        entries[e["key"]] = RunEntry(
            run_name=run_dir.name,
            run_dir=run_dir.resolve(),
            key=e["key"],
            pdf_stem=e["pdf_stem"],
            doc_dir=doc,
            dump_path=doc / "dump.json",
            gt=_merge_gt(doc),
        )
    return entries


def _score_text(
    entry: RunEntry, cfg: MetricConfig,
) -> "tuple[list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]], TextMetricSuiteResult | None]":
    dump = dump_io.load_dump(entry.dump_path)
    gt_by_type_all = adapters.gt_regions_by_text_type(entry.gt)
    entries_by_type_all = adapters.entries_by_text_type(entry.gt)
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]" = []
    for page in dump.pages:
        pi = page.page_meta.index
        ocr = [t for t in page.texts if t.source == "ocr"]
        preds = adapters.predictions_from_texts(ocr)
        gt_by_type = {t: [g for g in gt_by_type_all[t] if g.page_index == pi] for t in TEXT_TYPES}
        graphs_by_type = build_overlap_graphs_by_type(gt_by_type, preds, cfg)
        res = evaluate_text_metrics(gt_by_type, entries_by_type_all, preds, cfg=cfg)
        per_page.append((pi, res, graphs_by_type))
    return per_page, aggregate_text_metrics([r for _pi, r, _g in per_page])


def _score_vectors(
    entry: RunEntry, cfg: VectorMetricConfig,
) -> "tuple[list[tuple[int, VectorMetricSuiteResult]], VectorMetricSuiteResult | None]":
    """Vector metrics from `dump.json` alone. `original_vector` predictions
    come from that page's dump `vectors` (the final drawing content a
    pipeline run kept) -- weaker than the live `benchmark_jobs.py` path,
    which also includes `reassigned_text` (not serialized in a dump), a
    known accepted limitation. GT vectors are reconstructed by extracting
    fresh vectors from the run's own original PDF and matching
    `path_signature`s recorded in the labels."""
    dump = dump_io.load_dump(entry.dump_path)
    per_page: "list[tuple[int, VectorMetricSuiteResult]]" = []
    if not entry.gt.pdf_path or not Path(entry.gt.pdf_path).is_file():
        return per_page, None
    try:
        reader = Reader(entry.gt.pdf_path)
    except ValueError:
        return per_page, None

    for page in dump.pages:
        pi = page.page_meta.index
        try:
            fresh_page = reader.get_page(pi)
            fresh_vectors = extract_vectors(fresh_page)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("vector GT extraction failed for %s page %d: %s", entry.key, pi, exc)
            fresh_vectors = []

        page_gt = LabelSet(
            pdf_path=entry.gt.pdf_path,
            entries=[e for e in entry.gt.entries if e.page_index == pi],
            geometry_entries=[g for g in entry.gt.geometry_entries if g.page_index == pi],
        )
        inputs = adapters.build_vector_eval_inputs(page_gt, None, fresh_vectors)
        preds_by_type = dict(inputs.preds_by_type)
        preds_by_type["original_vector"] = [
            e for v in page.vectors for e in vector_metrics.geometry_entries_from_vector(v)
        ]
        res = vector_metrics.evaluate_vector_metrics(
            inputs.gt_by_type, preds_by_type, inputs.label_counts, cfg=cfg,
        )
        per_page.append((pi, res))
    return per_page, aggregate_vector_metrics([r for _pi, r in per_page])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "key"


def _short_slug(text: str, max_len: int = 16) -> str:
    """A filename-safe slug capped to `max_len` chars (+ a short hash
    suffix for uniqueness) -- unlike `_slug`, safe to compose several of
    into one path component without hitting Windows' ~260-char MAX_PATH."""
    import hashlib

    full = _slug(text)
    if len(full) <= max_len:
        return full
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{full[:max_len]}_{h}"


def _crop_to_png(pdf_path: Path, bbox, out_path: Path, *, dpi: float = 150.0) -> bool:
    """Crops `bbox` (page-space, unrotated MediaBox) out of `pdf_path`'s
    page 0 -- every `converted_p<N>.pdf` this benchmark scores against is a
    single-page vectorised render. Known accepted limitation: this does not
    apply the page's own rotation matrix, so it may crop the wrong region
    on a rotated page -- correct for the common rotation-0 case."""
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[0]
            rect = fitz.Rect(*bbox)
            if rect.is_empty or rect.is_infinite:
                return False
            pix = page.get_pixmap(clip=rect, dpi=int(dpi))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(out_path))
            return True
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("crop failed for %s %s: %s", pdf_path, bbox, exc)
        return False


def _collect_examples(
    entry: RunEntry,
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]",
    examples_dir: Path, kslug: str,
) -> "dict[tuple[str, str], list[ExampleCard]]":
    """Up to `_EXAMPLE_CAP` examples per (text_type, category) across all of
    this key/run's pages -- extra predictions (no GT overlap at all), missed
    GT (no prediction reached it), confusion misreads (GT word vs. its
    closest overlapping predicted word)."""
    out: "dict[tuple[str, str], list[ExampleCard]]" = {}
    rslug = _short_slug(entry.run_name)
    for text_type in TEXT_TYPES:
        extra: "list[ExampleCard]" = []
        missed: "list[ExampleCard]" = []
        confusion: "list[ExampleCard]" = []
        for pi, _res, graphs_by_type in per_page:
            graph = graphs_by_type[text_type]
            conv_pdf = entry.doc_dir / f"converted_p{pi}.pdf"

            if len(extra) < _EXAMPLE_CAP:
                for pj, p in enumerate(graph.preds):
                    if len(extra) >= _EXAMPLE_CAP:
                        break
                    if graph.edges_by_pred[pj]:
                        continue
                    n = len(extra)
                    img = examples_dir / f"{kslug}__{rslug}__{text_type}__extra_prediction__{n}.png"
                    ok = _crop_to_png(conv_pdf, p.bbox, img)
                    extra.append(ExampleCard(
                        caption=f"p{pi} pred={p.text!r}",
                        image_path=(Path("examples") / img.name) if ok else None,
                        run=entry.run_name, text_type=text_type,
                    ))

            if len(missed) < _EXAMPLE_CAP:
                for gi in graph.missed_gt_idxs:
                    if len(missed) >= _EXAMPLE_CAP:
                        break
                    g = graph.gt[gi]
                    n = len(missed)
                    img = examples_dir / f"{kslug}__{rslug}__{text_type}__missed_gt__{n}.png"
                    ok = _crop_to_png(conv_pdf, g.bbox, img)
                    missed.append(ExampleCard(
                        caption=f"p{pi} gt={g.text!r}",
                        image_path=(Path("examples") / img.name) if ok else None,
                        run=entry.run_name, text_type=text_type,
                    ))

            if len(confusion) < _EXAMPLE_CAP:
                for gi, g in enumerate(graph.gt):
                    if len(confusion) >= _EXAMPLE_CAP:
                        break
                    overlapping = graph.overlapping_preds_by_gt[gi]
                    if not overlapping:
                        continue
                    pred_words_all: "list[str]" = []
                    for pj in overlapping:
                        pred_words_all.extend(word_tokens(graph.preds[pj].text))
                    pred_word_set = set(pred_words_all)
                    for gt_word in word_tokens(g.text):
                        if gt_word in pred_word_set:
                            continue
                        closest = closest_pred_word(gt_word, pred_words_all)
                        if closest is None or closest == gt_word:
                            continue
                        n = len(confusion)
                        img = examples_dir / f"{kslug}__{rslug}__{text_type}__confusion__{n}.png"
                        ok = _crop_to_png(conv_pdf, g.bbox, img)
                        confusion.append(ExampleCard(
                            caption=f"p{pi} gt={gt_word!r} pred={closest!r}",
                            image_path=(Path("examples") / img.name) if ok else None,
                            run=entry.run_name, text_type=text_type,
                        ))
                        break  # one example per gt region

        out[(text_type, "extra_prediction")] = extra
        out[(text_type, "missed_gt")] = missed
        out[(text_type, "confusion")] = confusion
    return out


def _fmt_property_rows(agg_by_run: "dict[str, VectorMetricSuiteResult | None]") -> "tuple[list[str], list[list[str]]]":
    headers = ["vector_type", "property"] + list(agg_by_run)
    rows: "list[list[str]]" = []
    for t in VECTOR_TYPES:
        names: "list[str]" = []
        for res in agg_by_run.values():
            if res is not None:
                names = [r.property_name for r in res.by_type[t].property_rows]
                break
        for name in names:
            row = [t, name]
            for res in agg_by_run.values():
                if res is None:
                    row.append("n/a")
                    continue
                r = next((rr for rr in res.by_type[t].property_rows if rr.property_name == name), None)
                if r is None or not r.applicable:
                    row.append("n/a")
                else:
                    row.append(f"{r.kind}={r.metric_value:.3f} (n={r.n_applicable})")
            rows.append(row)
    return headers, rows


def _add_text_sections(builder: ReportBuilder, agg_by_run: "dict[str, TextMetricSuiteResult | None]") -> None:
    builder.add_group_header("Text")

    headers = ["text_type"] + list(agg_by_run)
    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ls = res.by_type[t].label_stats
            row.append(f"{ls.label_count} labels | {ls.char_count} chars | {ls.word_count} words | {ls.vector_count} vectors")
        rows.append(row)
    builder.add_text_subsection("Label description", headers, rows)

    for field, name in (("char_overlap", "Char overlap"), ("word_overlap", "Word overlap")):
        rows = []
        for t in TEXT_TYPES:
            row = [t]
            for res in agg_by_run.values():
                if res is None:
                    row.append("n/a")
                    continue
                co = getattr(res.by_type[t], field)
                row.append(
                    f"matched {co.matched}/{co.total_gt} | unclassified {co.unclassified} | "
                    f"missing {co.missing} | P {_fmt_ratio(co.precision)} | R {_fmt_ratio(co.recall)}"
                )
            rows.append(row)
        builder.add_text_subsection(name, headers, rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ba = res.by_type[t].bbox_accuracy
            row.append(f"mean IoU {_fmt_ratio(ba.mean_iou)} (n_gt={ba.n_gt} n_localized={ba.n_localized})")
        rows.append(row)
    row = ["unclassified"]
    for res in agg_by_run.values():
        if res is None:
            row.append("n/a")
            continue
        bu = res.bbox_unclassified
        row.append(f"spurious preds {bu.spurious_pred_count} (area frac {bu.spurious_pred_area_frac})")
    rows.append(row)
    builder.add_text_subsection("Bbox accuracy", headers, rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            rot = res.by_type[t].rotation
            row.append(
                f"correct={rot.buckets.correct} off90={rot.buckets.off_90} off180={rot.buckets.off_180} "
                f"(n={rot.n_localized}) | mean_err={rot.mean_error_deg:.1f} | rmse={rot.rmse_deg:.1f}"
            )
        rows.append(row)
    builder.add_text_subsection("Rotation accuracy", headers, rows)

    rows = []
    for t in ("native_to_vector", "original_vector"):
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            f_ = res.by_type[t].funnel
            row.append(f"{f_.n_survived}/{f_.n_gt_vectors} ({_fmt_ratio(f_.survival_rate)})" if f_ else "n/a")
        rows.append(row)
    builder.add_text_subsection("Vector classification funnel", ["text_type"] + list(agg_by_run), rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ro = res.by_type[t].reading_order
            row.append(
                f"in_order {ro.n_in_order}/{ro.n_with_overlap} ({_fmt_ratio(ro.in_order_rate)}) | "
                f"edit min/max/mean/median {ro.edit_distance_min}/{ro.edit_distance_max}/"
                f"{ro.edit_distance_mean}/{ro.edit_distance_median}"
            )
        rows.append(row)
    builder.add_text_subsection("Reading order accuracy", headers, rows)

    for t in TEXT_TYPES:
        chars: set = set()
        for res in agg_by_run.values():
            if res is not None:
                chars.update(res.by_type[t].confusion.keys())
        c_rows = []
        for ch in sorted(chars):
            row = [repr(ch)]
            for res in agg_by_run.values():
                if res is None or ch not in res.by_type[t].confusion:
                    row.append("")
                    continue
                counter = res.by_type[t].confusion[ch]
                total = sum(counter.values())
                top = counter.most_common(5)
                row.append(", ".join(
                    f"{repr(r) if r else '(none)'}: {c} ({100 * c / total:.0f}%)" for r, c in top
                ))
            c_rows.append(row)
        builder.add_text_subsection(f"OCR confusion characters -- {t}", ["gt_char"] + list(agg_by_run), c_rows)

    chars = set()
    for res in agg_by_run.values():
        if res is not None:
            chars.update(res.extra_chars.keys())
    ec_rows = []
    for ch in sorted(chars):
        row = [repr(ch)]
        for res in agg_by_run.values():
            if res is None:
                row.append("")
                continue
            c = res.extra_chars.get(ch, 0)
            total = sum(res.extra_chars.values()) or 1
            row.append(f"{c} ({100 * c / total:.0f}%)" if c else "")
        ec_rows.append(row)
    builder.add_text_subsection("Extra predicted characters (no GT overlap at all)", ["char"] + list(agg_by_run), ec_rows)


def _add_vector_sections(builder: ReportBuilder, agg_by_run: "dict[str, VectorMetricSuiteResult | None]") -> None:
    if not any(v is not None for v in agg_by_run.values()):
        return
    builder.add_group_header("Vector")

    headers = ["vector_type"] + list(agg_by_run)
    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            row.append(str(res.by_type[t].label_stats.count) if res else "n/a")
        rows.append(row)
    builder.add_vector_subsection("Label description", headers, rows)

    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            cs = res.by_type[t].count_stats
            row.append(
                f"paired={cs.n_paired} missed={cs.n_missed} spurious={cs.n_spurious} | "
                f"P {_fmt_ratio(cs.precision)} | R {_fmt_ratio(cs.recall)}"
            )
        rows.append(row)
    builder.add_vector_subsection("Count accuracy", headers, rows)

    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            es = res.by_type[t].endpoint_stats
            row.append(f"RMSE {es.rmse} (n_paired={es.n_paired})")
        rows.append(row)
    builder.add_vector_subsection("Endpoint accuracy", headers, rows)

    p_headers, p_rows = _fmt_property_rows(agg_by_run)
    builder.add_vector_subsection("Property accuracy", p_headers, p_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, type=Path,
                   help="Benchmark run folder (repeatable; 1 or 2).")
    p.add_argument("--iou-threshold", type=float, default=MetricConfig().iou_edge_min)
    p.add_argument("--out-root", type=Path, default=None,
                   help="Parent for the timestamped output folder "
                   "(default: outputs/pipeline_report_benchmark/).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if len(args.run) > 2:
        parser.error("at most 2 --run folders (benchmark compares two pipelines)")
    cfg = MetricConfig(iou_edge_min=args.iou_threshold)
    vcfg = VectorMetricConfig()

    runs = [_load_run(Path(d), parser) for d in args.run]
    shared_keys = set(runs[0])
    for r in runs[1:]:
        shared_keys &= set(r)
    if not shared_keys:
        parser.error("the run folders share no benchmark input")

    root = args.out_root or output_dir("pipeline_report_benchmark")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / ts
    charts_dir = out_dir / "charts"
    examples_dir = out_dir / "examples"
    charts_dir.mkdir(parents=True, exist_ok=True)

    builder = ReportBuilder(f"Pipeline benchmark report ({ts})")
    blocks: list[str] = []
    grand_text: "dict[str, list[TextMetricSuiteResult]]" = {}
    grand_vector: "dict[str, list[VectorMetricSuiteResult]]" = {}

    for key in sorted(shared_keys):
        kslug = _short_slug(key)
        text_scored = {r[key].run_name: _score_text(r[key], cfg) for r in runs}
        vector_scored = {r[key].run_name: _score_vectors(r[key], vcfg) for r in runs}

        text_by_run = {name: agg for name, (_pp, agg) in text_scored.items()}
        vector_by_run = {name: agg for name, (_pp, agg) in vector_scored.items()}
        blocks.append(format_aggregate_comparison(text_by_run, title=f"{key} -- text"))
        for name, agg in text_by_run.items():
            if agg is not None:
                blocks.append(f"[{key} / {name}] " + format_confusion_table(agg))
        blocks.append(format_vector_aggregate_comparison(vector_by_run, title=f"{key} -- vectors"))

        builder.add_key_section(key)
        _add_text_sections(builder, text_by_run)
        _add_vector_sections(builder, vector_by_run)

        charts.label_description_chart(
            text_by_run, title=f"{key} aggregate labels", path=charts_dir / f"{kslug}__aggregate__labels.png")
        charts.char_word_overlap_chart(
            text_by_run, field="char_overlap", title=f"{key} aggregate char recall",
            path=charts_dir / f"{kslug}__aggregate__char.png")
        charts.char_word_overlap_chart(
            text_by_run, field="word_overlap", title=f"{key} aggregate word recall",
            path=charts_dir / f"{kslug}__aggregate__word.png")
        charts.bbox_accuracy_chart(
            text_by_run, title=f"{key} aggregate bbox IoU", path=charts_dir / f"{kslug}__aggregate__bbox.png")
        charts.rotation_chart(
            text_by_run, title=f"{key} aggregate rotation", path=charts_dir / f"{kslug}__aggregate__rotation.png")
        charts.funnel_chart(
            text_by_run, title=f"{key} aggregate funnel", path=charts_dir / f"{kslug}__aggregate__funnel.png")
        charts.reading_order_chart(
            text_by_run, title=f"{key} aggregate reading order",
            path=charts_dir / f"{kslug}__aggregate__reading_order.png")

        chart_paths_by_subsection: "dict[str, list[Path]]" = {
            "labels": [charts_dir / f"{kslug}__aggregate__labels.png"],
            "char": [charts_dir / f"{kslug}__aggregate__char.png"],
            "word": [charts_dir / f"{kslug}__aggregate__word.png"],
            "bbox": [charts_dir / f"{kslug}__aggregate__bbox.png"],
            "rotation": [charts_dir / f"{kslug}__aggregate__rotation.png"],
            "funnel": [charts_dir / f"{kslug}__aggregate__funnel.png"],
            "reading_order": [charts_dir / f"{kslug}__aggregate__reading_order.png"],
        }
        for group, paths in chart_paths_by_subsection.items():
            existing = [p for p in paths if p.is_file()]
            if existing:
                builder.add_raw_html(
                    '<div class="charts">' + "".join(
                        f'<img src="{p.relative_to(out_dir).as_posix()}">' for p in existing
                    ) + "</div>"
                )

        for name, agg in text_by_run.items():
            nslug = _short_slug(name)
            for text_type in TEXT_TYPES:
                charts.font_size_histogram_chart(
                    agg, text_type, title=f"{key} {name} {text_type} font size",
                    path=charts_dir / f"{kslug}__{nslug}__{text_type}__font_size.png")
            charts.extra_chars_table_image(
                agg, title=f"{key} {name} extra predicted chars",
                path=charts_dir / f"{kslug}__{nslug}__extra_chars.png")

        if any(v is not None for v in vector_by_run.values()):
            charts.vector_count_chart(
                vector_by_run, title=f"{key} aggregate vector count",
                path=charts_dir / f"{kslug}__aggregate__vector_count.png")
            charts.endpoint_accuracy_chart(
                vector_by_run, title=f"{key} aggregate endpoint accuracy",
                path=charts_dir / f"{kslug}__aggregate__vector_endpoint.png")
            builder.add_raw_html(
                '<div class="charts">'
                f'<img src="{(charts_dir / f"{kslug}__aggregate__vector_count.png").relative_to(out_dir).as_posix()}">'
                f'<img src="{(charts_dir / f"{kslug}__aggregate__vector_endpoint.png").relative_to(out_dir).as_posix()}">'
                '</div>'
            )

        # ---- illustrated error examples (one run's pages at a time) ----
        for r in runs:
            entry = r[key]
            per_page, _agg = text_scored[entry.run_name]
            examples = _collect_examples(entry, per_page, examples_dir, kslug)
            for text_type in TEXT_TYPES:
                for category, label in (
                    ("extra_prediction", "Extra predictions"),
                    ("missed_gt", "Missed GT"),
                    ("confusion", "Confusion misreads"),
                ):
                    cards = examples.get((text_type, category), [])
                    if cards:
                        builder.add_error_examples(f"{label} [{entry.run_name}]", text_type, cards)

        # ---- pipeline diagnostic image galleries ----
        gallery_dir = out_dir / "gallery"
        for r in runs:
            entry = r[key]
            for folder in sorted(entry.doc_dir.glob("*_images")):
                files = sorted(folder.iterdir())[:5]
                if not files:
                    continue
                rslug = _short_slug(entry.run_name)
                gallery_dir.mkdir(parents=True, exist_ok=True)
                copied: "list[Path]" = []
                for f in files:
                    dst = gallery_dir / f"{kslug}__{rslug}__{folder.name}__{f.name}"
                    dst.write_bytes(f.read_bytes())
                    copied.append(Path("gallery") / dst.name)
                builder.add_image_gallery(entry.run_name, folder.name, copied)

        for name, (pp, _agg) in text_scored.items():
            grand_text.setdefault(name, []).extend(r for _pi, r, _g in pp)
        for name, (pp, _agg) in vector_scored.items():
            grand_vector.setdefault(name, []).extend(r for _pi, r in pp)

        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        builder.add_viewer_command(key, f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}")

    # ---- grand aggregate over every shared key ----
    grand_text_agg = {name: aggregate_text_metrics(rs) for name, rs in grand_text.items()}
    grand_vector_agg = {name: aggregate_vector_metrics(rs) for name, rs in grand_vector.items()}
    if grand_text_agg:
        charts.label_description_chart(
            grand_text_agg, title="aggregate labels (all inputs)", path=charts_dir / "aggregate__labels.png")
        charts.char_word_overlap_chart(
            grand_text_agg, field="char_overlap", title="aggregate char recall (all inputs)",
            path=charts_dir / "aggregate__char.png")
        charts.bbox_accuracy_chart(
            grand_text_agg, title="aggregate bbox IoU (all inputs)", path=charts_dir / "aggregate__bbox.png")

        builder.add_key_section("(grand aggregate over every shared input)")
        _add_text_sections(builder, grand_text_agg)
        _add_vector_sections(builder, grand_vector_agg)
        builder.add_raw_html(
            '<div class="charts">' + "".join(
                f'<img src="{p.relative_to(out_dir).as_posix()}">'
                for p in (charts_dir / "aggregate__labels.png", charts_dir / "aggregate__char.png", charts_dir / "aggregate__bbox.png")
                if p.is_file()
            ) + "</div>"
        )

    header = "compared runs:\n" + "\n".join(f"  {Path(d).name}" for d in args.run)
    (out_dir / "benchmark.txt").write_text(
        header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps({
        "runs": [str(Path(d).resolve()) for d in args.run],
        "iou_edge_min": args.iou_threshold,
        "shared_keys": sorted(shared_keys),
    }, indent=2), encoding="utf-8")

    lines = ["# Open each benchmarked input in the viewer (pipelines side by side):"]
    for key in sorted(shared_keys):
        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        lines.append(f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}")
    (out_dir / "viewer_commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "report.html").write_text(builder.render(), encoding="utf-8")

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
