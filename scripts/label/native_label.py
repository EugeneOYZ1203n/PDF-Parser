"""CLI: derive ground-truth text labels from a PDF page's own native text
and persist them, with a vectorised sidecar so the resulting
`vector_signatures` are re-resolvable later. Replaces `auto_label.py`'s CLI
(`source="auto"` -> `source="native"`).

The line-grouping/rotation logic itself lives in
`rastervec.Evaluation.Labelling.native_label.native_label_pdf` (also used
directly, without the vectorisation step, by the benchmark's per-page
ground-truth generation -- see that module's own docstring for why the two
are split: the benchmark runs this on a hot per-page loop and only needs
bbox/text/rotation, not a persisted vector-backed sidecar).

    .venv/Scripts/python.exe scripts/label/native_label.py path/to.pdf --page 0

Writes `<out_dir>/<stem>.json` (the labels) and
`<out_dir>/<stem>_p<N>_native_vectors.pdf` (the vectorised page render the
`vector_signatures` were resolved against).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rastervec.Evaluation.conversion import convert_page_text_only
from rastervec.Evaluation.Labelling.label_schema import load_labels, save_labels, LabelSet
from rastervec.Evaluation.Labelling.native_label import (
    attach_vector_signatures,
    native_label_pdf,
)
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir

_LOG = get_logger("native_label_cli")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive native-text ground-truth labels for one page.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out", default=None,
        help="Path to save/load the label JSON file "
        "(default: outputs/labels/<pdf stem>.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    out_path = args.out or str(output_dir("labels") / f"{Path(args.pdf).stem}.json")
    stem = Path(args.pdf).stem
    vectors_pdf_path = str(output_dir("labels") / f"{stem}_p{args.page}_native_vectors.pdf")

    labels = native_label_pdf(args.pdf, args.page)
    convert_page_text_only(args.pdf, args.page, vectors_pdf_path)
    attach_vector_signatures(labels, args.page, vectors_pdf_path)

    existing = load_labels(out_path) if Path(out_path).exists() else LabelSet(pdf_path=args.pdf)
    known = {e.label_id for e in existing.entries}
    added = [e for e in labels.entries if e.label_id not in known]
    existing.entries.extend(added)
    save_labels(existing, out_path)

    _LOG.info(
        "%d native label(s); %d new, %d total in %s (vectors: %s)",
        len(labels.entries), len(added), len(existing.entries), out_path, vectors_pdf_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
