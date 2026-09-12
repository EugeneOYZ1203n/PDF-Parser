"""Run `native_label_pdf` on one PDF page and open the vector-label window
to view (and optionally adjust) the result. Replaces `view_auto_labels.py`.

`native_label_pdf` produces `source="native"` line-region entries with no
backing vectors selectable in `VectorLabelApp` (it only ever draws/selects
the page's raw `Vector` pool) -- but any entry loaded from `--out` is still
shown, so the two data sets coexist in the one label file even though this
window can't select a native entry's own region directly.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe scripts/label/view_native_labels.py path/to.pdf --page 0

`--out` controls where the label JSON is written (default:
`outputs/labels/<pdf stem>_p<N>_native_labels.json`); pass an existing path
to append manual edits to a file you're already building.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels, save_labels
from rastervec.Evaluation.Labelling.native_label import native_label_pdf
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir

from vector_label import VectorLabelApp  # noqa: E402 -- sys.path set up above

_LOG = get_logger("view_native_labels")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View native_label_pdf output in the vector-label window.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out", default=None,
        help="Where to write the label JSON (default: outputs/labels/<pdf stem>_p<N>_native_labels.json). "
             "An existing file is loaded first, so manual edits accumulate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)

    native = native_label_pdf(args.pdf, args.page)
    out_path = args.out or str(
        output_dir("labels") / f"{Path(args.pdf).stem}_p{args.page}_native_labels.json"
    )

    # Merge onto any existing file so re-running never drops manual edits;
    # existing entries win on an id clash.
    labels = load_labels(out_path) if Path(out_path).exists() else LabelSet(pdf_path=args.pdf)
    known = {e.label_id for e in labels.entries}
    added = [e for e in native.entries if e.label_id not in known]
    labels.entries.extend(added)
    save_labels(labels, out_path)
    _LOG.info(
        "%d native label(s); %d new, %d total in %s",
        len(native.entries), len(added), len(labels.entries), out_path,
    )

    VectorLabelApp(args.pdf, args.page, out_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
