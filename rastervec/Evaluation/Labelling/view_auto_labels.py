"""Run `auto_label_pdf` on one PDF page and open the manual-label window
to view (and optionally adjust) the result.

`auto_label_pdf` produces `source="auto"` line-region entries with no
backing cluster; `ManualLabelApp` draws any entry whose signature matches
no live pipeline cluster as a dashed grey box with its text on hover, so
this is just:

    auto_label_pdf -> save_labels(<out>) -> ManualLabelApp(pdf, page, <out>)

The same window still shows the real text-candidate clusters, so you can
right-click one to add a manual label into the very same file.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.view_auto_labels \
        path/to.pdf --page 0

A window opens with a dashed grey box over every native-text line, each
showing its auto-derived text on hover. `--out` controls where the label
JSON is written (default: a temp file); pass an existing path to append
manual edits to a file you're already building.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels, save_labels
from rastervec.Evaluation.Labelling.manual_label import ManualLabelApp
from rastervec.logging_setup import configure_logging, get_logger

_LOG = get_logger("view_auto_labels")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View auto_label_pdf output in the label window.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out", default=None,
        help="Where to write the label JSON (default: a temp file). An existing file is loaded "
             "first, so manual edits accumulate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)

    auto = auto_label_pdf(args.pdf, args.page)
    out_path = args.out or str(
        Path(tempfile.gettempdir()) / f"{Path(args.pdf).stem}_p{args.page}_auto_labels.json"
    )

    # Merge onto any existing file so re-running never drops manual edits;
    # existing entries win on a signature clash.
    labels = load_labels(out_path) if Path(out_path).exists() else LabelSet(pdf_path=args.pdf)
    known = {e.cluster_signature for e in labels.entries}
    added = [e for e in auto.entries if e.cluster_signature not in known]
    labels.entries.extend(added)
    save_labels(labels, out_path)
    _LOG.info(
        "%d auto label(s); %d new, %d total in %s",
        len(auto.entries), len(added), len(labels.entries), out_path,
    )

    ManualLabelApp(args.pdf, args.page, out_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
