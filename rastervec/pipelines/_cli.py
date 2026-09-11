"""CLI for the pipeline modules (`python -m rastervec.pipelines.current ...`).
Ported from the old `pipeline.py::main`.
"""
from __future__ import annotations

import argparse
import logging

from rastervec.logging_setup import configure_logging, get_logger

_LOG = get_logger("pipelines.cli")


def build_arg_parser(pipeline_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the rastervec '{pipeline_name}' pipeline on one PDF page."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index (default: 0).")
    if pipeline_name == "current":
        parser.add_argument(
            "--no-fast", action="store_true",
            help="Turn FAST text detection into a pass-through (speed-testing).",
        )
        parser.add_argument(
            "--stop-after", default=None, metavar="STEP",
            help="Skip every pipeline step after STEP (one of the STEP_NAMES).",
        )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging + keep intermediates.")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Only WARNING and above.")
    return parser


def main(pipeline_name: str, argv: list[str] | None = None) -> int:
    args = build_arg_parser(pipeline_name).parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    configure_logging(level)

    if pipeline_name == "legacy":
        from rastervec.pipelines.legacy import run_pipeline

        result = run_pipeline(args.pdf, args.page, verbose=args.verbose)
    else:
        from rastervec.pipelines.current import run_pipeline

        result = run_pipeline(
            args.pdf, args.page, enable_fast=not getattr(args, "no_fast", False),
            verbose=args.verbose, stop_after=getattr(args, "stop_after", None),
        )

    n_native = sum(1 for t in result.texts if t.source == "native")
    n_ocr = sum(1 for t in result.texts if t.source == "ocr")
    n_paddle_boxes = len(result.paddle_boxes or [])
    _LOG.info(
        "page %d: %d native word(s), %d OCR reading(s), %d drawing vector(s), %d paddle detect box(es)",
        args.page, n_native, n_ocr, len(result.vectors), n_paddle_boxes,
    )
    for name, secs in result.step_durations.items():
        _LOG.info("  %-9s %6.2fs", name, secs)
    return 0
