"""The pluggable pipeline: Phase 1 (always the same) -> Phase 2 (P2_REGISTRY
choice) -> Phase 3 (P3_REGISTRY choice). See `core/registry.py` for the
backend names and `core/interfaces.py` for the contract each implements.

    phase1 = P1.read_and_extract(pdf_path, page_index)             # texts, images, vectors
    p2_vectors, p2_texts = P2_REGISTRY[p2](phase1.images, phase1.page)
    p3_vectors, p3_texts = P3_REGISTRY[p3](phase1.vectors, p2_vectors, phase1.page)
    -> PipelineResult(texts=phase1.texts + p2_texts + p3_texts, vectors=p3_vectors, ...)

CLI: `python -m rastervec.core.pipeline --pdf PATH --page N [--p2 Stub] [--p3 FastIntoPaddle]`
"""
from __future__ import annotations

import argparse
import sys
import time

from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.core.registry import DEFAULT_P2, DEFAULT_P3, resolve_p2, resolve_p3
from rastervec.core.result import PipelineResult, StepOutcome
from rastervec.P1_Reading_Native.phase1 import read_and_extract

_LOG = get_logger("core.pipeline")


class _StepTimer:
    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self.durations: dict[str, float] = {}
        self.outcomes: dict[str, StepOutcome] = {}
        self._name: str | None = None
        self._start = 0.0

    def __call__(self, name: str) -> "_StepTimer":
        self._name = name
        return self

    def __enter__(self) -> "_StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._start
        name = self._name or "?"
        self.durations[name] = elapsed
        if exc_type is None:
            self.outcomes[name] = StepOutcome(name, "ok", None, elapsed)
            return False
        _LOG.exception("pipeline phase %s failed", name)
        self.outcomes[name] = StepOutcome(name, "error", str(exc), elapsed)
        return self.verbose


def run_pipeline(
    pdf_path: str,
    page_index: int = 0,
    *,
    p2: str = DEFAULT_P2,
    p3: str = DEFAULT_P3,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> PipelineResult:
    """Run one page through Phase 1 (always) -> `p2` -> `p3`. `enable_fast`
    is forwarded to `p3` backends that accept it (ignored otherwise).
    `compute`/`progress_counter` forward to whichever backend accepts them
    (see `Reader/Parallel`/`core/parallel`)."""
    p2_fn = resolve_p2(p2)
    p3_fn = resolve_p3(p3)
    timer = _StepTimer(verbose=verbose)
    extra: dict = {}

    with timer("phase1"):
        phase1 = read_and_extract(pdf_path, page_index)

    with timer("phase2"):
        p2_vectors, p2_texts = p2_fn(phase1.images, phase1.page)

    with timer("phase3"):
        p3_kwargs = {}
        import inspect

        sig = inspect.signature(p3_fn)
        for name, value in (
            ("enable_fast", enable_fast), ("verbose", verbose),
            ("compute", compute), ("progress_counter", progress_counter),
        ):
            if name in sig.parameters:
                p3_kwargs[name] = value
        p3_vectors, p3_texts = p3_fn(phase1.vectors, p2_vectors, phase1.page, **p3_kwargs)

    if verbose:
        extra["phase1"] = phase1
        extra["phase2_vectors"] = p2_vectors
        extra["phase2_texts"] = p2_texts

    all_texts = list(phase1.texts) + list(p2_texts) + list(p3_texts)

    return PipelineResult(
        page=phase1.page,
        texts=all_texts,
        vectors=p3_vectors,
        step_durations=timer.durations,
        p2=p2,
        p3=p3,
        extra=extra,
        step_outputs=(timer.outcomes if verbose else None),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the rastervec pipeline on one PDF page.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--p2", default=DEFAULT_P2)
    parser.add_argument("--p3", default=DEFAULT_P3)
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO")
    result = run_pipeline(
        args.pdf, args.page, p2=args.p2, p3=args.p3,
        enable_fast=not args.no_fast, verbose=args.verbose,
    )
    print(f"p2={result.p2} p3={result.p3} texts={len(result.texts)} vectors={len(result.vectors)}")
    print("step durations:", result.step_durations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
