"""The pluggable pipeline: Phase 1 (always the same) -> Phase 2 (P2_REGISTRY
choice) -> Phase 3 (P3_REGISTRY choice) -> Phase 4 (always the same). See
`core/registry.py` for the backend names and `core/interfaces.py` for the
contract each implements.

    phase1 = P1.read_and_extract(pdf_path, page_index)             # texts, images, vectors
    p2_vectors, p2_texts = P2_REGISTRY[p2](phase1.images, phase1.page)
    p3_vectors, p3_texts = P3_REGISTRY[p3](phase1.vectors, p2_vectors, phase1.page)
    texts, vectors = P4.organize_outputs(phase1.texts, p2_texts, p3_texts, p3_vectors, phase1.page)
    -> PipelineResult(texts=texts, vectors=vectors, ...)

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
from rastervec.P4_Output_Organization.organize import organize_outputs

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
    on_debug_layer=None,
) -> PipelineResult:
    """Run one page through Phase 1 (always) -> `p2` -> `p3` -> Phase 4
    (always). `enable_fast` is forwarded to `p3` backends that accept it
    (ignored otherwise). `compute`/`progress_counter` forward to whichever
    backend accepts them (see `Reader/Parallel`/`core/parallel`).

    `on_debug_layer`, when given, is a `(stage, label, hex, pdf_bytes) ->
    None` callback forwarded to any `p2`/`p3` backend whose own signature
    declares it -- each backend calls it immediately after rendering each of
    its own debug layers, interleaved with its normal computation, instead
    of only after the whole run via the batch `render_debug`/`debug_out`
    path (`core/registry.py`'s `P2_RENDER_DEBUG`/`P3_RENDER_DEBUG`). This
    lets a caller (e.g. `scripts/generate_pipeline_report.py`) write each
    layer to disk as it's produced rather than holding every backend's
    heavier step-local debug data (render crops, masks) for the whole run.
    Independent of `verbose`/`debug_out` -- a caller may use either, both,
    or neither."""
    p2_fn = resolve_p2(p2)
    p3_fn = resolve_p3(p3)
    timer = _StepTimer(verbose=verbose)
    extra: dict = {}
    import inspect

    p2_debug: dict = {}
    p3_debug: dict = {}
    p3_substeps: dict = {}

    with timer("phase1"):
        phase1 = read_and_extract(pdf_path, page_index)

    with timer("phase2"):
        p2_kwargs = {}
        p2_params = inspect.signature(p2_fn).parameters
        if verbose and "debug_out" in p2_params:
            p2_kwargs["debug_out"] = p2_debug
        if on_debug_layer is not None and "on_debug_layer" in p2_params:
            p2_kwargs["on_debug_layer"] = on_debug_layer
        p2_vectors, p2_texts = p2_fn(phase1.images, phase1.page, **p2_kwargs)

    with timer("phase3"):
        p3_kwargs = {}

        sig = inspect.signature(p3_fn)
        for name, value in (
            ("enable_fast", enable_fast), ("verbose", verbose),
            ("compute", compute), ("progress_counter", progress_counter),
        ):
            if name in sig.parameters:
                p3_kwargs[name] = value
        if verbose and "debug_out" in sig.parameters:
            p3_kwargs["debug_out"] = p3_debug
        if on_debug_layer is not None and "on_debug_layer" in sig.parameters:
            p3_kwargs["on_debug_layer"] = on_debug_layer
        if "step_durations" in sig.parameters:
            p3_kwargs["step_durations"] = p3_substeps
        p3_vectors, p3_texts = p3_fn(phase1.vectors, p2_vectors, phase1.page, **p3_kwargs)

    with timer("phase4"):
        texts, vectors = organize_outputs(
            phase1.texts, p2_texts, p3_texts, p3_vectors, phase1.page,
        )

    if verbose:
        extra["phase1"] = phase1
        extra["phase2_vectors"] = p2_vectors
        extra["phase2_texts"] = p2_texts
        extra["p2_debug"] = p2_debug
        extra["p3_debug"] = p3_debug

    return PipelineResult(
        page=phase1.page,
        texts=texts,
        vectors=vectors,
        step_durations=timer.durations,
        p2=p2,
        p3=p3,
        extra=extra,
        step_outputs=(timer.outcomes if verbose else None),
        substep_durations=p3_substeps,
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
    print("phase3 sub-step durations:", result.substep_durations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
