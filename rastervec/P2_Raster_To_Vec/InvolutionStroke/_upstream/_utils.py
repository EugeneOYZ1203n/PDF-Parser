"""Stand-in for upstream's `main/utils/utils.py`: the vendored graph code only
uses `timer` from it (the rest is CLI / env / plotting plumbing)."""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from rastervec.commons.logging_setup import get_logger

_LOG = get_logger("P2.InvolutionStroke.upstream")


@contextmanager
def timer(desc=""):
    result = {"time": -1.0}
    start = perf_counter()
    yield result
    result["time"] = perf_counter() - start
    if desc:
        _LOG.debug("%s time: %.3fs", desc, result["time"])
