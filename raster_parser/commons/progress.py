"""Stage progress on stderr: a timestamped line per stage and tqdm bars over long loops.

Shown while the `raster_parser.progress` logger is enabled for DEBUG, which it is by default
(`logging_setup` puts the `raster_parser` logger at DEBUG). A host app silences it with
`logging.getLogger("raster_parser").setLevel(logging.INFO)`. Progress never changes a result.

    with stage("p3 v2VectorClass › ocr_detect"):
        for chunk in bar(chunks, desc="detect chunks"):
            ...

Writes go through `tqdm.write`, so lines and log records don't break an open bar. Only the
calling process shows progress; spawn workers configure no logging and are tracked by a bar over
`as_completed` in the parent.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from raster_parser.commons.logging_setup import get_logger

_LOG = get_logger("progress")


def enabled() -> bool:
    return _LOG.isEnabledFor(logging.DEBUG)


def _write(line: str) -> None:
    from tqdm import tqdm

    tqdm.write(f"{time.strftime('%H:%M:%S')}  {line}", file=sys.stderr)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Write `▶ name` before the block and `✔ name  1.23s` after it (`✖` if it raised)."""
    if not enabled():
        yield
        return
    _write(f"▶ {name}")
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        _write(f"✖ {name}  {time.perf_counter() - start:.2f}s")
        raise
    _write(f"✔ {name}  {time.perf_counter() - start:.2f}s")


def note(line: str) -> None:
    """One timestamped line, e.g. a skipped stage."""
    if enabled():
        _write(line)


def bar(iterable: Iterable | None = None, *, desc: str, total: int | None = None, unit: str = "it"):
    """A tqdm bar over `iterable` (or a manual one, with `total`), disabled unless `enabled()`."""
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False, file=sys.stderr,
                disable=not enabled(), dynamic_ncols=True)
