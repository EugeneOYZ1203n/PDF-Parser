"""Small generic iterable helpers shared across stages."""
from __future__ import annotations

from typing import Callable, Iterable, TypeVar

_T = TypeVar("_T")


def partition(
    items: Iterable[_T], predicate: Callable[[_T], bool]
) -> tuple[list[_T], list[_T]]:
    """Split `items` into `(kept, dropped)`: `kept` is everything the
    predicate returns truthy for, `dropped` the rest. Order within each
    list follows the input. The shared shape of every whole-group /
    whole-cluster filter step in `Vector_Classification`."""
    kept: list[_T] = []
    dropped: list[_T] = []
    for item in items:
        (kept if predicate(item) else dropped).append(item)
    return kept, dropped
