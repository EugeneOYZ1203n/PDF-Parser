"""JSON round-trip helper shared by the models' `from_dict`."""

from __future__ import annotations


def lists_to_tuples(obj):
    """Lists (from JSON) back to tuples, recursively, inside dicts too."""
    if isinstance(obj, list):
        return tuple(lists_to_tuples(x) for x in obj)
    if isinstance(obj, dict):
        return {k: lists_to_tuples(v) for k, v in obj.items()}
    return obj
