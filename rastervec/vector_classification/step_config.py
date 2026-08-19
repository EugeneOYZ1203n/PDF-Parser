"""`StepConfig`: one self-contained classification pipeline step instance.

Replaces the old `order: list[str]` + `step_params: dict[str, dict]` design
(step_params was keyed by step *name*, so two occurrences of the same step
at different ordinal positions shared one param dict -- the bug this
rework fixes). A `list[StepConfig]` has no such shared key: every instance
carries its own kind/metric/condition/scope/params, so two occurrences of
the same metric at different positions are simply two different list
entries, each independently tunable. The same list serializes directly
to/from JSON for save/load.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

_VALID_KINDS = ("filter", "cluster", "group")
_VALID_CONDITIONS = (">", ">=", "<", "<=", "==", "!=", "within", "outside")
_VALID_FILTER_SCOPES = ("item", "cluster", "items_in_cluster", "cluster_all_items")
_VALID_CLUSTER_METHODS = ("pairwise", "global")
_VALID_CLUSTER_SCOPES = ("within_group", "global_flatten")
_VALID_GROUP_SCOPES = ("path", "cluster")
_VALID_AGGREGATES = (None, "mean", "median")
_VALID_AGGREGATE_SCOPES = ("global", "group")


@dataclass
class StepConfig:
    kind: str  # "filter" | "cluster" | "group"

    metric: str = ""
    condition: str | None = None  # filter only
    threshold: float | list[float] | None = None  # [low, high] for within/outside
    scope: str = "item"
    method: str | None = None  # cluster only: "pairwise" | "global"
    aggregate: str | None = None  # filter only
    aggregate_scope: str = "global"  # filter only
    params: dict[str, float | int | str] = field(default_factory=dict)
    label: str = ""

    def validate(self) -> None:
        from rastervec.vector_classification.metrics import PAIRWISE_METRICS, SCALAR_METRICS

        if self.kind not in _VALID_KINDS:
            raise ValueError(f"invalid kind: {self.kind!r}")

        if self.kind == "filter":
            if self.metric not in SCALAR_METRICS:
                raise ValueError(f"unknown filter metric: {self.metric!r}")
            if self.condition not in _VALID_CONDITIONS:
                raise ValueError(f"invalid condition: {self.condition!r}")
            if self.scope not in _VALID_FILTER_SCOPES:
                raise ValueError(f"invalid filter scope: {self.scope!r}")
            if self.aggregate not in _VALID_AGGREGATES:
                raise ValueError(f"invalid aggregate: {self.aggregate!r}")
            if self.aggregate_scope not in _VALID_AGGREGATE_SCOPES:
                raise ValueError(f"invalid aggregate_scope: {self.aggregate_scope!r}")
            if self.condition in ("within", "outside"):
                if not (isinstance(self.threshold, list) and len(self.threshold) == 2):
                    raise ValueError(f"condition {self.condition!r} needs a [low, high] threshold")
            elif not isinstance(self.threshold, (int, float)):
                raise ValueError(f"condition {self.condition!r} needs a numeric threshold")

        elif self.kind == "cluster":
            if self.method not in _VALID_CLUSTER_METHODS:
                raise ValueError(f"invalid cluster method: {self.method!r}")
            if self.method == "pairwise":
                if self.metric not in SCALAR_METRICS:
                    raise ValueError(f"unknown cluster (pairwise) metric: {self.metric!r}")
                if self.scope not in _VALID_CLUSTER_SCOPES:
                    raise ValueError(f"invalid cluster scope: {self.scope!r}")
            else:  # "global"
                if self.metric != "spatial_gap" and self.metric not in PAIRWISE_METRICS:
                    raise ValueError(f"unknown cluster (global) metric: {self.metric!r}")
            if not isinstance(self.threshold, (int, float)):
                raise ValueError("cluster steps need a numeric threshold")

        elif self.kind == "group":
            if self.metric not in PAIRWISE_METRICS:
                raise ValueError(f"unknown group metric: {self.metric!r}")
            if self.scope not in _VALID_GROUP_SCOPES:
                raise ValueError(f"invalid group scope: {self.scope!r}")
            if not isinstance(self.threshold, (int, float)):
                raise ValueError("group steps need a numeric threshold")


def to_json(steps: list[StepConfig]) -> list[dict]:
    return [asdict(s) for s in steps]


def from_json(data: list[dict]) -> list[StepConfig]:
    steps = [StepConfig(**d) for d in data]
    for s in steps:
        s.validate()
    return steps


def save_to_file(steps: list[StepConfig], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(steps), f, indent=2)


def load_from_file(path: str) -> list[StepConfig]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_json(data)


# Reproduces the original fixed pipeline's 5 real operations (both
# path-level filters, one spatial clustering pass, both group-level
# filters) -- no "none" placeholders needed now that the step list is
# dynamic length.
DEFAULT_PIPELINE: tuple[StepConfig, ...] = (
    StepConfig(
        kind="filter", metric="layout_panel_singleton", condition="==", threshold=0.0,
        scope="item", label="Layout panels",
    ),
    StepConfig(
        kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2,
        scope="item", label="Large bbox (item)",
    ),
    StepConfig(
        kind="cluster", metric="spatial_gap", method="global", threshold=8.0,
        label="Spatial",
    ),
    StepConfig(
        kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2,
        scope="cluster", label="Large bbox (cluster)",
    ),
    StepConfig(
        kind="filter", metric="aspect_ratio", condition="<=", threshold=10.0,
        scope="cluster", label="Aspect ratio",
    ),
)
