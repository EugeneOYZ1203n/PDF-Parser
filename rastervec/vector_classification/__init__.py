"""Metric-driven classification engine for the Vector stage.

Three step kinds -- filter/cluster/group -- each a thin dispatcher
(filter_step.py/cluster_step.py/group_step.py) driven by a `StepConfig`
(step_config.py) naming a metric (metrics.py's SCALAR_METRICS/
PAIRWISE_METRICS) plus a condition/method and a scope. A pipeline is a
plain `list[StepConfig]`, each entry fully self-contained (no shared
name-keyed param dict), so two occurrences of the same metric at
different positions are simply two independent list entries -- and the
same list serializes directly to/from JSON (step_config.to_json/
from_json/save_to_file/load_from_file) for save/load.

Adding a new metric: add one entry to SCALAR_METRICS or PAIRWISE_METRICS
in metrics.py with a small compute function -- nothing here, in engine.py,
or in the debug app needs to change.
"""
from __future__ import annotations

from rastervec.vector_classification.engine import run_step
from rastervec.vector_classification.metrics import (
    MetricContext,
    MetricSpec,
    PAIRWISE_METRICS,
    SCALAR_METRICS,
    check_condition,
    resolve_value,
)
from rastervec.vector_classification.step_config import (
    DEFAULT_PIPELINE,
    StepConfig,
    from_json,
    load_from_file,
    save_to_file,
    to_json,
)

__all__ = [
    "run_step",
    "MetricContext",
    "MetricSpec",
    "PAIRWISE_METRICS",
    "SCALAR_METRICS",
    "check_condition",
    "resolve_value",
    "DEFAULT_PIPELINE",
    "StepConfig",
    "from_json",
    "load_from_file",
    "save_to_file",
    "to_json",
]
