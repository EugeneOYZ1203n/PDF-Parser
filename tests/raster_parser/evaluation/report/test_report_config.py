"""`ReportConfig`: backend names and run-setting defaults come from `core`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from raster_parser.models.settings import RunSettings
from raster_parser.evaluation.report.report_config import ReportConfig


def test_run_setting_defaults_follow_run_settings() -> None:
    cfg, rs = ReportConfig(), RunSettings()
    assert (cfg.ocr_dpi, cfg.use_angle_cls, cfg.workers) == (
        rs.ocr_dpi,
        rs.use_angle_cls,
        rs.workers,
    )


@pytest.mark.parametrize("field", ["p2", "p3"])
def test_unknown_backend_is_rejected_with_the_valid_names(field) -> None:
    with pytest.raises(ValidationError, match=rf"unknown {field} backend 'Nope'; choose from \["):
        ReportConfig(**{field: "Nope"})
