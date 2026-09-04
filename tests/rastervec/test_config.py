"""Value pins for rastervec/config.py.

Each test restates one constant's value alongside the behaviour it gates.
A failing test here means someone changed a pipeline threshold -- read the
test name / docstring to see what moved, then update the pin deliberately.
"""
from __future__ import annotations

from rastervec import config


# -- Vector Classification ---------------------------------------------------

def test_max_dimension_fraction():
    # Steps 1 & 5: 10% of the page's smaller side is the border/frame cutoff.
    assert config.MAX_DIMENSION_FRACTION == 0.10


def test_signature_round_px():
    # Step 2: shape-signature quantisation grid, in PDF points.
    assert config.SIGNATURE_ROUND_PX == 0.5


def test_duplicate_run_min_length():
    # Step 3a: 5+ consecutive same-shape items = a dropped hatching run.
    assert config.DUPLICATE_RUN_MIN_LENGTH == 5


def test_seq_overlap_tolerance_px():
    assert config.SEQ_OVERLAP_TOLERANCE_PX == 1.0


def test_min_group_size_px():
    assert config.MIN_GROUP_SIZE_PX == 1.0


def test_spatial_cluster_threshold():
    # Step 6: bbox-gap tolerance (points) for merging groups into clusters.
    assert config.SPATIAL_CLUSTER_THRESHOLD == 10.0


def test_spatial_size_tolerance():
    assert config.SPATIAL_SIZE_TOLERANCE == 0.30


def test_perimeter_margin_fraction():
    assert config.PERIMETER_MARGIN_FRACTION == 0.1


def test_density_grid_bounds():
    assert config.DENSITY_DEFAULT_GRID_SIZE == 4
    assert config.DENSITY_MIN_CELL_PX == 5.0
    assert config.DENSITY_MAX_CELL_PX == 40.0
    assert config.DENSITY_MAX_EMPTY_FRACTION == 0.70


def test_pattern_thresholds():
    assert config.PATTERN_SPACING_TOLERANCE == 0.20
    assert config.PATTERN_MIN_REPEAT_COUNT == 3
    assert config.PATTERN_FRACTION_THRESHOLD == 0.70


def test_low_variety_ramp():
    assert config.LOW_VARIETY_MIN_MEMBER_COUNT == 5
    assert config.LOW_VARIETY_MIN_REQUIRED == 1
    assert config.LOW_VARIETY_MAX_MEMBER_COUNT == 300
    assert config.LOW_VARIETY_MAX_REQUIRED == 10


def test_unique_cluster_tolerance():
    assert config.UNIQUE_CLUSTER_TOLERANCE == 0.04


# -- Pipeline stages -------------------------------------------------------

def test_fast_thresholds():
    assert config.FAST_PAGE_RENDER_DPI == 150
    assert config.FAST_COMBINED_KEEP_THRESHOLD == 0.2
    assert config.FAST_TILE_BLOCK_SIZE == 2048
    assert config.FAST_TILE_SCALE_FACTOR == 5
    assert config.FAST_TILE_ROTATION_COUNT == 4


def test_spatial_regroup_tolerance_px():
    assert config.SPATIAL_REGROUP_TOLERANCE_PX == 1.0


def test_ocr_backend_default_is_light():
    assert config.USE_LIGHT_OCR_BACKEND is True


def test_min_render_side_px():
    assert config.MIN_RENDER_SIDE_PX == 50


# -- Renderer / OCR crop --------------------------------------------------

def test_cluster_render_padding():
    assert config.MIN_CLUSTER_PADDING == 4.0
    assert config.OCR_VERTICAL_PADDING_FRACTION == 0.05
    assert config.OCR_HORIZONTAL_PADDING_FRACTION == 0.30


def test_rec_line_geometry():
    assert config.REC_LINE_HEIGHT_PX == 48
    assert config.REC_LINE_MAX_WIDTH_PX == 1024


def test_light_backend_knobs():
    assert config.DOC_ORI_MIN_CONFIDENCE == 0.7
    assert config.VERTICAL_ASPECT == 1.5
    assert config.REC_BATCH_SIZE == 128
