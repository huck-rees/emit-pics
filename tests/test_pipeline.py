"""Tests for the pipeline stage/mode -> script resolution."""

import pytest

from emit_pics import pipeline
from emit_pics.paths import REPO_ROOT


def test_setup_and_drift_are_mode_independent():
    for stage in ("setup", "drift"):
        assert pipeline.scripts_for(stage, "single") == pipeline.scripts_for(stage, "allpatches")


def test_extract_forks_single_vs_allpatches():
    assert pipeline.scripts_for("extract", "single") == [
        "run_single_patch_cube_extract.py", "run_single_patch_qc_products.py"]
    assert pipeline.scripts_for("extract", "allpatches") == [
        "run_PICS_allpatches_cube_extract.py", "run_PICS_allpatches_qc_products.py"]


def test_aggregate_is_allpatches_only():
    single = pipeline.scripts_for("analyze", "single")
    allp = pipeline.scripts_for("analyze", "allpatches")
    assert single == ["run_id_matched_pairs.py", "run_delta_calculations.py"]
    assert "run_PICS_allpatches_aggregate_pairs.py" in allp
    assert not any("aggregate" in s for s in single)


def test_drift_is_the_cross_pics_step():
    assert pipeline.scripts_for("drift", "allpatches") == ["run_matched_pairs_drift_analysis.py"]


def test_invalid_stage_or_mode_raises():
    with pytest.raises(ValueError):
        pipeline.scripts_for("nope", "single")
    with pytest.raises(ValueError):
        pipeline.scripts_for("setup", "nope")


def test_every_referenced_script_actually_exists():
    # Guards against a typo in STEPS pointing at a non-existent script.
    for stage in pipeline.STAGE_ORDER:
        for mode in pipeline.MODES:
            for name in pipeline.scripts_for(stage, mode):
                assert (REPO_ROOT / "scripts" / name).exists(), f"missing script: {name}"
