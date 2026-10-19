"""Tests for the pipeline stage -> script resolution."""

import pytest

from emit_pics import pipeline
from emit_pics.paths import REPO_ROOT


def test_stage_order():
    assert pipeline.STAGE_ORDER == (
        "setup", "extract", "analyze", "drift", "brdf",
        "landsat", "compare", "figures", "appendix")


def test_extract_is_allpatches():
    assert pipeline.scripts_for("extract") == [
        "run_PICS_allpatches_cube_extract.py",
        "run_PICS_allpatches_qc_products.py"]


def test_analyze_ends_with_aggregate():
    steps = pipeline.scripts_for("analyze")
    assert steps[-1] == "run_PICS_allpatches_aggregate_pairs.py"
    assert "run_PICS_allpatches_id_matched_pairs.py" in steps


def test_drift_is_the_cross_pics_step():
    assert pipeline.scripts_for("drift") == ["run_matched_pairs_drift_analysis.py"]


def test_brdf_stage_scripts_in_order():
    assert pipeline.scripts_for("brdf") == [
        "run_brdf_assemble.py", "run_brdf_single_channel.py",
        "run_brdf_drift_spectrum.py"]


def test_landsat_compare_figures_stages():
    assert pipeline.scripts_for("landsat")[0] == "run_landsat_cube_extract.py"
    assert pipeline.scripts_for("compare") == ["run_emit_oli2_band_comparison.py"]
    assert "run_publication_figures.py" in pipeline.scripts_for("figures")


def test_invalid_stage_raises():
    with pytest.raises(ValueError):
        pipeline.scripts_for("nope")


def test_every_referenced_script_exists():
    # Guards against a typo in STEPS pointing at a non-existent script.
    for stage in pipeline.STAGE_ORDER:
        for name in pipeline.scripts_for(stage):
            assert (REPO_ROOT / "scripts" / name).exists(), f"missing script: {name}"
