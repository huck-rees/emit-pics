"""Lock the output filename + directory contract (guards the whole #1 sweep).

If any of these change, an output would silently move to a new location — these
tests make that impossible to do by accident.
"""

import pytest

from emit_pics import paths

TS = "20230101T100000"


@pytest.fixture
def reset_variant():
    """Restore the canonical (baseline) variant after a test mutates it."""
    yield
    paths.set_variant("")
    paths.set_brdf_label("")


def test_filenames_match_historical_convention():
    assert paths.patch_dict("Libya_4").name == "patch_EMIT_dict_Libya_4.csv"
    assert paths.patch_bnds("Libya_4").name == "patches_Libya_4_bnds.gpkg"
    assert paths.patch_atts("Libya_4").name == "patches_Libya_4_atts.csv"

    assert paths.geometry("Libya_4", "10_7_1").name == "geometry_Libya_4_10_7_1.csv"
    assert paths.matched_pairs("Libya_4", "10_7_1").name == "Libya_4_10_7_1_matched_pairs.csv"
    assert paths.delta_per_patch("Libya_4", "10_7_1").name == "Libya_4_10_7_1_delta.csv"
    assert paths.delta_aggregated("Libya_4", "10").name == "Libya_4_10_delta.csv"
    assert paths.drift("10").name == "10_drift.csv"

    assert paths.perchannel("Libya_4", "10_7_1", TS, "rfl").name == \
        "Libya_4_10_7_1_20230101T100000_perchannel_rfl.csv"
    assert paths.perchannel("Libya_4", "10_7_1", TS, "rad").name == \
        "Libya_4_10_7_1_20230101T100000_perchannel_rad.csv"
    assert paths.mask_summary("Libya_4", "10_7_1", TS).name == \
        "Libya_4_10_7_1_20230101T100000_mask_summary.csv"
    assert paths.pixel_cube("Libya_4", "10_7_1", TS).name == \
        "cube_Libya_4_10_7_1_20230101T100000.nc"

    assert paths.qc_sheet("Libya_4", "10_7_1").name == "Libya_4_10_7_1_qc.csv"
    assert paths.qc_pdf("Libya_4", "10_7_1").name == "Libya_4_10_7_1_QC_snapshots.pdf"


def test_figure_filenames():
    assert paths.matched_pairs_fig("Libya_4", "10_7_1").name == \
        "Libya_4_10_7_1_matched_pairs.png"
    assert paths.spectra_fig("Libya_4", "10_7_1", "median", "rfl").name == \
        "Libya_4_10_7_1_median_rfl_spectra.png"
    assert paths.spectra_fig("Libya_4", "10_7_1", "median", "rad").name == \
        "Libya_4_10_7_1_median_rad_spectra.png"
    assert paths.drift_spectrum_fig("10").name == "10_drift_spectrum.png"
    assert paths.channel_fig("10", 500.0).name == "10_0500.png"     # {wl:04.0f}
    assert paths.channel_fig("10", 1234.5).name == "10_1234.png"
    assert paths.granule_coverage_fig("Libya_4", 10).name == \
        "Libya_4_10km_EMITgranule_coverage.png"


def test_brdf_data_filenames():
    assert paths.brdf_design().name == "brdf_design.csv"
    assert paths.brdf_leverage().name == "brdf_leverage.csv"
    assert paths.brdf_persite_gamma(858.0).name == "brdf_persite_gamma_858nm.csv"
    assert paths.brdf_joint_fit(858.0).name == "brdf_joint_fit_858nm.csv"
    assert paths.brdf_bootstrap_gamma(858.0).name == "brdf_bootstrap_gamma_858nm.csv"
    assert paths.brdf_drift_spectrum().name == "brdf_drift_spectrum.csv"
    # wavelength is rounded to a whole nm in the token.
    assert paths.brdf_joint_fit(857.6).name == "brdf_joint_fit_858nm.csv"


def test_brdf_figure_filenames():
    assert paths.brdf_leverage_fig(858.0).name == "brdf_leverage_scatter_858nm.png"
    assert paths.brdf_persite_gamma_fig(858.0).name == "brdf_persite_gamma_858nm.png"
    assert paths.brdf_corrected_fig(858.0).name == "brdf_corrected_858nm.png"
    assert paths.brdf_bootstrap_fig(858.0).name == "brdf_bootstrap_858nm.png"
    assert paths.brdf_drift_spectrum_fig().name == "brdf_drift_spectrum.png"


def test_brdf_paths_land_in_expected_dirs():
    assert paths.brdf_design().parent == paths.BRDF_DIR
    assert paths.brdf_drift_spectrum().parent == paths.BRDF_DIR
    assert paths.brdf_persite_gamma(858.0).parent == paths.BRDF_DIR
    assert paths.brdf_leverage_fig(858.0).parent == paths.BRDF_FIG_DIR
    assert paths.brdf_drift_spectrum_fig().parent == paths.BRDF_FIG_DIR


def test_site_spelling_is_normalised():
    # Any accepted spelling resolves to the same on-disk file.
    for spelling in ["libya_4", "Libya 4", "Libya_4"]:
        assert paths.delta_per_patch(spelling, "10_7_1").name == \
            "Libya_4_10_7_1_delta.csv"


def test_paths_land_in_expected_dirs():
    assert paths.delta_per_patch("Libya_4", "10_7_1").parent == paths.DELTA_PER_PATCH_DIR
    assert paths.delta_aggregated("Libya_4", "10").parent == paths.DELTA_AGG_DIR
    assert paths.perchannel("Libya_4", "10_7_1", TS, "rfl").parent == paths.RFL_DIR
    assert paths.perchannel("Libya_4", "10_7_1", TS, "rad").parent == paths.RAD_DIR
    assert paths.channel_fig("10", 500.0).parent == paths.CHANNEL_FIG_DIR


# ----------------------------------------------------- radiometry output variant
def test_baseline_variant_is_empty_and_layout_flat():
    # Default state: no variant -> historical flat layout (regression guard).
    assert paths.active_variant() == ""
    assert paths.DRIFT_DIR == paths.DATA / "drift"
    assert paths.BRDF_DIR == paths.DATA / "brdf"
    assert paths.DELTA_AGG_DIR == paths.DATA / "delta_r" / "aggregated"
    assert paths.drift("10") == paths.DATA / "drift" / "10_drift.csv"


def test_variant_nests_only_derived_dirs(reset_variant):
    paths.set_variant("no_earthsun")
    assert paths.active_variant() == "no_earthsun"
    # Derived outputs nest under the variant subfolder...
    assert paths.DELTA_PER_PATCH_DIR == paths.DATA / "delta_r" / "per_patch" / "no_earthsun"
    assert paths.DELTA_AGG_DIR == paths.DATA / "delta_r" / "aggregated" / "no_earthsun"
    assert paths.DRIFT_DIR == paths.DATA / "drift" / "no_earthsun"
    assert paths.BRDF_DIR == paths.DATA / "brdf" / "no_earthsun"
    assert paths.CHANNEL_FIG_DIR == \
        paths.FIGURES / "matched_pairs" / "no_earthsun" / "individual_channels"
    assert paths.BRDF_FIG_DIR == paths.FIGURES / "brdf" / "no_earthsun"
    # ...and the builder functions follow the redirect.
    assert paths.drift("10") == paths.DATA / "drift" / "no_earthsun" / "10_drift.csv"
    assert paths.brdf_design() == paths.DATA / "brdf" / "no_earthsun" / "brdf_design.csv"
    assert paths.delta_aggregated("Libya_4", "10").parent == paths.DELTA_AGG_DIR
    # Raw-input dirs are NEVER moved (shared across variants).
    assert paths.RAD_DIR == paths.DATA / "radiance"
    assert paths.RFL_DIR == paths.DATA / "reflectance"
    assert paths.GEOM_DIR == paths.DATA / "geometry"
    assert paths.MATCHED_DIR == paths.DATA / "matched_pairs"
    assert paths.perchannel("Libya_4", "10_7_1", TS, "rad").parent == paths.DATA / "radiance"


def test_filenames_are_variant_independent(reset_variant):
    # Only the parent folder changes; the file basename is identical.
    paths.set_variant("no_earthsun")
    assert paths.drift("10").name == "10_drift.csv"
    assert paths.delta_aggregated("Libya_4", "10").name == "Libya_4_10_delta.csv"
    assert paths.brdf_drift_spectrum().name == "brdf_drift_spectrum.csv"


def test_set_variant_round_trips_back_to_baseline(reset_variant):
    baseline = paths.DRIFT_DIR
    paths.set_variant("no_earthsun")
    assert paths.DRIFT_DIR != baseline
    paths.set_variant("")
    assert paths.DRIFT_DIR == baseline


# ------------------------------------------------------- BRDF-only output label
def test_brdf_label_nests_only_brdf_dirs(reset_variant):
    paths.set_brdf_label("20km")
    assert paths.active_brdf_label() == "20km"
    # BRDF data + figure dirs nest under the label...
    assert paths.BRDF_DIR == paths.DATA / "brdf" / "20km"
    assert paths.BRDF_FIG_DIR == paths.FIGURES / "brdf" / "20km"
    assert paths.brdf_design() == paths.DATA / "brdf" / "20km" / "brdf_design.csv"
    assert paths.brdf_drift_spectrum_fig() == paths.FIGURES / "brdf" / "20km" / "brdf_drift_spectrum.png"
    # ...but nothing else moves (matched-pairs drift, deltas, inputs untouched).
    assert paths.DRIFT_DIR == paths.DATA / "drift"
    assert paths.DELTA_AGG_DIR == paths.DATA / "delta_r" / "aggregated"
    assert paths.RAD_DIR == paths.DATA / "radiance"


def test_brdf_label_composes_with_variant(reset_variant):
    paths.set_variant("no_earthsun")
    paths.set_brdf_label("20km")
    # variant is the outer folder, the BRDF label nests inside it.
    assert paths.BRDF_DIR == paths.DATA / "brdf" / "no_earthsun" / "20km"
    assert paths.brdf_drift_spectrum() == \
        paths.DATA / "brdf" / "no_earthsun" / "20km" / "brdf_drift_spectrum.csv"
    # the non-BRDF derived dirs still carry only the variant, not the BRDF label.
    assert paths.DRIFT_DIR == paths.DATA / "drift" / "no_earthsun"


def test_brdf_label_empty_is_flat_baseline():
    assert paths.active_brdf_label() == ""
    assert paths.BRDF_DIR == paths.DATA / "brdf"
