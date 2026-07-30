"""Lock the output filename + directory contract (guards the whole #1 sweep).

If any of these change, an output would silently move to a new location — these
tests make that impossible to do by accident.
"""

from emit_pics import paths

TS = "20230101T100000"


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
