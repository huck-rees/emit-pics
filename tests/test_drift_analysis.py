"""Unit tests for the drift statistics: Theil-Sen fit + BH FDR control."""

import numpy as np
import pandas as pd

from emit_pics.matched_pairs_drift_analysis import (
    DAYS_PER_YEAR, benjamini_hochberg, compute_drift_spectrum,
    fit_channel_drift, select_one_patch_per_pics,
)
from tests._helpers import build_golden_delta_df


# ------------------------------------------------- select_one_patch_per_pics
def _pool(rows):
    return pd.DataFrame(rows, columns=["PICS_ID", "patch_id", "val"])


def test_select_single_patch_pics_taken_as_is():
    df = _pool([("Algeria_3", "10_3_2", 1), ("Libya_1", "10_1_0", 2)])
    sel, amb = select_one_patch_per_pics(df)
    assert amb == []
    assert set(sel["PICS_ID"]) == {"Algeria_3", "Libya_1"}


def test_select_roster_disambiguates_polluted_pics():
    # Libya_4 has three patches (the all-patches experiment); roster picks one.
    df = _pool([("Libya_4", "10_7_1", 1), ("Libya_4", "10_0_0", 9),
                ("Libya_4", "10_1_1", 9), ("Algeria_5", "10_5_5", 2)])
    sel, amb = select_one_patch_per_pics(df, {"Libya_4": "10_7_1"})
    assert amb == []
    lib = sel[sel["PICS_ID"] == "Libya_4"]
    assert set(lib["patch_id"]) == {"10_7_1"} and lib["val"].tolist() == [1]


def test_select_reports_ambiguous_without_roster():
    df = _pool([("Libya_4", "10_7_1", 1), ("Libya_4", "10_0_0", 9)])
    sel, amb = select_one_patch_per_pics(df)
    assert sel.empty
    assert amb == [("Libya_4", ["10_0_0", "10_7_1"])]


# ------------------------------------------------------------- benjamini_hochberg
def test_bh_rejects_only_the_small_pvalue():
    reject, q = benjamini_hochberg([0.001, 0.5, 0.5, 0.5], alpha=0.05)
    assert reject.tolist() == [True, False, False, False]
    np.testing.assert_allclose(q[0], 0.004, rtol=1e-6)   # 0.001 * 4 / 1


def test_bh_nan_passes_through_and_is_excluded_from_ranking():
    reject, q = benjamini_hochberg([0.001, np.nan, 0.5], alpha=0.05)
    assert bool(reject[0]) is True
    assert bool(reject[1]) is False
    assert np.isnan(q[1])


def test_bh_qvalues_are_monotonic():
    _, q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    # q-values are non-decreasing in p-value order
    assert np.all(np.diff(q) >= -1e-12)


# --------------------------------------------------------------- fit_channel_drift
def test_fit_channel_drift_recovers_known_slope():
    dt = np.array([100.0, 200.0, 300.0, 400.0])
    dR = 0.01 * dt                               # exactly 0.01 %/day
    out = fit_channel_drift(dt, dR)
    assert out["n"] == 4
    np.testing.assert_allclose(out["slope_pct_yr"], 0.01 * DAYS_PER_YEAR, rtol=1e-6)
    assert abs(out["intercept_pct"]) < 1e-9
    np.testing.assert_allclose(out["kendall_tau"], 1.0)


def test_fit_channel_drift_too_few_points_is_nan():
    out = fit_channel_drift([1.0, 2.0], [1.0, 2.0])
    assert out["n"] == 2
    assert np.isnan(out["slope_pct_yr"])


def test_fit_channel_drift_degenerate_x_is_nan():
    out = fit_channel_drift([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert out["n"] == 3                         # counts the points...
    assert np.isnan(out["slope_pct_yr"])         # ...but cannot fit a slope


def test_fit_channel_drift_drops_nan_pairs():
    out = fit_channel_drift([100.0, 200.0, np.nan, 300.0],
                            [1.0, 2.0, 99.0, 3.0])
    assert out["n"] == 3                          # the NaN-x pair is dropped


# ------------------------------------------------------------ compute_drift_spectrum
def test_compute_drift_spectrum_excludes_absorption_and_recovers_slopes():
    spec = compute_drift_spectrum(build_golden_delta_df(), "rad")
    spec = spec.set_index("band_index")

    np.testing.assert_allclose(spec.loc[0, "slope_pct_yr"], 2.0, rtol=1e-6)
    np.testing.assert_allclose(spec.loc[1, "slope_pct_yr"], -1.0, rtol=1e-6)

    # band 2 is a water-vapor absorption band -> excluded (NaN stats)
    assert bool(spec.loc[2, "absorption_band"]) is True
    assert np.isnan(spec.loc[2, "slope_pct_yr"])
