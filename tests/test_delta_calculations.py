"""Unit tests for the per-wavelength delta computation (the band-order join)."""

import numpy as np

from emit_pics.delta_calculations import (
    DELTA_CSV_COLS, _pct_change, absorption_band_flag, compute_pair_delta,
)
from tests._helpers import make_meta, make_spec


# ------------------------------------------------------------------ _pct_change
def test_pct_change_basic():
    out = _pct_change(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    np.testing.assert_allclose(out, [10.0, -10.0])


def test_pct_change_near_zero_denominator_is_nan():
    out = _pct_change(np.array([1e-9, 100.0]), np.array([5.0, 150.0]))
    assert np.isnan(out[0])                 # near-zero denom guarded
    np.testing.assert_allclose(out[1], 50.0)


def test_pct_change_non_finite_is_nan():
    out = _pct_change(np.array([100.0]), np.array([np.inf]))
    assert np.isnan(out[0])


# ----------------------------------------------------------- absorption_band_flag
def test_absorption_flag_explicit_wls_within_tol():
    wl = np.array([500.0, 1400.4, 1400.6, 1900.0])
    flag = absorption_band_flag(wl, bad_band_wls=np.array([1400.0]), tol=0.5)
    assert flag.tolist() == [False, True, False, False]   # 0.4<=tol, 0.6>tol


def test_absorption_flag_fallback_ranges():
    # No explicit bad bands -> fall back to ABSORPTION_BANDS_NM windows.
    wl = np.array([1000.0, 1400.0, 1850.0])
    flag = absorption_band_flag(wl, bad_band_wls=None)
    assert flag.tolist() == [False, True, True]


# ---------------------------------------------------------- compute_pair_delta
def test_compute_pair_delta_values_and_schema():
    t1 = make_spec([500.0, 600.0, 700.0],
                   med_rfl=[0.10, 0.20, 0.30], mean_rfl=[0.10, 0.20, 0.30],
                   med_rad=[10.0, 20.0, 30.0], mean_rad=[10.0, 20.0, 30.0])
    t2 = make_spec([500.0, 600.0, 700.0],
                   med_rfl=[0.11, 0.19, 0.30], mean_rfl=[0.11, 0.19, 0.30],
                   med_rad=[11.0, 20.0, 33.0], mean_rad=[11.0, 20.0, 33.0])
    out = compute_pair_delta(t1, t2, make_meta())

    assert list(out.columns) == DELTA_CSV_COLS
    assert out["band_index"].tolist() == [0, 1, 2]
    assert out["wavelength"].tolist() == [500.0, 600.0, 700.0]
    np.testing.assert_allclose(out["delta_rfl_pct_median"], [10.0, -5.0, 0.0])
    np.testing.assert_allclose(out["delta_rad_pct_median"], [10.0, 0.0, 10.0])
    assert out["absorption_band_flag"].tolist() == [False, False, False]


def test_compute_pair_delta_unequal_band_count_returns_empty():
    t1 = make_spec([500.0, 600.0, 700.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                   [10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    t2 = make_spec([500.0, 600.0], [0.1, 0.2], [0.1, 0.2], [10.0, 20.0],
                   [10.0, 20.0])
    out = compute_pair_delta(t1, t2, make_meta())
    assert out.empty
    assert list(out.columns) == DELTA_CSV_COLS   # empty but schema-correct


def test_compute_pair_delta_wavelength_gap_beyond_tol_returns_empty():
    # band grids diverge by 5 nm > WL_MATCH_TOL_NM (3) -> deemed non-corresponding
    t1 = make_spec([500.0, 600.0, 700.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                   [10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    t2 = make_spec([500.0, 600.0, 705.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                   [10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert compute_pair_delta(t1, t2, make_meta()).empty


def test_compute_pair_delta_small_wavelength_drift_within_tol_kept():
    # 2 nm cross-era drift (<= tol) is accepted; t1's grid is canonical.
    t1 = make_spec([500.0, 600.0, 700.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                   [10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    t2 = make_spec([500.0, 600.0, 702.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                   [10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    out = compute_pair_delta(t1, t2, make_meta())
    assert len(out) == 3
    assert out["wavelength"].tolist() == [500.0, 600.0, 700.0]       # t1 grid
    assert out["wavelength_t2"].tolist() == [500.0, 600.0, 702.0]    # t2 label
