"""Unit tests for the Earth–Sun (E0) radiance normalization.

Covers the Spencer-1971 factor itself, the per-scene spectrum normalizer, and —
the key correctness claim — that normalizing each scene's radiance BEFORE the
delta equals the notebook's ratio-domain correction applied AFTER the delta.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from emit_pics.earth_sun import (
    normalize_radiance_spectrum, spencer_E0, RAD_STAT_COLS,
)
from emit_pics.delta_calculations import compute_pair_delta
from tests._helpers import make_meta, make_spec


# ------------------------------------------------------------------ spencer_E0
def test_spencer_E0_perihelion_above_aphelion():
    # E0 peaks near perihelion (early Jan) and troughs near aphelion (early Jul).
    e_jan = spencer_E0(datetime(2023, 1, 3))
    e_jul = spencer_E0(datetime(2023, 7, 4))
    assert e_jan > 1.0 > e_jul
    assert e_jan == max(e_jan, e_jul)


def test_spencer_E0_annual_span_is_about_3_4_percent():
    # Full-year span of E0 is ~6.9% peak-to-peak (~±3.4%).
    doy = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    e = spencer_E0(doy)
    span = e.max() / e.min() - 1.0
    assert 0.06 < span < 0.075


def test_spencer_E0_scalar_vs_array_agree():
    dt = datetime(2023, 4, 15)
    scalar = spencer_E0(dt)
    arr = spencer_E0([dt, dt])
    assert np.isscalar(scalar)
    np.testing.assert_allclose(arr, [scalar, scalar])


def test_spencer_E0_parses_strings():
    np.testing.assert_allclose(spencer_E0("2023-01-03"),
                               spencer_E0(datetime(2023, 1, 3)))


# ----------------------------------------------------- normalize_radiance_spectrum
def test_normalize_divides_only_radiance_by_E0():
    dt = datetime(2023, 1, 3)
    spec = make_spec([500.0, 600.0], med_rfl=[0.1, 0.2], mean_rfl=[0.1, 0.2],
                     med_rad=[10.0, 20.0], mean_rad=[10.0, 20.0])
    out = normalize_radiance_spectrum(spec, dt)
    e0 = spencer_E0(dt)
    # radiance divided by E0 ...
    np.testing.assert_allclose(out["median_rad"], np.array([10.0, 20.0]) / e0)
    np.testing.assert_allclose(out["mean_rad"], np.array([10.0, 20.0]) / e0)
    # ... reflectance untouched, input not mutated.
    np.testing.assert_allclose(out["median_rfl"], [0.1, 0.2])
    np.testing.assert_allclose(spec["median_rad"], [10.0, 20.0])


def test_normalize_none_passthrough():
    assert normalize_radiance_spectrum(None, datetime(2023, 1, 1)) is None


def test_normalize_only_touches_present_columns():
    # A wide per-channel CSV shape (median_rad/mean_rad present) is handled; a
    # frame missing a rad column is left alone for that column.
    df = pd.DataFrame({"central_wl": [500.0], "median_rad": [10.0]})
    out = normalize_radiance_spectrum(df, datetime(2023, 1, 3))
    assert set(RAD_STAT_COLS) - set(df.columns) == {"mean_rad"}   # only one present
    assert out["median_rad"].iloc[0] == 10.0 / spencer_E0(datetime(2023, 1, 3))


# ---------------------------------------------- equivalence with the notebook
def _remove_illumination(delta_pct, ratio):
    """The notebook's ratio-domain correction (earth_sun_distance_diagnostic)."""
    return 100.0 * ((1.0 + np.asarray(delta_pct) / 100.0) / ratio - 1.0)


def test_per_scene_normalize_equals_ratio_domain_correction():
    # Two scenes six months apart (large E0 contrast), one channel.
    t1_dt, t2_dt = datetime(2023, 1, 3), datetime(2023, 7, 4)
    e1, e2 = spencer_E0(t1_dt), spencer_E0(t2_dt)

    t1 = make_spec([500.0], med_rfl=[0.10], mean_rfl=[0.10],
                   med_rad=[10.0], mean_rad=[10.0])
    t2 = make_spec([500.0], med_rfl=[0.11], mean_rfl=[0.11],
                   med_rad=[12.0], mean_rad=[12.0])

    # (a) notebook path: raw delta, then divide the ratio out afterwards.
    raw = compute_pair_delta(t1, t2, make_meta(t1_dt=t1_dt, t2_dt=t2_dt))
    corrected_after = _remove_illumination(raw["delta_rad_pct_median"], e2 / e1)

    # (b) pipeline path: normalize each scene's radiance first, then delta.
    corrected_before = compute_pair_delta(
        normalize_radiance_spectrum(t1, t1_dt),
        normalize_radiance_spectrum(t2, t2_dt),
        make_meta(t1_dt=t1_dt, t2_dt=t2_dt))["delta_rad_pct_median"]

    np.testing.assert_allclose(corrected_before, corrected_after)
    # And reflectance delta is identical either way (never touched).
    np.testing.assert_allclose(raw["delta_rfl_pct_median"],
                               compute_pair_delta(
                                   normalize_radiance_spectrum(t1, t1_dt),
                                   normalize_radiance_spectrum(t2, t2_dt),
                                   make_meta())["delta_rfl_pct_median"])
