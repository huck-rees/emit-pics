"""Unit tests for the one-use-per-scene matching-bootstrap drift estimator."""

import numpy as np
import pandas as pd
import pytest

from emit_pics.matched_pairs_drift_analysis import DAYS_PER_YEAR
from emit_pics.matched_pairs_bootstrap import (
    build_edges,
    random_matching,
    bootstrap_channel,
    compute_bootstrap_drift_spectrum,
)


def _band_df(*, slope_per_day=-0.01, n_scenes=12, n_patches=5, band=0,
             pics="Libya_4", noise=0.1, seed=0):
    """Synthetic per-patch delta rows for ONE band with a known drift slope.

    Every unordered scene pair is a matched pair; each pair is replicated across
    *n_patches* patches (the spatial replicates), with ΔR = slope·Δt + noise.
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 1000, n_scenes))       # acquisition times (days)
    rows = []
    for i in range(n_scenes):
        for j in range(i + 1, n_scenes):
            dt = float(times[j] - times[i])
            for _ in range(n_patches):
                dR = slope_per_day * dt + rng.normal(0, noise)
                rows.append({
                    "PICS_ID": pics, "pair_id": f"S{i}_S{j}", "band_index": band,
                    "wavelength": 500.0 + band, "dt_days": dt,
                    "t1_scene_id": f"{pics}_S{i}", "t2_scene_id": f"{pics}_S{j}",
                    "absorption_band_flag": False,
                    "delta_rad_pct_median": dR, "delta_rfl_pct_median": dR,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ edge list
def test_build_edges_collects_patch_replicates_per_pics():
    df = _band_df(n_scenes=4, n_patches=5)
    edges = build_edges(df, "delta_rad_pct_median")
    assert set(edges) == {"Libya_4"}
    e = edges["Libya_4"]
    assert len(e) == 6                                   # 4 scenes -> 6 pairs
    assert all(x["dR"].size == 5 for x in e)             # 5 patch replicates each


def test_build_edges_groups_within_pics():
    df = pd.concat([_band_df(pics="Libya_4", n_scenes=5, seed=1),
                    _band_df(pics="Algeria_3", n_scenes=5, seed=2)],
                   ignore_index=True)
    edges = build_edges(df, "delta_rad_pct_median")
    assert set(edges) == {"Libya_4", "Algeria_3"}
    # scene ids are namespaced by PICS, so no edge spans two sites
    for pics, elist in edges.items():
        for e in elist:
            assert e["u"].startswith(pics) and e["v"].startswith(pics)


# ------------------------------------------------------------------ matching
def test_random_matching_is_scene_disjoint():
    edges = build_edges(_band_df(n_scenes=12), "delta_rad_pct_median")["Libya_4"]
    rng = np.random.default_rng(0)
    for _ in range(20):                                  # every draw must be disjoint
        chosen = random_matching(edges, rng)
        used = [s for e in chosen for s in (e["u"], e["v"])]
        assert len(used) == len(set(used))               # no scene reused
        assert len(chosen) >= 1


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_recovers_known_slope_with_bracketing_ci():
    slope = -0.01                                        # per day
    df = _band_df(slope_per_day=slope, n_scenes=14, n_patches=6, seed=1)
    edges = build_edges(df, "delta_rad_pct_median")
    fit = bootstrap_channel(edges, 500, np.random.default_rng(42))

    expected = slope * DAYS_PER_YEAR
    assert abs(fit["slope_pct_yr"] - expected) < abs(expected) * 0.25
    assert fit["ci_low_pct_yr"] < fit["slope_pct_yr"] < fit["ci_high_pct_yr"]
    assert fit["raw_significant"] is True                # strong signal, CI excludes 0
    assert fit["n_scenes"] == 14
    # one-use-per-scene: pooled pairs per iteration cannot exceed floor(scenes/2).
    assert fit["n_pairs"] <= fit["n_scenes"] / 2 + 1e-9


def test_bootstrap_too_few_pairs_returns_nan():
    # 2 scenes -> 1 pair -> 1 pooled point/iteration, below MIN_PAIRS -> unfittable.
    edges = build_edges(_band_df(n_scenes=2, n_patches=3), "delta_rad_pct_median")
    fit = bootstrap_channel(edges, 100, np.random.default_rng(0))
    assert np.isnan(fit["slope_pct_yr"])
    assert fit["raw_significant"] is False


# ------------------------------------------------------------------ spectrum
def test_compute_spectrum_excludes_absorption_and_exposes_plot_columns():
    good = _band_df(band=0, seed=1)
    absorp = _band_df(band=1, seed=2)
    absorp["absorption_band_flag"] = True
    spec = compute_bootstrap_drift_spectrum(
        pd.concat([good, absorp], ignore_index=True), "rad", n_boot=100, seed=0)

    # columns plot_drift_spectrum reads must all be present
    assert {"wavelength", "slope_pct_yr", "ci_low_pct_yr", "ci_high_pct_yr",
            "raw_significant", "absorption_band", "p_value"} <= set(spec.columns)
    row_abs = spec[spec["band_index"] == 1].iloc[0]
    assert bool(row_abs["absorption_band"]) is True
    assert np.isnan(row_abs["slope_pct_yr"])             # absorption band excluded
    row_ok = spec[spec["band_index"] == 0].iloc[0]
    assert np.isfinite(row_ok["slope_pct_yr"])


def test_compute_spectrum_missing_column_raises():
    df = _band_df().drop(columns=["dt_days"])
    with pytest.raises(ValueError):
        compute_bootstrap_drift_spectrum(df, "rad", n_boot=10)


def test_bootstrap_is_reproducible_under_seed():
    df = _band_df(n_scenes=10, seed=3)
    a = compute_bootstrap_drift_spectrum(df, "rad", n_boot=200, seed=5)
    b = compute_bootstrap_drift_spectrum(df, "rad", n_boot=200, seed=5)
    pd.testing.assert_frame_equal(a, b)
