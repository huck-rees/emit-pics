"""Unit tests for the quasi-BRDF pooled-drift model & fits.

The fits have no closed form, so correctness is checked by **recovery**: build
scenes from a known ``(alpha_s, beta_s, gamma)``, and assert the fitters recover
them. Zero-noise data must be recovered exactly; small-noise data must land close
with the bootstrap CI bracketing the truth. ``consistency_check`` is exercised on
hand-built overlapping / disjoint / degenerate CI tables.
"""

import types

import numpy as np
import pandas as pd
import pytest

from emit_pics.brdf_drift import (
    apply_min_scenes_gate, boot_site_gamma, bootstrap_joint_gamma, channel_gamma,
    consistency_check, fit_joint, fit_site, leverage_summary, resolve_roster,
    water_vapor_mask,
)
from emit_pics.config import load_sites


# ------------------------------------------------------------------ synthetic data
def _design(alpha, beta, gamma, *, noise=0.0, seed=0):
    """Scenes for sites from known params. g,t vary on a grid (identifiable)."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, (a, b) in enumerate(zip(alpha, beta)):
        # a small non-collinear g x t grid per site (8 scenes)
        gg = np.linspace(0.75, 0.98, 4)
        tt = np.linspace(0.0, 3.0, 2)
        for g in gg:
            for t in tt:
                rho = (a + b * g) * (1 + gamma * t)
                if noise:
                    rho += rng.normal(0.0, noise)
                rows.append({"site": f"S{si}", "g": g, "t": t,
                             "rho_rad": rho, "rho_rfl": rho})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ fit_site
def test_fit_site_recovers_exactly_zero_noise():
    df = _design([100.0], [12.0], 0.02)
    r = fit_site(df["g"], df["t"], df["rho_rad"])
    assert r["alpha"] == pytest.approx(100.0, abs=1e-6)
    assert r["beta"] == pytest.approx(12.0, abs=1e-6)
    assert r["gamma"] == pytest.approx(0.02, abs=1e-9)
    assert r["n"] == 8


def test_boot_site_gamma_zero_noise_is_degenerate_ci():
    df = _design([100.0], [12.0], -0.015)
    g0, (lo, hi) = boot_site_gamma(df["g"], df["t"], df["rho_rad"], n_boot=50)
    # noiseless -> every resample yields the same gamma -> CI collapses to a point
    assert g0 == pytest.approx(-0.015, abs=1e-9)
    assert lo == pytest.approx(-0.015, abs=1e-6)
    assert hi == pytest.approx(-0.015, abs=1e-6)


# ----------------------------------------------------------------------- fit_joint
def test_fit_joint_recovers_shared_gamma_and_per_site_terms():
    alpha, beta, gamma = [100.0, 60.0, 130.0], [12.0, -8.0, 20.0], 0.018
    df = _design(alpha, beta, gamma)
    r = fit_joint(df, "rad")
    assert r["sites"] == ["S0", "S1", "S2"]
    assert r["success"]
    assert r["gamma"] == pytest.approx(gamma, abs=1e-7)
    assert np.allclose(r["alpha"], alpha, atol=1e-5)
    assert np.allclose(r["beta"], beta, atol=1e-5)
    assert r["n"] == 24


def test_fit_joint_noisy_is_close_and_bootstrap_ci_brackets_truth():
    alpha, beta, gamma = [100.0, 60.0], [12.0, -8.0], 0.02
    df = _design(alpha, beta, gamma, noise=0.5, seed=7)
    r = fit_joint(df, "rad")
    assert r["gamma"] == pytest.approx(gamma, abs=0.01)     # close despite noise
    b = bootstrap_joint_gamma(df, "rad", n_boot=200, seed=42)
    lo, hi = b["ci"]
    assert lo <= gamma <= hi                                 # CI brackets the truth
    assert b["se"] > 0


def test_bootstrap_joint_gamma_zero_noise_collapses():
    df = _design([100.0, 60.0], [12.0, -8.0], 0.02)
    b = bootstrap_joint_gamma(df, "rad", n_boot=100, seed=42)
    assert b["gamma"] == pytest.approx(0.02, abs=1e-7)
    assert b["se"] == pytest.approx(0.0, abs=1e-7)


# -------------------------------------------------------------------- channel_gamma
def test_channel_gamma_recovers_and_flags_significance():
    alpha, beta, gamma = [100.0, 60.0], [12.0, -8.0], 0.03
    df = _design(alpha, beta, gamma)
    rng = np.random.default_rng(2024)
    gam, (lo, hi), pval, n = channel_gamma(
        df["rho_rad"].to_numpy(), df["g"].to_numpy(), df["t"].to_numpy(),
        df["site"].to_numpy(), n_boot=100, rng=rng)
    assert gam == pytest.approx(gamma, abs=1e-6)
    assert n == 16
    # zero-noise, non-zero gamma -> all draws same sign -> smallest possible p
    assert pval == pytest.approx(2 / 101, abs=1e-9)


def test_channel_gamma_degenerate_returns_nan():
    # 2 sites but only 4 scenes total (n == 2S) -> under-determined -> NaN
    df = pd.DataFrame({
        "site": ["S0", "S0", "S1", "S1"],
        "g": [0.8, 0.9, 0.8, 0.9], "t": [0.0, 1.0, 0.0, 1.0],
        "rho_rad": [100.0, 101.0, 60.0, 61.0],
    })
    rng = np.random.default_rng(0)
    gam, ci, pval, n = channel_gamma(
        df["rho_rad"].to_numpy(), df["g"].to_numpy(), df["t"].to_numpy(),
        df["site"].to_numpy(), n_boot=10, rng=rng)
    assert np.isnan(gam) and np.isnan(pval)


# ----------------------------------------------------------------- consistency_check
def _persite(rows):
    return pd.DataFrame(rows, columns=["product", "site", "gamma_pct_yr",
                                       "ci_lo", "ci_hi"])


def test_consistency_pass_on_wide_overlap():
    df = _persite([
        ("rad", "A", -0.5, -1.5, 0.5),
        ("rad", "B", -0.4, -1.2, 0.6),
        ("rad", "C", -0.6, -1.4, 0.4),
    ])
    out = consistency_check(df)
    assert out["overall"] == "pass"
    assert out["per_product"]["rad"]["overlap_lo"] == pytest.approx(-1.2)
    assert out["per_product"]["rad"]["overlap_hi"] == pytest.approx(0.4)


def test_consistency_fail_on_disjoint_cis():
    df = _persite([
        ("rad", "A", -2.0, -3.0, -1.0),
        ("rad", "B", 1.5, 0.8, 2.2),        # entirely above A's CI -> no common value
    ])
    out = consistency_check(df)
    assert out["overall"] == "fail"
    assert out["per_product"]["rad"]["overlap_width"] < 0


def test_consistency_marginal_on_thin_overlap():
    # CIs overlap only in a thin sliver relative to their width -> marginal
    df = _persite([
        ("rfl", "A", 0.0, -0.90, 1.05),
        ("rfl", "B", -2.1, -4.32, -0.53),   # overlap [-0.90, -0.53], width 0.37
    ])
    out = consistency_check(df)
    assert out["per_product"]["rfl"]["verdict"] == "marginal"
    assert out["overall"] == "marginal"


def test_consistency_overall_is_worst_product():
    df = _persite([
        ("rad", "A", -0.5, -1.5, 0.5), ("rad", "B", -0.4, -1.2, 0.6),   # pass
        ("rfl", "A", -2.0, -3.0, -1.0), ("rfl", "B", 1.5, 0.8, 2.2),    # fail
    ])
    out = consistency_check(df)
    assert out["per_product"]["rad"]["verdict"] == "pass"
    assert out["per_product"]["rfl"]["verdict"] == "fail"
    assert out["overall"] == "fail"


def test_consistency_single_site_and_nan_handling():
    df = _persite([("rad", "A", -0.5, -1.5, 0.5)])       # one site -> trivially ok
    assert consistency_check(df)["overall"] == "pass"
    nan_df = _persite([("rad", "A", np.nan, np.nan, np.nan)])
    assert consistency_check(nan_df)["overall"] == "n/a"


# ---------------------------------------------------------------- resolve_roster
def _cfg(site_patches, exclude=()):
    return types.SimpleNamespace(brdf_site_patches=dict(site_patches),
                                 brdf_exclude_sites=list(exclude))


def test_resolve_roster_keeps_valid_drops_excluded_and_unknown():
    valid = sorted({info["site_name"].replace(" ", "_") for info in load_sites().values()})
    a, b = valid[0], valid[1]                         # two real sites.yaml tokens
    roster, excluded, unknown = resolve_roster(
        _cfg({a: "10_7_1", b: "1_2_3", "Not_A_Site": "9_9_9"}, exclude=[b]))
    assert roster == {a: "10_7_1"}                    # a kept
    assert excluded == [b]                            # b excluded by config
    assert unknown == ["Not_A_Site"]                  # bogus site flagged


# ------------------------------------------------- leverage_summary / min-scenes gate
def _obs():
    # two sites: A has 5 scenes, B has 2; spread over time and SZA.
    rows = []
    for i in range(5):
        rows.append({"site": "A", "datetime": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * i),
                     "sza_mean": 20.0 + i, "t": 0.25 * i, "g": np.cos(np.radians(20.0 + i))})
    for i in range(2):
        rows.append({"site": "B", "datetime": pd.Timestamp("2023-02-01") + pd.Timedelta(days=90 * i),
                     "sza_mean": 30.0 + i, "t": 0.25 * i, "g": np.cos(np.radians(30.0 + i))})
    return pd.DataFrame(rows)


def test_leverage_summary_columns_and_counts():
    s = leverage_summary(_obs())
    assert list(s["site"]) == ["A", "B"]
    assert list(s["n"]) == [5, 2]
    assert set(["cosSZA_span", "t_span_yr", "corr_g_t"]).issubset(s.columns)
    # B has <3 scenes -> corr is NaN (undefined leverage)
    assert np.isnan(s.loc[s.site == "B", "corr_g_t"].iloc[0])


def test_min_scenes_gate_drops_small_sites():
    obs = _obs()
    summary = leverage_summary(obs)
    obs2, summary2, dropped = apply_min_scenes_gate(obs, summary, min_scenes=4)
    assert dropped == ["B"]
    assert set(obs2["site"]) == {"A"}
    assert set(summary2["site"]) == {"A"}


# ---------------------------------------------------------------- water_vapor_mask
def test_water_vapor_mask_flags_nan_reflectance():
    ref = pd.DataFrame({"central_wl": [400.0, 500.0, 1400.0, 1900.0, 2000.0],
                        "median_rfl": [0.3, 0.35, np.nan, np.nan, 0.4]})
    wlgrid, excl, fit_bands = water_vapor_mask(ref)
    assert list(wlgrid) == [400.0, 500.0, 1400.0, 1900.0, 2000.0]
    assert list(excl) == [False, False, True, True, False]
    assert list(fit_bands) == [0, 1, 4]
