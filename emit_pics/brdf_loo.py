"""Leave-one-out (jackknife) pooled quasi-BRDF drift spectra.

The standalone per-site drift fits (one PICS at a time) have very few scenes, so
each site's ``gamma`` carries wide bootstrap uncertainty. This module inverts that
picture: for every PICS it **drops that site** and fits the pooled shared-``gamma``
drift on the remaining **four** sites, at every EMIT channel, using the same
stratified within-site bootstrap as the primary drift spectrum
(:func:`emit_pics.brdf_drift.channel_gamma`). Five leave-one-out combinations give
five ``gamma(lambda)`` spectra, each far better constrained than a single-site fit;
their mutual agreement is a direct robustness check that no single site drives the
pooled drift.

Pure NumPy/pandas orchestration over the existing BRDF fitter — the runner owns all
I/O and figure rendering.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from emit_pics.brdf_drift import channel_gamma

#: Tidy per-channel output columns (a superset of what
#: :func:`emit_pics.viz.figures.drift_spectrum` needs, so a single-combo slice
#: drops straight into that builder).
LOO_SPECTRUM_COLS = [
    "combo", "left_out", "product", "band_index", "wavelength",
    "gamma_pct_yr", "ci_lo_pct_yr", "ci_hi_pct_yr", "p_value", "n",
    "sig_raw", "unstable",
]


def abbrev_site(site: str) -> str:
    """Compact PICS token: first letter of the name + its trailing number.

    ``Libya_1`` → ``L1``, ``Algeria_3`` → ``A3``, ``Mauritania_1`` → ``M1``.
    """
    parts = str(site).split("_")
    num = parts[-1] if parts[-1].isdigit() else ""
    return f"{parts[0][0].upper()}{num}"


def combo_label(sites) -> str:
    """Comma-joined abbreviations of a combo's INCLUDED sites (sorted)."""
    return ", ".join(abbrev_site(s) for s in sorted(sites))


def leave_one_out_combos(sites):
    """One ``{left_out, included, label}`` per site dropped in turn (sorted order)."""
    sites = sorted(sites)
    combos = []
    for out in sites:
        inc = [s for s in sites if s != out]
        combos.append({"left_out": out, "included": inc, "label": combo_label(inc)})
    return combos


def leave_one_out_drift_spectra(obs, mats, wlgrid, fit_bands, *, n_boot,
                                gamma_cap, alpha, seed=2024):
    """Per-channel pooled shared-``gamma`` drift for every leave-one-out combo.

    Parameters
    ----------
    obs : pandas.DataFrame
        Full pooled design table (all sites) with ``site``, ``g`` (= cos SZA), and
        ``t`` (years). Exactly :func:`emit_pics.brdf_drift.assemble_design`'s output.
    mats : dict
        ``{product: scenes×nband ndarray}`` aligned row-for-row with *obs*
        (:func:`emit_pics.brdf_drift.channel_matrix`).
    wlgrid : numpy.ndarray
        Band-centre wavelengths (nm), length ``nband``.
    fit_bands : numpy.ndarray
        Indices of the channels to fit (water-vapor bands already excluded).
    n_boot : int
        Bootstrap resamples per channel.
    gamma_cap : float
        %/yr; a channel whose ``|gamma|`` or CI half-width exceeds this is flagged
        ``unstable`` (degenerate near-zero baseline) and kept off the plot scale.
    alpha : float
        Per-channel raw significance level (two-sided bootstrap p).
    seed : int
        Base RNG seed; each combo advances its own reproducible stream.

    Returns
    -------
    pandas.DataFrame
        Tidy per-channel table, columns :data:`LOO_SPECTRUM_COLS`, stacked over
        (combo × product × fit channel).
    """
    products = list(mats)
    site_all = obs["site"].to_numpy()
    g_all = obs["g"].to_numpy(float)
    t_all = obs["t"].to_numpy(float)
    combos = leave_one_out_combos(pd.unique(site_all))

    rows = []
    for ci, combo in enumerate(combos):
        keep = site_all != combo["left_out"]
        g, t, site = g_all[keep], t_all[keep], site_all[keep]
        rng = np.random.default_rng(seed + ci)          # one stream per combo
        for prod in products:
            M = mats[prod][keep]
            for b in fit_bands:
                gam, (lo, hi), pval, n = channel_gamma(
                    M[:, int(b)], g, t, site, n_boot, rng)
                rows.append({
                    "combo": combo["label"], "left_out": combo["left_out"],
                    "product": prod, "band_index": int(b),
                    "wavelength": float(wlgrid[int(b)]),
                    "gamma_pct_yr": 100.0 * gam,
                    "ci_lo_pct_yr": 100.0 * lo, "ci_hi_pct_yr": 100.0 * hi,
                    "p_value": pval, "n": int(n)})

    df = pd.DataFrame(rows)
    # Significance = the 95% bootstrap CI excludes zero (matches the plotted
    # whiskers and the primary drift-spectrum convention; see run_brdf_drift_spectrum).
    df["sig_raw"] = (df["ci_lo_pct_yr"] > 0) | (df["ci_hi_pct_yr"] < 0)
    ci_hw = (df["ci_hi_pct_yr"] - df["ci_lo_pct_yr"]).abs() / 2
    df["unstable"] = (~np.isfinite(df["gamma_pct_yr"])
                      | (df["gamma_pct_yr"].abs() > gamma_cap)
                      | (ci_hw > gamma_cap))
    return df[LOO_SPECTRUM_COLS]


def leave_k_out_combos(sites, k):
    """One ``{left_out, left_out_label, included, label}`` per set of *k* sites dropped.

    Generalises :func:`leave_one_out_combos` to dropping *k* sites at a time and
    pooling the rest. With ``n`` sites there are ``C(n, k)`` combinations. ``label``
    is the comma-joined abbreviations of the INCLUDED sites, ``left_out_label`` the
    dropped ones.
    """
    sites = sorted(sites)
    combos = []
    for drop in combinations(sites, k):
        inc = [s for s in sites if s not in drop]
        combos.append({"left_out": list(drop), "left_out_label": combo_label(drop),
                       "included": inc, "label": combo_label(inc)})
    return combos


def leave_k_out_drift_spectra(obs, mats, wlgrid, fit_bands, *, k, n_boot,
                              gamma_cap, alpha, seed=2024):
    """Per-channel pooled shared-``gamma`` drift for every leave-*k*-out combo.

    The leave-one-out generalisation (:func:`leave_one_out_drift_spectra`): for each
    of the ``C(n, k)`` ways to drop *k* sites, fit the pooled shared-``gamma`` drift
    on the remaining ``n - k`` sites at every fit channel. The ``left_out`` column
    holds the abbreviated dropped-site label; ``combo`` holds the kept-site label.
    Same stratified within-site bootstrap and column schema
    (:data:`LOO_SPECTRUM_COLS`) as the leave-one-out version, so a single-combo slice
    drops straight into the spectrum/overlay builders.
    """
    products = list(mats)
    site_all = obs["site"].to_numpy()
    g_all = obs["g"].to_numpy(float)
    t_all = obs["t"].to_numpy(float)
    combos = leave_k_out_combos(pd.unique(site_all), k)

    rows = []
    for ci, combo in enumerate(combos):
        keep = ~np.isin(site_all, combo["left_out"])
        g, t, site = g_all[keep], t_all[keep], site_all[keep]
        rng = np.random.default_rng(seed + 1000 * k + ci)   # stream per (k, combo)
        for prod in products:
            M = mats[prod][keep]
            for b in fit_bands:
                gam, (lo, hi), pval, n = channel_gamma(
                    M[:, int(b)], g, t, site, n_boot, rng)
                rows.append({
                    "combo": combo["label"], "left_out": combo["left_out_label"],
                    "product": prod, "band_index": int(b),
                    "wavelength": float(wlgrid[int(b)]),
                    "gamma_pct_yr": 100.0 * gam,
                    "ci_lo_pct_yr": 100.0 * lo, "ci_hi_pct_yr": 100.0 * hi,
                    "p_value": pval, "n": int(n)})

    df = pd.DataFrame(rows)
    df["sig_raw"] = (df["ci_lo_pct_yr"] > 0) | (df["ci_hi_pct_yr"] < 0)
    ci_hw = (df["ci_hi_pct_yr"] - df["ci_lo_pct_yr"]).abs() / 2
    df["unstable"] = (~np.isfinite(df["gamma_pct_yr"])
                      | (df["gamma_pct_yr"].abs() > gamma_cap)
                      | (ci_hw > gamma_cap))
    return df[LOO_SPECTRUM_COLS]


def combo_summary(df):
    """Compact per-(combo, product) rollup of the tidy spectrum table.

    Returns one row per combo/product with the fit-channel count, the number of
    channels significant at the raw level, and the median drift rate and median CI
    width over the stable (non-degenerate) channels.
    """
    rows = []
    for (combo, prod), g in df.groupby(["combo", "product"], sort=False):
        stable = g[~g["unstable"]]
        rows.append({
            "combo": combo, "left_out": g["left_out"].iloc[0], "product": prod,
            "n_channels": int(len(g)), "n_stable": int(len(stable)),
            "n_sig_raw": int(stable["sig_raw"].sum()),
            "median_gamma_pct_yr": float(stable["gamma_pct_yr"].median()),
            "median_ci_width_pct_yr": float(
                (stable["ci_hi_pct_yr"] - stable["ci_lo_pct_yr"]).median()),
        })
    return pd.DataFrame(rows)
