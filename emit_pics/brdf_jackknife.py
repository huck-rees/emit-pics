"""Temporal delete-one-block jackknife of the pooled quasi-BRDF drift.

Leave-one-PICS-out (see :mod:`emit_pics.brdf_loo`) shows no single *site* drives
the shared drift. This module is its temporal counterpart: it asks whether any
single *epoch* drives it. The pooled acquisition timeline is partitioned into a
handful of contiguous blocks; each block is dropped in turn and the pooled
shared-``gamma`` per-channel spectrum is refit on the rest — a block (grouped)
jackknife over time, using the same stratified within-site bootstrap as the
primary fit (:func:`emit_pics.brdf_drift.channel_gamma`).

Because time is the regressor, dropping an end block shortens the baseline and
shifts the temporal centroid, so ``gamma`` moves at the ends *by construction*
(leverage), not because anything anomalous happened there. The test is therefore
**not** "did ``gamma`` change" but "does each block-deleted ``gamma`` stay inside
the full-data fit's 95% bootstrap CI?" (:func:`flag_ci_escapes`). A block-deleted
estimate escaping that band flags a genuinely influential epoch — the kind of
single-period bias a bootstrap CI, which measures variance not bias, does not see.

Pure NumPy/pandas orchestration over the existing BRDF fitter; the runner owns I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emit_pics.brdf_drift import channel_gamma

#: Tidy per-channel output columns (superset of what viz.figures.drift_spectrum needs).
JACKKNIFE_SPECTRUM_COLS = [
    "block_out", "block_id", "product", "band_index", "wavelength",
    "gamma_pct_yr", "ci_lo_pct_yr", "ci_hi_pct_yr", "p_value", "n",
    "sig_raw", "unstable",
]


def make_time_blocks(obs, *, scheme="equal_count", n_blocks=5):
    """Assign each scene to a contiguous temporal block on the pooled timeline.

    Parameters
    ----------
    obs : pandas.DataFrame
        Pooled per-scene design table with a ``datetime`` column (order preserved).
    scheme : str
        ``"equal_count"`` splits the time-sorted scenes into *n_blocks* contiguous
        groups of near-equal size (balanced influence per block); ``"calendar_year"``
        bins by acquisition year (interpretable, but end-years are partial).
    n_blocks : int
        Number of blocks for ``equal_count`` (ignored for ``calendar_year``).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        *obs* with an added integer ``block`` column (aligned to the input row
        order, so it lines up with matrices built on the same *obs*), and a
        ``blocks`` metadata frame (``block_id``, ``short``, ``label``, ``start``,
        ``end``, ``n_scenes``), ordered by time.
    """
    o = obs.copy()
    dt = pd.to_datetime(o["datetime"])
    order = np.argsort(dt.to_numpy(), kind="stable")   # row indices in time order

    if scheme == "equal_count":
        ids = np.empty(len(o), dtype=int)
        for k, chunk in enumerate(np.array_split(order, n_blocks)):
            ids[chunk] = k
        o["block"] = ids
    elif scheme == "calendar_year":
        years = dt.dt.year
        lut = {y: i for i, y in enumerate(sorted(years.unique()))}
        o["block"] = years.map(lut).to_numpy()
    else:
        raise ValueError(f"unknown block scheme {scheme!r} "
                         "(use 'equal_count' or 'calendar_year').")

    rows = []
    for bid, sub in o.groupby("block"):
        sdt = pd.to_datetime(sub["datetime"])
        start, end = sdt.min(), sdt.max()
        if scheme == "calendar_year":
            short = str(start.year)
            label = short
        else:
            short = f"B{int(bid) + 1}"
            label = f"{short} ({start:%Y-%m} to {end:%Y-%m})"
        rows.append({"block_id": int(bid), "short": short, "label": label,
                     "start": start.date().isoformat(),
                     "end": end.date().isoformat(), "n_scenes": int(len(sub))})
    blocks = pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)
    return o, blocks


def leave_one_block_out_drift_spectra(obs, mats, wlgrid, fit_bands, blocks, *,
                                      n_boot, gamma_cap, alpha, seed=2024):
    """Per-channel pooled shared-``gamma`` drift, refit dropping each time block.

    Parameters mirror :func:`emit_pics.brdf_loo.leave_one_out_drift_spectra`, but
    the held-out unit is a temporal block (from :func:`make_time_blocks`) rather
    than a PICS. *obs* must carry the ``block`` column; *mats* is
    ``{product: scenes×nband ndarray}`` aligned row-for-row with *obs*.

    Returns a tidy per-channel table, columns :data:`JACKKNIFE_SPECTRUM_COLS`,
    stacked over (block-left-out × product × fit channel).
    """
    products = list(mats)
    site_all = obs["site"].to_numpy()
    g_all = obs["g"].to_numpy(float)
    t_all = obs["t"].to_numpy(float)
    block_all = obs["block"].to_numpy()

    rows = []
    for _, blk in blocks.iterrows():
        bid = int(blk["block_id"])
        keep = block_all != bid
        g, t, site = g_all[keep], t_all[keep], site_all[keep]
        rng = np.random.default_rng(seed + bid)          # one stream per block
        for prod in products:
            M = mats[prod][keep]
            for b in fit_bands:
                gam, (lo, hi), pval, n = channel_gamma(
                    M[:, int(b)], g, t, site, n_boot, rng)
                rows.append({
                    "block_out": blk["label"], "block_id": bid,
                    "product": prod, "band_index": int(b),
                    "wavelength": float(wlgrid[int(b)]),
                    "gamma_pct_yr": 100.0 * gam,
                    "ci_lo_pct_yr": 100.0 * lo, "ci_hi_pct_yr": 100.0 * hi,
                    "p_value": pval, "n": int(n)})

    df = pd.DataFrame(rows)
    # Significance = 95% bootstrap CI excludes zero (matches the plotted whiskers
    # and the primary drift-spectrum convention).
    df["sig_raw"] = (df["ci_lo_pct_yr"] > 0) | (df["ci_hi_pct_yr"] < 0)
    ci_hw = (df["ci_hi_pct_yr"] - df["ci_lo_pct_yr"]).abs() / 2
    df["unstable"] = (~np.isfinite(df["gamma_pct_yr"])
                      | (df["gamma_pct_yr"].abs() > gamma_cap)
                      | (ci_hw > gamma_cap))
    return df[JACKKNIFE_SPECTRUM_COLS]


def flag_ci_escapes(jack_df, full_df):
    """Add ``outside_full_ci``: does each block-deleted gamma leave the full-fit CI?

    Joins the leave-one-block-out table to the full-data reference spectrum
    (*full_df*, e.g. ``brdf_drift_spectrum.csv``) on (product, band_index) and
    flags channels whose block-deleted point estimate falls outside the full-fit
    95% bootstrap CI — the epoch-robustness test. Returns a copy of *jack_df* with
    the boolean column added.
    """
    ref = (full_df[["product", "band_index", "ci_lo_pct_yr", "ci_hi_pct_yr"]]
           .rename(columns={"ci_lo_pct_yr": "_full_lo", "ci_hi_pct_yr": "_full_hi"}))
    m = jack_df.merge(ref, on=["product", "band_index"], how="left")
    m["outside_full_ci"] = ((m["gamma_pct_yr"] < m["_full_lo"])
                            | (m["gamma_pct_yr"] > m["_full_hi"]))
    return m.drop(columns=["_full_lo", "_full_hi"])


def block_summary(jack_df):
    """Per-(block, product) rollup of the jackknife spectrum table.

    One row per block/product with the fit-channel count, significant-channel
    count, median drift and median CI width over stable channels, and — when
    :func:`flag_ci_escapes` has been applied — how many / what fraction of stable
    channels escape the full-fit CI (the headline robustness number).
    """
    rows = []
    for (blk, prod), g in jack_df.groupby(["block_out", "product"], sort=False):
        stable = g[~g["unstable"]]
        row = {
            "block_out": blk, "block_id": int(g["block_id"].iloc[0]), "product": prod,
            "n_channels": int(len(g)), "n_stable": int(len(stable)),
            "n_sig_raw": int(stable["sig_raw"].sum()),
            "median_gamma_pct_yr": float(stable["gamma_pct_yr"].median()),
            "median_ci_width_pct_yr": float(
                (stable["ci_hi_pct_yr"] - stable["ci_lo_pct_yr"]).median()),
        }
        if "outside_full_ci" in g.columns:
            row["n_outside_full_ci"] = int(stable["outside_full_ci"].sum())
            row["pct_outside_full_ci"] = round(
                100.0 * float(stable["outside_full_ci"].mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["product", "block_id"]).reset_index(drop=True)
