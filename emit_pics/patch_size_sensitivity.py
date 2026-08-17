"""10 km vs 20 km patch-size sensitivity for the pooled BRDF drift spectrum.

A **read-only** diagnostic: it never recomputes a drift fit. It reads the two
already-computed BRDF configurations from disk — the main (flat ``data/brdf/``)
run and the labelled sweep (e.g. ``data/brdf/20km/``) — and quantifies how much
the representative-patch size moves the answer, so we can judge which
configuration is better constrained.

The two configurations must have been produced by the ordinary BRDF workflow
(``run_brdf_assemble`` + ``run_brdf_drift_spectrum``) at each patch size; here we
only compare their outputs:

* per-channel shared-``gamma`` median + 95 % CI for both sizes and their
  difference (``gamma_20 - gamma_10``),
* summary diagnostics per product (median CI width = precision; RMS / max
  difference; count of channels significant at ``alpha``),
* a per-site leverage comparison (scene counts + ``corr(cos SZA, t)``), which
  exposes geometry confounding behind any difference.

Nothing here writes files or plots; the runner owns I/O. Functions are pure so
they are unit-testable with tiny synthetic frames.
"""

import numpy as np
import pandas as pd

#: Per-channel value columns carried from each drift-spectrum CSV into the merge.
_SPECTRUM_VALS = ["wavelength", "gamma_pct_yr", "ci_lo_pct_yr", "ci_hi_pct_yr",
                  "p_value", "n", "sig_raw", "unstable"]

#: Keys that align a channel across the two configurations (same instrument grid).
_KEYS = ["product", "band_index"]


def compare_spectra(df10, df20, *, tag10="10km", tag20="20km"):
    """Merge two BRDF drift spectra channel-by-channel into a tidy comparison.

    Both frames are the native ``brdf_drift_spectrum.csv`` schema (``product``,
    ``band_index``, ``wavelength``, ``gamma_pct_yr``, ``ci_{lo,hi}_pct_yr``,
    ``p_value``, ``n``, ``sig_raw``, ``unstable``). Channels are matched on
    ``(product, band_index)`` — the EMIT band grid is instrument-fixed, so the
    two sizes share it. The merge is an outer join, so a channel fit in only one
    configuration is kept (its other side is NaN).

    Returns
    -------
    pandas.DataFrame
        One row per (product, channel), sorted by product then wavelength, with:
        ``wavelength``; per side ``gamma_pct_yr_<tag>``, ``ci_lo/hi_pct_yr_<tag>``,
        ``ci_width_<tag>``, ``p_value_<tag>``, ``n_<tag>``, ``sig_raw_<tag>``,
        ``unstable_<tag>``; ``gamma_diff_pct_yr`` (``tag20 - tag10``); and
        ``stable_both`` (finite, non-``unstable`` on both sides).
    """
    def _prep(df, tag):
        d = df[_KEYS + _SPECTRUM_VALS].copy()
        return d.rename(columns={c: f"{c}_{tag}" for c in _SPECTRUM_VALS})

    m = _prep(df10, tag10).merge(_prep(df20, tag20), on=_KEYS, how="outer")
    # Coalesce the (identical) wavelength grid into one column, drop the pair.
    m["wavelength"] = m[f"wavelength_{tag10}"].combine_first(m[f"wavelength_{tag20}"])
    m = m.drop(columns=[f"wavelength_{tag10}", f"wavelength_{tag20}"])

    for tag in (tag10, tag20):
        m[f"ci_width_{tag}"] = m[f"ci_hi_pct_yr_{tag}"] - m[f"ci_lo_pct_yr_{tag}"]
        # A missing side reads as unstable so it is excluded from "stable_both".
        m[f"unstable_{tag}"] = m[f"unstable_{tag}"].fillna(True).astype(bool)

    m["gamma_diff_pct_yr"] = m[f"gamma_pct_yr_{tag20}"] - m[f"gamma_pct_yr_{tag10}"]
    m["stable_both"] = ~(m[f"unstable_{tag10}"] | m[f"unstable_{tag20}"])
    return m.sort_values(["product", "wavelength"]).reset_index(drop=True)


def summarize(cmp, *, tag10="10km", tag20="20km", alpha=0.05):
    """Per-product headline diagnostics from :func:`compare_spectra` output.

    Precision, agreement, and detection power — computed over the channels that
    are stable in **both** configurations (an apples-to-apples set):

    * ``median_ci_width_<tag>`` — median 95 % CI width (%/yr); narrower = the
      better-constrained configuration.
    * ``rms_gamma_diff`` / ``max_abs_gamma_diff`` — how far the two sizes' drift
      estimates sit apart across the spectrum.
    * ``n_sig_<tag>`` — channels significant at ``alpha`` (uncorrected bootstrap
      p), over the stable-both set; higher = more detection power.

    Returns a tidy DataFrame, one row per product.
    """
    rows = []
    for prod, g in cmp.groupby("product"):
        s = g[g["stable_both"]]
        diff = s["gamma_diff_pct_yr"].to_numpy(float)
        rows.append({
            "product": prod,
            "n_channels_stable_both": int(len(s)),
            f"median_ci_width_{tag10}": float(np.nanmedian(s[f"ci_width_{tag10}"])),
            f"median_ci_width_{tag20}": float(np.nanmedian(s[f"ci_width_{tag20}"])),
            "rms_gamma_diff_pct_yr": float(np.sqrt(np.nanmean(diff ** 2))) if len(diff) else np.nan,
            "max_abs_gamma_diff_pct_yr": float(np.nanmax(np.abs(diff))) if len(diff) else np.nan,
            "median_abs_gamma_diff_pct_yr": float(np.nanmedian(np.abs(diff))) if len(diff) else np.nan,
            f"n_sig_{tag10}": int((s[f"p_value_{tag10}"] <= alpha).sum()),
            f"n_sig_{tag20}": int((s[f"p_value_{tag20}"] <= alpha).sum()),
        })
    return pd.DataFrame(rows)


#: Per-site leverage columns worth comparing across the two sizes.
_LEVERAGE_VALS = ["n", "first", "last", "sza_lo", "sza_hi", "cosSZA_span",
                  "t_span_yr", "corr_g_t", "kept"]


def compare_leverage(lev10, lev20, *, tag10="10km", tag20="20km"):
    """Join the two configs' per-site leverage tables side by side.

    Each input is a ``brdf_leverage.csv`` (from ``run_brdf_assemble``). Matched
    on ``site`` (outer join, so a site present at only one size is kept). Columns
    that exist are carried with a ``_<tag>`` suffix; ``corr_g_t`` (``corr(cos SZA,
    t)``) and ``n`` are the ones that flag geometry confounding / thin sampling.
    """
    def _prep(df, tag):
        cols = [c for c in _LEVERAGE_VALS if c in df.columns]
        return df[["site"] + cols].rename(columns={c: f"{c}_{tag}" for c in cols})

    return (_prep(lev10, tag10).merge(_prep(lev20, tag20), on="site", how="outer")
            .sort_values("site").reset_index(drop=True))


def spectrum_segments(wavelength, *, gap_factor=3.0):
    """Index slices of contiguous channels, split where the wavelength step jumps.

    The BRDF drift spectrum simply omits the excluded (water-vapor) channels, so
    they show up as large gaps in an otherwise ~7 nm grid rather than as flagged
    rows. Splitting the spectrum at any step wider than ``gap_factor`` times the
    median channel spacing lets a smoothed line / CI band **break** across the
    missing region instead of bridging it — the data's own missingness is the
    exclusion window, so nothing is hand-set. Returns a list of ``slice`` objects
    covering *wavelength* in order.
    """
    wl = np.asarray(wavelength, dtype=float)
    if wl.size < 2:
        return [slice(0, wl.size)]
    dwl = np.diff(wl)
    thr = gap_factor * np.median(dwl)
    breaks = (np.where(dwl > thr)[0] + 1).tolist()
    bounds = [0, *breaks, wl.size]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def smooth_series(wavelength, values, *, window=11):
    """Centered rolling-median smooth of *values* ordered by *wavelength*.

    NaNs (e.g. unstable channels masked out) are tolerated: the rolling median
    uses ``min_periods=1`` so gaps don't blank the whole series. Returns a NumPy
    array aligned to the input order (assumed already wavelength-sorted).
    """
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()
