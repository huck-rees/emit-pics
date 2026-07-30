"""Cross-PICS radiometric-drift analysis from the per-patch delta tables.

Consumes the tidy long delta CSVs and, per EMIT channel, quantifies radiometric
drift by regressing the pairwise percent change ΔR against the pair's time
separation Δt across all PICS of one patch size. The intended input is the
**aggregated** table (``data/delta_r/aggregated/{PICS}_{size}_delta.csv``, one row
per acquisition pair per channel — spatial-replicate patches already collapsed;
see :func:`load_aggregated_delta_tables`); the raw per-patch tables
(``data/delta_r/per_patch/{PICS}_{PATCH}_delta.csv``, :func:`load_delta_tables`)
are still loadable but pseudo-replicate each acquisition pair across patches.

**Model.** Over a pseudo-invariant calibration site with geometry-matched pairs
the surface is assumed stable, so if the sensor drifts linearly at rate ``m`` then
``R(t) = R0 (1 + m t)`` and

    ΔR = (R(t2) - R(t1)) / R(t1) ≈ m · Δt.

So the **slope of ΔR-vs-Δt is the drift rate**. Each matched pair is one estimate
of accumulated change over its interval. We fit the slope robustly with
**Theil–Sen** (outlier-resistant), test significance with **Kendall's τ** (the
rank test that matches Theil–Sen), and control the false-discovery rate across the
285 channels with **Benjamini–Hochberg**. Theil–Sen's intercept is reported as a
**bias diagnostic** (expected ≈ 0 for a valid matched-pairs design).

As a **secondary sensitivity estimator** we also report an **origin-forced** drift
rate: the robust median of the per-pair ratios ``ΔR_i / Δt_i`` (each pair is one
standalone drift estimate under the ``ΔR(Δt=0)=0`` anchor), with a distribution-
free (order-statistic) 95% CI. It is the robust through-origin analog of the free
fit; its **agreement** with the Theil–Sen slope corroborates the zero-intercept
premise, while a systematic gap flags residual bias (see ``intercept_pct``). It is
diagnostic only — the free-intercept Theil–Sen slope remains the primary estimate.

Channels are keyed by ``band_index`` (0..284), not by float wavelength, so the
same physical channel groups correctly across PICS and across EMIT's per-era
wavelength drift. The **median** in-patch statistic is used (robust to residual
edge/cloud pixels). Two products throughout: radiance (``rad``) and reflectance
(``rfl``).

Pure library — no I/O side effects on import; figures are **returned**, not saved
(the runner handles paths and saving).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes

# ============================================================
# CONSTANTS
# ============================================================
DAYS_PER_YEAR = 365.25
MIN_PAIRS = 3               #: fewest points needed to fit a channel
DEFAULT_ALPHA = 0.05        #: BH false-discovery-rate target
_Z95 = 1.959963985          #: normal quantile for the 95% distribution-free median CI

#: Stable colour + marker per PICS site (underscored ids, as in the delta CSVs).
PICS_STYLE = {
    "Libya_4":      ("#1f77b4", "o"),
    "Algeria_3":    ("#ff7f0e", "s"),
    "Algeria_5":    ("#2ca02c", "^"),
    "Libya_1":      ("#d62728", "D"),
    "Mauritania_1": ("#9467bd", "v"),
    "Mauritania_2": ("#8c564b", "P"),
}
_FALLBACK_STYLE = ("#7f7f7f", "X")

#: Median % change column per product, and a human label.
_DELTA_COL = {"rad": "delta_rad_pct_median", "rfl": "delta_rfl_pct_median"}
_PRODUCT_LABEL = {"rad": "Radiance (L1B)", "rfl": "Reflectance (L2A)"}

#: Columns every delta CSV must carry for this analysis (post step-6 schema).
_REQUIRED_COLS = {
    "PICS_ID", "patch_id", "pair_id", "dt_days", "band_index", "wavelength",
    "SZA_t1", "SZA_t2", "delta_rad_pct_median", "delta_rfl_pct_median",
    "absorption_band_flag",
}

#: Fixed SZA -> marker-size mapping so point sizes are comparable across figures.
_SZA_RANGE = (10.0, 65.0)      # degrees mapped to...
_SIZE_RANGE = (25.0, 260.0)    # ...marker area (points^2)

#: Fixed replicate-count -> marker-size mapping for the aggregated tables, so a
#: more-replicated acquisition pair draws a larger dot (option A does not weight
#: the fit by it, but replication stays visible in the figures).
_NPATCH_RANGE = (1.0, 12.0)    # patch counts mapped to the same _SIZE_RANGE


# ============================================================
# STYLING / SMALL HELPERS
# ============================================================
def pics_style(pics_id):
    """(colour, marker) for a PICS id, with a grey fallback for unknown sites."""
    return PICS_STYLE.get(str(pics_id), _FALLBACK_STYLE)


def patch_size_of(patch_id) -> str:
    """First token of a patch_id (e.g. ``'10_7_1' -> '10'``)."""
    return str(patch_id).split("_")[0]


def sza_marker_size(sza):
    """Map (absolute) SZA in degrees to a marker area — larger SZA, larger point."""
    return np.interp(np.asarray(sza, dtype=float), _SZA_RANGE, _SIZE_RANGE)


def n_patches_marker_size(n):
    """Map replicate patch count to a marker area — more patches, larger point."""
    return np.interp(np.asarray(n, dtype=float), _NPATCH_RANGE, _SIZE_RANGE)


def _truthy_any(series) -> bool:
    """True if any cell of *series* reads truthy (bool True or 'True'/'1' text)."""
    return bool(series.astype(str).str.strip().str.lower().isin(["true", "1"]).any())


def absorption_band_indices(df):
    """Sorted ``band_index`` values flagged as water-vapor absorption bands.

    These are excluded from the drift analysis for **both** products: reflectance
    is fill (NaN) there, and radiance, though numerically present, is
    atmospherically contaminated and not a usable drift signal.
    """
    if "absorption_band_flag" not in df:
        return []
    return sorted(int(b) for b, g in df.groupby("band_index")
                  if _truthy_any(g["absorption_band_flag"]))


# ============================================================
# LOADING
# ============================================================
def load_delta_tables(delta_dir, patch_size, *, glob_pattern="*_delta.csv"):
    """Concatenate all delta CSVs of one patch size across PICS/patches.

    Parameters
    ----------
    delta_dir : path-like
        Directory of per-patch delta CSVs (``data/delta_r``).
    patch_size : str | int
        Patch-size token to keep (first field of ``patch_id``, e.g. ``10``).

    Returns
    -------
    pd.DataFrame
        Combined rows for that size, plus a ``sza_size`` column =
        ``mean(|SZA_t1|, |SZA_t2|)`` (the marker-size driver). Empty if no CSVs
        match.

    Raises
    ------
    ValueError
        If matched CSVs are missing any required (post step-6) column — the delta
        tables predate the SZA/band_index schema and must be regenerated.
    """
    delta_dir = Path(delta_dir)
    frames = []
    for csv in sorted(delta_dir.glob(glob_pattern)):
        df = pd.read_csv(csv)
        if "patch_id" not in df.columns:
            continue
        df = df[df["patch_id"].map(patch_size_of) == str(patch_size)]
        if df.empty:
            continue
        # Validate per file (a post-concat check would miss an old-schema file
        # whose columns are supplied as NaN by a newer sibling).
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv.name} is missing columns {sorted(missing)}. Re-run "
                "scripts/run_delta_calculations.py (step 6) to regenerate the "
                "delta tables with the SZA_t1/SZA_t2/band_index schema."
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["sza_size"] = np.nanmean(
        np.abs(out[["SZA_t1", "SZA_t2"]].to_numpy(dtype=float)), axis=1)
    return out


def load_aggregated_delta_tables(agg_dir, patch_size, *, glob_pattern="*_delta.csv"):
    """Concatenate the per-PICS *aggregated* delta CSVs of one patch size.

    Counterpart to :func:`load_delta_tables` for the option-A aggregated tables
    (``data/delta_r/aggregated/{PICS}_{size}_delta.csv``, from
    ``run_PICS_allpatches_aggregate_pairs.py``), where each unique acquisition
    pair contributes exactly one row per channel — the spatial-replicate patches
    have already been collapsed. Because ``patch_id`` is gone, these tables carry
    an explicit ``patch_size`` column (used here for the size filter) and an
    ``n_patches`` diagnostic (used as the marker size in the per-channel figures).

    Parameters
    ----------
    agg_dir : path-like
        Directory of aggregated delta CSVs (``data/delta_r/aggregated``).
    patch_size : str | int
        Patch-size token to keep (matched against the ``patch_size`` column).

    Returns
    -------
    pd.DataFrame
        Combined rows for that size, plus a ``sza_size`` column
        (``mean(|SZA_t1|, |SZA_t2|)``). Empty if no CSVs match.

    Raises
    ------
    ValueError
        If a matched CSV is missing a required column (regenerate with
        ``run_PICS_allpatches_aggregate_pairs.py``).
    """
    agg_dir = Path(agg_dir)
    # patch_id is intentionally absent from aggregated tables; everything else the
    # analysis reads must still be present.
    need = (_REQUIRED_COLS - {"patch_id"}) | {"patch_size", "n_patches"}
    frames = []
    for csv in sorted(agg_dir.glob(glob_pattern)):
        df = pd.read_csv(csv)
        if "patch_size" not in df.columns:
            continue
        df = df[df["patch_size"].astype(str) == str(patch_size)]
        if df.empty:
            continue
        missing = need - set(df.columns)
        if missing:
            raise ValueError(
                f"{csv.name} is missing columns {sorted(missing)}. Re-run "
                "scripts/run_PICS_allpatches_aggregate_pairs.py to regenerate the "
                "aggregated delta tables."
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["sza_size"] = np.nanmean(
        np.abs(out[["SZA_t1", "SZA_t2"]].to_numpy(dtype=float)), axis=1)
    return out


# ============================================================
# STATISTICS
# ============================================================
def select_one_patch_per_pics(df, roster=None):
    """Reduce a pooled per-patch delta table to ONE representative patch per PICS.

    The original cross-PICS drift analysis pools a single patch per PICS. When the
    parallel all-patches experiment has also written extra per-patch tables for
    some PICS (they share ``data/delta_r/per_patch/``), this picks the
    representative one so replicate patches are never silently pooled:

    * *roster* (``{PICS_ID: patch_id}``) wins where given;
    * any PICS with exactly one patch is taken as-is;
    * a PICS with several patches and no (valid) roster entry is left *out* and
      reported back as ambiguous for the caller to fail on.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled per-patch delta rows (needs ``PICS_ID`` and ``patch_id``).
    roster : dict or None
        ``{PICS_ID: patch_id}`` representative-patch map.

    Returns
    -------
    tuple
        ``(selected_df, ambiguous)`` — the one-patch-per-PICS subset and a list of
        ``(PICS_ID, [patch_ids])`` that could not be resolved to a single patch.
    """
    roster = {str(k): str(v) for k, v in (roster or {}).items()}
    keep, ambiguous = [], []
    for pics, g in df.groupby("PICS_ID"):
        patches = sorted(g["patch_id"].astype(str).unique())
        if str(pics) in roster:
            sel = g[g["patch_id"].astype(str) == roster[str(pics)]]
            keep.append(sel) if not sel.empty else ambiguous.append((str(pics), patches))
        elif len(patches) == 1:
            keep.append(g)
        else:
            ambiguous.append((str(pics), patches))
    selected = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0].copy()
    return selected, ambiguous


def benjamini_hochberg(pvals, alpha=DEFAULT_ALPHA):
    """Benjamini–Hochberg FDR control.

    Returns ``(reject, qvalues)`` — a boolean array of BH-significant channels and
    the adjusted p-values. NaN inputs (unfittable channels) pass through as
    not-rejected / NaN and are excluded from the ranking.
    """
    p = np.asarray(pvals, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return reject, q
    ps = p[idx]
    order = np.argsort(ps)
    m = ps.size
    ranked = ps[order]
    qs = ranked * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]     # enforce monotonicity
    qs = np.clip(qs, 0.0, 1.0)
    q_unsorted = np.empty(m)
    q_unsorted[order] = qs
    q[idx] = q_unsorted
    reject[idx] = q_unsorted <= alpha
    return reject, q


def _nan_stats(n=0):
    """A drift-stat dict with NaN estimates (unfittable / excluded channel)."""
    return {"n": int(n), "slope_pct_yr": np.nan, "ci_low_pct_yr": np.nan,
            "ci_high_pct_yr": np.nan, "intercept_pct": np.nan,
            "kendall_tau": np.nan, "p_value": np.nan,
            "origin_slope_pct_yr": np.nan, "origin_ci_low_pct_yr": np.nan,
            "origin_ci_high_pct_yr": np.nan}


def median_ratio_slope(dt_days, dR_pct):
    """Origin-forced robust slope: median of the per-pair ratios ΔR/Δt (per day).

    Each matched pair is a standalone drift estimate ``s_i = ΔR_i / Δt_i`` under the
    ``ΔR(Δt=0)=0`` anchor; the median is the robust through-origin rate (the analog
    of Theil–Sen that keeps its outlier resistance while forcing the intercept to
    zero). The CI is the **distribution-free** 95% interval for that median, from
    order statistics of the ratios via the normal approximation to the binomial
    rank bounds — no normality assumption, and wild small-Δt ratios only ever
    become extreme order statistics the median ignores. Pairs with non-finite or
    non-positive Δt are dropped.

    Returns ``(slope, ci_low, ci_high)`` in **percent per day**; all NaN if fewer
    than ``MIN_PAIRS`` usable ratios.
    """
    x = np.asarray(dt_days, dtype=float)
    y = np.asarray(dR_pct, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    r = np.sort(y[ok] / x[ok])
    n = r.size
    if n < MIN_PAIRS:
        return np.nan, np.nan, np.nan
    med = float(np.median(r))
    # Distribution-free CI for the median: 1-indexed lower rank l (clamped >= 1),
    # symmetric upper rank n-l+1 -> 0-indexed r[l-1] and r[n-l].
    l = int(np.floor(n / 2.0 - _Z95 * np.sqrt(n) / 2.0))
    l = max(l, 1)
    return med, float(r[l - 1]), float(r[n - l])


def fit_channel_drift(dt_days, dR_pct):
    """Robust drift fit for one channel: ΔR (%) vs Δt (days).

    Returns a dict with ``n``, ``slope_pct_yr`` (drift rate) + 95% CI,
    ``intercept_pct`` (bias diagnostic), ``kendall_tau``, ``p_value``, and the
    secondary origin-forced estimator ``origin_slope_pct_yr`` + its 95% CI
    (``origin_ci_low_pct_yr`` / ``origin_ci_high_pct_yr``; see
    :func:`median_ratio_slope`). Too-few / all-NaN / degenerate-x cases return NaN
    stats (never raise).
    """
    x = np.asarray(dt_days, dtype=float)
    y = np.asarray(dR_pct, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = _nan_stats(x.size)
    if x.size < MIN_PAIRS or np.ptp(x) == 0:
        return out
    slope, intercept, lo, hi = theilslopes(y, x)      # per-day
    tau, p = kendalltau(x, y)
    o_slope, o_lo, o_hi = median_ratio_slope(x, y)    # per-day, through origin
    out.update(slope_pct_yr=slope * DAYS_PER_YEAR,
               ci_low_pct_yr=lo * DAYS_PER_YEAR,
               ci_high_pct_yr=hi * DAYS_PER_YEAR,
               intercept_pct=intercept, kendall_tau=tau, p_value=p,
               origin_slope_pct_yr=o_slope * DAYS_PER_YEAR,
               origin_ci_low_pct_yr=o_lo * DAYS_PER_YEAR,
               origin_ci_high_pct_yr=o_hi * DAYS_PER_YEAR)
    return out


def compute_drift_spectrum(df, product, *, alpha=DEFAULT_ALPHA):
    """Per-channel drift fit across all bands for one product.

    Returns a DataFrame (one row per ``band_index``, sorted) with the fit stats,
    a representative ``wavelength`` (median across pairs), an ``absorption_band``
    flag (water-vapor bands, excluded -> NaN stats), ``raw_significant``
    (p<=alpha), and BH results (``bh_qvalue``, ``bh_significant``).
    """
    dcol = _DELTA_COL[product]
    rows = []
    for bidx, g in df.groupby("band_index"):
        is_abs = _truthy_any(g["absorption_band_flag"])
        # Water-vapor absorption bands are excluded from BOTH products: reflectance
        # is fill (NaN) there, and radiance is atmospherically contaminated — not a
        # usable drift signal. They become gaps in the spectrum.
        fit = (_nan_stats(len(g)) if is_abs
               else fit_channel_drift(g["dt_days"], g[dcol]))
        rows.append({"band_index": int(bidx),
                     "wavelength": float(np.nanmedian(g["wavelength"])),
                     "absorption_band": is_abs, **fit})
    spec = pd.DataFrame(rows).sort_values("band_index").reset_index(drop=True)
    reject, q = benjamini_hochberg(spec["p_value"].to_numpy(), alpha)
    spec["raw_significant"] = spec["p_value"] <= alpha
    spec["bh_qvalue"] = q
    spec["bh_significant"] = reject
    spec.insert(0, "product", product)
    return spec


# ============================================================
# FIGURES (returned, not saved)
# ============================================================
def _channel_legend(ax, sub, *, size_by="sza"):
    """PICS colour/shape legend + a marker-size reference (SZA or #patches)."""
    handles = [mlines.Line2D([], [], color=pics_style(p)[0],
                             marker=pics_style(p)[1], ls="", markeredgecolor="k",
                             markersize=8, label=str(p))
               for p in sorted(sub["PICS_ID"].unique())]
    handles.append(mlines.Line2D([], [], ls="", label=""))   # spacer
    if size_by == "n_patches":
        for n in (1, 5, 10):
            handles.append(mlines.Line2D([], [], color="gray", marker="o", ls="",
                                         markeredgecolor="k",
                                         markersize=np.sqrt(n_patches_marker_size(n)),
                                         label=f"{n} patch(es)"))
    else:
        for sza in (20, 40, 60):
            handles.append(mlines.Line2D([], [], color="gray", marker="o", ls="",
                                         markeredgecolor="k",
                                         markersize=np.sqrt(sza_marker_size(sza)),
                                         label=f"SZA {sza}°"))
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)


def _lookup_stats(spectra, product, band_index):
    """Fetch a channel's precomputed drift stats from a spectrum, or None."""
    if not spectra or product not in spectra:
        return None
    spec = spectra[product]
    row = spec[spec["band_index"] == band_index]
    return None if row.empty else row.iloc[0].to_dict()


def _draw_fit(ax, stats, x):
    """Overlay the Theil–Sen regression line + a stats box on one panel.

    *stats* is a fit dict (from :func:`fit_channel_drift` or a spectrum row);
    *x* is the channel's Δt values (to bound the line). The solid black line is the
    free-intercept Theil–Sen fit; the dashed grey line is the secondary origin-
    forced (median-of-ratios) fit, drawn from the origin. Draws nothing but the
    n-annotation when the channel is unfit (NaN slope).
    """
    n = int(stats.get("n", 0) or 0)
    slope = stats.get("slope_pct_yr", np.nan)
    x = np.asarray(x, dtype=float)
    if x.size and np.isfinite(slope):
        xr = np.array([np.nanmin(x), np.nanmax(x)])
        yline = stats["intercept_pct"] + (slope / DAYS_PER_YEAR) * xr
        ax.plot(xr, yline, color="k", lw=1.6, zorder=5, label="Theil–Sen")
        txt = (f"drift = {slope:+.3f} %/yr\n"
               f"95% CI [{stats['ci_low_pct_yr']:+.2f}, {stats['ci_high_pct_yr']:+.2f}]\n"
               f"intercept = {stats['intercept_pct']:+.2f} %\n"
               f"Kendall p = {stats['p_value']:.2g}   n = {n}")
        # Secondary origin-forced fit: dashed grey line through (0, 0) + a box line.
        o_slope = stats.get("origin_slope_pct_yr", np.nan)
        if pd.notna(o_slope):
            xr0 = np.array([0.0, np.nanmax(x)])
            ax.plot(xr0, (o_slope / DAYS_PER_YEAR) * xr0, color="0.4", lw=1.2,
                    ls="--", zorder=4, label="origin-forced")
            txt += (f"\norigin-forced = {o_slope:+.3f} %/yr\n"
                    f"  95% CI [{stats['origin_ci_low_pct_yr']:+.2f}, "
                    f"{stats['origin_ci_high_pct_yr']:+.2f}]")
        if pd.notna(stats.get("bh_significant", np.nan)):
            txt += "\n" + ("BH-significant" if bool(stats["bh_significant"])
                           else "not BH-significant")
    else:
        txt = f"no drift fit (n = {n})"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def plot_channel_scatter(df, band_index, *, spectra=None, stat_label="median",
                         size_by="sza"):
    """Two-panel ΔR-vs-Δt scatter for one channel: RAD (left), RFL (right).

    Points are coloured + shaped by PICS. Marker size encodes *size_by*:
    ``"sza"`` (default) sizes by ``mean(|SZA_t1|,|SZA_t2|)``; ``"n_patches"``
    sizes by the replicate-patch count (for the aggregated tables, so a more-
    replicated acquisition pair draws a larger dot). If ``"n_patches"`` is
    requested but the column is absent, it falls back to SZA. Each panel overlays
    its Theil–Sen regression line and a stats box (drift rate, 95% CI, intercept,
    Kendall p, n, BH-significance). Stats come from *spectra*
    (``{"rad": spec_rad, "rfl": spec_rfl}``) when given — the same values as the
    drift spectrum — otherwise they are fit on the fly.

    Returns ``(fig, wavelength)``, or ``(None, None)`` if the channel has no rows
    or is a water-vapor absorption band (excluded from the analysis).
    """
    sub = df[df["band_index"] == band_index]
    if sub.empty or _truthy_any(sub["absorption_band_flag"]):
        return None, None
    use_npatch = size_by == "n_patches" and "n_patches" in sub.columns
    wl = float(np.nanmedian(sub["wavelength"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for ax, product in zip(axes, ("rad", "rfl")):
        dcol = _DELTA_COL[product]
        for pics, g in sub.groupby("PICS_ID"):
            color, marker = pics_style(pics)
            sizes = (n_patches_marker_size(g["n_patches"]) if use_npatch
                     else sza_marker_size(g["sza_size"]))
            ax.scatter(g["dt_days"], g[dcol], s=sizes,
                       c=color, marker=marker, edgecolors="k", linewidths=0.4,
                       alpha=0.8, label=str(pics))
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Δt — days between pair")
        ax.set_title(_PRODUCT_LABEL[product])
        stats = _lookup_stats(spectra, product, band_index)
        if stats is None:
            stats = fit_channel_drift(sub["dt_days"], sub[dcol])
        _draw_fit(ax, stats, sub["dt_days"].to_numpy())
    axes[0].set_ylabel(f"ΔR — % change ({stat_label})")
    _channel_legend(axes[1], sub, size_by="n_patches" if use_npatch else "sza")
    size_note = ("marker size ∝ #patches (replication)" if use_npatch
                 else "marker size ∝ mean |SZA|")
    fig.suptitle(f"Channel {int(band_index)} — {wl:.1f} nm    "
                 f"(colour/shape = PICS, {size_note}; black line = "
                 f"Theil–Sen fit)")
    fig.tight_layout()
    return fig, wl


def _absorption_spans(spec):
    """Contiguous (wl_lo, wl_hi) wavelength ranges flagged as absorption bands."""
    if "absorption_band" not in spec:
        return []
    s = spec.sort_values("band_index")
    spans, run = [], []
    for _, r in s.iterrows():
        if bool(r["absorption_band"]):
            run.append(float(r["wavelength"]))
        elif run:
            spans.append((min(run), max(run)))
            run = []
    if run:
        spans.append((min(run), max(run)))
    return spans


def plot_drift_spectrum(spec_rad, spec_rfl, *, patch_size, alpha=DEFAULT_ALPHA):
    """Two-panel drift spectrum: drift rate (%/yr) vs wavelength, RAD then RFL.

    Per-channel 95% CI whiskers; BH-significant channels filled/red, the rest
    hollow/grey. Water-vapor absorption windows (excluded from the analysis) are
    shaded. Returns the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    spans = _absorption_spans(spec_rad)
    for ax, spec, product in zip(axes, (spec_rad, spec_rfl), ("rad", "rfl")):
        for i, (lo, hi) in enumerate(spans):
            ax.axvspan(lo, hi, color="0.9", zorder=0,
                       label="water-vapor band (excluded)" if i == 0 else None)
        s = spec.dropna(subset=["slope_pct_yr"])
        wl = s["wavelength"].to_numpy()
        y = s["slope_pct_yr"].to_numpy()
        lo = s["ci_low_pct_yr"].to_numpy()
        hi = s["ci_high_pct_yr"].to_numpy()
        sig = s["bh_significant"].to_numpy(dtype=bool)
        # Whiskers = 95% CI on the slope; each marker = that channel's Theil–Sen
        # drift rate (filled/red if BH-significant, hollow/grey otherwise).
        ax.errorbar(wl, y, yerr=[y - lo, hi - y], fmt="none", ecolor="darkgray",
                    elinewidth=0.8, capsize=2, zorder=1, label="95% CI")
        ax.scatter(wl[~sig], y[~sig], s=14, facecolors="none",
                   edgecolors="tab:gray", linewidths=0.6, zorder=2,
                   label="drift rate — not significant")
        ax.scatter(wl[sig], y[sig], s=20, c="tab:red", zorder=3,
                   label=f"drift rate — BH-significant (FDR < {alpha})")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel(f"{_PRODUCT_LABEL[product]}\ndrift (%/yr)")
        ax.set_xlabel("Wavelength (nm)")     # x-axis on BOTH panels
        ax.tick_params(labelbottom=True)     # re-enable top panel's tick labels
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    n_ch = int(spec_rad["slope_pct_yr"].notna().sum())
    fig.suptitle(f"Radiometric drift spectrum — {patch_size} km patches   "
                 f"(marker = per-channel Theil–Sen drift rate, whisker = 95% CI; "
                 f"{n_ch} channels fit)")
    fig.tight_layout()
    return fig
