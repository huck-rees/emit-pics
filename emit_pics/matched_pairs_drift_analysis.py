"""Cross-PICS radiometric-drift analysis from the per-patch delta tables.

Consumes the tidy long delta CSVs (step 6, ``data/delta_r/{PICS}_{PATCH}_delta.csv``)
and, per EMIT channel, quantifies radiometric drift by regressing the pairwise
percent change ΔR against the pair's time separation Δt across all PICS/patches
of one patch size.

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


# ============================================================
# STATISTICS
# ============================================================
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
            "kendall_tau": np.nan, "p_value": np.nan}


def fit_channel_drift(dt_days, dR_pct):
    """Robust drift fit for one channel: ΔR (%) vs Δt (days).

    Returns a dict with ``n``, ``slope_pct_yr`` (drift rate) + 95% CI,
    ``intercept_pct`` (bias diagnostic), ``kendall_tau`` and ``p_value``.
    Too-few / all-NaN / degenerate-x cases return NaN stats (never raise).
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
    out.update(slope_pct_yr=slope * DAYS_PER_YEAR,
               ci_low_pct_yr=lo * DAYS_PER_YEAR,
               ci_high_pct_yr=hi * DAYS_PER_YEAR,
               intercept_pct=intercept, kendall_tau=tau, p_value=p)
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
def _channel_legend(ax, sub):
    """PICS colour/shape legend + an SZA marker-size reference."""
    handles = [mlines.Line2D([], [], color=pics_style(p)[0],
                             marker=pics_style(p)[1], ls="", markeredgecolor="k",
                             markersize=8, label=str(p))
               for p in sorted(sub["PICS_ID"].unique())]
    handles.append(mlines.Line2D([], [], ls="", label=""))   # spacer
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
    *x* is the channel's Δt values (to bound the line). Draws nothing but the
    n-annotation when the channel is unfit (NaN slope).
    """
    n = int(stats.get("n", 0) or 0)
    slope = stats.get("slope_pct_yr", np.nan)
    x = np.asarray(x, dtype=float)
    if x.size and np.isfinite(slope):
        xr = np.array([np.nanmin(x), np.nanmax(x)])
        yline = stats["intercept_pct"] + (slope / DAYS_PER_YEAR) * xr
        ax.plot(xr, yline, color="k", lw=1.6, zorder=5)
        txt = (f"drift = {slope:+.3f} %/yr\n"
               f"95% CI [{stats['ci_low_pct_yr']:+.2f}, {stats['ci_high_pct_yr']:+.2f}]\n"
               f"intercept = {stats['intercept_pct']:+.2f} %\n"
               f"Kendall p = {stats['p_value']:.2g}   n = {n}")
        if pd.notna(stats.get("bh_significant", np.nan)):
            txt += "\n" + ("BH-significant" if bool(stats["bh_significant"])
                           else "not BH-significant")
    else:
        txt = f"no drift fit (n = {n})"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def plot_channel_scatter(df, band_index, *, spectra=None, stat_label="median"):
    """Two-panel ΔR-vs-Δt scatter for one channel: RAD (left), RFL (right).

    Points are coloured + shaped by PICS and sized by ``mean(|SZA_t1|,|SZA_t2|)``.
    Each panel overlays its Theil–Sen regression line and a stats box (drift
    rate, 95% CI, intercept, Kendall p, n, BH-significance). Stats come from
    *spectra* (``{"rad": spec_rad, "rfl": spec_rfl}``) when given — the same
    values as the drift spectrum — otherwise they are fit on the fly.

    Returns ``(fig, wavelength)``, or ``(None, None)`` if the channel has no rows
    or is a water-vapor absorption band (excluded from the analysis).
    """
    sub = df[df["band_index"] == band_index]
    if sub.empty or _truthy_any(sub["absorption_band_flag"]):
        return None, None
    wl = float(np.nanmedian(sub["wavelength"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for ax, product in zip(axes, ("rad", "rfl")):
        dcol = _DELTA_COL[product]
        for pics, g in sub.groupby("PICS_ID"):
            color, marker = pics_style(pics)
            ax.scatter(g["dt_days"], g[dcol], s=sza_marker_size(g["sza_size"]),
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
    _channel_legend(axes[1], sub)
    fig.suptitle(f"Channel {int(band_index)} — {wl:.1f} nm    "
                 f"(colour/shape = PICS, marker size ∝ mean |SZA|; black line = "
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
