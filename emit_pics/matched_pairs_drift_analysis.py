"""Shared plumbing + figure for the matched-pairs cross-PICS drift analysis.

Home for the pieces the matched-pairs drift method shares: the constants, the
per-patch delta-table loader, and the drift-spectrum figure. The estimator itself
— a PICS-stratified matched-pairs **bootstrap** — lives in
:mod:`emit_pics.matched_pairs_bootstrap`.

**Model.** Over a pseudo-invariant calibration site with geometry-matched pairs
the surface is assumed stable, so if the sensor drifts linearly at rate ``m`` then
``R(t) = R0 (1 + m t)`` and

    ΔR = (R(t2) - R(t1)) / R(t1) ≈ m · Δt.

So the **slope of ΔR-vs-Δt is the drift rate**. Each matched pair is one estimate
of accumulated change over its interval; the slope is fit robustly with Theil-Sen
and its uncertainty is bootstrapped over the pairs (see the bootstrap module).

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
import matplotlib.pyplot as plt

# CONSTANTS
DAYS_PER_YEAR = 365.25
MIN_PAIRS = 3               #: fewest points needed to fit a channel
DEFAULT_ALPHA = 0.05        #: per-channel significance level (uncorrected bootstrap p)

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


# STYLING / SMALL HELPERS
def pics_style(pics_id):
    """(colour, marker) for a PICS id, with a grey fallback for unknown sites."""
    return PICS_STYLE.get(str(pics_id), _FALLBACK_STYLE)


def patch_size_of(patch_id) -> str:
    """First token of a patch_id (e.g. ``'10_7_1' -> '10'``)."""
    return str(patch_id).split("_")[0]


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


# LOADING
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
        ``mean(|SZA_t1|, |SZA_t2|)`` (a per-pair geometry diagnostic). Empty if no
        CSVs match.

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


# FIGURE (returned, not saved)
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

    Per-channel bootstrap 95% CI whiskers; significant channels (CI excludes 0,
    raw p < alpha) filled/red, the rest hollow/grey. Water-vapor absorption windows
    (excluded from the analysis) are shaded. Returns the figure.
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
        sig = s["raw_significant"].to_numpy(dtype=bool)
        # Whiskers = bootstrap 95% CI on the slope; each marker = that channel's
        # matched-pairs (Theil-Sen) drift rate (filled/red if the CI excludes 0 at
        # raw p < alpha, hollow/grey otherwise).
        ax.errorbar(wl, y, yerr=[y - lo, hi - y], fmt="none", ecolor="darkgray",
                    elinewidth=0.8, capsize=2, zorder=1, label="95% CI")
        ax.scatter(wl[~sig], y[~sig], s=14, facecolors="none",
                   edgecolors="tab:gray", linewidths=0.6, zorder=2,
                   label="drift rate — not significant")
        ax.scatter(wl[sig], y[sig], s=20, c="tab:red", zorder=3,
                   label=f"drift rate — significant (p < {alpha})")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel(f"{_PRODUCT_LABEL[product]}\ndrift (%/yr)")
        ax.set_xlabel("Wavelength (nm)")     # x-axis on BOTH panels
        ax.tick_params(labelbottom=True)     # re-enable top panel's tick labels
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    n_ch = int(spec_rad["slope_pct_yr"].notna().sum())
    fig.suptitle(f"Radiometric drift spectrum — {patch_size} km patches   "
                 f"(marker = per-channel matched-pairs drift rate, whisker = "
                 f"bootstrap 95% CI; {n_ch} channels fit)")
    fig.tight_layout()
    return fig
