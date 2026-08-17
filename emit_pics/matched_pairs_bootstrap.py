"""PICS-stratified matched-pairs bootstrap for cross-PICS radiometric drift.

The single cross-PICS drift estimator. Per EMIT channel it fits the drift rate as
the robust (Theil-Sen) slope of the pairwise percent change ΔR against the pair's
time separation Δt, pooled across all PICS, and quantifies the uncertainty with a
genuine nonparametric bootstrap over the matched pairs.

**Data structure.** Reorganize the per-patch delta rows into an *edge list per
PICS*: nodes are scenes, an edge is a unique matched scene-pair carrying its Δt
and a bag of per-patch ΔR values (the spatial replicates at that pair).

**Point estimate.** One representative ΔR per edge (the median across its patch
replicates), pooled across PICS, fit with Theil-Sen. This is the reported slope.

**Uncertainty (the bootstrap).** For each of ``n_boot`` iterations:

* **resample the edges with replacement, stratified by PICS** — within each PICS
  draw as many edges as it has, with replacement, so the five-site composition is
  preserved every iteration and no single PICS can dominate or vanish;
* draw **one patch's ΔR** at random for each resampled edge, so between-patch
  surface heterogeneity propagates into the spread rather than being averaged away;
* pool the resulting ``(Δt, ΔR)`` points across PICS and fit one Theil-Sen slope.

The 2.5/97.5 percentiles of the ``n_boot`` slopes give the 95% CI, and the tail
fraction about 0 gives a two-sided bootstrap p-value.

**Acknowledged limitation — scene reuse.** A scene appears in many pairs, so the
edges are not independent (it is a network). This bootstrap resamples pairs, not
scenes, so it treats the pairs as exchangeable and its CI runs slightly optimistic
(a touch too tight). A scene-level cluster bootstrap would remove that bias at the
cost of complexity; it is left as future work and stated as a limitation.

Pure function library — no I/O, no hardcoded paths, no side effects on import.
The per-channel spectrum it returns carries the columns
:func:`emit_pics.matched_pairs_drift_analysis.plot_drift_spectrum` reads, so the
figure code renders it unchanged.
"""

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from emit_pics.matched_pairs_drift_analysis import (
    DAYS_PER_YEAR,
    DEFAULT_ALPHA,
    MIN_PAIRS,
    _DELTA_COL,
    _truthy_any,
)

#: Per-product seed offset so RAD and RFL draw independent, reproducible streams.
_PRODUCT_CODE = {"rad": 0, "rfl": 1}

#: Columns the per-patch delta table must carry to build the edge list.
_REQUIRED_COLS = {
    "PICS_ID", "pair_id", "band_index", "wavelength", "dt_days",
    "t1_scene_id", "t2_scene_id", "absorption_band_flag",
}


# ============================================================
# EDGE LIST
# ============================================================
def build_edges(band_df, delta_col):
    """Group one band's per-patch delta rows into an edge list per PICS.

    Parameters
    ----------
    band_df : pd.DataFrame
        Per-patch delta rows for a *single* ``band_index`` (all PICS).
    delta_col : str
        The ΔR column to read (e.g. ``delta_rad_pct_median``).

    Returns
    -------
    dict
        ``{PICS_ID: [edge, ...]}`` where each *edge* is
        ``{"u", "v": scene ids, "dt": Δt days, "dR": np.ndarray of per-patch ΔR}``.
        Edges with no finite ΔR are dropped; a PICS with no usable edge is absent.
    """
    edges_by_pics = {}
    for pics, gp in band_df.groupby("PICS_ID"):
        edges = []
        for _, gpair in gp.groupby("pair_id"):
            dr = pd.to_numeric(gpair[delta_col], errors="coerce").to_numpy(dtype=float)
            dr = dr[np.isfinite(dr)]
            if dr.size == 0:
                continue
            first = gpair.iloc[0]
            edges.append({
                "u": str(first["t1_scene_id"]),
                "v": str(first["t2_scene_id"]),
                "dt": float(gpair["dt_days"].iloc[0]),
                "dR": dr,
            })
        if edges:
            edges_by_pics[str(pics)] = edges
    return edges_by_pics


# ============================================================
# SLOPE (POINT ESTIMATE + ONE BOOTSTRAP DRAW)
# ============================================================
def point_slope(edges_by_pics):
    """Point estimate: Theil-Sen on the median-ΔR-per-edge, pooled across PICS.

    Each edge contributes ONE point: its Δt and the median of its patch-replicate
    ΔR bag (all patches used, not resampled). Returns ``(slope_per_day, n_points)``,
    or ``(np.nan, n_points)`` if too few points remain or all share one Δt.
    """
    xs, ys = [], []
    for edges in edges_by_pics.values():
        for e in edges:
            xs.append(e["dt"])
            ys.append(float(np.median(e["dR"])))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < MIN_PAIRS or np.ptp(x) == 0:
        return np.nan, x.size
    return float(theilslopes(y, x)[0]), x.size


def resample_slope(edges_by_pics, rng):
    """One bootstrap draw: PICS-stratified edge resample + one patch ΔR, then fit.

    Within each PICS, draw as many edges as it has **with replacement** (preserving
    the per-site count); for each drawn edge draw one patch's ΔR at random from its
    bag; pool across PICS and fit Theil-Sen. Returns ``(slope_per_day, n_points)``,
    or ``(np.nan, n_points)`` for a degenerate draw (too few points / single Δt).
    """
    xs, ys = [], []
    for edges in edges_by_pics.values():
        k = len(edges)
        if k == 0:
            continue
        for j in rng.integers(0, k, k):                   # edges, with replacement
            e = edges[j]
            xs.append(e["dt"])
            ys.append(e["dR"][rng.integers(e["dR"].size)])   # one patch draw
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < MIN_PAIRS or np.ptp(x) == 0:
        return np.nan, x.size
    return float(theilslopes(y, x)[0]), x.size


# ============================================================
# CHANNEL BOOTSTRAP
# ============================================================
def _nan_boot(n_scenes, n_edges, n_valid):
    """Bootstrap-stat dict with NaN estimates (absorption / unfittable channel)."""
    return {"slope_pct_yr": np.nan, "ci_low_pct_yr": np.nan,
            "ci_high_pct_yr": np.nan, "p_value": np.nan, "raw_significant": False,
            "n_scenes": int(n_scenes), "n_edges": int(n_edges), "n_boot": int(n_valid)}


def bootstrap_channel(edges_by_pics, n_boot, rng, *, alpha=DEFAULT_ALPHA):
    """Point estimate + bootstrap CI for one channel from its edge list.

    The reported ``slope_pct_yr`` is :func:`point_slope` on the observed data (the
    median-ΔR-per-edge Theil-Sen fit) — NOT a bootstrap average. The CI and p-value
    come from *n_boot* :func:`resample_slope` draws (PICS-stratified edge resampling
    with replacement + one patch draw per edge). Returns a stat dict with the point
    estimate, the ``100(1-alpha)%`` percentile CI, a two-sided bootstrap ``p_value``,
    ``raw_significant`` (CI excludes 0), and diagnostic counts ``n_scenes`` /
    ``n_edges`` / ``n_boot`` (valid draws). NaN stats when the channel is unfittable
    or fewer than ``MIN_PAIRS`` draws produced a slope.
    """
    n_edges = sum(len(e) for e in edges_by_pics.values())
    scenes = {s for edges in edges_by_pics.values()
              for e in edges for s in (e["u"], e["v"])}
    n_scenes = len(scenes)

    point, _ = point_slope(edges_by_pics)
    if not np.isfinite(point):
        return _nan_boot(n_scenes, n_edges, 0)

    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draws[b], _ = resample_slope(edges_by_pics, rng)
    valid = draws[np.isfinite(draws)]

    slope_pct = point * DAYS_PER_YEAR
    if valid.size < MIN_PAIRS:                            # no usable CI
        return {"slope_pct_yr": slope_pct, "ci_low_pct_yr": np.nan,
                "ci_high_pct_yr": np.nan, "p_value": np.nan,
                "raw_significant": False, "n_scenes": int(n_scenes),
                "n_edges": int(n_edges), "n_boot": int(valid.size)}

    lo = np.percentile(valid, 100.0 * alpha / 2.0)
    hi = np.percentile(valid, 100.0 * (1.0 - alpha / 2.0))
    # Two-sided bootstrap p: twice the smaller tail mass on the wrong side of 0.
    p = 2.0 * min(float(np.mean(valid <= 0)), float(np.mean(valid >= 0)))
    return {
        "slope_pct_yr": slope_pct,
        "ci_low_pct_yr": lo * DAYS_PER_YEAR,
        "ci_high_pct_yr": hi * DAYS_PER_YEAR,
        "p_value": min(p, 1.0),
        "raw_significant": bool(lo > 0 or hi < 0),
        "n_scenes": int(n_scenes),
        "n_edges": int(n_edges),
        "n_boot": int(valid.size),
    }


# ============================================================
# SPECTRUM
# ============================================================
def compute_bootstrap_drift_spectrum(df, product, *, n_boot=1000,
                                     alpha=DEFAULT_ALPHA, seed=0):
    """Per-channel matched-pairs bootstrap drift spectrum for one product.

    Consumes the pooled **per-patch** delta table (all PICS of one size). Water-
    vapor absorption bands are excluded (NaN stats). Each channel gets an
    independent, reproducible RNG stream keyed by ``(seed, product, band_index)``.

    Returns a DataFrame (one row per ``band_index``, sorted) whose columns are a
    superset of what :func:`matched_pairs_drift_analysis.plot_drift_spectrum` reads
    (``wavelength``, ``slope_pct_yr``, ``ci_low_pct_yr``, ``ci_high_pct_yr``,
    ``raw_significant``, ``absorption_band``), plus ``p_value`` and the ``n_*``
    diagnostics.
    """
    missing = _REQUIRED_COLS | {_DELTA_COL[product]}
    missing -= set(df.columns)
    if missing:
        raise ValueError(
            f"Per-patch delta rows are missing columns {sorted(missing)}. "
            "Re-run scripts/run_delta_calculations.py (step 6) to regenerate the "
            "delta tables with the full schema."
        )

    dcol = _DELTA_COL[product]
    pcode = _PRODUCT_CODE[product]
    rows = []
    for bidx, g in df.groupby("band_index"):
        is_abs = _truthy_any(g["absorption_band_flag"])
        wl = float(np.nanmedian(g["wavelength"]))
        if is_abs:
            fit = _nan_boot(0, 0, 0)
        else:
            edges_by_pics = build_edges(g, dcol)
            rng = np.random.default_rng([seed, pcode, int(bidx)])
            fit = bootstrap_channel(edges_by_pics, n_boot, rng, alpha=alpha)
        rows.append({"product": product, "band_index": int(bidx),
                     "wavelength": wl, "absorption_band": is_abs, **fit})
    return pd.DataFrame(rows).sort_values("band_index").reset_index(drop=True)
