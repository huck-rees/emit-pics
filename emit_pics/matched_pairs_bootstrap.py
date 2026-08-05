"""One-use-per-scene matching bootstrap for cross-PICS radiometric drift.

An alternative to the median-collapse + Theil-Sen estimator
(:mod:`emit_pics.matched_pairs_drift_analysis`) that fixes two defects of pooling
matched pairs directly:

1. **Graph-degree leverage.** A matched pair is defined by two *scenes*, and one
   scene appears in many pairs. Fitting all pairs treats a network of correlated
   observations as independent, overstates precision, and lets one biased scene
   contaminate every pair it touches.
2. **Hidden surface heterogeneity.** The median-collapse across spatial-replicate
   patches treats different ground as interchangeable and discards the between-
   patch spread — real uncertainty that never reaches the fit.

**The method.** Reorganize the per-patch delta rows into an *edge list per PICS*:
nodes are scenes, an edge is a unique matched scene-pair carrying its Δt and a bag
of per-patch ΔR values (the spatial replicates). Then, for each of ``n_boot``
iterations:

* draw a **random maximal matching** within each PICS (greedy, scene-disjoint —
  every scene is used at most once, so within one fit the points are independent
  and no scene can dominate),
* draw **one patch's ΔR** per chosen edge (so surface heterogeneity enters as
  variability, not an averaged-away constant),
* pool the resulting ``(Δt, ΔR)`` points **across PICS** and fit one Theil-Sen
  slope.

The distribution of the ``n_boot`` slopes gives the point estimate (its median),
an honest 95% CI (its percentiles), and a bootstrap p-value. Matching is *within*
a PICS (scenes never cross PICS); the fit pools *across* PICS, exactly as the
Theil-Sen path does — so the two estimators are directly comparable.

Pure function library — no I/O, no hardcoded paths, no side effects on import.
The per-band per-channel spectrum it returns carries the same columns
:func:`emit_pics.matched_pairs_drift_analysis.plot_drift_spectrum` reads, so the
existing figure code renders it unchanged.
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


def random_matching(edges, rng):
    """Greedy scene-disjoint matching: keep pairs that share no scene.

    Walks the edges in a random order and accepts an edge only if neither of its
    two scenes has been used yet, so every scene appears in at most one kept pair.
    Returns the list of chosen edges (a subset of *edges*).
    """
    used = set()
    chosen = []
    for k in rng.permutation(len(edges)):
        e = edges[k]
        if e["u"] in used or e["v"] in used:
            continue
        used.add(e["u"])
        used.add(e["v"])
        chosen.append(e)
    return chosen


# ============================================================
# ONE ITERATION + BOOTSTRAP
# ============================================================
def one_slope(edges_by_pics, rng):
    """One bootstrap iteration: match within each PICS, draw a patch, pool, fit.

    Returns ``(slope_per_day, n_points)`` — the Theil-Sen slope of the pooled
    ``(Δt, ΔR)`` points, or ``(np.nan, n_points)`` when too few points remain or
    all share one Δt (degenerate x).
    """
    xs, ys = [], []
    for edges in edges_by_pics.values():
        for e in random_matching(edges, rng):
            xs.append(e["dt"])
            ys.append(e["dR"][rng.integers(e["dR"].size)])   # one patch draw
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < MIN_PAIRS or np.ptp(x) == 0:
        return np.nan, x.size
    return float(theilslopes(y, x)[0]), x.size


def _nan_boot(n_scenes, n_edges, n_valid):
    """Bootstrap-stat dict with NaN estimates (absorption / unfittable channel)."""
    return {"slope_pct_yr": np.nan, "ci_low_pct_yr": np.nan,
            "ci_high_pct_yr": np.nan, "p_value": np.nan, "raw_significant": False,
            "n_pairs": np.nan, "n_scenes": int(n_scenes),
            "n_edges": int(n_edges), "n_boot": int(n_valid)}


def bootstrap_channel(edges_by_pics, n_boot, rng, *, alpha=DEFAULT_ALPHA):
    """Bootstrap the drift slope for one channel from its edge list.

    Runs *n_boot* matching+draw+fit iterations and summarizes the slope
    distribution. Returns a stat dict with the point estimate (median slope, %/yr),
    the ``100(1-alpha)%`` percentile CI, a two-sided bootstrap ``p_value``,
    ``raw_significant`` (CI excludes 0), the mean pooled pairs per iteration, and
    diagnostic counts. NaN stats when fewer than ``MIN_PAIRS`` iterations produced
    a fit.
    """
    n_edges = sum(len(e) for e in edges_by_pics.values())
    scenes = {s for edges in edges_by_pics.values()
              for e in edges for s in (e["u"], e["v"])}
    n_scenes = len(scenes)

    slopes = np.full(n_boot, np.nan)
    npts = np.zeros(n_boot)
    for b in range(n_boot):
        slopes[b], npts[b] = one_slope(edges_by_pics, rng)

    ok = np.isfinite(slopes)
    valid = slopes[ok]
    if valid.size < MIN_PAIRS:
        return _nan_boot(n_scenes, n_edges, valid.size)

    per_day = np.median(valid)
    lo = np.percentile(valid, 100.0 * alpha / 2.0)
    hi = np.percentile(valid, 100.0 * (1.0 - alpha / 2.0))
    # Two-sided bootstrap p: twice the smaller tail mass on the wrong side of 0.
    p = 2.0 * min(float(np.mean(valid <= 0)), float(np.mean(valid >= 0)))
    return {
        "slope_pct_yr": per_day * DAYS_PER_YEAR,
        "ci_low_pct_yr": lo * DAYS_PER_YEAR,
        "ci_high_pct_yr": hi * DAYS_PER_YEAR,
        "p_value": min(p, 1.0),
        "raw_significant": bool(lo > 0 or hi < 0),
        "n_pairs": float(np.mean(npts[ok])),
        "n_scenes": int(n_scenes),
        "n_edges": int(n_edges),
        "n_boot": int(valid.size),
    }


# ============================================================
# SPECTRUM
# ============================================================
def compute_bootstrap_drift_spectrum(df, product, *, n_boot=1000,
                                     alpha=DEFAULT_ALPHA, seed=0):
    """Per-channel matching-bootstrap drift spectrum for one product.

    Consumes the pooled **per-patch** delta table (all PICS of one size). Water-
    vapor absorption bands are excluded (NaN stats) exactly as in the Theil-Sen
    path. Each channel gets an independent, reproducible RNG stream keyed by
    ``(seed, product, band_index)``.

    Returns a DataFrame (one row per ``band_index``, sorted) whose columns are a
    superset of what :func:`matched_pairs_drift_analysis.plot_drift_spectrum`
    reads (``wavelength``, ``slope_pct_yr``, ``ci_low_pct_yr``, ``ci_high_pct_yr``,
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
