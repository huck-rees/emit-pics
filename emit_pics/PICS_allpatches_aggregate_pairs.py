"""Collapse spatial-replicate patches to one ΔR per acquisition pair (option A).

The cross-PICS drift analysis regresses the pairwise percent change ΔR against
each pair's time separation Δt. Its raw input — the per-patch delta tables
(``data/delta_r/per_patch/{PICS}_{PATCH}_delta.csv``) — carries one row per
``(patch × acquisition-pair × channel)``. Pooling those directly is
**pseudo-replication**: the *same* acquisition pair (same two acquisition
timestamps, hence the same sensor state and atmosphere) observed by many spatial-
replicate patches enters the Theil–Sen fit as many near-identical points,
artificially narrowing the uncertainty and inflating the effective sample size.

**Option A** fixes this by collapsing, per channel, every spatial replicate of a
unique acquisition pair down to a single ΔR *before* the fit — so each
acquisition pair contributes exactly one point regardless of how many patches
saw it. The collapse is the **median across patches** (robust to an anomalous
patch, and matching the Theil–Sen estimator downstream).

Aggregation is **within one PICS**: a ``pair_id`` (an acquisition-timestamp pair)
is unique to a single PICS because the sites are far apart, so there is no cross-
PICS collision — ``PICS_ID`` only scopes the input, it is not part of the group
key. The diagnostic ``n_patches`` and across-patch spread columns are recorded
(not used to weight anything here) because they are free to compute and are
exactly what a future option-B inverse-variance weighting would need.

Pure function library — no top-level execution, no hardcoded paths, no side
effects on import.

Typical usage from a runner::

    from emit_pics.PICS_allpatches_aggregate_pairs import (
        load_patch_delta_tables, aggregate_by_pair,
        write_aggregated_table, collapse_summary,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emit_pics.matched_pairs_drift_analysis import load_delta_tables, patch_size_of

# CONSTANTS
#: Per-channel delta columns collapsed by the across-patch MEDIAN.
DELTA_VALUE_COLS = [
    "delta_rfl_pct_median", "delta_rfl_pct_mean",
    "delta_rad_pct_median", "delta_rad_pct_mean",
]

#: Scalar pair metadata identical across replicate patches -> carried unchanged
#: (taken from the first replicate).
_META_FIRST_COLS = [
    "PICS_ID", "t1_scene_id", "t2_scene_id",
    "t1_dt", "t2_dt", "dt_days", "absorption_band_flag",
]

#: In-patch MEAN geometry -> varies slightly per patch (different pixels), so it
#: is collapsed by the across-patch median, like the delta columns.
_GEOM_MEDIAN_COLS = ["SZA_t1", "SZA_t2", "delta_SZA", "delta_VZA"]

#: Delta columns whose across-patch spread is retained as a diagnostic. The
#: median statistic is the one that matches the downstream estimator; recording
#: its spread (std/min/max) is what an option-B inverse-variance scheme needs.
_SPREAD_SRC = ["delta_rfl_pct_median", "delta_rad_pct_median"]

#: Group key: one output row per unique acquisition pair per channel.
_GROUP_KEYS = ["pair_id", "band_index"]

#: Columns any per-patch delta table must carry to be aggregatable.
_REQUIRED_COLS = (
    set(_GROUP_KEYS) | set(DELTA_VALUE_COLS) | set(_META_FIRST_COLS)
    | set(_GEOM_MEDIAN_COLS) | {"patch_id", "wavelength"}
)

#: Output column order — a superset of what the drift analysis reads, so the
#: aggregated table is a drop-in for :func:`matched_pairs_drift_analysis`.
AGG_CSV_COLS = (
    ["PICS_ID", "patch_size", "pair_id",
     "t1_scene_id", "t2_scene_id", "t1_dt", "t2_dt", "dt_days",
     "SZA_t1", "SZA_t2", "delta_SZA", "delta_VZA",
     "band_index", "wavelength", "absorption_band_flag"]
    + DELTA_VALUE_COLS
    + ["n_patches"]
    + [f"{c}_{s}" for c in _SPREAD_SRC for s in ("std", "min", "max")]
)


# LOADING
def load_patch_delta_tables(delta_dir, pics_id, patch_size, *,
                            glob_pattern="*_delta.csv"):
    """Pool the per-patch delta CSVs for ONE PICS at one patch size.

    Reuses :func:`matched_pairs_drift_analysis.load_delta_tables` (which globs the
    directory, filters by the size token of ``patch_id``, validates the step-6
    schema, and appends ``sza_size``), then scopes the result to a single PICS —
    aggregation never crosses PICS.

    Parameters
    ----------
    delta_dir : path-like
        Directory of per-patch delta CSVs (``data/delta_r/per_patch``).
    pics_id : str
        Underscored site id to keep (e.g. ``"Libya_4"``).
    patch_size : str | int
        Patch-size token (first field of ``patch_id``, e.g. ``10``).

    Returns
    -------
    pd.DataFrame
        Per-patch delta rows for that (PICS, size); empty if none match.

    Raises
    ------
    ValueError
        Propagated from the underlying loader if a matched CSV predates the
        SZA/band_index schema.
    """
    df = load_delta_tables(delta_dir, patch_size, glob_pattern=glob_pattern)
    if df.empty:
        return df
    df = df[df["PICS_ID"].astype(str) == str(pics_id)].reset_index(drop=True)
    return df


# AGGREGATION
def _median(series):
    """Across-patch median of a column coerced to float (NaNs ignored)."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan


def aggregate_by_pair(df):
    """Collapse spatial-replicate patches to one row per ``(pair_id, band_index)``.

    For each group of replicate patches sharing one acquisition pair and channel:

    * the four :data:`DELTA_VALUE_COLS` and the four :data:`_GEOM_MEDIAN_COLS`
      (in-patch mean geometry) are collapsed by the **median across patches**;
    * ``wavelength`` is taken as the across-patch median (robust to EMIT's per-era
      calibration drift);
    * the identical scalar metadata (:data:`_META_FIRST_COLS`) is carried from the
      first replicate; ``absorption_band_flag`` becomes the group's logical OR;
    * ``n_patches`` (distinct ``patch_id`` count) and the across-patch spread
      (``*_std`` / ``*_min`` / ``*_max`` of the median-delta columns) are recorded
      as diagnostics.

    The whole input must be a single patch size (``pair_id`` repeats across sizes
    within a PICS, so mixing sizes would merge unrelated groups). ``patch_size``
    is derived and written as a scalar column so the drift loader can size-filter
    the aggregated tables without a ``patch_id``.

    Parameters
    ----------
    df : pd.DataFrame
        Per-patch delta rows for one PICS + one size (from
        :func:`load_patch_delta_tables`).

    Returns
    -------
    pd.DataFrame
        One row per ``(pair_id, band_index)``, columns in :data:`AGG_CSV_COLS`
        order, sorted by ``(pair_id, band_index)``.

    Raises
    ------
    ValueError
        If *df* is empty, is missing a required column, or mixes patch sizes.
    """
    if df is None or df.empty:
        raise ValueError("No per-patch delta rows to aggregate (empty input).")
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Per-patch delta rows are missing columns {sorted(missing)}. "
            "Re-run scripts/run_delta_calculations.py to regenerate the delta "
            "tables with the full step-6 schema."
        )
    sizes = sorted(set(df["patch_id"].map(patch_size_of)))
    if len(sizes) != 1:
        raise ValueError(
            f"aggregate_by_pair expects a single patch size, got {sizes}. "
            "Aggregation is per (PICS, size); load one size at a time."
        )
    patch_size = sizes[0]

    records = []
    for (pair_id, bidx), g in df.groupby(_GROUP_KEYS, sort=True):
        first = g.iloc[0]
        rec = {"PICS_ID": first["PICS_ID"], "patch_size": patch_size,
               "pair_id": pair_id, "band_index": int(bidx)}
        for c in _META_FIRST_COLS:
            if c == "PICS_ID":
                continue
            rec[c] = first[c]
        # absorption flag: identical across patches, but OR is the safe reduction.
        rec["absorption_band_flag"] = bool(
            g["absorption_band_flag"].astype(str).str.strip()
             .str.lower().isin(["true", "1"]).any())
        # per-patch mean geometry + wavelength -> across-patch median.
        for c in _GEOM_MEDIAN_COLS:
            rec[c] = _median(g[c])
        rec["wavelength"] = _median(g["wavelength"])
        # the collapse itself: median ΔR across the spatial-replicate patches.
        for c in DELTA_VALUE_COLS:
            rec[c] = _median(g[c])
        # diagnostics (recorded, not used to weight in option A).
        rec["n_patches"] = int(g["patch_id"].nunique())
        for c in _SPREAD_SRC:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            has = bool(np.isfinite(vals).any())
            rec[f"{c}_std"] = float(np.nanstd(vals)) if has else np.nan
            rec[f"{c}_min"] = float(np.nanmin(vals)) if has else np.nan
            rec[f"{c}_max"] = float(np.nanmax(vals)) if has else np.nan
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    out = out.sort_values(_GROUP_KEYS).reset_index(drop=True)
    return out.reindex(columns=AGG_CSV_COLS)


# WRITING / SUMMARY
def write_aggregated_table(agg_df, out_path):
    """Write the aggregated tidy table (one row per pair × channel).

    Columns are forced to :data:`AGG_CSV_COLS` order; the parent directory is
    created if needed. Returns the path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.reindex(columns=AGG_CSV_COLS).to_csv(out_path, index=False)
    return out_path


def collapse_summary(raw_df, agg_df):
    """Quantify the pseudo-replication removed by the collapse.

    Returns a dict with the ``(patch × pair)`` row count, the unique-pair count,
    the per-pair patch-count distribution (median/min/max), and the per-channel
    raw→aggregated row totals — enough for the runner to print how much
    effective-N inflation option A corrected.
    """
    per_pair = raw_df.groupby("pair_id")["patch_id"].nunique()
    return {
        "n_patch_pairs": int(raw_df.groupby(["pair_id", "patch_id"]).ngroups),
        "n_pairs": int(raw_df["pair_id"].nunique()),
        "n_channels": int(raw_df["band_index"].nunique()),
        "n_raw_rows": int(len(raw_df)),
        "n_agg_rows": int(len(agg_df)),
        "patches_per_pair_median": float(per_pair.median()) if len(per_pair) else float("nan"),
        "patches_per_pair_min": int(per_pair.min()) if len(per_pair) else 0,
        "patches_per_pair_max": int(per_pair.max()) if len(per_pair) else 0,
    }
