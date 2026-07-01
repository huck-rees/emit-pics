"""Per-wavelength % change in reflectance/radiance across matched scene pairs.

Pure function library — no top-level execution, no hardcoded paths, no side
effects on import. Given the matched-pair list (from ``id_matched_pairs``) and
the per-acquisition per-channel stats CSVs (from ``run_patch_extract``), this
computes, for each matched pair and each wavelength, the pairwise **t1 -> t2**
percent change of the median and mean in-patch value, for both products.

The premise: over a pseudo-invariant calibration site with matched viewing
geometry, the surface is assumed stable, so a non-zero delta is a proxy for
sensor drift (plus residual atmosphere / BRDF). No aggregation or plotting is
done here — the single output is a tidy long table, one row per
``(matched pair x wavelength)``; downstream analysis across all PICS consumes it.

Typical usage from a runner::

    from emit_pics.delta_calculations import (
        load_matched_pairs, load_scene_spectrum, compute_pair_delta,
        DELTA_CSV_COLS,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emit_pics.id_matched_pairs import acq_timestamp

# ============================================================
# CONSTANTS
# ============================================================
#: Output column order of the tidy long delta table.
DELTA_CSV_COLS = [
    "PICS_ID", "patch_id", "pair_id",
    "t1_scene_id", "t2_scene_id", "t1_dt", "t2_dt", "dt_days",
    "delta_SZA", "delta_VZA",
    "wavelength",
    "rfl_t1_median", "rfl_t2_median", "rfl_t1_mean", "rfl_t2_mean",
    "delta_rfl_pct_median", "delta_rfl_pct_mean",
    "rad_t1_median", "rad_t2_median", "rad_t1_mean", "rad_t2_mean",
    "delta_rad_pct_median", "delta_rad_pct_mean",
    "absorption_band_flag",
]

#: Deep water-vapor absorption regions (nm) where L2A reflectance approaches
#: zero and the percent-change denominator is unreliable. Rows in these ranges
#: are FLAGGED (``absorption_band_flag = True``), never dropped — the downstream
#: all-PICS analysis decides whether to exclude them.
ABSORPTION_BANDS_NM = [(1340.0, 1460.0), (1790.0, 1960.0)]

#: Denominator magnitude below which a percent change is set to NaN (guards the
#: near-zero reflectance in absorption bands from producing +/-inf).
_EPS = 1e-6


# ============================================================
# LOADING
# ============================================================
def load_matched_pairs(matched_csv: Path) -> pd.DataFrame:
    """Load the matched-pairs CSV, returning only the TRUE (matched) rows.

    Parameters
    ----------
    matched_csv : Path
        ``{PICS}_{PATCH}_matched_pairs.csv`` from ``run_id_matched_pairs``.

    Returns
    -------
    pd.DataFrame
        Matched-only rows, with ``t1_scene_dt`` / ``t2_scene_dt`` coerced to
        datetimes. Empty (with the right columns) if none are matched.
    """
    df = pd.read_csv(matched_csv)
    is_matched = df["matched_pair"].astype(str).str.strip().str.upper().eq("TRUE")
    df = df[is_matched].copy()
    df["t1_scene_dt"] = pd.to_datetime(df["t1_scene_dt"])
    df["t2_scene_dt"] = pd.to_datetime(df["t2_scene_dt"])
    return df.reset_index(drop=True)


def _perchannel_path(stats_dir: Path, pics_id: str, patch_id: str,
                     ts: str, product: str) -> Path:
    """Path to a per-acquisition per-channel stats CSV (product = 'rfl'|'rad')."""
    return stats_dir / f"{pics_id}_{patch_id}_{ts}_perchannel_{product}.csv"


def load_scene_spectrum(scene_ur: str, pics_id: str, patch_id: str,
                        rfl_dir: Path, rad_dir: Path) -> pd.DataFrame | None:
    """Load one scene's median/mean spectrum for both products, keyed by wavelength.

    The scene is located by its acquisition timestamp (extracted from the RAD
    GranuleUR *scene_ur*), which names both per-channel CSVs. Radiance is
    required (it anchors the join); reflectance is optional — its columns are
    NaN when that scene has no RFL per-channel CSV (absent product, or screened
    out upstream).

    Parameters
    ----------
    scene_ur : str
        The scene's RAD GranuleUR (the ``scene_id`` from the matched-pairs CSV).
    pics_id, patch_id : str
        Identify the per-channel CSV filenames.
    rfl_dir, rad_dir : Path
        Directories holding the reflectance / radiance per-channel CSVs.

    Returns
    -------
    pd.DataFrame or None
        Columns ``central_wl, median_rfl, mean_rfl, median_rad, mean_rad``.
        ``None`` if the radiance CSV is missing (the scene cannot be used).
    """
    ts = acq_timestamp(scene_ur)

    rad_path = _perchannel_path(rad_dir, pics_id, patch_id, ts, "rad")
    if not rad_path.exists():
        return None
    rad = pd.read_csv(rad_path)[["central_wl", "median_rad", "mean_rad"]]

    rfl_path = _perchannel_path(rfl_dir, pics_id, patch_id, ts, "rfl")
    if rfl_path.exists():
        rfl = pd.read_csv(rfl_path)[["central_wl", "median_rfl", "mean_rfl"]]
        spec = rad.merge(rfl, on="central_wl", how="outer")
    else:
        spec = rad.assign(median_rfl=np.nan, mean_rfl=np.nan)

    return spec.sort_values("central_wl").reset_index(drop=True)


# ============================================================
# COMPUTATION
# ============================================================
def absorption_band_flag(wavelengths: np.ndarray) -> np.ndarray:
    """Boolean mask: True where a wavelength falls in a deep-absorption region."""
    wl = np.asarray(wavelengths, dtype=float)
    flag = np.zeros(wl.shape, dtype=bool)
    for lo, hi in ABSORPTION_BANDS_NM:
        flag |= (wl >= lo) & (wl <= hi)
    return flag


def _pct_change(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Percent change 100*(t2 - t1)/t1, with near-zero / non-finite -> NaN."""
    a = np.asarray(t1, dtype=float)
    b = np.asarray(t2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (b - a) / a
    out[np.abs(a) < _EPS] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def compute_pair_delta(t1_spec: pd.DataFrame, t2_spec: pd.DataFrame,
                       meta: dict) -> pd.DataFrame:
    """Build the per-wavelength delta rows for one matched pair.

    Parameters
    ----------
    t1_spec, t2_spec : pd.DataFrame
        Per-scene spectra from :func:`load_scene_spectrum` (earlier scene is
        *t1*). Joined on ``central_wl`` so only shared wavelengths are kept.
    meta : dict
        Scalar pair metadata broadcast onto every row: ``PICS_ID``,
        ``patch_id``, ``pair_id``, ``t1_scene_id``, ``t2_scene_id``, ``t1_dt``,
        ``t2_dt``, ``dt_days``, ``delta_SZA``, ``delta_VZA``.

    Returns
    -------
    pd.DataFrame
        One row per shared wavelength, columns in :data:`DELTA_CSV_COLS` order.
    """
    m = t1_spec.merge(t2_spec, on="central_wl", suffixes=("_t1", "_t2"))
    wl = m["central_wl"].to_numpy()

    out = pd.DataFrame({
        "PICS_ID": meta["PICS_ID"],
        "patch_id": meta["patch_id"],
        "pair_id": meta["pair_id"],
        "t1_scene_id": meta["t1_scene_id"],
        "t2_scene_id": meta["t2_scene_id"],
        "t1_dt": meta["t1_dt"],
        "t2_dt": meta["t2_dt"],
        "dt_days": meta["dt_days"],
        "delta_SZA": meta["delta_SZA"],
        "delta_VZA": meta["delta_VZA"],
        "wavelength": wl,
        "rfl_t1_median": m["median_rfl_t1"].to_numpy(),
        "rfl_t2_median": m["median_rfl_t2"].to_numpy(),
        "rfl_t1_mean": m["mean_rfl_t1"].to_numpy(),
        "rfl_t2_mean": m["mean_rfl_t2"].to_numpy(),
        "delta_rfl_pct_median": _pct_change(m["median_rfl_t1"], m["median_rfl_t2"]),
        "delta_rfl_pct_mean": _pct_change(m["mean_rfl_t1"], m["mean_rfl_t2"]),
        "rad_t1_median": m["median_rad_t1"].to_numpy(),
        "rad_t2_median": m["median_rad_t2"].to_numpy(),
        "rad_t1_mean": m["mean_rad_t1"].to_numpy(),
        "rad_t2_mean": m["mean_rad_t2"].to_numpy(),
        "delta_rad_pct_median": _pct_change(m["median_rad_t1"], m["median_rad_t2"]),
        "delta_rad_pct_mean": _pct_change(m["mean_rad_t1"], m["mean_rad_t2"]),
        "absorption_band_flag": absorption_band_flag(wl),
    })
    return out[DELTA_CSV_COLS]


def pair_metadata(row: pd.Series, pics_id: str, patch_id: str) -> dict:
    """Assemble the scalar per-pair metadata dict from a matched-pairs row."""
    t1_dt, t2_dt = row["t1_scene_dt"], row["t2_scene_dt"]
    t1_ts = acq_timestamp(row["t1_scene_id"])
    t2_ts = acq_timestamp(row["t2_scene_id"])
    return {
        "PICS_ID": pics_id,
        "patch_id": patch_id,
        "pair_id": f"{t1_ts}_{t2_ts}",
        "t1_scene_id": row["t1_scene_id"],
        "t2_scene_id": row["t2_scene_id"],
        "t1_dt": t1_dt,
        "t2_dt": t2_dt,
        "dt_days": (t2_dt - t1_dt).total_seconds() / 86400.0,
        "delta_SZA": row.get("delta_SZA", np.nan),
        "delta_VZA": row.get("delta_VZA", np.nan),
    }
