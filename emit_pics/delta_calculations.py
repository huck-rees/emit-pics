"""Per-wavelength % change in reflectance/radiance across matched scene pairs.

Pure function library — no top-level execution, no hardcoded paths, no side
effects on import. Given the matched-pair list (from ``id_matched_pairs``) and
the per-acquisition per-channel stats CSVs (from ``run_single_patch_cube_extract``), this
computes, for each matched pair and each wavelength, the pairwise **t1 -> t2**
percent change of the median and mean in-patch value, for both products.

The premise: over a pseudo-invariant calibration site with matched viewing
geometry, the surface is assumed stable, so a non-zero delta is a proxy for
sensor drift (plus residual atmosphere / BRDF). No aggregation or plotting is
done here — the single output is a tidy long table, one row per
``(matched pair x wavelength)``; downstream analysis across all PICS consumes it.

Typical usage from a runner::

    from emit_pics.delta_calculations import (
        load_matched_pairs, load_scene_spectrum, absorption_band_wavelengths,
        compute_pair_delta, DELTA_CSV_COLS,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emit_pics.earth_sun import normalize_radiance_spectrum
from emit_pics.id_matched_pairs import acq_datetime, acq_timestamp

# ============================================================
# CONSTANTS
# ============================================================
#: Output column order of the tidy long delta table.
DELTA_CSV_COLS = [
    "PICS_ID", "patch_id", "pair_id",
    "t1_scene_id", "t2_scene_id", "t1_dt", "t2_dt", "dt_days",
    "SZA_t1", "SZA_t2", "delta_SZA", "delta_VZA",
    "band_index", "wavelength", "wavelength_t2",
    "rfl_t1_median", "rfl_t2_median", "rfl_t1_mean", "rfl_t2_mean",
    "delta_rfl_pct_median", "delta_rfl_pct_mean",
    "rad_t1_median", "rad_t2_median", "rad_t1_mean", "rad_t2_mean",
    "delta_rad_pct_median", "delta_rad_pct_mean",
    "absorption_band_flag",
]

#: *Fallback* deep water-vapor absorption regions (nm). The preferred source for
#: ``absorption_band_flag`` is the data itself — the exact wavelengths where L2A
#: reflectance is fill (see :func:`absorption_band_wavelengths`), which coincide
#: with EMIT's fixed ``good_wavelengths`` bad-bands. These round-number ranges are
#: only used when no reflectance-bearing scene is available to derive the set.
#: Flagged rows are never dropped — the downstream all-PICS analysis decides.
ABSORPTION_BANDS_NM = [(1340.0, 1460.0), (1790.0, 1960.0)]

#: Denominator magnitude below which a percent change is set to NaN (guards the
#: near-zero reflectance in absorption bands from producing +/-inf).
_EPS = 1e-6

#: Sanity bound (nm) on how far a band's centre wavelength may differ between two
#: scenes and still be accepted as the same channel. Scenes are aligned by BAND
#: ORDER (see :func:`compute_pair_delta`), NOT by matching wavelengths — EMIT
#: carries the same 285-band grid in every granule and its calibration drifts
#: only ~0.09 nm between processing eras, far below the ~7.4 nm band spacing. This
#: is only a guard: if any band's cross-scene wavelength gap exceeds it, the grids
#: are deemed non-corresponding and the pair is reported degenerate (never
#: silently misaligned).
WL_MATCH_TOL_NM = 3.0


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


def scene_perchannel_path(scene_ur: str, pics_id: str, patch_id: str,
                          stats_dir: Path, product: str) -> Path:
    """Public: expected per-channel CSV path for a scene UR (product 'rfl'|'rad').

    Resolves the scene's acquisition timestamp from *scene_ur* the same way
    :func:`load_scene_spectrum` does, so runners can report the exact file a scene
    maps to (e.g. naming a missing radiance CSV in a skip message).
    """
    return _perchannel_path(stats_dir, pics_id, patch_id,
                            acq_timestamp(scene_ur), product)


def load_scene_spectrum(scene_ur: str, pics_id: str, patch_id: str,
                        rfl_dir: Path, rad_dir: Path,
                        *, earth_sun_normalize: bool = True) -> pd.DataFrame | None:
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
    earth_sun_normalize : bool, keyword-only
        Divide the radiance columns by the scene's Earth–Sun eccentricity factor
        ``E0`` (Spencer 1971), bringing every scene to a common 1-AU illumination
        so the seasonal ``1/d²`` solar-flux term does not leak into the drift
        (L1B radiance is not distance-normalized; L2A reflectance already is).
        Exact — ``E0`` is a per-scene scalar. Default on.

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

    spec = spec.sort_values("central_wl").reset_index(drop=True)
    if earth_sun_normalize:
        spec = normalize_radiance_spectrum(spec, acq_datetime(scene_ur))
    return spec


# ============================================================
# COMPUTATION
# ============================================================
def absorption_band_wavelengths(spec: pd.DataFrame,
                                rfl_col: str = "median_rfl") -> np.ndarray:
    """Water-vapor bad-band centers, derived from a reflectance-bearing spectrum.

    EMIT's L2A reflectance is set to fill (-> NaN here) in the deep water-vapor
    absorption windows. Those bands are a fixed instrument property — identical
    in every L2A granule — so a single reflectance-bearing scene defines the set
    for the whole patch. Returns the wavelengths where *rfl_col* is NaN.

    Returns an empty array when *spec* has no usable reflectance: an all-NaN
    column means the RFL product was absent (a RAD-only scene), which is a data
    gap, NOT a bad-band pattern — the caller should try another scene or fall back
    to :data:`ABSORPTION_BANDS_NM`.
    """
    if spec is None or rfl_col not in spec or "central_wl" not in spec:
        return np.array([], dtype=float)
    bad = spec[rfl_col].isna().to_numpy()
    if not bad.any() or bad.all():
        # none NaN => no bad bands observed (unexpected); all NaN => RAD-only.
        return np.array([], dtype=float)
    return np.sort(spec.loc[bad, "central_wl"].to_numpy(dtype=float))


def absorption_band_flag(wavelengths: np.ndarray,
                         bad_band_wls: np.ndarray | None = None,
                         tol: float = 0.5) -> np.ndarray:
    """Boolean mask marking deep water-vapor absorption bands.

    Preferred path: pass *bad_band_wls* from :func:`absorption_band_wavelengths`
    (the real fill wavelengths of a reflectance-bearing scene); a wavelength is
    flagged when it matches one within *tol* nm. This is exact — the flag then
    coincides with the reflectance NaNs — and, because it is keyed on wavelength,
    it flags the same bands in *radiance* rows too (radiance is never NaN there,
    but is atmospherically contaminated in those windows).

    Fallback: when *bad_band_wls* is None/empty (no reflectance-bearing scene was
    available), use the fixed :data:`ABSORPTION_BANDS_NM` ranges.
    """
    wl = np.asarray(wavelengths, dtype=float)
    if bad_band_wls is not None and len(bad_band_wls):
        bad = np.asarray(bad_band_wls, dtype=float)
        return (np.abs(wl[:, None] - bad[None, :]) <= tol).any(axis=1)
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


def _empty_delta() -> pd.DataFrame:
    """An empty delta frame with the canonical columns (a degenerate pair)."""
    return pd.DataFrame(columns=DELTA_CSV_COLS)


def compute_pair_delta(t1_spec: pd.DataFrame, t2_spec: pd.DataFrame,
                       meta: dict,
                       bad_band_wls: np.ndarray | None = None) -> pd.DataFrame:
    """Build the per-wavelength delta rows for one matched pair.

    Parameters
    ----------
    t1_spec, t2_spec : pd.DataFrame
        Per-scene spectra from :func:`load_scene_spectrum` (earlier scene is
        *t1*). Aligned by **band order** (band *i* <-> band *i*), NOT by matching
        wavelengths — EMIT scenes share one 285-band grid whose labels drift only
        slightly between calibration eras. ``wavelength`` reports *t1*'s grid (the
        canonical value) and ``wavelength_t2`` reports *t2*'s label for the same
        band, so the alignment is self-checking by eye. Returns an empty frame if
        the grids don't correspond (unequal band count, or a per-band gap beyond
        :data:`WL_MATCH_TOL_NM`).
    meta : dict
        Scalar pair metadata broadcast onto every row: ``PICS_ID``,
        ``patch_id``, ``pair_id``, ``t1_scene_id``, ``t2_scene_id``, ``t1_dt``,
        ``t2_dt``, ``dt_days``, ``SZA_t1``, ``SZA_t2`` (absolute per-scene mean
        SZA), ``delta_SZA``, ``delta_VZA``. A per-band ``band_index`` (0..N-1) is
        added from the sorted grid as a stable cross-scene channel identity.
    bad_band_wls : np.ndarray, optional
        Water-vapor bad-band wavelengths (from
        :func:`absorption_band_wavelengths`) used to set ``absorption_band_flag``.
        If None/empty, the flag falls back to :data:`ABSORPTION_BANDS_NM`.

    Returns
    -------
    pd.DataFrame
        One row per shared wavelength, columns in :data:`DELTA_CSV_COLS` order.
    """
    # Align the two scenes by BAND ORDER, not by matching wavelengths: EMIT
    # carries the same 285 bands in the same order in every granule, so band i is
    # the same channel even though its labelled centre drifts slightly between
    # calibration eras (no dates / per-era logic needed). Guard only: a mismatch
    # in band count, or a per-band wavelength gap beyond mere drift, means the
    # grids don't correspond -> return empty so the runner reports a degenerate
    # pair instead of silently misaligning.
    a = t1_spec.sort_values("central_wl").reset_index(drop=True)
    b = t2_spec.sort_values("central_wl").reset_index(drop=True)
    if a.empty or len(a) != len(b):
        return _empty_delta()
    wl = a["central_wl"].to_numpy()
    wl_t2 = b["central_wl"].to_numpy()          # t2's label for the same band (QC)
    if np.nanmax(np.abs(wl - wl_t2)) > WL_MATCH_TOL_NM:
        return _empty_delta()

    m = pd.concat(
        [a.add_suffix("_t1").rename(columns={"central_wl_t1": "central_wl"}),
         b.drop(columns="central_wl").add_suffix("_t2")],
        axis=1,
    )

    out = pd.DataFrame({
        "PICS_ID": meta["PICS_ID"],
        "patch_id": meta["patch_id"],
        "pair_id": meta["pair_id"],
        "t1_scene_id": meta["t1_scene_id"],
        "t2_scene_id": meta["t2_scene_id"],
        "t1_dt": meta["t1_dt"],
        "t2_dt": meta["t2_dt"],
        "dt_days": meta["dt_days"],
        "SZA_t1": meta["SZA_t1"],
        "SZA_t2": meta["SZA_t2"],
        "delta_SZA": meta["delta_SZA"],
        "delta_VZA": meta["delta_VZA"],
        "band_index": np.arange(len(wl), dtype=int),
        "wavelength": wl,
        "wavelength_t2": wl_t2,
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
        "absorption_band_flag": absorption_band_flag(wl, bad_band_wls),
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
        "SZA_t1": np.abs(row.get("t1_scene_SZA", np.nan)),
        "SZA_t2": np.abs(row.get("t2_scene_SZA", np.nan)),
        "delta_SZA": row.get("delta_SZA", np.nan),
        "delta_VZA": row.get("delta_VZA", np.nan),
    }
