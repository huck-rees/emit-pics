"""Identify matched pairs of EMIT acquisitions over a single patch.

A *matched pair* is two EMIT acquisitions whose in-patch mean solar and view
geometry are similar (``|ΔSZA| <= SZA_TOL`` and ``|ΔVZA| <= VZA_TOL``).  The
geometry is streamed from the EMIT L1B OBS file (the OBS asset that lives in
each L1B Radiance granule); because SZA/VZA come from that shared OBS geometry,
matching is identical for the L1B_RAD and L2A_RFL products.

Pure function library — no top-level execution, no hardcoded parameters,
no ``__main__`` block, and zero side effects on import.  ``earthaccess.login``
is *not* called here; the caller authenticates before invoking the streaming
functions.

Typical usage from a runner script::

    from emit_pics.id_matched_pairs import (
        load_patch_dict, split_products, build_acquisitions,
        load_patch_geom_4326, fetch_granule, scene_geometry_stats,
        find_matched_pairs, plot_matched_pairs,
    )
"""

import re
import time
from datetime import datetime
from pathlib import Path

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.spatial import cKDTree

# ============================================================
# CONSTANTS
# ============================================================
#: OBS band labels, looked up by name (never by hardcoded index).
SZA_LABEL = "To-sun zenith (0 to 90 degrees from zenith)"
VZA_LABEL = "To-sensor zenith (0 to 90 degrees from zenith)"

#: EMIT fill value used in OBS / location arrays.
FILL = -9999.0

#: Substring identifying the acquisition timestamp inside a GranuleUR.
_TS_RE = re.compile(r"_(\d{8}T\d{6})_")

#: Column order of the matched-pairs CSV.
PAIR_CSV_COLS = [
    "t1_scene_id", "t1_scene_dt", "t1_scene_SZA", "t1_scene_VZA",
    "t2_scene_id", "t2_scene_dt", "t2_scene_SZA", "t2_scene_VZA",
    "delta_SZA", "delta_VZA", "delta_time", "matched_pair",
]

#: Per-granule geometry stat columns appended to the patch–granule dict.
STAT_COLS = [
    "sza_min", "sza_std", "sza_mean", "sza_max",
    "vza_min", "vza_std", "vza_mean", "vza_max",
]


# ============================================================
# IO HELPERS
# ============================================================
def sniff_delimiter(path: Path) -> str:
    """Return ``"\\t"`` if the file's header line contains a tab, else ``","``.

    Lets a single reader handle both the tab-separated patch–granule dict and
    comma-separated attribute tables without corrupting either on write-back.
    """
    with open(path, "r", newline="") as fh:
        header = fh.readline()
    return "\t" if "\t" in header else ","


def load_patch_dict(dict_path: Path, sep: str = "\t") -> pd.DataFrame:
    """Load the patch–granule dictionary CSV.

    Parameters
    ----------
    dict_path : Path
        Path to ``patch_EMIT_dict_{PICS_ID}.csv`` (tab-separated by default).
    sep : str
        Field delimiter; pass the result of :func:`sniff_delimiter` to be
        robust to comma- or tab-separated files.

    Returns
    -------
    pd.DataFrame
        Columns: ``patch_id``, ``size_km``, ``product``, ``granule_ur``.
    """
    return pd.read_csv(dict_path, sep=sep)


def split_products(dict_df: pd.DataFrame, patch_id: str) -> tuple[list[str], list[str]]:
    """Split a patch's granule URs into (L1B_RAD, L2A_RFL) lists.

    Parameters
    ----------
    dict_df : pd.DataFrame
        Full patch–granule dictionary (all patches).
    patch_id : str
        Patch to select (e.g. ``"40_0_0"``).

    Returns
    -------
    tuple of list of str
        ``(rad_urs, rfl_urs)`` — GranuleUR strings for each product.
    """
    sub = dict_df[dict_df["patch_id"] == patch_id]
    rad = sub[sub["product"] == "L1B_RAD"]["granule_ur"].tolist()
    rfl = sub[sub["product"] == "L2A_RFL"]["granule_ur"].tolist()
    return rad, rfl


def load_patch_geom_4326(bnds_path: Path, patch_id: str):
    """Load one patch polygon and reproject it to EPSG:4326.

    The GeoPackage CRS (typically UTM) is preserved on disk; only a copy is
    reprojected to EPSG:4326 for the lon/lat pixel-membership test.

    Parameters
    ----------
    bnds_path : Path
        Path to ``patches_{PICS_ID}_bnds.gpkg``.
    patch_id : str
        Value of the ``patch_id`` column to select.

    Returns
    -------
    shapely.geometry.Polygon
        The patch polygon in EPSG:4326 (lon/lat).

    Raises
    ------
    ValueError
        If *patch_id* is absent, listing the available ids.
    """
    gdf = gpd.read_file(bnds_path)
    sub = gdf[gdf["patch_id"] == patch_id]
    if sub.empty:
        available = sorted(gdf["patch_id"].unique())
        raise ValueError(
            f"patch_id '{patch_id}' not found in {bnds_path}. "
            f"Available: {available}"
        )
    return sub.to_crs("EPSG:4326").geometry.iloc[0]


# ============================================================
# TIMESTAMPS & COINCIDENCE
# ============================================================
def acq_timestamp(granule_ur: str) -> str:
    """Extract the ``YYYYMMDDThhmmss`` acquisition timestamp from a GranuleUR.

    Raises
    ------
    ValueError
        If no ``_(\\d{8}T\\d{6})_`` substring is present.
    """
    m = _TS_RE.search(granule_ur)
    if not m:
        raise ValueError(f"No acquisition timestamp found in '{granule_ur}'.")
    return m.group(1)


def acq_datetime(granule_ur: str) -> datetime:
    """Parse the acquisition timestamp of a GranuleUR into a ``datetime``."""
    return datetime.strptime(acq_timestamp(granule_ur), "%Y%m%dT%H%M%S")


def build_acquisitions(rad_urs: list[str], rfl_urs: list[str]) -> list[dict]:
    """Build the list of acquisitions to process from the RAD granules.

    Geometry is always pulled from the RAD granule (OBS lives in L1B); the RFL
    list is only checked for *presence*.  An acquisition is "coincident" when
    its timestamp appears in both the RAD and RFL lists.

    Parameters
    ----------
    rad_urs, rfl_urs : list of str
        GranuleUR strings for the L1B_RAD and L2A_RFL products.

    Returns
    -------
    list of dict
        One dict per RAD granule, sorted by acquisition time, with keys
        ``granule_ur``, ``timestamp`` (str), ``datetime``, and ``has_rfl``
        (bool — whether a coincident RFL granule exists).
    """
    rfl_ts = {acq_timestamp(u) for u in rfl_urs}
    acqs = []
    for ur in rad_urs:
        ts = acq_timestamp(ur)
        acqs.append({
            "granule_ur": ur,
            "timestamp": ts,
            "datetime": acq_datetime(ur),
            "has_rfl": ts in rfl_ts,
        })
    acqs.sort(key=lambda a: a["datetime"])
    return acqs


# ============================================================
# EARTHACCESS GRANULE LOOKUP
# ============================================================
def fetch_granule(short_name: str, granule_ur: str):
    """Resolve a GranuleUR to its earthaccess result object via CMR.

    The caller must have authenticated (``earthaccess.login``) first.

    Parameters
    ----------
    short_name : str
        Collection short name (e.g. ``"EMITL1BRAD"``).
    granule_ur : str
        Exact GranuleUR to fetch.

    Returns
    -------
    earthaccess result object
        The granule whose ``GranuleUR`` matches, exposing ``data_links()``.

    Raises
    ------
    ValueError
        If CMR returns no granule for *granule_ur*.
    """
    results = earthaccess.search_data(short_name=short_name,
                                      granule_name=granule_ur)
    for r in results:
        if r["umm"].get("GranuleUR") == granule_ur:
            return r
    if results:
        return results[0]
    raise ValueError(f"Granule not found in CMR: {granule_ur}")


def find_obs_url(granule) -> str:
    """Return the data link in *granule* whose URL contains ``"OBS"``.

    Raises
    ------
    ValueError
        If no OBS asset is present among the granule's links.
    """
    links = granule.data_links()
    obs = [u for u in links if "OBS" in u]
    if not obs:
        raise ValueError(f"No OBS file among granule links: {links}")
    return obs[0]


# ============================================================
# OBS STREAMING
# ============================================================
def band_index(band_labels, label: str) -> int:
    """Return the index of *label* within the observation-band label array.

    Looking the band up by label (never a hardcoded index) keeps SZA/VZA
    selection robust to band-order changes.

    Raises
    ------
    ValueError
        If *label* is not present, listing the available labels.
    """
    labels = [str(b) for b in np.asarray(band_labels).ravel()]
    for i, b in enumerate(labels):
        if b == label:
            return i
    raise ValueError(f"Band label '{label}' not found. Available: {labels}")


def in_patch_mask(lon: np.ndarray, lat: np.ndarray,
                  patch_geom_4326, pad: float = 1.10) -> np.ndarray:
    """Boolean mask of pixels whose lon/lat fall inside the patch polygon.

    A KDTree disc pre-filter (radius = patch half-diagonal × *pad*, centered on
    the patch centroid) narrows candidates before an exact
    :func:`shapely.contains_xy` test.  Pixels with ``-9999`` fill in lon or lat
    are excluded.

    Parameters
    ----------
    lon, lat : np.ndarray
        2-D longitude/latitude arrays from the OBS ``location`` group
        (EPSG:4326).
    patch_geom_4326 : shapely Polygon
        Patch polygon in EPSG:4326.
    pad : float
        Multiplier applied to the half-diagonal pre-filter radius.

    Returns
    -------
    np.ndarray
        2-D boolean array (same shape as *lon*); ``True`` inside the patch.
    """
    shape = lon.shape
    lon_f = np.asarray(lon, dtype=float).ravel()
    lat_f = np.asarray(lat, dtype=float).ravel()

    valid = ~(np.isclose(lon_f, FILL) | np.isclose(lat_f, FILL))
    idx_valid = np.flatnonzero(valid)
    mask_flat = np.zeros(lon_f.size, dtype=bool)
    if idx_valid.size == 0:
        return mask_flat.reshape(shape)

    pts = np.column_stack([lon_f[idx_valid], lat_f[idx_valid]])

    minx, miny, maxx, maxy = patch_geom_4326.bounds
    centroid = patch_geom_4326.centroid
    radius = 0.5 * np.hypot(maxx - minx, maxy - miny) * pad

    tree = cKDTree(pts)
    cand = np.asarray(
        tree.query_ball_point([centroid.x, centroid.y], radius), dtype=int
    )
    if cand.size:
        inside = shapely.contains_xy(
            patch_geom_4326, pts[cand, 0], pts[cand, 1]
        )
        mask_flat[idx_valid[cand[inside]]] = True

    return mask_flat.reshape(shape)


def _dist_stats(values: np.ndarray, prefix: str) -> dict:
    """Mean/std/min/max and 25/50/75 percentiles of *values* (NaNs dropped)."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    keys = ["mean", "std", "min", "max", "p25", "p50", "p75"]
    if a.size == 0:
        return {f"{prefix}_{k}": np.nan for k in keys}
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std":  float(np.std(a)),
        f"{prefix}_min":  float(np.min(a)),
        f"{prefix}_max":  float(np.max(a)),
        f"{prefix}_p25":  float(np.percentile(a, 25)),
        f"{prefix}_p50":  float(np.percentile(a, 50)),
        f"{prefix}_p75":  float(np.percentile(a, 75)),
    }


def open_obs(granule):
    """Open the OBS file in *granule* once; return the shared handle + groups.

    The ``obs`` variable and the ``location`` / ``sensor_band_parameters``
    groups are all opened against the *same* file handle, satisfying the
    "open once, read both from the same handle" requirement.

    Parameters
    ----------
    granule : earthaccess result object
        The L1B_RAD granule containing the OBS asset.

    Returns
    -------
    tuple
        ``(fh, ds_obs, ds_loc, ds_bands)`` — the fsspec file handle, the root
        dataset (holding ``obs``), the ``location`` group, and the
        ``sensor_band_parameters`` group.  The caller must close *fh*.
    """
    url = find_obs_url(granule)
    fh = earthaccess.open([url])[0]
    ds_obs = xr.open_dataset(fh, engine="h5netcdf")
    ds_loc = xr.open_dataset(fh, engine="h5netcdf", group="location")
    ds_bands = xr.open_dataset(
        fh, engine="h5netcdf", group="sensor_band_parameters"
    )
    return fh, ds_obs, ds_loc, ds_bands


def open_obs_with_retry(granule, n_retries: int = 3, backoff: int = 5):
    """Call :func:`open_obs`, retrying transient server errors with backoff.

    LP DAAC intermittently returns 502/503/timeout on long runs; the open is
    retried *n_retries* times with increasing sleeps (``backoff*1``,
    ``backoff*2``, ...) before the error is propagated.

    Parameters
    ----------
    granule : earthaccess result object
        The L1B_RAD granule to open.
    n_retries : int
        Number of attempts before giving up.
    backoff : int
        Base sleep (seconds); attempt *k* sleeps ``backoff * (k + 1)``.

    Returns
    -------
    tuple
        Same as :func:`open_obs`.
    """
    for attempt in range(n_retries):
        try:
            return open_obs(granule)
        except Exception as ex:
            if attempt < n_retries - 1:
                print(f"    retry {attempt + 1}/{n_retries} after "
                      f"{type(ex).__name__}: {ex}")
                time.sleep(backoff * (attempt + 1))
            else:
                raise


def scene_geometry_stats(granule, patch_geom_4326,
                         n_retries: int = 3, backoff: int = 5) -> dict:
    """Stream OBS geometry for one granule and summarize it over the patch.

    The OBS file is opened **once** (with retry/backoff); the ``obs`` variable
    and the ``location`` group are read from that same handle.  SZA/VZA bands
    are located by label.  Over the in-patch pixels, ``-9999`` fills are set to
    NaN before computing per-scene statistics.

    Parameters
    ----------
    granule : earthaccess result object
        The L1B_RAD granule containing the OBS asset.
    patch_geom_4326 : shapely Polygon
        Patch polygon in EPSG:4326 for the pixel-membership test.
    n_retries, backoff : int
        Passed to :func:`open_obs_with_retry`.

    Returns
    -------
    dict
        ``sza_*`` and ``vza_*`` stats (mean/std/min/max/p25/p50/p75) plus
        ``n_pixels`` (count of in-patch pixels used).
    """
    fh, ds_obs, ds_loc, ds_bands = open_obs_with_retry(
        granule, n_retries=n_retries, backoff=backoff
    )
    try:
        labels = ds_bands["observation_bands"].values
        sza_i = band_index(labels, SZA_LABEL)
        vza_i = band_index(labels, VZA_LABEL)

        lon = np.asarray(ds_loc["lon"].values)
        lat = np.asarray(ds_loc["lat"].values)
        mask = in_patch_mask(lon, lat, patch_geom_4326)

        obs = ds_obs["obs"]
        sza = np.asarray(obs[..., sza_i].values)[mask]
        vza = np.asarray(obs[..., vza_i].values)[mask]
    finally:
        for d in (ds_bands, ds_loc, ds_obs):
            try:
                d.close()
            except Exception:
                pass
        try:
            fh.close()
        except Exception:
            pass

    # NaN out fill values before summarizing.
    sza = np.where(np.isclose(sza, FILL), np.nan, sza)
    vza = np.where(np.isclose(vza, FILL), np.nan, vza)

    stats = {}
    stats.update(_dist_stats(sza, "sza"))
    stats.update(_dist_stats(vza, "vza"))
    stats["n_pixels"] = int(mask.sum())
    return stats


# ============================================================
# MATCHED PAIRS
# ============================================================
def find_matched_pairs(scenes: pd.DataFrame,
                       sza_tol: float, vza_tol: float,
                       min_dt_days: float = 0.0) -> pd.DataFrame:
    """Compute all unique scene pairs and flag the matched ones.

    For every unordered pair of processed acquisitions, computes
    ``delta_SZA``, ``delta_VZA`` (absolute mean differences) and ``delta_time``
    (absolute difference in days).  A pair is *matched* when
    ``delta_SZA <= sza_tol`` and ``delta_VZA <= vza_tol`` and
    ``delta_time >= min_dt_days``.

    Parameters
    ----------
    scenes : pd.DataFrame
        One row per acquisition with at least ``scene_id``, ``datetime``,
        ``sza_mean``, ``vza_mean`` and ``has_rfl`` columns.
    sza_tol, vza_tol : float
        Matching tolerances in degrees.
    min_dt_days : float
        Minimum time separation (days) for a pair to count as matched; ``0.0``
        (default) imposes no floor. Near-simultaneous pairs carry a drift signal
        below the per-pair noise floor, so they only add origin scatter.

    Returns
    -------
    pd.DataFrame
        All pairs, with the :data:`PAIR_CSV_COLS` columns plus the helper
        columns ``t1_has_rfl`` / ``t2_has_rfl`` (used to count RFL pairs).
    """
    recs = scenes.to_dict("records")
    rows = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            dsza = abs(a["sza_mean"] - b["sza_mean"])
            dvza = abs(a["vza_mean"] - b["vza_mean"])
            dt_days = abs((a["datetime"] - b["datetime"]).total_seconds()) / 86400.0
            matched = bool(dsza <= sza_tol and dvza <= vza_tol
                           and dt_days >= min_dt_days)
            rows.append({
                "t1_scene_id": a["scene_id"], "t1_scene_dt": a["datetime"],
                "t1_scene_SZA": a["sza_mean"], "t1_scene_VZA": a["vza_mean"],
                "t2_scene_id": b["scene_id"], "t2_scene_dt": b["datetime"],
                "t2_scene_SZA": b["sza_mean"], "t2_scene_VZA": b["vza_mean"],
                "delta_SZA": dsza, "delta_VZA": dvza, "delta_time": dt_days,
                "matched_pair": matched,
                "t1_has_rfl": a["has_rfl"], "t2_has_rfl": b["has_rfl"],
            })
    cols = PAIR_CSV_COLS + ["t1_has_rfl", "t2_has_rfl"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================
# PLOTTING
# ============================================================
def plot_matched_pairs(scenes: pd.DataFrame, matched: pd.DataFrame,
                       patch_id: str, sza_tol: float, vza_tol: float):
    """Build the two-panel SZA/VZA-over-time figure with matched-pair links.

    Each scene is drawn as a box-and-whisker (per-scene percentiles) at its
    acquisition date, with a black dot at the mean; red lines connect the two
    scenes of every matched pair on both panels.  The figure is returned, not
    saved — the caller decides.  ``matplotlib`` is imported lazily.

    Parameters
    ----------
    scenes : pd.DataFrame
        Per-scene stats with ``datetime``, ``sza_*``/``vza_*`` percentile
        columns and ``sza_mean``/``vza_mean``.
    matched : pd.DataFrame
        The matched-only pairs (``t1_scene_dt``, ``t2_scene_dt``,
        ``t1_scene_SZA``/``VZA``, ``t2_scene_SZA``/``VZA``).
    patch_id : str
        Patch label for titles.
    sza_tol, vza_tol : float
        Tolerances, shown in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    scenes = scenes.sort_values("datetime").reset_index(drop=True)
    positions = mdates.date2num(scenes["datetime"])
    box_w = 6  # days

    def bxp_from(df, pfx):
        return [{"med": r[f"{pfx}_p50"], "q1": r[f"{pfx}_p25"], "q3": r[f"{pfx}_p75"],
                 "whislo": r[f"{pfx}_min"], "whishi": r[f"{pfx}_max"], "fliers": []}
                for _, r in df.iterrows()]

    fig, (axS, axV) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axS.bxp(bxp_from(scenes, "sza"), positions=positions, widths=box_w, showfliers=False,
            patch_artist=True, boxprops=dict(facecolor="C1", alpha=0.5),
            medianprops=dict(color="black"))
    axS.set_ylabel("SZA (degrees)")
    axS.set_title(f"{patch_id} — in-patch SZA distribution per scene")
    axV.bxp(bxp_from(scenes, "vza"), positions=positions, widths=box_w, showfliers=False,
            patch_artist=True, boxprops=dict(facecolor="C0", alpha=0.5),
            medianprops=dict(color="black"))
    axV.set_ylabel("VZA (degrees)")
    axV.set_title(f"{patch_id} — in-patch VZA distribution per scene")

    dt_to_pos = dict(zip(scenes["datetime"], positions))
    for _, r in matched.iterrows():
        x1, x2 = dt_to_pos[r["t1_scene_dt"]], dt_to_pos[r["t2_scene_dt"]]
        axS.plot([x1, x2], [r["t1_scene_SZA"], r["t2_scene_SZA"]],
                 color="red", lw=0.8, alpha=0.5, zorder=5)
        axV.plot([x1, x2], [r["t1_scene_VZA"], r["t2_scene_VZA"]],
                 color="red", lw=0.8, alpha=0.5, zorder=5)
    axS.plot(positions, scenes["sza_mean"], "k.", ms=4, zorder=6)
    axV.plot(positions, scenes["vza_mean"], "k.", ms=4, zorder=6)

    axV.xaxis.set_major_locator(mdates.YearLocator())
    axV.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axV.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axV.set_xlabel("Acquisition date")

    fig.suptitle(f"{patch_id} — patch SZA/VZA over time (red lines = matched pairs, "
                 f"|ΔSZA|≤{sza_tol}°, |ΔVZA|≤{vza_tol}°)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
