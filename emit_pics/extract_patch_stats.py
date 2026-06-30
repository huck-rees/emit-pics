"""Extract per-channel L1B radiance / L2A reflectance stats over a patch.

For one patch, this library streams every covering EMIT acquisition and
summarizes the in-patch pixels of each product, channel by channel.  It is
*extraction only* — no differencing or matched-pair logic.

Pure function library — no top-level execution, no hardcoded parameters,
no ``__main__`` block, and zero side effects on import.  ``earthaccess.login``
is *not* called here; the caller authenticates before streaming.

Low-level geometry/IO helpers proven in :mod:`emit_pics.id_matched_pairs`
(timestamp parsing, band-by-label lookup, the KDTree patch mask, granule
lookup) are reused rather than duplicated.
"""

import os
import time
import warnings
from datetime import datetime

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

from emit_pics.id_matched_pairs import acq_timestamp, band_index

# ============================================================
# CONSTANTS
# ============================================================
#: Fill values to NaN out before computing stats (np.isclose comparison).
RFL_FILLS = (-9999.0, -0.01)   # L2A reflectance
RAD_FILLS = (-9999.0,)         # L1B radiance

#: L2A MASK band located by label (never a hardcoded index).
AGG_FLAG_LABEL = "Aggregate Flag"

#: Asset-URL tokens identifying each file within a granule.
RFL_TOKEN  = "_RFL_"    # L2A reflectance (excludes _RFLUNCERT_)
RAD_TOKEN  = "_RAD_"    # L1B radiance (excludes _OBS_, _GLT_)
MASK_TOKEN = "MASK"     # L2A mask

#: Output column order per product (central_wl first, no channel index).
_STAT_KEYS = ["min", "1qt", "median", "3qt", "max", "mean", "std"]


# ============================================================
# DICT LOADING & ACQUISITION PAIRING
# ============================================================
def load_patch_dict(dict_path, sep: str = "\t") -> pd.DataFrame:
    """Load the tab-separated patch–granule dictionary CSV."""
    return pd.read_csv(dict_path, sep=sep)


def split_products(dict_df: pd.DataFrame, patch_id: str) -> tuple[list[str], list[str]]:
    """Return ``(rad_urs, rfl_urs)`` GranuleURs for *patch_id*."""
    sub = dict_df[dict_df["patch_id"] == patch_id]
    rad = sub[sub["product"] == "L1B_RAD"]["granule_ur"].tolist()
    rfl = sub[sub["product"] == "L2A_RFL"]["granule_ur"].tolist()
    return rad, rfl


def pair_acquisitions(rad_urs: list[str], rfl_urs: list[str]) -> list[dict]:
    """Pair RAD and RFL granules by acquisition timestamp.

    Parameters
    ----------
    rad_urs, rfl_urs : list of str
        L1B_RAD and L2A_RFL GranuleURs for the patch.

    Returns
    -------
    list of dict
        One dict per acquisition (sorted by time) with keys ``timestamp``
        (``YYYYMMDDThhmmss``), ``datetime``, ``rad_ur`` and ``rfl_ur`` — either
        UR is ``None`` if that product is absent for the acquisition.
    """
    rad_by = {acq_timestamp(u): u for u in rad_urs}
    rfl_by = {acq_timestamp(u): u for u in rfl_urs}
    out = []
    for ts in sorted(set(rad_by) | set(rfl_by)):
        out.append({
            "timestamp": ts,
            "datetime": datetime.strptime(ts, "%Y%m%dT%H%M%S"),
            "rad_ur": rad_by.get(ts),
            "rfl_ur": rfl_by.get(ts),
        })
    return out


# ============================================================
# RETRY WRAPPER
# ============================================================
def with_retry(open_fn, n_retries: int = 3, backoff: int = 5, label: str = ""):
    """Call *open_fn* (zero-arg), retrying transient server errors w/ backoff.

    LP DAAC intermittently returns 502/503/timeouts on long runs; the call is
    retried *n_retries* times with increasing sleeps (``backoff*1``,
    ``backoff*2``, ...) before the error is propagated.
    """
    for attempt in range(n_retries):
        try:
            return open_fn()
        except Exception as ex:
            if attempt < n_retries - 1:
                print(f"    retry {attempt + 1}/{n_retries} {label} after "
                      f"{type(ex).__name__}: {ex}")
                time.sleep(backoff * (attempt + 1))
            else:
                raise


# ============================================================
# GRANULE OPENERS
# ============================================================
def find_asset_url(granule, token: str) -> str:
    """Return the data link in *granule* whose **file name** contains *token*.

    Matching is against the URL basename only, not the full URL: every file in
    a granule lives under a directory named after the granule itself (e.g.
    ``.../EMIT_L2A_RFL_001_<ts>_.../``), so a token like ``_RFL_`` or ``_RAD_``
    appears in the directory portion of *every* sibling file's URL.  Restricting
    the match to the basename uniquely selects the intended file (e.g. ``_RFL_``
    matches ``EMIT_L2A_RFL_...`` but not ``EMIT_L2A_RFLUNCERT_...``).
    """
    links = granule.data_links()
    hit = [u for u in links if token in u.split("/")[-1]]
    if not hit:
        raise ValueError(
            f"No asset whose file name contains '{token}' among "
            f"granule links: {links}"
        )
    return hit[0]


def open_l2a(granule, want_agg: bool) -> dict:
    """Open the L2A reflectance file (and, if requested, the MASK file) once.

    Reflectance, wavelengths and lon/lat are read from the single reflectance
    handle; the Aggregate-Flag band is read from the MASK file when *want_agg*.

    Parameters
    ----------
    granule : earthaccess result object
        The L2A_RFL granule.
    want_agg : bool
        Whether to also open the MASK file and expose the Aggregate-Flag band.

    Returns
    -------
    dict
        ``refl`` (lazy DataArray, bands last), ``wl`` (1-D wavelengths),
        ``lon``/``lat`` (2-D), ``agg`` (lazy 2-D Aggregate-Flag DataArray or
        ``None``), and ``_handles`` (objects to close).
    """
    rfl_url = find_asset_url(granule, RFL_TOKEN)
    fh = earthaccess.open([rfl_url])[0]
    ds_rfl = xr.open_dataset(fh, engine="h5netcdf")
    ds_wl = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")
    ds_loc = xr.open_dataset(fh, engine="h5netcdf", group="location")

    out = {
        "refl": ds_rfl["reflectance"],
        "wl": np.asarray(ds_wl["wavelengths"].values),
        "lon": np.asarray(ds_loc["lon"].values),
        "lat": np.asarray(ds_loc["lat"].values),
        "agg": None,
        "_handles": [ds_rfl, ds_wl, ds_loc, fh],
    }

    if want_agg:
        mask_url = find_asset_url(granule, MASK_TOKEN)
        fh_m = earthaccess.open([mask_url])[0]
        ds_m = xr.open_dataset(fh_m, engine="h5netcdf")
        ds_mb = xr.open_dataset(
            fh_m, engine="h5netcdf", group="sensor_band_parameters"
        )
        agg_i = band_index(ds_mb["mask_bands"].values, AGG_FLAG_LABEL)
        out["agg"] = ds_m["mask"][..., agg_i]
        out["_handles"] += [ds_m, ds_mb, fh_m]

    return out


def open_l1b(granule, want_loc: bool = False) -> dict:
    """Open the L1B radiance file once.

    Parameters
    ----------
    granule : earthaccess result object
        The L1B_RAD granule.
    want_loc : bool
        Also read lon/lat from the radiance file's ``location`` group (only
        needed when the matching L2A is absent, since RAD and RFL share the
        (downtrack, crosstrack) grid).

    Returns
    -------
    dict
        ``rad`` (lazy DataArray, bands last), ``wl`` (1-D wavelengths),
        optionally ``lon``/``lat`` (2-D), and ``_handles`` (objects to close).
    """
    rad_url = find_asset_url(granule, RAD_TOKEN)
    fh = earthaccess.open([rad_url])[0]
    ds_rad = xr.open_dataset(fh, engine="h5netcdf")
    ds_wl = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")

    out = {
        "rad": ds_rad["radiance"],
        "wl": np.asarray(ds_wl["wavelengths"].values),
        "_handles": [ds_rad, ds_wl, fh],
    }
    if want_loc:
        ds_loc = xr.open_dataset(fh, engine="h5netcdf", group="location")
        out["lon"] = np.asarray(ds_loc["lon"].values)
        out["lat"] = np.asarray(ds_loc["lat"].values)
        out["_handles"].append(ds_loc)
    return out


def close_handles(handles) -> None:
    """Close every dataset/file handle, ignoring individual close errors."""
    for h in handles:
        try:
            h.close()
        except Exception:
            pass


# ============================================================
# WINDOWING
# ============================================================
def window_bounds(mask: np.ndarray):
    """Tight (r0, r1, c0, c1) bounding the ``True`` pixels of *mask*.

    Returns ``None`` if *mask* has no ``True`` pixels.  Slicing a cube to this
    window keeps only a small block in memory instead of the full granule.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def window_2d(da, bounds) -> np.ndarray:
    """Read a 2-D DataArray over *bounds* = (r0, r1, c0, c1) as a NumPy array."""
    r0, r1, c0, c1 = bounds
    return np.asarray(da[r0:r1, c0:c1].values)


def window_pixels(cube_da, bounds, keep_sub: np.ndarray) -> np.ndarray:
    """Read a (downtrack, crosstrack, band) cube over *bounds* and select pixels.

    Parameters
    ----------
    cube_da : xr.DataArray
        Lazy reflectance/radiance cube (bands last).
    bounds : tuple
        ``(r0, r1, c0, c1)`` window from :func:`window_bounds`.
    keep_sub : np.ndarray
        2-D boolean mask over the *window* selecting pixels to keep.

    Returns
    -------
    np.ndarray
        ``(n_kept_pixels, n_bands)`` array of values.
    """
    r0, r1, c0, c1 = bounds
    win = np.asarray(cube_da[r0:r1, c0:c1, :].values)
    return win[keep_sub]


# ============================================================
# PER-CHANNEL STATISTICS
# ============================================================
def per_channel_stats(values: np.ndarray, wavelengths: np.ndarray,
                      fills: tuple, suffix: str) -> pd.DataFrame:
    """Per-channel summary stats over kept in-patch pixels.

    For each channel, *fills* are NaN'd out (``np.isclose``) and
    min / 25th / median / 75th / max / mean / std are computed over the finite
    values (NaN-aware reductions, empty-slice warnings suppressed).

    Parameters
    ----------
    values : np.ndarray
        ``(n_pixels, n_bands)`` kept-pixel values.
    wavelengths : np.ndarray
        ``(n_bands,)`` central wavelengths for the ``central_wl`` column.
    fills : tuple of float
        Fill values to drop (e.g. :data:`RFL_FILLS` / :data:`RAD_FILLS`).
    suffix : str
        Column suffix (``"rad"`` or ``"rfl"``).

    Returns
    -------
    pd.DataFrame
        One row per channel: ``central_wl``, ``min_{suffix}``, ``1qt_{suffix}``,
        ``median_{suffix}``, ``3qt_{suffix}``, ``max_{suffix}``,
        ``mean_{suffix}``, ``std_{suffix}``.
    """
    arr = np.asarray(values, dtype=float)
    for f in fills:
        arr[np.isclose(arr, f)] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cols = {
            f"min_{suffix}":    np.nanmin(arr, axis=0),
            f"1qt_{suffix}":    np.nanpercentile(arr, 25, axis=0),
            f"median_{suffix}": np.nanpercentile(arr, 50, axis=0),
            f"3qt_{suffix}":    np.nanpercentile(arr, 75, axis=0),
            f"max_{suffix}":    np.nanmax(arr, axis=0),
            f"mean_{suffix}":   np.nanmean(arr, axis=0),
            f"std_{suffix}":    np.nanstd(arr, axis=0),
        }

    return pd.DataFrame({
        "central_wl": np.asarray(wavelengths, dtype=float),
        **cols,
    })


# ============================================================
# CSV OUTPUT
# ============================================================
def write_perchannel_csv(df: pd.DataFrame, path) -> None:
    """Write *df* to *path*, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
