"""Unified per-acquisition extraction over a patch — open once, mask once.

This module consolidates the streaming/masking layer that used to be duplicated
across :mod:`emit_pics.id_matched_pairs` (geometry), :mod:`emit_pics.extract_patch_stats`
(per-channel radiance/reflectance) and :mod:`emit_pics.display_qc_visuals` (QC
rendering).  For one acquisition it:

1. opens each EMIT asset **once** (L2A reflectance + MASK, L1B radiance, L1B OBS),
2. builds the in-patch pixel mask **once** from the shared geolocation grid,
3. windows every array to the in-patch block and selects the polygon-contained
   pixels, returning a single *cube dict* of raw, unscreened per-pixel arrays.

From that one in-memory cube the caller derives everything it needs locally —
per-channel stats, per-scene SZA/VZA geometry stats (with percentiles and the
valid-pixel count), MASK-band summaries — and optionally persists the cube to a
NetCDF store so that future screening and BRDF-correction passes re-process the
*same pixels* with a different transform without ever touching the network.

Design rules (consistent with the rest of the package):

* The cube is stored **raw and unscreened**: only the polygon-containment test is
  applied.  Fill values are kept intact; per-pixel SZA/VZA, lon/lat and every
  MASK band are stored alongside reflectance/radiance.  Aggregate-Flag / AOD
  screening is therefore a *read-time* choice, never baked into the cache.
* Pixels are stored flattened to ``(n_pixels, band)`` — the polygon-contained
  pixels only, not the rectangular bounding window.  Per-pixel ``row``/``col``
  (full-scene downtrack/crosstrack indices) are stored too, so a 2-D image can be
  reconstructed later (e.g. for QC rendering from cache).

Low-level helpers proven elsewhere are reused rather than duplicated; only the
two granule openers that previously lived in the matplotlib-importing
:mod:`emit_pics.display_qc_visuals` are re-stated here so this module stays free
of any plotting dependency.
"""

from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from emit_pics.id_matched_pairs import (
    FILL,
    SZA_LABEL,
    VZA_LABEL,
    _dist_stats,
    band_index,
    fetch_granule,
    in_patch_mask,
)
from emit_pics.extract_patch_stats import (
    AGG_FLAG_LABEL,
    MASK_TOKEN,
    OBS_TOKEN,
    RAD_FILLS,
    RFL_FILLS,
    RFL_TOKEN,
    close_handles,
    find_asset_url,
    open_l1b,
    per_channel_stats,
    window_2d,
    window_bounds,
    window_pixels,
    with_retry,
)

# ============================================================
# CONSTANTS
# ============================================================
#: MASK band labels that are binary flags (summarized as a flagged percentage).
#: Any MASK band whose label is NOT in this set is treated as continuous.
BINARY_FLAGS = {
    "Cloud flag", "Cirrus flag", "Water flag",
    "Spacecraft Flag", "Dilated Cloud Flag", "Aggregate Flag",
}


# ============================================================
# GRANULE OPENERS (neutral; no plotting dependency)
# ============================================================
def open_rfl_mask(granule) -> dict:
    """Open the L2A reflectance file and its sibling MASK file once.

    Reflectance / wavelengths / lon / lat come from the reflectance handle; the
    full mask cube and its band labels come from the MASK handle.  The whole mask
    band set is exposed so downstream code can summarize or cache every band.

    Returns a dict with ``refl`` (lazy cube, bands last), ``wl`` (1-D), ``lon`` /
    ``lat`` (2-D), ``mask_cube`` (lazy, bands last), ``mask_labels`` (list of str)
    and ``_handles`` (objects to close).
    """
    rfl_url = find_asset_url(granule, RFL_TOKEN)
    fh = earthaccess.open([rfl_url])[0]
    ds_rfl = xr.open_dataset(fh, engine="h5netcdf")
    ds_wl = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")
    ds_loc = xr.open_dataset(fh, engine="h5netcdf", group="location")

    mask_url = find_asset_url(granule, MASK_TOKEN)
    fh_m = earthaccess.open([mask_url])[0]
    ds_m = xr.open_dataset(fh_m, engine="h5netcdf")
    ds_mb = xr.open_dataset(fh_m, engine="h5netcdf", group="sensor_band_parameters")

    return {
        "refl": ds_rfl["reflectance"],
        "wl": np.asarray(ds_wl["wavelengths"].values, dtype=float),
        "lon": np.asarray(ds_loc["lon"].values, dtype=float),
        "lat": np.asarray(ds_loc["lat"].values, dtype=float),
        "mask_cube": ds_m["mask"],
        "mask_labels": [str(x) for x in ds_mb["mask_bands"].values],
        "_handles": [ds_rfl, ds_wl, ds_loc, fh, ds_m, ds_mb, fh_m],
    }


def open_obs(granule) -> dict:
    """Open the bundled OBS geometry file from an L1B_RAD granule once.

    OBS shares the (downtrack, crosstrack) grid with RAD/RFL, so lon/lat are not
    re-read here; the caller reuses the reflectance/radiance geolocation.

    Returns a dict with ``obs_cube`` (lazy, bands last), ``obs_bands`` (raw label
    array for :func:`band_index`) and ``_handles`` (objects to close).
    """
    obs_url = find_asset_url(granule, OBS_TOKEN)
    fh = earthaccess.open([obs_url])[0]
    ds_obs = xr.open_dataset(fh, engine="h5netcdf")
    ds_ob = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")
    return {
        "obs_cube": ds_obs["obs"],
        "obs_bands": ds_ob["observation_bands"].values,
        "_handles": [ds_obs, ds_ob, fh],
    }


# ============================================================
# WINDOWING HELPER
# ============================================================
def _win_np(arr: np.ndarray, bounds) -> np.ndarray:
    """Slice a full-scene 2-D NumPy array to the (r0, r1, c0, c1) window."""
    r0, r1, c0, c1 = bounds
    return np.asarray(arr)[r0:r1, c0:c1]


def _footprint_wkt(umm: dict):
    """Granule footprint (EPSG:4326) as a WKT string from the UMM GPolygons.

    All GPolygons are unioned so a multi-polygon footprint is preserved.
    Returns ``None`` if the UMM lacks the expected GPolygon structure — the
    footprint is only a cosmetic context overlay, so its absence must not fail
    the extraction.
    """
    try:
        gp = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
        polys = [
            shapely.geometry.Polygon(
                [(p["Longitude"], p["Latitude"]) for p in g["Boundary"]["Points"]]
            )
            for g in gp
        ]
        poly = polys[0] if len(polys) == 1 else shapely.unary_union(polys)
        return poly.wkt
    except (KeyError, IndexError, TypeError):
        return None


# ============================================================
# UNIFIED EXTRACTION
# ============================================================
def extract_acquisition(acq: dict, patch_geom_4326, *, patch_id: str,
                        rfl_short: str, rad_short: str, want_obs: bool = True,
                        n_retries: int = 3, backoff: int = 5):
    """Open every asset for one acquisition once and return a raw cube dict.

    The reflectance + MASK file, the radiance file and the OBS file are each
    opened a single time; the in-patch mask is built once from whichever
    geolocation grid is available (L2A preferred, L1B fallback) and reused for
    all products.  Every returned array is **raw and unscreened** — only the
    polygon-containment test is applied — with fill values intact.

    Parameters
    ----------
    acq : dict
        One entry from :func:`emit_pics.extract_patch_stats.pair_acquisitions`
        (keys ``timestamp``, ``datetime``, ``rad_ur``, ``rfl_ur``).
    patch_geom_4326 : shapely Polygon
        Patch polygon in EPSG:4326 for the pixel-membership test.
    patch_id : str
        Patch label, stored on the cube for provenance.
    rfl_short, rad_short : str
        Earthdata collection short names for L2A_RFL and L1B_RAD.
    want_obs : bool
        Whether to also stream the OBS geometry (SZA/VZA) for this acquisition.

    Returns
    -------
    dict or None
        A *cube dict* (see module docstring) of per-pixel arrays plus metadata,
        ``{"n_pixels": 0, ...}`` if the acquisition has no in-patch pixels, or
        ``None`` if neither product was available to build a mask.
    """
    rfl_ur = acq.get("rfl_ur")
    rad_ur = acq.get("rad_ur")
    ts = acq["timestamp"]

    handles: list = []
    cube = {
        "patch_id": patch_id, "timestamp": ts, "datetime": acq["datetime"],
        "rad_ur": rad_ur, "rfl_ur": rfl_ur, "has_rfl": rfl_ur is not None,
        "has_rad": rad_ur is not None, "n_pixels": 0, "bounds": None,
        "row": None, "col": None, "lon": None, "lat": None,
        "reflectance": None, "rfl_wl": None,
        "radiance": None, "rad_wl": None,
        "mask": None, "mask_labels": None,
        "sza": None, "vza": None, "footprint_wkt": None,
    }

    try:
        lon = lat = None
        rfl = None
        gran_rad = None
        umm_fp = None       # UMM whose footprint labels the QC context panel

        # --- L2A reflectance + MASK (preferred geolocation source) ---
        if rfl_ur is not None:
            gran_rfl = fetch_granule(rfl_short, rfl_ur)
            rfl = with_retry(lambda: open_rfl_mask(gran_rfl), label=f"{ts} L2A")
            handles += rfl["_handles"]
            lon, lat = rfl["lon"], rfl["lat"]
            umm_fp = gran_rfl["umm"]

        # --- L1B radiance (and lon/lat fallback if no L2A) ---
        l1b = None
        if rad_ur is not None:
            gran_rad = fetch_granule(rad_short, rad_ur)
            need_loc = lon is None
            l1b = with_retry(lambda: open_l1b(gran_rad, want_loc=need_loc),
                             label=f"{ts} L1B")
            handles += l1b["_handles"]
            if lon is None:
                lon, lat = l1b["lon"], l1b["lat"]
            if umm_fp is None:
                umm_fp = gran_rad["umm"]

        if lon is None:
            return None  # nothing to build a mask from

        # --- mask + window, built ONCE and shared by every product ---
        mask = in_patch_mask(lon, lat, patch_geom_4326)
        bounds = window_bounds(mask)
        if bounds is None:
            return cube  # n_pixels == 0
        r0, r1, c0, c1 = bounds
        keep_sub = mask[r0:r1, c0:c1]
        n_pixels = int(keep_sub.sum())

        # Full-scene pixel positions, in the same C-order as window_pixels().
        wr, wc = np.nonzero(keep_sub)
        cube["row"] = (r0 + wr).astype(np.int32)
        cube["col"] = (c0 + wc).astype(np.int32)
        cube["lon"] = _win_np(lon, bounds)[keep_sub]
        cube["lat"] = _win_np(lat, bounds)[keep_sub]
        cube["bounds"] = bounds
        cube["n_pixels"] = n_pixels

        # --- reflectance + mask bands ---
        if rfl is not None:
            cube["reflectance"] = window_pixels(rfl["refl"], bounds, keep_sub)
            cube["rfl_wl"] = rfl["wl"]
            cube["mask"] = window_pixels(rfl["mask_cube"], bounds, keep_sub)
            cube["mask_labels"] = rfl["mask_labels"]

        # --- radiance ---
        if l1b is not None:
            cube["radiance"] = window_pixels(l1b["rad"], bounds, keep_sub)
            cube["rad_wl"] = l1b["wl"]

        # --- OBS geometry (SZA/VZA), reusing the already-fetched RAD granule ---
        if want_obs and gran_rad is not None:
            obs = with_retry(lambda: open_obs(gran_rad), label=f"{ts} OBS")
            handles += obs["_handles"]
            si = band_index(obs["obs_bands"], SZA_LABEL)
            vi = band_index(obs["obs_bands"], VZA_LABEL)
            cube["sza"] = window_2d(obs["obs_cube"][:, :, si], bounds)[keep_sub]
            cube["vza"] = window_2d(obs["obs_cube"][:, :, vi], bounds)[keep_sub]

        # Footprint polygon for the QC context overlay (cosmetic; may be None).
        cube["footprint_wkt"] = _footprint_wkt(umm_fp) if umm_fp else None

        return cube
    finally:
        close_handles(handles)


# ============================================================
# IN-MEMORY DERIVATIONS (no network)
# ============================================================
def geometry_stats_from_cube(cube: dict):
    """Per-scene SZA/VZA distribution stats from a cube's geometry pixels.

    Returns mean/std/min/max and the 25/50/75 percentiles for both angles
    (fills NaN'd first) plus ``n_pixels``, or ``None`` if the cube carries no
    OBS geometry.
    """
    if cube["sza"] is None or cube["vza"] is None:
        return None
    sza = np.where(np.isclose(cube["sza"], FILL), np.nan, cube["sza"])
    vza = np.where(np.isclose(cube["vza"], FILL), np.nan, cube["vza"])
    out = {}
    out.update(_dist_stats(sza, "sza"))
    out.update(_dist_stats(vza, "vza"))
    out["n_pixels"] = int(cube["n_pixels"])
    return out


def channel_stats_from_cube(cube: dict, keep: np.ndarray = None):
    """Per-channel ``(rad_df, rfl_df)`` from a cube, optionally row-screened.

    *keep* is a per-pixel boolean selection (e.g. from :func:`aggregate_keep`);
    ``None`` uses every in-patch pixel.  Either frame is ``None`` if that product
    is absent from the cube.
    """
    sel = slice(None) if keep is None else keep
    rad_df = rfl_df = None
    if cube["reflectance"] is not None:
        rfl_df = per_channel_stats(cube["reflectance"][sel], cube["rfl_wl"],
                                   RFL_FILLS, "rfl")
    if cube["radiance"] is not None:
        rad_df = per_channel_stats(cube["radiance"][sel], cube["rad_wl"],
                                   RAD_FILLS, "rad")
    return rad_df, rfl_df


def aggregate_keep(cube: dict, min_valid_frac: float):
    """Read-time Aggregate-Flag screen: ``(keep_mask, valid_fraction)``.

    Pixels are kept where the L2A MASK Aggregate-Flag band equals 0; fill / NaN
    pixels are treated as flagged (dropped).  Returns ``(None, None)`` if the
    cube has no MASK (so the caller cannot screen this acquisition).
    """
    if cube["mask"] is None or cube["mask_labels"] is None:
        return None, None
    if AGG_FLAG_LABEL not in cube["mask_labels"]:
        return None, None
    i = cube["mask_labels"].index(AGG_FLAG_LABEL)
    agg = np.asarray(cube["mask"][:, i], dtype=float)
    agg = np.where(np.isclose(agg, FILL), np.nan, agg)
    keep = np.isclose(np.nan_to_num(agg, nan=1.0), 0.0)
    n = cube["n_pixels"]
    vf = float(keep.sum()) / n if n else 0.0
    return keep, vf


def mask_summaries_from_cube(cube: dict):
    """One summary row per MASK band: flagged-% for binary, stats for continuous.

    Returns a DataFrame (columns ``mask_band``, ``kind``, ``pct_flagged``,
    ``mean``, ``std``, ``min``, ``max``, ``n_valid``) or ``None`` if no MASK.
    """
    if cube["mask"] is None or cube["mask_labels"] is None:
        return None
    rows = []
    for i, lbl in enumerate(cube["mask_labels"]):
        a = np.asarray(cube["mask"][:, i], dtype=float)
        a = np.where(np.isclose(a, FILL), np.nan, a)
        finite = a[np.isfinite(a)]
        row = {"mask_band": lbl, "pct_flagged": np.nan, "mean": np.nan,
               "std": np.nan, "min": np.nan, "max": np.nan,
               "n_valid": int(finite.size)}
        if lbl in BINARY_FLAGS:
            row["kind"] = "binary"
            row["pct_flagged"] = (100.0 * np.count_nonzero(finite > 0) / finite.size
                                  if finite.size else np.nan)
        else:
            row["kind"] = "continuous"
            if finite.size:
                row.update(mean=float(finite.mean()), std=float(finite.std()),
                           min=float(finite.min()), max=float(finite.max()))
        rows.append(row)
    cols = ["mask_band", "kind", "pct_flagged", "mean", "std", "min", "max",
            "n_valid"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================
# PIXEL-CUBE PERSISTENCE (NetCDF)
# ============================================================
def write_cube_netcdf(cube: dict, path: Path) -> Path:
    """Persist a raw, unscreened in-patch pixel cube to NetCDF.

    Dimensions: ``pixel`` (the polygon-contained in-patch pixels), ``rfl_band``,
    ``rad_band`` and ``mask_band``.  Reflectance / radiance / mask are stored
    flattened to ``(pixel, band)`` with fills intact; per-pixel ``sza``/``vza``,
    ``lon``/``lat`` and full-scene ``row``/``col`` indices are stored alongside.
    Band wavelengths label the spectral coords; MASK band labels are stored in
    the ``mask_labels`` attribute (avoids NetCDF string-coordinate pitfalls).

    No screening (Aggregate-Flag, AOD, valid-fraction) is applied — that is a
    deliberate read-time choice so screening and BRDF experiments re-process
    these pixels locally without re-streaming.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coords = {"pixel": np.arange(cube["n_pixels"], dtype=np.int32)}
    data_vars = {
        "row": ("pixel", cube["row"]),
        "col": ("pixel", cube["col"]),
        "lon": ("pixel", cube["lon"]),
        "lat": ("pixel", cube["lat"]),
    }
    if cube["sza"] is not None:
        data_vars["sza"] = ("pixel", np.asarray(cube["sza"], dtype="float32"))
    if cube["vza"] is not None:
        data_vars["vza"] = ("pixel", np.asarray(cube["vza"], dtype="float32"))
    if cube["reflectance"] is not None:
        coords["rfl_band"] = np.asarray(cube["rfl_wl"], dtype="float32")
        data_vars["reflectance"] = (
            ("pixel", "rfl_band"), np.asarray(cube["reflectance"], dtype="float32"))
    if cube["radiance"] is not None:
        coords["rad_band"] = np.asarray(cube["rad_wl"], dtype="float32")
        data_vars["radiance"] = (
            ("pixel", "rad_band"), np.asarray(cube["radiance"], dtype="float32"))
    if cube["mask"] is not None:
        coords["mask_band"] = np.arange(len(cube["mask_labels"]), dtype=np.int32)
        data_vars["mask"] = (
            ("pixel", "mask_band"), np.asarray(cube["mask"], dtype="float32"))

    attrs = {
        "patch_id": str(cube["patch_id"]),
        "timestamp": str(cube["timestamp"]),
        "rad_ur": str(cube["rad_ur"]),
        "rfl_ur": str(cube["rfl_ur"]),
        "n_pixels": int(cube["n_pixels"]),
        "fill_value": FILL,
        "screening": "none (raw in-patch pixels; screen at read time)",
        "note": "Polygon-contained pixels only, flattened; fills intact.",
    }
    # Window bounds + footprint let QC rebuild 2-D panels offline from this cube.
    if cube.get("bounds") is not None:
        attrs["bounds"] = [int(x) for x in cube["bounds"]]
    if cube.get("footprint_wkt"):
        attrs["footprint_wkt"] = cube["footprint_wkt"]
    if cube["mask_labels"] is not None:
        attrs["mask_labels"] = list(cube["mask_labels"])

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path, engine="netcdf4")
    return path


def read_cube_netcdf(path: Path) -> xr.Dataset:
    """Open a persisted pixel cube. MASK band labels are in ``ds.attrs['mask_labels']``."""
    return xr.open_dataset(Path(path), engine="netcdf4")


# ============================================================
# CUBE -> QC RENDER INPUTS (offline)
# ============================================================
def cube_to_render_inputs(ds: xr.Dataset):
    """Rebuild the windowed 2-D inputs that ``render_scene_figure`` expects.

    A cube stores only the flattened polygon-contained pixels plus their
    full-scene ``row``/``col`` indices and the window ``bounds``.  This scatters
    those pixels back into dense ``(window_height, window_width)`` grids (cells
    outside the patch filled with the EMIT fill value, which the renderer blanks
    via the in-patch mask), wraps the spectral cubes as DataArrays so the
    renderer's ``.values`` slicing works, and synthesizes a 2-band OBS cube
    (SZA, VZA) labelled exactly as the renderer looks them up.

    The figure is produced entirely from cached data — no network, no
    re-masking.  The nadir track is unavailable from a cube (it needs whole-scene
    VZA), so callers must render with ``draw_nadir=False``.

    Parameters
    ----------
    ds : xr.Dataset
        A cube opened by :func:`read_cube_netcdf`.

    Returns
    -------
    dict or None
        ``{"rfl", "obs", "mask", "bounds", "granule_poly"}`` ready to splat into
        :func:`emit_pics.display_qc_visuals.render_scene_figure`, or ``None`` if the
        cube lacks the reflectance / MASK / geometry needed for a full QC page.
    """
    needed = ("reflectance", "mask", "sza", "vza")
    if any(v not in ds for v in needed):
        return None
    bounds_attr = ds.attrs.get("bounds")
    if bounds_attr is None or len(bounds_attr) != 4:
        return None

    r0, r1, c0, c1 = (int(x) for x in bounds_attr)
    h, w = r1 - r0, c1 - c0
    wr = ds["row"].values.astype(int) - r0
    wc = ds["col"].values.astype(int) - c0

    in_sub = np.zeros((h, w), dtype=bool)
    in_sub[wr, wc] = True

    def _scatter2d(vals):
        g = np.full((h, w), FILL, dtype=float)
        g[wr, wc] = np.asarray(vals, dtype=float)
        return g

    def _scatter3d(vals):
        vals = np.asarray(vals, dtype=float)
        g = np.full((h, w, vals.shape[1]), FILL, dtype=float)
        g[wr, wc, :] = vals
        return g

    refl_g = _scatter3d(ds["reflectance"].values)
    mask_g = _scatter3d(ds["mask"].values)
    obs_g = np.stack([_scatter2d(ds["sza"].values),
                      _scatter2d(ds["vza"].values)], axis=-1)

    rfl = {
        "refl": xr.DataArray(refl_g, dims=("downtrack", "crosstrack", "band")),
        "wl": np.asarray(ds["rfl_band"].values, dtype=float),
        "lon": _scatter2d(ds["lon"].values),
        "lat": _scatter2d(ds["lat"].values),
        "mask_cube": xr.DataArray(
            mask_g, dims=("downtrack", "crosstrack", "mask_band")),
        "mask_labels": list(ds.attrs["mask_labels"]),
    }
    obs = {
        "obs_cube": xr.DataArray(
            obs_g, dims=("downtrack", "crosstrack", "obs_band")),
        "obs_bands": np.array([SZA_LABEL, VZA_LABEL], dtype=object),
    }

    fp = ds.attrs.get("footprint_wkt")
    granule_poly = shapely.from_wkt(fp) if fp else None

    return {"rfl": rfl, "obs": obs, "mask": in_sub,
            "bounds": (0, h, 0, w), "granule_poly": granule_poly}
