"""Batched extraction of **every patch of one size** across a whole PICS.

This is the loop-inverted counterpart to
:mod:`emit_pics.single_patch_cube_extract`.  The single-patch path opens each
granule, reads its full-scene geolocation and builds a KDTree *per patch*; run it
over N patches (a naive wrapper) and every granule that covers K patches is
re-opened and its lon/lat re-streamed and re-indexed K times.  Within one PICS a
single EMIT granule (~85 km swath) blankets essentially the whole tiling, so K is
large and that redundancy dominates.

The fix is to **invert the loop nesting**: iterate acquisitions (granules) on the
outside and patches on the inside.  For each acquisition this module

1. opens every asset **once** (L2A reflectance + MASK, L1B radiance, L1B OBS),
2. reads the full-scene lon/lat **once** and builds the geolocation KDTree
   **once** (:func:`build_geo_index`), then
3. for each patch the granule covers, queries that shared index (cheap) and
   windows only that patch's pixels (:func:`extract_patch_from_grid`).

So granule opens, lon/lat reads and KDTree builds become ``O(#granules)`` instead
of ``O(#granules × #patches)``.  Only the windowed spectral reads stay per-patch
(different pixels each), which is irreducible for a non-overlapping tiling.

Every per-patch *cube dict* returned here is byte-for-byte the same structure the
single-patch extractor produces, so all the in-memory derivations and NetCDF
persistence (``geometry_stats_from_cube``, ``channel_stats_from_cube``,
``aggregate_keep``, ``mask_summaries_from_cube``, ``write_cube_netcdf``) are
imported from :mod:`emit_pics.single_patch_cube_extract` and reused unchanged.

Pure function library: no top-level execution, no hardcoded paths, no
``earthaccess.login`` (the runner authenticates).  A ``grid`` returned by
:func:`open_acquisition_grid` owns open file handles and MUST be closed with
:func:`close_grid` once its patches are done.
"""

import numpy as np
import shapely
from scipy.spatial import cKDTree

from emit_pics.id_matched_pairs import (
    FILL,
    SZA_LABEL,
    VZA_LABEL,
    band_index,
    fetch_granule,
)
from emit_pics.extract_patch_stats import (
    close_handles,
    open_l1b,
    pair_acquisitions,
    split_products,
    window_2d,
    window_bounds,
    window_pixels,
    with_retry,
)
from emit_pics.single_patch_cube_extract import (
    _footprint_wkt,
    _win_np,
    open_obs,
    open_rfl_mask,
)


# ACQUISITION -> PATCHES INVERSE MAP (no network)
def patch_ids_for_size(dict_df, size_token) -> list:
    """Every ``patch_id`` in the dict whose size field matches *size_token*.

    ``patch_id`` is ``"{size}_{i}_{j}"`` (e.g. ``"10_7_1"``), so the size is the
    first underscore-delimited field.  Returns them sorted.
    """
    ids = sorted(set(str(p) for p in dict_df["patch_id"]))
    return [p for p in ids if p.split("_")[0] == str(size_token)]


def build_acquisition_patch_map(dict_df, patch_ids):
    """Invert the patch-granule dict into an acquisition-outer schedule.

    For every patch, its acquisitions are formed exactly as the single-patch path
    does (:func:`split_products` + :func:`pair_acquisitions`), then collated by
    acquisition timestamp.  The same granule covering many patches therefore
    yields **one** acquisition entry plus the set of patches it serves.

    Parameters
    ----------
    dict_df : pd.DataFrame
        The loaded patch-granule dictionary.
    patch_ids : list of str
        Patches to schedule (e.g. from :func:`patch_ids_for_size`).

    Returns
    -------
    (acquisitions, patches_by_ts) : (list of dict, dict)
        *acquisitions* — one merged acq dict per unique timestamp
        (``timestamp``, ``datetime``, ``rad_ur``, ``rfl_ur``), sorted by time.
        *patches_by_ts* — ``timestamp -> sorted list of patch_ids`` covered.
    """
    acq_by_ts, patches_by_ts = {}, {}
    for pid in patch_ids:
        rad_urs, rfl_urs = split_products(dict_df, pid)
        for acq in pair_acquisitions(rad_urs, rfl_urs):
            ts = acq["timestamp"]
            patches_by_ts.setdefault(ts, set()).add(pid)
            if ts not in acq_by_ts:
                acq_by_ts[ts] = dict(acq)
            else:  # merge in case a patch supplies a product the first lacked
                a = acq_by_ts[ts]
                if a.get("rad_ur") is None:
                    a["rad_ur"] = acq.get("rad_ur")
                if a.get("rfl_ur") is None:
                    a["rfl_ur"] = acq.get("rfl_ur")
    acquisitions = [acq_by_ts[ts] for ts in sorted(acq_by_ts)]
    patches_by_ts = {ts: sorted(v) for ts, v in patches_by_ts.items()}
    return acquisitions, patches_by_ts


# REUSABLE GEOLOCATION INDEX (built once per granule)
def build_geo_index(lon: np.ndarray, lat: np.ndarray) -> dict:
    """Build the per-granule geolocation index reused by every patch mask.

    This is the granule-scoped half of
    :func:`emit_pics.id_matched_pairs.in_patch_mask` — the valid-pixel point
    cloud and its KDTree, which depend only on the granule's lon/lat, not on any
    patch.  Building it once and querying it per patch (see
    :func:`mask_from_geo_index`) is what removes the redundant per-patch KDTree
    builds.

    Returns a dict with ``shape`` (the 2-D grid shape), ``idx_valid`` (flat
    indices of non-fill pixels), ``pts`` (``(n_valid, 2)`` lon/lat) and ``tree``
    (a ``cKDTree`` over ``pts``, or ``None`` if the scene has no valid pixels).
    """
    shape = lon.shape
    lon_f = np.asarray(lon, dtype=float).ravel()
    lat_f = np.asarray(lat, dtype=float).ravel()
    valid = ~(np.isclose(lon_f, FILL) | np.isclose(lat_f, FILL))
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        return {"shape": shape, "idx_valid": idx_valid,
                "pts": np.empty((0, 2)), "tree": None}
    pts = np.column_stack([lon_f[idx_valid], lat_f[idx_valid]])
    return {"shape": shape, "idx_valid": idx_valid, "pts": pts,
            "tree": cKDTree(pts)}


def mask_from_geo_index(geo: dict, patch_geom_4326, pad: float = 1.10) -> np.ndarray:
    """In-patch boolean mask from a prebuilt :func:`build_geo_index`.

    A KDTree disc pre-filter (radius = patch half-diagonal × *pad*, centred on the
    patch centroid) narrows candidates before an exact
    :func:`shapely.contains_xy` test — identical logic and result to
    :func:`emit_pics.id_matched_pairs.in_patch_mask`, but reusing the granule's
    shared tree instead of rebuilding it.

    Returns a 2-D boolean array of the grid shape; ``True`` inside the patch.
    """
    shape = geo["shape"]
    mask_flat = np.zeros(int(np.prod(shape)), dtype=bool)
    if geo["tree"] is None or geo["idx_valid"].size == 0:
        return mask_flat.reshape(shape)

    minx, miny, maxx, maxy = patch_geom_4326.bounds
    centroid = patch_geom_4326.centroid
    radius = 0.5 * np.hypot(maxx - minx, maxy - miny) * pad
    cand = np.asarray(
        geo["tree"].query_ball_point([centroid.x, centroid.y], radius), dtype=int
    )
    if cand.size:
        pts = geo["pts"]
        inside = shapely.contains_xy(patch_geom_4326, pts[cand, 0], pts[cand, 1])
        mask_flat[geo["idx_valid"][cand[inside]]] = True
    return mask_flat.reshape(shape)


# PER-GRANULE OPEN (once) + PER-PATCH EXTRACT (many)
def open_acquisition_grid(acq: dict, *, rfl_short: str, rad_short: str,
                          want_obs: bool = True):
    """Open one acquisition's assets once and index its geolocation once.

    The L2A reflectance + MASK, the L1B radiance and (optionally) the OBS file are
    each opened a single time; the full-scene lon/lat is read once and its KDTree
    built once.  The returned *grid* keeps the lazy spectral cubes and open file
    handles alive so :func:`extract_patch_from_grid` can window each covered patch
    without re-opening anything.  **Close it with :func:`close_grid`.**

    Parameters
    ----------
    acq : dict
        A merged acquisition from :func:`build_acquisition_patch_map`
        (``timestamp``, ``datetime``, ``rad_ur``, ``rfl_ur``).
    rfl_short, rad_short : str
        Earthdata collection short names for L2A_RFL and L1B_RAD.
    want_obs : bool
        Also open the OBS geometry (SZA/VZA) for the geometry table.

    Returns
    -------
    dict or None
        A grid dict (lazy cubes, geolocation index, open handles), or ``None`` if
        neither product was available to build a geolocation grid.  On error the
        handles are closed before the exception propagates.
    """
    rfl_ur = acq.get("rfl_ur")
    rad_ur = acq.get("rad_ur")
    ts = acq["timestamp"]

    handles: list = []
    grid = {
        "timestamp": ts, "datetime": acq["datetime"],
        "rad_ur": rad_ur, "rfl_ur": rfl_ur,
        "has_rfl": rfl_ur is not None, "has_rad": rad_ur is not None,
        "lon": None, "lat": None, "geo": None,
        "refl": None, "rfl_wl": None, "mask_cube": None, "mask_labels": None,
        "radiance": None, "rad_wl": None,
        "obs_cube": None, "sza_idx": None, "vza_idx": None,
        "footprint_wkt": None, "_handles": handles,
    }

    try:
        lon = lat = None
        umm_fp = None

        # --- L2A reflectance + MASK (preferred geolocation source) ---
        if rfl_ur is not None:
            gran_rfl = fetch_granule(rfl_short, rfl_ur)
            rfl = with_retry(lambda: open_rfl_mask(gran_rfl), label=f"{ts} L2A")
            handles += rfl["_handles"]
            lon, lat = rfl["lon"], rfl["lat"]
            grid["refl"], grid["rfl_wl"] = rfl["refl"], rfl["wl"]
            grid["mask_cube"], grid["mask_labels"] = rfl["mask_cube"], rfl["mask_labels"]
            umm_fp = gran_rfl["umm"]

        # --- L1B radiance (and lon/lat fallback if no L2A) + OBS geometry ---
        if rad_ur is not None:
            gran_rad = fetch_granule(rad_short, rad_ur)
            need_loc = lon is None
            l1b = with_retry(lambda: open_l1b(gran_rad, want_loc=need_loc),
                             label=f"{ts} L1B")
            handles += l1b["_handles"]
            if lon is None:
                lon, lat = l1b["lon"], l1b["lat"]
            grid["radiance"], grid["rad_wl"] = l1b["rad"], l1b["wl"]
            if umm_fp is None:
                umm_fp = gran_rad["umm"]
            if want_obs:
                obs = with_retry(lambda: open_obs(gran_rad), label=f"{ts} OBS")
                handles += obs["_handles"]
                grid["obs_cube"] = obs["obs_cube"]
                grid["sza_idx"] = band_index(obs["obs_bands"], SZA_LABEL)
                grid["vza_idx"] = band_index(obs["obs_bands"], VZA_LABEL)

        if lon is None:
            close_handles(handles)
            return None  # nothing to build a geolocation index from

        grid["lon"], grid["lat"] = lon, lat
        grid["geo"] = build_geo_index(lon, lat)
        grid["footprint_wkt"] = _footprint_wkt(umm_fp) if umm_fp else None
        return grid
    except Exception:
        close_handles(handles)
        raise


def extract_patch_from_grid(grid: dict, patch_geom_4326, *, patch_id: str) -> dict:
    """Extract one patch's raw cube from an already-open acquisition *grid*.

    Queries the grid's shared geolocation index for this patch, windows only the
    patch's pixels out of the lazy spectral cubes, and returns a *cube dict* with
    exactly the fields :func:`emit_pics.single_patch_cube_extract.extract_acquisition`
    produces — so every downstream derivation works unchanged.

    Returns the cube (``n_pixels == 0`` if the patch has no in-patch pixels for
    this acquisition).
    """
    cube = {
        "patch_id": patch_id, "timestamp": grid["timestamp"],
        "datetime": grid["datetime"], "rad_ur": grid["rad_ur"],
        "rfl_ur": grid["rfl_ur"], "has_rfl": grid["has_rfl"],
        "has_rad": grid["has_rad"], "n_pixels": 0, "bounds": None,
        "row": None, "col": None, "lon": None, "lat": None,
        "reflectance": None, "rfl_wl": None, "radiance": None, "rad_wl": None,
        "mask": None, "mask_labels": None,
        "sza": None, "vza": None,
        "footprint_wkt": grid["footprint_wkt"],
    }

    mask = mask_from_geo_index(grid["geo"], patch_geom_4326)
    bounds = window_bounds(mask)
    if bounds is None:
        return cube  # n_pixels == 0

    r0, r1, c0, c1 = bounds
    keep_sub = mask[r0:r1, c0:c1]
    wr, wc = np.nonzero(keep_sub)
    cube["row"] = (r0 + wr).astype(np.int32)
    cube["col"] = (c0 + wc).astype(np.int32)
    cube["lon"] = _win_np(grid["lon"], bounds)[keep_sub]
    cube["lat"] = _win_np(grid["lat"], bounds)[keep_sub]
    cube["bounds"] = bounds
    cube["n_pixels"] = int(keep_sub.sum())

    if grid["refl"] is not None:
        cube["reflectance"] = window_pixels(grid["refl"], bounds, keep_sub)
        cube["rfl_wl"] = grid["rfl_wl"]
        cube["mask"] = window_pixels(grid["mask_cube"], bounds, keep_sub)
        cube["mask_labels"] = grid["mask_labels"]
    if grid["radiance"] is not None:
        cube["radiance"] = window_pixels(grid["radiance"], bounds, keep_sub)
        cube["rad_wl"] = grid["rad_wl"]
    if grid["obs_cube"] is not None:
        cube["sza"] = window_2d(grid["obs_cube"][:, :, grid["sza_idx"]], bounds)[keep_sub]
        cube["vza"] = window_2d(grid["obs_cube"][:, :, grid["vza_idx"]], bounds)[keep_sub]

    return cube


def close_grid(grid) -> None:
    """Close every open file handle a *grid* owns (no-op for ``None``)."""
    if grid is not None:
        close_handles(grid.get("_handles", []))
