"""Per-scene Landsat 9 TOA-reflectance extraction over a single PICS patch.

The Landsat control analog to :mod:`emit_pics.single_patch_cube_extract`. Where the
EMIT path streams granules via ``earthaccess`` and derives per-channel stats + a
per-scene SZA/VZA geometry table, this module does the same for Landsat 9 OLI-2
**TOA reflectance** (Collection 2 Level-1), read as windowed COGs from a STAC
catalog. The outputs are written in the SAME schemas the EMIT side uses, so the
sensor-agnostic BRDF fit (:mod:`emit_pics.brdf_drift`) consumes them unchanged:

* a per-scene **geometry table** — ``sza_mean``/``vza_mean`` (+ percentiles) and
  ``n_pixels`` — matching :func:`emit_pics.paths.ls_geometry`'s schema, and
* one per-scene **per-band table** (``central_wl``, ``median_toa``, ``mean_toa``,
  ...) with a fixed 7-row band grid, matching the ``perchannel`` schema.

Design choices:

* **TOA reflectance only** — Landsat's surface-reflectance retrieval (LaSRC)
  differs too much from EMIT's to be a fair instrument comparison. TOA is computed
  from the L1 DN via the MTL rescaling + scene sun elevation.
* **All 7 reflective OLI-2 bands** (coastal..swir22); pan/cirrus excluded.
* **No Earth-Sun normalization** — Landsat TOA reflectance already folds in the
  Earth-Sun distance and solar irradiance (unlike EMIT L1B *radiance*, which we
  normalize to 1 AU). So there is no ``E0`` step here.
* **Geometry from the angle-band rasters** — ``sza_mean``/``vza_mean`` are the
  in-patch means of the L1 ``SZA``/``VZA`` bands (int16, 0.01 deg), paralleling
  EMIT's in-patch OBS means. (The scene ``SUN_ELEVATION`` from the MTL is used only
  for the TOA reflectance conversion, per the USGS convention.)
* **QC is NOT auto-populated** — like EMIT, quality control is a separate manual
  worksheet (opt-out keep/remove + reviewed flag). This module writes only the raw
  geometry/stats; it never screens on ``qa_pixel``.

Access is via STAC + windowed ``rasterio`` reads (no full-tile downloads). TOA
(collection ``landsat-c2l1`` on the USGS LandsatLook STAC) lives in the
requester-pays ``usgs-landsat`` S3 bucket, so a run needs AWS credentials in the
boto3 default chain (see ``notebooks/landsat9_single_scene_extract.ipynb``).

This module is pure library: no printing, no ``sys.exit`` — the runner
(``scripts/run_landsat_cube_extract.py``) owns I/O and messaging.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import shape


# BAND METADATA  (OLI-2 reflective bands 1-7; pan/cirrus excluded)
#: (asset key, MTL band number, nominal band-center wavelength nm). ``center_nm``
#: becomes the ``central_wl`` grid the per-band table is keyed on. The two flagged
#: bands land on EMIT's significant-drift regions (858 nm; 2000-2370 nm).
OLI2_BANDS = [
    {"asset": "coastal", "band_n": 1, "center_nm": 443},
    {"asset": "blue",    "band_n": 2, "center_nm": 482},
    {"asset": "green",   "band_n": 3, "center_nm": 561},
    {"asset": "red",     "band_n": 4, "center_nm": 654},
    {"asset": "nir08",   "band_n": 5, "center_nm": 865},   # <- EMIT 858 nm
    {"asset": "swir16",  "band_n": 6, "center_nm": 1609},
    {"asset": "swir22",  "band_n": 7, "center_nm": 2201},  # <- EMIT 2000-2370 nm
]

#: STAC asset keys for the per-pixel solar/view zenith angle rasters (int16, 0.01 deg).
SZA_ASSET, VZA_ASSET = "SZA", "VZA"

#: DN fill value in Landsat Collection 2 rasters (bands and angle grids).
FILL = 0


# STAC SEARCH / SCENE DISCOVERY
def search_scenes(catalog, collection, patch_wgs84, date_start, date_end, *,
                  platform="landsat-9", max_cloud=100.0, dedupe_overpass=True,
                  max_retries=4, backoff=3):
    """Return Landsat 9 scenes that FULLY CONTAIN the patch, sorted by datetime.

    A plain STAC ``intersects`` also returns edge-clipping / adjacent-WRS-path
    scenes that cover the patch only partially, so we keep only scenes whose
    footprint fully contains it and whose scene-level cloud cover is under
    *max_cloud* (client-side, so it behaves identically across STAC servers).

    When *dedupe_overpass* is set, WRS row-overlap duplicates are collapsed to one
    frame per overpass (see :func:`_dedupe_overpass`) — a patch in the along-track
    overlap of two rows is otherwise imaged twice per pass, which would double-count
    in the fit. The LandsatLook STAC intermittently 500s on pagination, so the
    materialization is retried up to *max_retries* times with linear *backoff*.

    Returns a list of ``pystac.Item`` in acquisition-time order (all qualifying
    overpasses — the caller loops them into the time series).
    """
    import time
    from pystac_client.exceptions import APIError

    search = catalog.search(
        collections=[collection],
        intersects=patch_wgs84.__geo_interface__,
        datetime=f"{date_start}/{date_end}",
    )
    items = None
    for attempt in range(1, max_retries + 1):
        try:
            items = list(search.items())
            break
        except APIError:
            if attempt == max_retries:
                raise
            time.sleep(backoff * attempt)      # LandsatLook 500s intermittently on paging

    def _plat(it):
        return str(it.properties.get("platform", "")).lower().replace("_", "-")
    def _cloud(it):
        return it.properties.get("eo:cloud_cover", 100.0)
    def _covers(it):
        try:
            return shape(it.geometry).buffer(0).contains(patch_wgs84)
        except Exception:
            return False

    keep = [it for it in items
            if _plat(it) == platform and _cloud(it) <= max_cloud and _covers(it)]
    if dedupe_overpass:
        keep = _dedupe_overpass(keep, patch_wgs84)
    return sorted(keep, key=lambda it: it.properties.get("datetime", ""))


def _dedupe_overpass(items, patch_wgs84):
    """Collapse WRS row-overlap duplicates to one frame per overpass.

    A patch sitting in the along-track overlap of two WRS rows is imaged twice in
    a single overpass (adjacent frames ~24 s apart, near-identical geometry);
    letting both into the fit double-counts that acquisition. Scenes are grouped by
    ``(acquisition date, WRS path)`` — which uniquely identifies one overpass (a
    path revisits only every 16 days) without merging genuinely distinct paths —
    and within each group the frame whose footprint centroid is nearest the patch
    centroid (the most centered) is kept.
    """
    from collections import defaultdict

    pc = patch_wgs84.centroid
    groups = defaultdict(list)
    for it in items:
        date = str(it.properties.get("datetime", ""))[:10]
        path = it.properties.get("landsat:wrs_path")
        groups[(date, path)].append(it)

    kept = []
    for grp in groups.values():
        if len(grp) == 1:
            kept.append(grp[0])
        else:
            kept.append(min(grp, key=lambda it: shape(it.geometry).centroid.distance(pc)))
    return kept


# ACCESS HELPERS  (STAC asset -> windowed COG read)
def resolve_href(asset):
    """GDAL-openable href for a STAC asset (USGS requester-pays s3 preferred)."""
    alt = (asset.extra_fields or {}).get("alternate", {})
    if "s3" in alt and alt["s3"].get("href"):
        return alt["s3"]["href"]
    return asset.href


def read_window(href, patch_bounds, patch_crs):
    """Windowed read of one single-band COG, clipped to the patch bounds.

    Returns ``(data, valid, win_transform, crs)``: *data* is a float64 array (raw
    DN) over the patch window, *valid* is a boolean mask (unmasked pixels that are
    not the fill value), and *win_transform* / *crs* georeference the window (for
    the optional QC cube). Only the patch window is fetched from the remote COG,
    not the full ~185 km tile.
    """
    with rasterio.open(href) as src:
        l, b, r, t = transform_bounds(patch_crs, src.crs, *patch_bounds)
        win = from_bounds(l, b, r, t, src.transform)
        dn = src.read(1, window=win, masked=True)
        win_transform = src.window_transform(win)
        crs = src.crs
    data = np.asarray(dn, dtype="float64")
    valid = ~np.ma.getmaskarray(dn) & (data != FILL)
    return data, valid, win_transform, crs


def read_mtl(item):
    """Scene MTL (JSON) — REFLECTANCE_MULT/ADD_BAND_* + SUN_ELEVATION for TOA."""
    key = next((k for k in item.assets
                if k.upper().endswith("MTL.JSON") or k.upper() == "MTL.JSON"), None)
    if key is None:
        key = next(k for k in item.assets if item.assets[k].href.upper().endswith("MTL.JSON"))
    href = resolve_href(item.assets[key])
    if href.startswith("s3://"):
        import boto3
        u = urlparse(href)
        obj = boto3.client("s3", region_name="us-west-2").get_object(
            Bucket=u.netloc, Key=u.path.lstrip("/"), RequestPayer="requester")
        return json.load(obj["Body"])
    import urllib.request
    with urllib.request.urlopen(href) as fh:
        return json.load(fh)


def to_toa_reflectance(dn, band_n, mtl):
    """L1 DN -> sun-angle-corrected TOA reflectance for one band (from the MTL).

    ``rho = (M * DN + A) / sin(SUN_ELEVATION)``. Uses the scene-center sun
    elevation (USGS L1 convention); within a 10 km patch the sun-angle variation
    is negligible for the reflectance conversion. (The per-pixel SZA band still
    drives ``sza_mean`` in the geometry table — the BRDF ``cos theta`` term.)
    """
    m = mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"]
    mult = float(m[f"REFLECTANCE_MULT_BAND_{band_n}"])
    add = float(m[f"REFLECTANCE_ADD_BAND_{band_n}"])
    sun_elev = float(mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"])
    return (mult * dn + add) / np.sin(np.radians(sun_elev))


# UNIFIED PER-SCENE EXTRACTION
def extract_scene(item, patch_bounds, patch_crs, *, bands=OLI2_BANDS):
    """Window every band + the SZA/VZA rasters for one scene; return a cube dict.

    Reads each of the 7 reflective bands and the two angle rasters over the patch
    window, converts DN -> TOA reflectance (per :func:`to_toa_reflectance`), and
    keeps the raw per-band windows plus in-patch geometry. No QC/cloud screening
    is applied here (that is the manual worksheet's job downstream).

    Returns
    -------
    dict
        ``{scene_id, timestamp, datetime, sun_elevation, cloud_cover, wrs_path,
        wrs_row, refl: {asset: 2-D TOA refl (NaN where invalid)}, sza, vza (2-D deg,
        NaN where invalid), n_pixels, win_px, coverage}``.
    """
    mtl = read_mtl(item)
    props = item.properties

    refl, per_band_valid = {}, {}
    transform = crs = None
    for b in bands:
        dn, valid, tr, cr = read_window(resolve_href(item.assets[b["asset"]]),
                                        patch_bounds, patch_crs)
        if transform is None:
            transform, crs = tr, cr                    # shared 30 m grid for all bands
        rho = to_toa_reflectance(dn, b["band_n"], mtl)
        refl[b["asset"]] = np.where(valid, rho, np.nan)
        per_band_valid[b["asset"]] = valid

    # In-patch solar/view zenith (deg) from the L1 angle rasters (0.01 deg int16).
    sza_dn, sza_ok, _, _ = read_window(resolve_href(item.assets[SZA_ASSET]),
                                       patch_bounds, patch_crs)
    vza_dn, vza_ok, _, _ = read_window(resolve_href(item.assets[VZA_ASSET]),
                                       patch_bounds, patch_crs)
    sza = np.where(sza_ok, sza_dn * 0.01, np.nan)
    vza = np.where(vza_ok, vza_dn * 0.01, np.nan)

    ref_valid = per_band_valid[bands[0]["asset"]]        # coverage vs a reference band
    win_px = int(ref_valid.size)
    n_pixels = int(ref_valid.sum())
    return {
        "scene_id": item.id,
        "timestamp": _ts_from_datetime(props.get("datetime")),
        "datetime": props.get("datetime"),
        "sun_elevation": (float(mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]
                                ["SUN_ELEVATION"]) if mtl
                          else props.get("view:sun_elevation")),
        "cloud_cover": props.get("eo:cloud_cover"),
        "wrs_path": props.get("landsat:wrs_path"),
        "wrs_row": props.get("landsat:wrs_row"),
        "refl": refl, "sza": sza, "vza": vza, "bands": list(bands),
        "transform": transform, "crs": crs,       # georeferencing for the optional QC cube
        "n_pixels": n_pixels, "win_px": win_px,
        "coverage": (n_pixels / win_px) if win_px else 0.0,
    }


# IN-MEMORY DERIVATIONS  (schemas matched to the EMIT side)
def _dist_stats(a, prefix):
    """mean/std/min/max + p25/p50/p75 of the finite values of *a*, keyed by prefix.

    Reproduces the EMIT geometry-table columns (``{prefix}_mean``, ``_std``,
    ``_min``, ``_max``, ``_p25``, ``_p50``, ``_p75``) so the tables share a schema.
    """
    v = np.asarray(a, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        keys = ["mean", "std", "min", "max", "p25", "p50", "p75"]
        return {f"{prefix}_{k}": np.nan for k in keys}
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()),
            f"{prefix}_min": float(v.min()), f"{prefix}_max": float(v.max()),
            f"{prefix}_p25": float(p25), f"{prefix}_p50": float(p50),
            f"{prefix}_p75": float(p75)}


def geometry_record_from_cube(cube):
    """Per-scene geometry row matching :func:`emit_pics.paths.ls_geometry`'s schema.

    Columns: ``scene_id, timestamp, datetime, has_rfl, sza_mean(+stats),
    vza_mean(+stats), n_pixels``. ``has_rfl`` is always True for the TOA product
    (every scene carries reflectance) so the BRDF loader's reflectance-survivor
    filter passes.
    """
    rec = {"scene_id": cube["scene_id"], "timestamp": cube["timestamp"],
           "datetime": cube["datetime"], "has_rfl": True}
    rec.update(_dist_stats(cube["sza"], "sza"))
    rec.update(_dist_stats(cube["vza"], "vza"))
    rec["n_pixels"] = int(cube["n_pixels"])
    return rec


def channel_stats_from_cube(cube, bands=OLI2_BANDS):
    """Per-band TOA-reflectance stats as a DataFrame in the ``perchannel`` schema.

    One row per band, sorted by ``central_wl``, with the columns the BRDF
    ``channel_matrix`` reads (``median_toa``) plus the full stat set mirrored from
    the EMIT per-channel table (``min/1qt/median/3qt/max/mean/std``, suffix ``toa``).
    """
    rows = []
    for b in sorted(bands, key=lambda x: x["center_nm"]):
        v = cube["refl"][b["asset"]]
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append({"central_wl": float(b["center_nm"]),
                         "min_toa": float(v.min()), "1qt_toa": float(q1),
                         "median_toa": float(med), "3qt_toa": float(q3),
                         "max_toa": float(v.max()), "mean_toa": float(v.mean()),
                         "std_toa": float(v.std())})
        else:
            rows.append({"central_wl": float(b["center_nm"]),
                         **{f"{s}_toa": np.nan for s in
                            ["min", "1qt", "median", "3qt", "max", "mean", "std"]}})
    return pd.DataFrame(rows)


# PIXEL-CUBE PERSISTENCE  (GeoTIFF, optional — for offline manual QC)
def write_cube_geotiff(cube, path):
    """Persist the windowed in-patch cube to a georeferenced multi-band GeoTIFF.

    Off by default (``extract.cache_pixels``); enable it to keep the raw per-pixel
    patch imagery for **offline manual QC** (RGB/NIR panels, cloud/dust inspection)
    without re-streaming. Bands 1-7 are the OLI-2 TOA reflectances (in
    :data:`OLI2_BANDS` order); bands 8-9 are per-pixel SZA/VZA (deg). Invalid
    pixels are NaN. Scene metadata + the band/wavelength order are stored as tags.

    Unlike the EMIT cube (NetCDF via xarray), Landsat uses GeoTIFF: it needs only
    ``rasterio`` (already a dependency), is raster-native for QC and QGIS, and the
    patch window is already a rectangular grid — so no flatten/scatter round-trip.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = [cube["refl"][b["asset"]] for b in bands_of(cube)] + [cube["sza"], cube["vza"]]
    arr = np.stack([np.asarray(x, dtype="float32") for x in layers], axis=0)
    descriptions = ([f'{b["asset"]}_{b["center_nm"]}nm' for b in bands_of(cube)]
                    + ["SZA_deg", "VZA_deg"])
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype="float32", crs=cube["crs"], transform=cube["transform"],
        nodata=float("nan"), compress="deflate",
    ) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions, 1):
            dst.set_band_description(i, d)
        dst.update_tags(
            scene_id=str(cube["scene_id"]), timestamp=str(cube["timestamp"]),
            datetime=str(cube["datetime"]), source="landsat9_toa", product="toa",
            sun_elevation=str(cube["sun_elevation"]), cloud_cover=str(cube["cloud_cover"]),
            wrs_path=str(cube["wrs_path"]), wrs_row=str(cube["wrs_row"]),
            band_order=",".join(descriptions),
            note="Windowed in-patch TOA reflectance (7 OLI-2 bands) + per-pixel "
                 "SZA/VZA; NaN = invalid. Raw, unscreened — QC at read time.",
        )
    return path


def read_cube_geotiff(path):
    """Open a Landsat QC cube GeoTIFF.

    Returns ``{data: (band, y, x) array, descriptions, tags, transform, crs}`` —
    ready to slice for RGB/NIR QC panels (the reflectance bands are the first 7,
    in :data:`OLI2_BANDS` order; SZA/VZA are the last two).
    """
    with rasterio.open(Path(path)) as src:
        return {"data": src.read(), "descriptions": list(src.descriptions),
                "tags": src.tags(), "transform": src.transform, "crs": src.crs}


def bands_of(cube):
    """The band list a cube was built from (defaults to the full OLI-2 set)."""
    return cube.get("bands", OLI2_BANDS)


# PATCH GEOMETRY  (same polygon the EMIT pipeline uses)
def load_patch(bnds_path, patch_id):
    """Return ``(patch_bounds, patch_crs, patch_wgs84)`` for one patch id.

    Reads the EMIT patch-boundary GeoPackage (site UTM CRS). *patch_bounds* +
    *patch_crs* drive the windowed COG reads; *patch_wgs84* (lon/lat polygon)
    drives the STAC search + containment test.
    """
    patches = gpd.read_file(bnds_path)
    sel = patches[patches["patch_id"] == patch_id]
    if sel.empty:
        raise ValueError(f"patch {patch_id!r} not in {bnds_path}")
    return (sel.geometry.iloc[0].bounds, patches.crs,
            sel.to_crs(4326).geometry.iloc[0])


# INTERNAL
def _ts_from_datetime(dt_iso):
    """ISO acquisition datetime -> the canonical ``YYYYMMDDThhmmss`` timestamp token."""
    if dt_iso is None:
        return None
    return pd.to_datetime(dt_iso).strftime("%Y%m%dT%H%M%S")


# TODO / open items (see the module docstring + the Landsat memory note)
# * Pixel-cube cache: available via write_cube_geotiff() (opt-in, off by default).
#   Windowed COG reads are cheap, so it is not needed for the drift fit — its
#   purpose is offline manual QC (RGB/NIR panels) without re-streaming.
# * QC worksheet: create the opt-out keep/remove + reviewed worksheet (reuse
#   emit_pics.qc_gate's schema) in a SEPARATE step — NOT auto-populated from
#   qa_pixel (per decision). The extractor writes only geometry + per-band stats.
# * RSR convolution (downstream comparison, not extraction): convolve EMIT channels
#   with the OLI-2 relative spectral response before comparing at shared bands.
