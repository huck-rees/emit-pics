"""EMIT granule search, download, and manifest generation.

This module handles the first pipeline stage: finding and retrieving
EMIT granules from NASA Earthdata that intersect a given PICS site
within a date range.

EMIT data products used:
    - EMITL1BRAD (L1B Radiance)
    - EMITL1BOBS (L1B Observation geometry — SZA, VZA, etc.)
    - EMITL2ARFL (L2A Reflectance, uncertainty, and mask)

Each product is a separate NetCDF4 granule. We search and download all
three so downstream steps have radiance, geometry, and reflectance in
one place.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import earthaccess

logger = logging.getLogger(__name__)

# EMIT short_name identifiers on Earthdata.
EMIT_PRODUCTS = {
    "L1B_RAD": "EMITL1BRAD",
    "L1B_OBS": "EMITL1BOBS",
    "L2A_RFL": "EMITL2ARFL",
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate() -> None:
    """Authenticate to NASA Earthdata.

    Uses earthaccess.login() which checks, in order:
        1. Existing valid token in ~/.earthaccess/
        2. Credentials in .netrc / _netrc
        3. Interactive prompt

    No credentials are stored in this codebase.
    """
    earthaccess.login()
    logger.info("Authenticated to NASA Earthdata.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_granules(
    short_name: str,
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
) -> list:
    """Search Earthdata for EMIT granules intersecting a point and date range.

    Parameters
    ----------
    short_name : str
        Earthdata collection short_name, e.g. 'EMITL1BRAD'.
    lat, lon : float
        Center coordinates of the site (decimal degrees).
    date_start, date_end : str
        ISO-8601 date strings, e.g. '2022-08-01'.

    Returns
    -------
    list
        earthaccess granule result objects.
    """
    results = earthaccess.search_data(
        short_name=short_name,
        point=(lon, lat),  # earthaccess expects (lon, lat)
        temporal=(date_start, date_end),
    )
    logger.info(
        "Found %d granules for %s at (%.2f, %.2f) [%s to %s]",
        len(results), short_name, lat, lon, date_start, date_end,
    )
    return results


def search_all_products(
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
) -> dict[str, list]:
    """Search for all three EMIT products at a site.

    Returns a dict keyed by product label (e.g. 'L1B_RAD') with lists
    of granule results.
    """
    results = {}
    for label, short_name in EMIT_PRODUCTS.items():
        results[label] = search_granules(
            short_name, lat, lon, date_start, date_end,
        )
    return results


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_granules(
    granules: list,
    dest_dir: Path = DEFAULT_DATA_DIR,
) -> list[Path]:
    """Download a list of granules to a local directory.

    Parameters
    ----------
    granules : list
        earthaccess granule objects from a search call.
    dest_dir : Path
        Local directory to write files into.

    Returns
    -------
    list[Path]
        Paths to downloaded files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(granules, str(dest_dir))
    paths = [Path(f) for f in files]
    logger.info("Downloaded %d files to %s", len(paths), dest_dir)
    return paths


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _extract_granule_metadata(granule) -> dict[str, Any]:
    """Pull provenance-relevant metadata from an earthaccess granule object.

    We capture the processing/software build version so that downstream
    analyses can flag granules processed with different algorithm versions.
    """
    umm = granule.get("umm", granule) if isinstance(granule, dict) else granule

    # earthaccess granule objects expose a dict-like interface.
    # The exact structure depends on the CMR UMM response; we pull what
    # we can and degrade gracefully.
    meta: dict[str, Any] = {}

    try:
        meta["granule_ur"] = granule["umm"]["GranuleUR"]
    except (KeyError, TypeError):
        meta["granule_ur"] = str(granule)

    try:
        meta["collection"] = granule["umm"]["CollectionReference"]["ShortName"]
    except (KeyError, TypeError):
        meta["collection"] = None

    try:
        meta["version"] = granule["umm"]["CollectionReference"]["Version"]
    except (KeyError, TypeError):
        meta["version"] = None

    # PGEVersion / software build version — lives in
    # AdditionalAttributes or DataGranule depending on the provider.
    try:
        attrs = granule["umm"].get("AdditionalAttributes", [])
        for attr in attrs:
            if attr.get("Name") in ("PGEVersion", "SOFTWARE_BUILD_VERSION"):
                meta["pge_version"] = attr["Values"][0]
                break
    except (KeyError, TypeError):
        pass
    if "pge_version" not in meta:
        meta["pge_version"] = None

    try:
        time_range = granule["umm"]["TemporalExtent"]["RangeDateTime"]
        meta["time_start"] = time_range.get("BeginningDateTime")
        meta["time_end"] = time_range.get("EndingDateTime")
    except (KeyError, TypeError):
        meta["time_start"] = None
        meta["time_end"] = None

    return meta


def write_manifest(
    all_granules: dict[str, list],
    site_id: str,
    dest_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Write a JSON manifest of downloaded granules with provenance metadata.

    Parameters
    ----------
    all_granules : dict[str, list]
        Dict keyed by product label ('L1B_RAD', etc.) mapping to lists
        of earthaccess granule objects.
    site_id : str
        Identifier for the PICS site (used in the filename).
    dest_dir : Path
        Directory to write the manifest into.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest: dict[str, Any] = {
        "site": site_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "products": {},
    }

    for label, granules in all_granules.items():
        manifest["products"][label] = [
            _extract_granule_metadata(g) for g in granules
        ]

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"manifest_{site_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote manifest to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Stubs — next pipeline stages (not yet implemented)
# ---------------------------------------------------------------------------

def apply_glt(radiance_path: Path, glt_path: Path) -> Any:
    """Apply the Geographic Lookup Table to orthorectify a granule.

    EMIT distributes data in spatially raw (non-orthorectified) format.
    Each granule ships with a GLT that maps (ortho_row, ortho_col) to
    (raw_row, raw_col). Applying the GLT resamples the raw swath onto
    a regular geographic grid so that pixel locations correspond to real
    lat/lon coordinates.

    Parameters
    ----------
    radiance_path : Path
        Path to the L1B RAD NetCDF4 file.
    glt_path : Path
        Path to the GLT file (may be embedded in the RAD file or
        distributed separately depending on the product version).

    Returns
    -------
    xarray.Dataset
        Orthorectified radiance on a regular lat/lon grid.

    TODO
    ----
    - Read the GLT x/y index arrays from the NetCDF4 file.
    - Handle the GLT sign convention (negative indices indicate
      aggregated/fill pixels in some versions).
    - Apply the lookup to remap each band of the radiance cube.
    - Attach georeferencing metadata (CRS, transform) to the output.
    """
    raise NotImplementedError("GLT orthorectification not yet implemented.")


def extract_geometry_bands(obs_path: Path) -> dict[str, Any]:
    """Extract SZA, VZA, and other geometry bands from the OBS file.

    The EMIT L1B OBS file stores observation geometry as an unlabeled
    multiband array. Band semantics must be determined from the
    'band_parameters' metadata group, which lists band names/descriptions
    in order.

    Known band layout (verify against metadata in the exploration notebook):
        Band  0: path_length
        Band  1: to_sensor_azimuth
        Band  2: to_sensor_zenith   (VZA)
        Band  3: to_sun_azimuth
        Band  4: to_sun_zenith      (SZA)
        Band  5: solar_phase
        Band  6: slope
        Band  7: aspect
        Band  8: cosine(i)
        Band  9: UTC_time

    Parameters
    ----------
    obs_path : Path
        Path to the L1B OBS NetCDF4 file.

    Returns
    -------
    dict
        Keys 'sza', 'vza' (and potentially others) mapping to 2-D numpy
        arrays in the raw (non-ortho) pixel grid.

    TODO
    ----
    - Open OBS file with xarray, read the obs data variable.
    - Read band names from the 'observation_bands' variable or
      sensor_band_parameters group to confirm index mapping.
    - Slice out SZA and VZA bands by index.
    - Mask fill/no-data values.
    """
    raise NotImplementedError("OBS geometry extraction not yet implemented.")


def compute_roi_mean(
    data: Any,
    center_lat: float,
    center_lon: float,
    roi_size_km: float,
) -> Any:
    """Compute the spatial mean over a square ROI centered on a PICS site.

    After orthorectification (apply_glt), this function clips the data
    to a square region defined by center coordinates and a side length,
    then computes the per-band spatial mean.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Orthorectified data with lat/lon coordinates.
    center_lat, center_lon : float
        ROI center in decimal degrees.
    roi_size_km : float
        Side length of the square ROI in kilometers.

    Returns
    -------
    xarray.DataArray or pandas.Series
        Per-band spatial mean within the ROI.

    TODO
    ----
    - Convert roi_size_km to degree offsets (approximate at this latitude).
    - Clip the dataset to the bounding box.
    - Apply the quality/mask layer to exclude cloudy or flagged pixels.
    - Compute per-band mean, also record pixel count and std for QC.
    """
    raise NotImplementedError("ROI mean computation not yet implemented.")
