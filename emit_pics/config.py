"""Load PICS site definitions from YAML and build their geometry.

``config/sites.yaml`` is the single source of truth for PICS site geometry.
Each site defines an axis-aligned bounding box in EPSG:4326 (decimal degrees);
this module turns that box into the shapely polygons and GeoDataFrames the rest
of the pipeline consumes, and can regenerate the companion
``northafrican_pics.gpkg`` from the YAML — so no external GeoPackage seed is
required to run the pipeline.

Site lookup accepts either the YAML slug key (``"libya_4"``), the spaced
``site_name`` (``"Libya 4"``), or the underscored id (``"Libya_4"``);
all are matched case-insensitively.
"""

from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sites.yaml"

# The gpkg column order to reproduce when exporting from YAML (mirrors the
# original northafrican_pics.gpkg schema so downstream GIS use is unchanged).
_GPKG_COLUMNS = [
    "site_name", "country", "roi_type", "center_lat", "center_lon",
    "wrs2_path_row", "utm_zone", "altitude_m", "usable_area_km",
    "min_lon", "min_lat", "max_lon", "max_lat", "source",
]


def _norm(s: str) -> str:
    """Normalize a site identifier for matching (case/space/underscore-insensitive)."""
    return str(s).strip().lower().replace(" ", "_")


# ============================================================
# LOADING
# ============================================================
def load_sites(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load all PICS site definitions, keyed by their YAML slug (e.g. ``libya_4``)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sites", {})


def get_site(site_id: str, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate a single site by slug, ``site_name``, or underscored id.

    Returns the full site dict (including ``bbox`` and all metadata fields).

    Raises
    ------
    KeyError
        If no site matches *site_id* (lists the available site names).
    ValueError
        If the matched site has no valid 4-value ``bbox``.
    """
    sites = load_sites(config_path)
    target = _norm(site_id)

    match = None
    for slug, info in sites.items():
        if target in (_norm(slug), _norm(info.get("site_name", ""))):
            match = info
            break
    if match is None:
        names = sorted(info.get("site_name", slug) for slug, info in sites.items())
        raise KeyError(f"Unknown site '{site_id}'. Available: {names}")

    bbox = match.get("bbox") or {}
    missing = [k for k in ("min_lon", "min_lat", "max_lon", "max_lat")
               if bbox.get(k) is None]
    if missing:
        raise ValueError(
            f"Site '{match.get('site_name', site_id)}' is missing bbox "
            f"field(s) {missing} in {config_path}."
        )
    return match


def canonical_token(site_id: str, config_path: Path = DEFAULT_CONFIG) -> str:
    """Return the canonical filename token for a site (e.g. ``"Libya_4"``).

    Resolves any accepted spelling — slug (``"libya_4"``), spaced ``site_name``
    (``"Libya 4"``) or underscored id (``"Libya_4"``) — to the single token used
    in every output filename: the site's ``site_name`` with spaces replaced by
    underscores. This reproduces the historical convention
    (``patch_generator._safe_filename`` applied to the site name) exactly, so
    existing on-disk filenames are unchanged. Routing all filename construction
    through this makes the token derived one way, in one place.
    """
    return get_site(site_id, config_path)["site_name"].replace(" ", "_")


# ============================================================
# GEOMETRY
# ============================================================
def pics_polygon(site_id: str, config_path: Path = DEFAULT_CONFIG):
    """Return a site's ROI as a shapely rectangle (Polygon) in EPSG:4326.

    Built directly from the YAML ``bbox`` — this replaces reading the boundary
    from a GeoPackage.
    """
    from shapely.geometry import box

    b = get_site(site_id, config_path)["bbox"]
    return box(b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])


def pics_geodataframe(site_id: Optional[str] = None,
                      config_path: Path = DEFAULT_CONFIG):
    """Build a GeoDataFrame of PICS site rectangles in EPSG:4326.

    This is a drop-in replacement for reading ``northafrican_pics.gpkg``: it
    carries the same metadata columns plus a rectangle ``geometry`` per site,
    so callers can use ``.total_bounds``, ``.to_crs(...)``,
    ``.estimate_utm_crs()``, ``.geometry.iloc[0]``, etc. unchanged.

    Parameters
    ----------
    site_id : str, optional
        If given, return only that site (matched by slug / site_name / id).
        If ``None``, return every site in the config.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per site, columns as in :data:`_GPKG_COLUMNS` plus ``geometry``,
        CRS EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import box

    if site_id is not None:
        sites = {"_": get_site(site_id, config_path)}
    else:
        sites = load_sites(config_path)

    records, geoms = [], []
    for info in sites.values():
        b = info["bbox"]
        records.append({
            "site_name":     info.get("site_name"),
            "country":       info.get("country"),
            "roi_type":      info.get("roi_type"),
            "center_lat":    info.get("center_lat"),
            "center_lon":    info.get("center_lon"),
            "wrs2_path_row": info.get("wrs2_path_row"),
            "utm_zone":      info.get("utm_zone"),
            "altitude_m":    info.get("altitude_m"),
            "usable_area_km": info.get("usable_area_km"),
            "min_lon": b["min_lon"], "min_lat": b["min_lat"],
            "max_lon": b["max_lon"], "max_lat": b["max_lat"],
            "source":        info.get("source"),
        })
        geoms.append(box(b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"]))

    return gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")[
        _GPKG_COLUMNS + ["geometry"]
    ]


# ============================================================
# GPKG EXPORT (option B — derived, for GIS use only)
# ============================================================
def write_pics_gpkg(out_path: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    """Regenerate ``northafrican_pics.gpkg`` from the YAML (a derived output).

    The YAML remains the single source of truth; this just materializes a
    GeoPackage for QGIS / GIS inspection. Returns the written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = pics_geodataframe(None, config_path)
    gdf.to_file(out_path, driver="GPKG")
    return out_path
