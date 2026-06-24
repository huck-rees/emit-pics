"""Generate nested, centroid-anchored square patches for a PICS site.

Reads a PICS boundary from a GeoPackage, reprojects to UTM, and tiles the
site into square patches at increasing sizes (MIN_PATCH_KM, MIN_PATCH_KM +
INCREMENT_KM, ...).  Only patches that fall entirely within the PICS polygon
are retained.  Outputs are a GeoPackage of patch polygons and a companion CSV
of patch attributes.
"""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box


# ============================================================
# LOADING
# ============================================================
def load_pics(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Load all PICS boundaries from the GeoPackage."""
    return gpd.read_file(gpkg_path)


def select_site(pics_gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Return the single-row GeoDataFrame for *name*.

    Raises ValueError (listing available names) if not found.
    """
    site = pics_gdf[pics_gdf["site_name"] == name]
    if site.empty:
        available = sorted(pics_gdf["site_name"].unique())
        raise ValueError(
            f"PICS site '{name}' not found. Available sites: {available}"
        )
    return site


# ============================================================
# TILING
# ============================================================
def _tile_one_size(cx: float, cy: float, size_m: float,
                   bounds: tuple, polygon) -> list[dict]:
    """Generate square patches on a centroid-anchored grid at one size.

    The grid origin is the polygon centroid (cx, cy).  Grid indices (i, j) are
    computed so that each cell spans [cx + i*size_m, cx + (i+1)*size_m] in
    easting and similarly in northing.  Only cells whose area overlaps the
    polygon by > 99.9 % are kept (i.e. must be fully contained).

    Row/col IDs are then normalized over the *retained* set so the top-left
    kept patch is (row=0, col=0): col increases eastward, row increases
    southward (top-left origin).

    Returns a list of dicts with keys: patch_id, size_km, geometry.
    """
    minx, miny, maxx, maxy = bounds
    size_km = int(size_m / 1000)

    # Integer index range that covers the bounding box.
    i0 = math.floor((minx - cx) / size_m)
    i1 = math.ceil((maxx - cx) / size_m)
    j0 = math.floor((miny - cy) / size_m)
    j1 = math.ceil((maxy - cy) / size_m)

    # First pass: keep only fully-contained boxes, recording grid (i, j).
    kept: list[tuple[int, int, object]] = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            x0, y0 = cx + i * size_m, cy + j * size_m
            patch = box(x0, y0, x0 + size_m, y0 + size_m)
            frac = patch.intersection(polygon).area / patch.area
            if frac > 0.999:
                kept.append((i, j, patch))

    if not kept:
        return []

    # Normalize row/col over the retained set (contiguous, no gaps).
    i_vals = sorted({i for i, _, _ in kept})
    j_vals = sorted({j for _, j, _ in kept})
    col_of = {i: c for c, i in enumerate(i_vals)}            # west -> east
    row_of = {j: r for r, j in enumerate(reversed(j_vals))}  # north -> south

    out = []
    for i, j, geom in kept:
        row, col = row_of[j], col_of[i]
        out.append({
            "patch_id": f"{size_km}_{row}_{col}",
            "size_km": size_km,
            "geometry": geom,
        })
    return out


def generate_patches(site_utm: gpd.GeoDataFrame,
                     min_km: int, inc_km: int) -> gpd.GeoDataFrame:
    """Build nested patches from *min_km* up to the largest that fits.

    The centroid of the UTM-projected site polygon anchors the grid.
    Patch sizes are min_km, min_km + inc_km, min_km + 2*inc_km, ...
    up to the largest size whose edge length does not exceed the shorter
    axis of the site bounding box.

    Returns a GeoDataFrame (in the site's UTM CRS) of fully-contained
    patches with columns: patch_id, size_km, geometry.
    """
    polygon = site_utm.geometry.iloc[0]
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y

    bounds = site_utm.total_bounds  # (minx, miny, maxx, maxy)
    max_extent_km = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 1000

    # Candidate sizes: min_km, min_km + inc_km, ... up to max_extent_km.
    sizes_km = list(range(min_km, int(max_extent_km) + 1, inc_km))

    records: list[dict] = []
    for size_km in sizes_km:
        records.extend(
            _tile_one_size(cx, cy, size_km * 1000, tuple(bounds), polygon)
        )

    if not records:
        print("Warning: no fully-contained patches at any candidate size.")
        return gpd.GeoDataFrame(
            columns=["patch_id", "size_km", "geometry"],
            crs=site_utm.crs,
        )

    return gpd.GeoDataFrame(records, crs=site_utm.crs)


# ============================================================
# OUTPUT
# ============================================================
def _safe_filename(pics_name: str) -> str:
    """Replace spaces with underscores for use in filenames."""
    return pics_name.replace(" ", "_")


def write_outputs(patches_gdf: gpd.GeoDataFrame, pics_name: str,
                  out_dir: Path) -> tuple[Path, Path]:
    """Write the GeoPackage and companion CSV.

    Returns (gpkg_path, csv_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_filename(pics_name)

    gpkg_path = out_dir / f"patches_{tag}_bnds.gpkg"
    csv_path = out_dir / f"patches_{tag}_atts.csv"

    attr_cols = ["patch_id", "size_km"]

    # GeoPackage — polygons in the site's UTM CRS.
    patches_gdf[attr_cols + ["geometry"]].to_file(gpkg_path, driver="GPKG")

    # CSV — attribute table only (no geometry column).
    patches_gdf[attr_cols].to_csv(csv_path, index=False)

    return gpkg_path, csv_path


