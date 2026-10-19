"""Identify EMIT granules that intersect a PICS site and fully cover patches.

Pure function library — no top-level execution, no hardcoded parameters,
no ``__main__`` block, and zero side effects on import.

Typical usage from a runner script::

    from emit_pics.id_granules import (
        load_pics, load_patches, search_granules,
        footprints_to_gdf, patches_covered_by_granule,
    )
"""

from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


# LOADING
def load_pics(gpkg_path: Path, pics_name: str) -> gpd.GeoDataFrame:
    """Load a single PICS site row from a GeoPackage.

    Parameters
    ----------
    gpkg_path : Path
        Path to the PICS GeoPackage (e.g. ``northafrican_pics.gpkg``).
    pics_name : str
        Value of the ``site_name`` column to select (e.g. ``"Libya 4"``).

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame for the requested site (native CRS).

    Raises
    ------
    ValueError
        If *pics_name* is not found, listing all available site names.
    """
    gdf = gpd.read_file(gpkg_path)
    site = gdf[gdf["site_name"] == pics_name]
    if site.empty:
        available = sorted(gdf["site_name"].unique())
        raise ValueError(
            f"PICS site '{pics_name}' not found. "
            f"Available sites: {available}"
        )
    return site


def load_patches(bnds_path: Path, size_km: int) -> gpd.GeoDataFrame:
    """Load patches from a GeoPackage, filtered to a single size.

    Parameters
    ----------
    bnds_path : Path
        Path to the patches GeoPackage
        (e.g. ``patches_Libya_4_bnds.gpkg``).
    size_km : int
        Patch edge length in km to select.

    Returns
    -------
    gpd.GeoDataFrame
        Patches of the requested size in the file's native CRS
        (typically a UTM projection).

    Raises
    ------
    ValueError
        If no patches match *size_km*, listing available sizes.
    """
    gdf = gpd.read_file(bnds_path)
    subset = gdf[gdf["size_km"] == size_km]
    if subset.empty:
        available = sorted(gdf["size_km"].unique())
        raise ValueError(
            f"No patches with size_km={size_km}. "
            f"Available sizes: {available}"
        )
    return subset


# EARTHACCESS SEARCH
def search_granules(
    short_name: str,
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
) -> list:
    """Search NASA Earthdata for EMIT granules intersecting a bounding box.

    The caller must have already called ``earthaccess.login()`` before
    invoking this function.

    Parameters
    ----------
    short_name : str
        Earthdata collection short name (e.g. ``"EMITL1BRAD"``).
    bbox : tuple of float
        ``(west, south, east, north)`` in EPSG:4326 decimal degrees.
    date_range : tuple of str
        ``(start, end)`` as ISO-8601 date strings,
        e.g. ``("2023-01-01", "2023-12-31")``.

    Returns
    -------
    list
        earthaccess granule result objects.
    """
    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=date_range,
        count=-1,
    )


# UMM METADATA HELPERS
def granule_footprint(umm: dict) -> Polygon:
    """Reconstruct a granule's footprint polygon from its UMM GPolygons.

    Parameters
    ----------
    umm : dict
        The ``granule["umm"]`` dict from an earthaccess result object.

    Returns
    -------
    shapely.geometry.Polygon
        Footprint polygon in EPSG:4326 (lon/lat coordinates).

    Raises
    ------
    KeyError
        If the required GPolygon structure is absent from the UMM record,
        with a message describing the missing key path.
    """
    try:
        gpolygons = (
            umm["SpatialExtent"]
               ["HorizontalSpatialDomain"]
               ["Geometry"]
               ["GPolygons"]
        )
        points = gpolygons[0]["Boundary"]["Points"]
    except (KeyError, IndexError, TypeError) as exc:
        raise KeyError(
            "UMM record is missing the expected GPolygon structure "
            "(SpatialExtent > HorizontalSpatialDomain > Geometry > "
            f"GPolygons > [0] > Boundary > Points): {exc}"
        ) from exc

    coords = [(pt["Longitude"], pt["Latitude"]) for pt in points]
    return Polygon(coords)


def granule_ur(umm: dict) -> str:
    """Extract the GranuleUR identifier string from UMM metadata.

    Parameters
    ----------
    umm : dict
        The ``granule["umm"]`` dict from an earthaccess result object.

    Returns
    -------
    str
        The GranuleUR string
        (e.g. ``"EMIT_L1B_RAD_001_20230302T071844_2306105_003"``).
    """
    return umm["GranuleUR"]


# FOOTPRINT GEODATAFRAME
def footprints_to_gdf(
    granules: list,
    product: str,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of granule footprints.

    Parameters
    ----------
    granules : list
        earthaccess granule result objects (as returned by
        :func:`search_granules`).
    product : str
        Product label to store in the ``product`` column
        (e.g. ``"L1B_RAD"``).

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``granule_ur``, ``product``, ``geometry``.
        CRS is EPSG:4326.
    """
    records = []
    for g in granules:
        umm = g["umm"]
        records.append({
            "granule_ur": granule_ur(umm),
            "product": product,
            "geometry": granule_footprint(umm),
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# PATCH COVERAGE
def patches_covered_by_granule(
    patch_gdf: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
) -> dict[str, list[str]]:
    """For each patch, find granules whose footprint fully contains it.

    Parameters
    ----------
    patch_gdf : gpd.GeoDataFrame
        Patches with a ``patch_id`` column.  May be in any CRS; will be
        reprojected to EPSG:4326 internally if needed.
    footprints : gpd.GeoDataFrame
        Granule footprints with ``granule_ur`` and ``geometry`` columns,
        in EPSG:4326 (as returned by :func:`footprints_to_gdf`).

    Returns
    -------
    dict[str, list[str]]
        Mapping from ``patch_id`` to a list of ``granule_ur`` strings
        whose footprint fully contains that patch.
    """
    # Ensure patches are in EPSG:4326 for the containment test.
    if patch_gdf.crs is None or patch_gdf.crs.to_epsg() != 4326:
        patches_4326 = patch_gdf.to_crs("EPSG:4326")
    else:
        patches_4326 = patch_gdf

    coverage: dict[str, list[str]] = {}
    for _, prow in patches_4326.iterrows():
        pid = prow["patch_id"]
        patch_geom = prow.geometry
        covering = [
            frow["granule_ur"]
            for _, frow in footprints.iterrows()
            if frow.geometry.contains(patch_geom)
        ]
        coverage[pid] = covering

    return coverage


# PLOTTING
def plot_patch_coverage(
    patches: gpd.GeoDataFrame,
    site: gpd.GeoDataFrame,
    coverage: dict[str, dict[str, list[str]]],
    product_titles: dict[str, str],
    suptitle: str,
):
    """Build a one-map-per-product figure of patches colored by granule count.

    Each panel shows the patch grid (drawn in the patches' native CRS, which
    is typically a UTM projection) shaded by the number of granules that fully
    cover each patch, with the PICS site boundary overlaid and every patch
    annotated with its id and granule count.

    The figure is assembled and returned but **not** saved or shown — the
    caller decides what to do with it, keeping this library free of side
    effects.  ``matplotlib`` is imported lazily inside the function so that
    callers who only use the search/coverage helpers do not need it installed.

    Parameters
    ----------
    patches : gpd.GeoDataFrame
        Patches with ``patch_id`` and ``geometry`` columns, in a projected
        CRS (e.g. UTM).  Panels are drawn in this CRS.
    site : gpd.GeoDataFrame
        Single-row PICS site GeoDataFrame; its boundary is overlaid on every
        panel.  Reprojected to the patches' CRS internally.
    coverage : dict[str, dict[str, list[str]]]
        Mapping ``product_label -> {patch_id -> [granule_ur, ...]}`` as built
        by :func:`patches_covered_by_granule` for each product.
    product_titles : dict[str, str]
        Ordered mapping ``product_label -> human-readable title``.  One panel
        is drawn per entry, in this order.
    suptitle : str
        Figure-level title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    import matplotlib.pyplot as plt

    labels = list(product_titles)
    site_proj = site.to_crs(patches.crs)

    fig, axes = plt.subplots(1, len(labels), figsize=(8 * len(labels), 8))
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        gmap = patches.copy()
        gmap["n_granules"] = gmap["patch_id"].map(
            lambda pid: len(coverage.get(label, {}).get(pid, []))
        )

        site_proj.boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=3)
        gmap.plot(
            ax=ax, column="n_granules", cmap="viridis_r",
            edgecolor="white", linewidth=0.5, legend=True,
            legend_kwds={"label": f"{label} granules", "shrink": 0.7},
        )

        for _, prow in gmap.iterrows():
            cpt = prow.geometry.centroid
            ax.annotate(
                f"{prow['patch_id']}\n{prow['n_granules']}",
                (cpt.x, cpt.y), ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=5,
            )

        ax.set_title(
            f"{product_titles[label]} — fully-covering scenes per patch",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
