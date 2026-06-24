import time
import random
import re
import numpy as np
import os
import rioxarray
import s3fs
import click
import requests
import pandas
import geopandas
import shapely
from shapely import geometry
import rasterio
import xarray as xr
import earthaccess
from matplotlib import pyplot as plt
from isofit import ray
from pyproj import Transformer
from scipy.spatial import KDTree
from spectral import envi
from isofit.core.common import envi_header



emit_path = '/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_data.gpkg'
grwl_path = '/Users/bgreenbe/Projects/SpectralHydrograph/Data/grwl.gpkg'

emit_df = geopandas.read_file(emit_path)
grwl_df = geopandas.read_file(grwl_path)


total_intersections
for i, irow in emit_df.iterrows():
    intersections = []
    for j, jrow in emit_df.iterrows():
        if i == j:
            continue
        inter = shapely.intersection(jrow['geometry'], irow['geometry'])
        if not inter:
            continue
        for k, krow in emit_df.iterrows():
            if (i == k) | (j == k):
                continue
            inter = shapely.intersection(inter, krow['geometry'])
        break
    break


import geopandas as gpd
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # suppress some shapely warnings

gpkg_path = '/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_data.gpkg'
layer_name = None

gdf = gpd.read_file(gpkg_path, layer=layer_name)
# gdf = gdf.to_crs(gdf.estimate_utm_crs())
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].reset_index(drop=True)
gdf["uid"] = gdf.index

geom_list = np.array(list(gdf.geometry))
tree = STRtree(geom_list)                         # spatial index on the geometries
geom_to_uid = dict(zip(gdf["uid"], geom_list))
uid_list = list(geom_to_uid.keys())

intersections = []
# Iterate over each polygon only once (a < b) to avoid duplicate pairs.
for idx_a, uid_a in tqdm(enumerate(uid_list), total=len(geom_list),
                          desc="Scanning polygons"):
    geom_a = geom_list[uid_a]
    # Query the index – return geometries whose bounding boxes intersect geom_a
    cand_uids = tree.query(geom_a)

    for uid_b in cand_uids:
        # Skip the same polygon and the symmetric copy (a >= b)
        if uid_b <= uid_a:
            continue

        geom_b = geom_list[uid_b]

        # Fast‑reject: if the bounding boxes intersect but the actual geometries do not,
        # `intersection` will be empty – we can test with `intersects`.
        if not geom_a.intersects(geom_b):
            continue

        inter = geom_a.intersection(geom_b)
        if not inter.is_empty:
            intersections.append({
                "uid_1": uid_a,
                "uid_2": uid_b,
                "geometry": inter
            })

# Convert to GeoDataFrame
inter_gdf = gpd.GeoDataFrame(intersections, crs='epsg:4326', geometry="geometry")
print(f"Found {len(inter_gdf)} pairwise intersection geometries")
# The pairwise intersections may still overlap each other (e.g. a region
# covered by 3 polygons will appear three times).  Dissolving them
# produces a clean, non‑overlapping set of cells.
# While dissolving we also keep a count of how many input polygons contributed
# to each cell.
# merged = (
#     inter_gdf
#     .assign(dummy=1)
#     .dissolve(by="dummy", aggfunc="size")
#     .reset_index(drop=True)
# )

merged = inter_gdf.assign(dummy=1)
merged = merged.dissolve(by="dummy", aggfunc="count")


# `merged.geometry` now holds the union of *all* overlapped areas.
# If you need a per‑cell count (e.g. “this patch is overlapped by exactly 3 polygons”)
# you can compute it by a second pass that uses a point‑in‑polygon count:

def count_covering_polys(cell_geom, tree, uid_lookup):
    """Return how many original polygons contain (or intersect) `cell_geom`."""
    # Take a single point interior to the cell – `representative_point` works for all polygon types.
    pt = cell_geom.representative_point()
    # Candidate polygons from the spatial index
    candidates = tree.query(pt)
    # Count how many of those really contain the point
    return sum(1 for cand in candidates if geom_to_uid[cand].contains(pt))

# Split the merged geometry into individual (non‑overlapping) polygons first:
merged = merged.explode(index_parts=False).reset_index(drop=True)

# Compute the coverage count for each piece (≥2 by design)
coverage_counts = []
for geom in tqdm(merged.geometry, desc="Counting overlaps"):
    coverage_counts.append(count_covering_polys(geom, tree, geom_to_uid))

merged["overlap_count"] = coverage_counts

# Keep only the pieces that are really overlapped by ≥2 polygons
final_overlap = merged[merged["overlap_count"] >= 2].copy()
print(f"Result contains {len(final_overlap)} distinct overlapped polygons")

# --------------------------------------------------------------
# 5️⃣  Save the result
# --------------------------------------------------------------
out_gpkg = "/Users/bgreenbe/Projects/SpectralHydrograph/Data/overlaps_2plus_polygons.gpkg"
final_overlap.to_file(out_gpkg, driver="GPKG")
print(f"Written overlapped regions (with count) to {out_gpkg}")






inter = [grwl_df['geometry'].intersection(row['geometry'])[0] for i, row in final_overlap.iterrows()]
inter_poly = [shapely.buffer(i, 0.01, mitre_limit=2, quad_segs=8, cap_style='flat') for i in inter]

df = pandas.DataFrame(data={'i': [i for i in range(len(inter))]})
inter_gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=inter)
out_gpkg = "/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_overlap_rivers.gpkg"
inter_gdf.to_file(out_gpkg, driver="GPKG")

inter_gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=inter_poly)
out_gpkg = "/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_overlap_rivers_polys.gpkg"
inter_gdf.to_file(out_gpkg, driver="GPKG")
