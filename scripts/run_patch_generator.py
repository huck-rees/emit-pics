#!/usr/bin/env python
"""Generate nested, centroid-anchored square patches for a PICS site.

Set the analysis target in run_config.yaml, then run:

    python scripts/run_patch_generator.py
"""

import sys

from emit_pics import paths, provenance
from emit_pics.config import canonical_token, pics_geodataframe
from emit_pics.run_config import load_config
from emit_pics.patch_generator import (
    generate_patches,
    write_outputs,
)

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PICS_NAME = canonical_token(cfg.site)   # canonical filename token, e.g. "Libya_4"
MIN_PATCH_KM = cfg.min_patch_km         # smallest patch edge length (km)
INCREMENT_KM = cfg.increment_km         # step between successive patch sizes (km)

# Repo-relative paths — every name is built in emit_pics.paths.
OUT_DIR = paths.PATCH_DIR


def main() -> None:
    # 1. Build the PICS site rectangle from sites.yaml (EPSG:4326).
    site = pics_geodataframe(PICS_NAME)
    print(f"Selected site: {PICS_NAME}")

    # 2. Reproject to the site's UTM zone; centroid becomes grid anchor.
    utm_crs = site.estimate_utm_crs()
    site_utm = site.to_crs(utm_crs)
    centroid = site_utm.geometry.iloc[0].centroid
    print(f"UTM CRS: {utm_crs}")
    print(f"Grid anchor (centroid): ({centroid.x:.1f}, {centroid.y:.1f})")

    extent = site_utm.total_bounds
    print(
        f"Site extent: {(extent[2] - extent[0]) / 1000:.1f} x "
        f"{(extent[3] - extent[1]) / 1000:.1f} km"
    )

    # 3. Generate fully-contained patches at all candidate sizes.
    patches_gdf = generate_patches(site_utm, MIN_PATCH_KM, INCREMENT_KM)
    n = len(patches_gdf)
    if n == 0:
        print("No patches generated. Exiting.")
        sys.exit(1)

    sizes = sorted(patches_gdf["size_km"].unique())
    print(f"\nGenerated {n} patch(es) across sizes: {sizes} km")

    summary = (
        patches_gdf.groupby("size_km")
        .agg(n_patches=("patch_id", "size"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    # 4. Write outputs.
    gpkg_path, csv_path = write_outputs(patches_gdf, PICS_NAME, OUT_DIR)
    provenance.write_sidecar(csv_path, cfg.as_dict())
    print(f"\nGeoPackage: {gpkg_path}")
    print(f"CSV:        {csv_path}")


if __name__ == "__main__":
    main()
