#!/usr/bin/env python
"""Generate nested, centroid-anchored square patches for a PICS site.

Edit the user-configurable inputs below, then run:

    python scripts/run_patch_generator.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_NAME = "Libya 4"       # site_name value in the GeoPackage
MIN_PATCH_KM = 10           # smallest patch edge length (km)
INCREMENT_KM = 10           # step between successive patch sizes (km)

# Paths
_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../Code/EMIT
_DATA_ROOT = _REPO_ROOT.parent.parent / "Data"               # .../JPL/Data
PICS_GPKG = _DATA_ROOT / "ROIs" / "northafrican_pics.gpkg"
OUT_DIR = _DATA_ROOT / "ROIs" / "patches"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

from emit_pics.patch_generator import (
    generate_patches,
    load_pics,
    select_site,
    write_outputs,
)


def main() -> None:
    # 1. Load PICS boundaries and select the requested site.
    pics_gdf = load_pics(PICS_GPKG)
    site = select_site(pics_gdf, PICS_NAME)
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
    print(f"\nGeoPackage: {gpkg_path}")
    print(f"CSV:        {csv_path}")


if __name__ == "__main__":
    main()
