#!/usr/bin/env python
"""Identify EMIT granules covering each patch of a PICS site.

For both L1B Radiance and L2A Reflectance, this script:

1. Searches NASA Earthdata for all granules intersecting the PICS bounding
   box over the date range.
2. Determines which granules fully cover each patch of the specified size.
3. Writes footprint, coverage-count, and patch–granule dictionary outputs.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_id_granules.py
"""

import sys

import earthaccess
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.config import pics_geodataframe
from emit_pics.id_granules import (
    load_patches,
    search_granules,
    footprints_to_gdf,
    patches_covered_by_granule,
    plot_patch_coverage,
)

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
SITE          = cfg.site                     # any spelling; pics_geodataframe resolves it
PATCH_SIZE_KM = cfg.search_patch_size_km     # patch edge length to analyze (km)
DATE_START    = cfg.date_start
DATE_END      = cfg.date_end

EMIT_PRODUCTS = {
    "L1B_RAD": "EMITL1BRAD",
    "L2A_RFL": "EMITL2ARFL",
}

# Human-readable titles used for the per-product figure panels.
EMIT_PRODUCT_TITLES = {
    "L1B_RAD": "L1B Radiance",
    "L2A_RFL": "L2A Reflectance",
}

# Repo-relative paths — every name is built in emit_pics.paths.
FP_OUT       = paths.granule_footprints(cfg.site)
BNDS_PATH    = paths.patch_bnds(cfg.site)
ATTS_PATH    = paths.patch_atts(cfg.site)
DICT_PATH    = paths.patch_dict(cfg.site)
COVERAGE_FIG = paths.granule_coverage_fig(cfg.site, PATCH_SIZE_KM)


def main() -> None:
    # 0. Authenticate to NASA Earthdata.
    earthaccess.login()
    print("Authenticated to NASA Earthdata.\n")

    # 1. Build PICS site rectangle (from sites.yaml) and derive bounding box.
    site = pics_geodataframe(SITE)
    bbox = tuple(site.total_bounds)  # (west, south, east, north) in EPSG:4326
    print(f"Site: {SITE}")
    print(f"Bounding box: {bbox}\n")

    # 2. For each product: search and build footprints.
    all_footprints = []
    for label, short_name in EMIT_PRODUCTS.items():
        print(f"Searching {label} ({short_name}) ...")
        granules = search_granules(short_name, bbox, (DATE_START, DATE_END))
        print(f"  Found {len(granules)} granule(s).")
        if granules:
            fp_gdf = footprints_to_gdf(granules, label)
            all_footprints.append(fp_gdf)

    if not all_footprints:
        print("\nNo granules found for any product. Exiting.")
        sys.exit(1)

    footprints = gpd.GeoDataFrame(
        pd.concat(all_footprints, ignore_index=True), crs="EPSG:4326"
    )

    # 3. Write combined footprints GeoPackage.
    FP_OUT.parent.mkdir(parents=True, exist_ok=True)
    footprints.to_file(FP_OUT, driver="GPKG")
    print(f"\nFootprints GeoPackage: {FP_OUT}")
    print(f"  {len(footprints)} footprint(s) across "
          f"{footprints['product'].nunique()} product(s)")

    # 4. Load patches for the requested size.
    bnds_path = BNDS_PATH
    csv_path  = ATTS_PATH
    patches = load_patches(bnds_path, PATCH_SIZE_KM)
    print(f"\nLoaded {len(patches)} patch(es) of size {PATCH_SIZE_KM} km")

    # 5. Per-product coverage: which granules fully cover each patch.
    coverage: dict[str, dict[str, list[str]]] = {}
    for label in EMIT_PRODUCTS:
        fp_product = footprints[footprints["product"] == label]
        if fp_product.empty:
            coverage[label] = {pid: [] for pid in patches["patch_id"]}
        else:
            coverage[label] = patches_covered_by_granule(patches, fp_product)

    # 6. Compute per-product overlap counts (per-size).
    #    Counts use pandas' nullable Int64 so rows for sizes not yet processed
    #    can hold <NA> while these columns stay integer-typed.
    patch_ids = patches["patch_id"].tolist()
    stats = pd.DataFrame({"patch_id": patch_ids})

    for label, suffix in [("L1B_RAD", "rad"), ("L2A_RFL", "rfl")]:
        counts = [len(coverage[label].get(pid, [])) for pid in patch_ids]
        stats[f"n_gran_ovlp_{suffix}"] = pd.array(counts, dtype="Int64")

    # 7. Amend existing GeoPackage and CSV — ROW-scoped to PATCH_SIZE_KM.
    #    Coverage columns are updated only for the current size's patch_ids;
    #    rows for other sizes (e.g. an earlier 10 km run) are left intact.
    cov_cols = [c for c in stats.columns if c != "patch_id"]
    updates = stats.set_index("patch_id")

    def _amend(existing):
        """Row-scoped, in-place update of coverage columns by patch_id."""
        existing = existing.set_index("patch_id")
        for col in cov_cols:
            # Create the column on first run; coerce to nullable Int64 so
            # untouched sizes read back as <NA>, not 0 or NaN-float.
            if col not in existing.columns:
                existing[col] = pd.Series(pd.NA, index=existing.index,
                                          dtype="Int64")
            else:
                existing[col] = existing[col].astype("Int64")
            # Overwrite only the rows for the size just processed.
            existing.loc[updates.index, col] = updates[col]
        return existing.reset_index()

    existing_gpkg = gpd.read_file(bnds_path)
    existing_csv  = pd.read_csv(csv_path)

    amended_gpkg = _amend(existing_gpkg)
    amended_csv  = _amend(existing_csv)

    amended_gpkg.to_file(bnds_path, driver="GPKG")
    amended_csv.to_csv(csv_path, index=False)
    print(f"\nAmended GeoPackage: {bnds_path}")
    print(f"Amended CSV:        {csv_path}")

    # 8. Write patch–granule dictionary (long format), ROW-scoped by size.
    #    A size_km column makes the per-size overwrite explicit: prior rows
    #    for PATCH_SIZE_KM are dropped, then the fresh ones appended.
    dict_rows = []
    for label in EMIT_PRODUCTS:
        for pid in patch_ids:
            for gur in coverage[label].get(pid, []):
                dict_rows.append({
                    "patch_id": pid,
                    "size_km": PATCH_SIZE_KM,
                    "product": label,
                    "granule_ur": gur,
                })

    new_dict = pd.DataFrame(
        dict_rows,
        columns=["patch_id", "size_km", "product", "granule_ur"],
    )

    dict_path = DICT_PATH
    if dict_path.exists():
        existing_dict = pd.read_csv(dict_path)
        if "size_km" not in existing_dict.columns:
            raise ValueError(
                f"{dict_path} predates the size_km schema and cannot be "
                f"row-scoped safely. Delete it to regenerate from scratch, "
                f"or add a size_km column to the legacy rows first."
            )
        # Drop prior rows for this size, then append the fresh ones.
        existing_dict = existing_dict[existing_dict["size_km"] != PATCH_SIZE_KM]
        out_dict = pd.concat([existing_dict, new_dict], ignore_index=True)
    else:
        out_dict = new_dict

    out_dict.to_csv(dict_path, index=False)
    print(f"\nPatch-granule dict:  {dict_path}")
    print(f"  {len(new_dict)} new row(s) for {PATCH_SIZE_KM} km; "
          f"{len(out_dict)} row(s) total")

    # 9. Two-panel coverage figure (one map per product).
    fig = plot_patch_coverage(
        patches=patches,
        site=site,
        coverage=coverage,
        product_titles={
            label: EMIT_PRODUCT_TITLES.get(label, label)
            for label in EMIT_PRODUCTS
        },
        suptitle=(
            f"{SITE} — {PATCH_SIZE_KM} km patches by granule count "
            f"(darker = more)"
        ),
    )
    COVERAGE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COVERAGE_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCoverage figure:    {COVERAGE_FIG}")

    # 10. Summary.
    print("\n--- Summary ---")
    for label, suffix in [("L1B_RAD", "rad"), ("L2A_RFL", "rfl")]:
        col = f"n_gran_ovlp_{suffix}"
        s = stats[col]
        print(f"  {label}: overlap count  min={s.min()}, max={s.max()}, "
              f"mean={s.mean():.1f}")


if __name__ == "__main__":
    main()
