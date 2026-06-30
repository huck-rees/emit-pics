#!/usr/bin/env python
"""Identify EMIT granules covering each patch of a PICS site.

For both L1B Radiance and L2A Reflectance, this script:

1. Searches NASA Earthdata for all granules intersecting the PICS bounding
   box over the date range.
2. Determines which granules fully cover each patch of the specified size.
3. Writes footprint, coverage-count, and patch–granule dictionary outputs.

Edit the user-configurable inputs below, then run::

    python scripts/run_id_granules.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_NAME     = "Mauritania 2"       # site_name value in the GeoPackage
PATCH_SIZE_KM = 10              # patch edge length to analyze (km)
DATE_START    = "2022-08-01"
DATE_END      = "2026-06-30"

EMIT_PRODUCTS = {
    "L1B_RAD": "EMITL1BRAD",
    "L2A_RFL": "EMITL2ARFL",
}

# Human-readable titles used for the per-product figure panels.
EMIT_PRODUCT_TITLES = {
    "L1B_RAD": "L1B Radiance",
    "L2A_RFL": "L2A Reflectance",
}

# Paths (relative to the repo root assumed two levels above this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../Code/EMIT
_DATA_ROOT = _REPO_ROOT.parent.parent / "Data"               # .../JPL/Data
PICS_GPKG  = _DATA_ROOT / "ROIs" / "northafrican_pics.gpkg"
PATCH_DIR  = _DATA_ROOT / "ROIs" / "patches"
OUT_DIR_FP = _DATA_ROOT / "ROIs"                             # footprints output
FIG_DIR    = Path(r"C:\Users\huckr\Desktop\JPL\Figures")     # coverage figure

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import earthaccess
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics.id_granules import (
    load_pics,
    load_patches,
    search_granules,
    footprints_to_gdf,
    patches_covered_by_granule,
    plot_patch_coverage,
)
from emit_pics.patch_generator import _safe_filename


def main() -> None:
    tag = _safe_filename(PICS_NAME)

    # 0. Authenticate to NASA Earthdata.
    earthaccess.login()
    print("Authenticated to NASA Earthdata.\n")

    # 1. Load PICS site and derive bounding box.
    site = load_pics(PICS_GPKG, PICS_NAME)
    bbox = tuple(site.total_bounds)  # (west, south, east, north) in EPSG:4326
    print(f"Site: {PICS_NAME}")
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
    fp_out = OUT_DIR_FP / f"emit_granule_footprints_{tag}.gpkg"
    footprints.to_file(fp_out, driver="GPKG")
    print(f"\nFootprints GeoPackage: {fp_out}")
    print(f"  {len(footprints)} footprint(s) across "
          f"{footprints['product'].nunique()} product(s)")

    # 4. Load patches for the requested size.
    bnds_path = PATCH_DIR / f"patches_{tag}_bnds.gpkg"
    csv_path  = PATCH_DIR / f"patches_{tag}_atts.csv"
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

    # 6. Compute overlap counts and dense ranks (per-size).
    #    Counts/ranks use pandas' nullable Int64 so rows for sizes not yet
    #    processed can hold <NA> while these columns stay integer-typed.
    patch_ids = patches["patch_id"].tolist()
    stats = pd.DataFrame({"patch_id": patch_ids})

    for label, suffix in [("L1B_RAD", "rad"), ("L2A_RFL", "rfl")]:
        counts = [len(coverage[label].get(pid, [])) for pid in patch_ids]
        stats[f"n_gran_ovlp_{suffix}"] = pd.array(counts, dtype="Int64")
        stats[f"n_rank_{suffix}"] = (
            stats[f"n_gran_ovlp_{suffix}"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )

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

    dict_path = PATCH_DIR / f"patch_EMIT_dict_{tag}.csv"
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
            f"{PICS_NAME} — {PATCH_SIZE_KM} km patches by granule count "
            f"(darker = more)"
        ),
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"{tag}_{PATCH_SIZE_KM}km_EMITgranule_coverage.png"
    fig.savefig(fig_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCoverage figure:    {fig_path}")

    # 10. Summary.
    print("\n--- Summary ---")
    for label, suffix in [("L1B_RAD", "rad"), ("L2A_RFL", "rfl")]:
        col = f"n_gran_ovlp_{suffix}"
        s = stats[col]
        print(f"  {label}: overlap count  min={s.min()}, max={s.max()}, "
              f"mean={s.mean():.1f}")


if __name__ == "__main__":
    main()
