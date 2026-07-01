#!/usr/bin/env python
"""Unified per-acquisition extraction over one patch — open once, mask once.

For every EMIT acquisition that covers the patch (read from the patch-granule
dictionary; no live search), this script opens each granule's files **once**,
builds the in-patch pixel mask **once**, and from that single in-memory cube
derives and persists everything downstream needs:

1. per-channel L1B radiance and L2A reflectance summary stats (one CSV each),
2. a per-scene SZA/VZA geometry table (means, percentiles and valid-pixel count)
   that the matched-pairs post-processor reads with no Earthdata calls,
3. per-MASK-band summaries (flagged-% for binary flags, stats for AOD/H2O), and
4. (optionally) the raw, unscreened in-patch pixel cube to NetCDF, so future
   screening and BRDF-correction passes re-process the same pixels locally
   without ever re-streaming.

This consolidates the streaming/masking work that previously lived separately in
``run_extract_patch_stats.py`` and the streaming half of ``run_id_matched_pairs.py``;
each granule is now opened once instead of two-to-three times.

Optionally screens per-channel stats via the L2A MASK Aggregate Flag (dropping an
acquisition whose in-patch valid fraction falls below a threshold).  Screening
affects only the per-channel stats — the geometry table and the cached pixel
cube are always raw and unscreened.

PIXEL-CUBE CACHE SIZING (per acquisition, per product): a 10 km patch is
~28k pixels (~32 MB); cache scales with patch area (quadratic in edge length), so
a 40 km patch is ~16x that (~500 MB). Enable CACHE_PIXELS only for the patches the
granule-set-cover selection actually uses; do not cache all nested sizes en masse.

Edit the user-configurable inputs below, then run::

    python scripts/run_patch_extract.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the dictionary / bnds gpkg

CACHE_PIXELS    = True      # write the raw in-patch pixel cube to NetCDF
WANT_OBS        = True      # stream OBS geometry (SZA/VZA) for the geometry table

FILTER_AGG_FLAG = False     # True -> screen per-channel stats via MASK Aggregate Flag
MIN_VALID_FRAC  = 0.60      # min in-patch valid fraction (only if filtering)

RAD_SHORT_NAME = "EMITL1BRAD"   # L1B radiance (+ bundled OBS) collection
RFL_SHORT_NAME = "EMITL2ARFL"   # L2A reflectance (+ MASK) collection

# Paths (repo-relative — identical on every machine)
_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics
DATA_ROOT = _REPO_ROOT / "data"
PATCH_DIR = DATA_ROOT / "ROIs" / "patches"
RAD_DIR   = DATA_ROOT / "radiance"
RFL_DIR   = DATA_ROOT / "reflectance"
MASK_DIR  = DATA_ROOT / "mask_summary"
GEOM_DIR  = DATA_ROOT / "geometry"          # per-scene SZA/VZA table (matched-pairs input)
CUBE_DIR  = DATA_ROOT / "pixel_cubes"       # raw in-patch pixel cubes (NetCDF)

DICT_PATH = PATCH_DIR / f"patch_EMIT_dict_{PICS_ID}.csv"
BNDS_PATH = PATCH_DIR / f"patches_{PICS_ID}_bnds.gpkg"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import earthaccess
import numpy as np
import pandas as pd

from emit_pics.id_matched_pairs import (
    STAT_COLS,
    acq_timestamp,
    load_patch_geom_4326,
    sniff_delimiter,
)
from emit_pics.extract_patch_stats import (
    load_patch_dict,
    pair_acquisitions,
    split_products,
    write_perchannel_csv,
)
from emit_pics.patch_extract import (
    aggregate_keep,
    channel_stats_from_cube,
    extract_acquisition,
    geometry_stats_from_cube,
    mask_summaries_from_cube,
    write_cube_netcdf,
)


def process_acquisition(acq, patch_geom):
    """Extract one acquisition once and write all of its outputs.

    Returns ``(status, geom_record)`` where *status* is ``"produced"`` (>=1
    per-channel CSV written), ``"screened"`` (per-channel stats dropped by the
    Aggregate-Flag guard), or ``"missing"`` (no in-patch pixels / no product).
    *geom_record* is the per-scene geometry row (or ``None`` if no OBS); it is
    recorded regardless of screening, since matched-pair geometry is independent
    of cloud cover.  Raises on transient errors that survive retries.
    """
    ts = acq["timestamp"]
    cube = extract_acquisition(
        acq, patch_geom, patch_id=PATCH_ID,
        rfl_short=RFL_SHORT_NAME, rad_short=RAD_SHORT_NAME, want_obs=WANT_OBS,
    )
    if cube is None:
        print(f"  skipped {ts}: no product available")
        return "missing", None
    if cube["n_pixels"] == 0:
        print(f"  skipped {ts}: no in-patch pixels")
        return "missing", None

    # --- per-scene geometry (raw; unaffected by screening) ---
    geom = geometry_stats_from_cube(cube)
    geom_record = None
    if geom is not None:
        geom_record = {
            "scene_id": cube["rad_ur"], "granule_ur": cube["rad_ur"],
            "timestamp": ts, "datetime": cube["datetime"],
            "has_rfl": cube["has_rfl"], **geom,
        }

    # --- raw pixel cube (always unscreened; the cache's whole point) ---
    if CACHE_PIXELS:
        cube_path = CUBE_DIR / f"cube_{PICS_ID}_{PATCH_ID}_{ts}.nc"
        write_cube_netcdf(cube, cube_path)
        print(f"  cached {cube_path.name} ({cube['n_pixels']} px)")

    # --- MASK-band summaries (raw) ---
    if cube["mask"] is not None:
        mdf = mask_summaries_from_cube(cube)
        mpath = MASK_DIR / f"{PICS_ID}_{PATCH_ID}_{ts}_mask_summary.csv"
        write_perchannel_csv(mdf, mpath)

    # --- per-channel stats (the only output gated by screening) ---
    keep = None
    if FILTER_AGG_FLAG:
        keep, vf = aggregate_keep(cube, MIN_VALID_FRAC)
        if keep is None:
            print(f"  {ts}: no L2A MASK -> cannot screen; per-channel stats skipped")
            return "screened", geom_record
        if vf < MIN_VALID_FRAC:
            print(f"  dropped {ts}: {vf:.0%} valid < {MIN_VALID_FRAC:.0%}")
            return "screened", geom_record

    rad_df, rfl_df = channel_stats_from_cube(cube, keep)
    n_written = 0
    if rfl_df is not None:
        rfl_path = RFL_DIR / f"{PICS_ID}_{PATCH_ID}_{ts}_perchannel_rfl.csv"
        write_perchannel_csv(rfl_df, rfl_path)
        print(f"  wrote {rfl_path.name}")
        n_written += 1
    if rad_df is not None:
        rad_path = RAD_DIR / f"{PICS_ID}_{PATCH_ID}_{ts}_perchannel_rad.csv"
        write_perchannel_csv(rad_df, rad_path)
        print(f"  wrote {rad_path.name}")
        n_written += 1

    return ("produced" if n_written else "missing"), geom_record


def amend_dict_geometry(dict_path, geom_records):
    """Amend the patch-granule dict with per-granule SZA/VZA stats (row-scoped).

    RAD rows get their streamed values by ``granule_ur``; RFL rows get the
    coincident RAD acquisition's values matched by timestamp.  Migrated here from
    the old matched-pairs script so all streaming-derived persistence lives in
    the single extraction pass.
    """
    sep = sniff_delimiter(dict_path)
    stats_by_ur = {r["granule_ur"]: r for r in geom_records}
    stats_by_ts = {r["timestamp"]: r for r in geom_records}

    ddf = pd.read_csv(dict_path, sep=sep)
    for col in STAT_COLS:
        if col not in ddf.columns:
            ddf[col] = np.nan

    for idx, drow in ddf.iterrows():
        if drow["patch_id"] != PATCH_ID:
            continue
        if drow["product"] == "L1B_RAD":
            rec = stats_by_ur.get(drow["granule_ur"])
        elif drow["product"] == "L2A_RFL":
            rec = stats_by_ts.get(acq_timestamp(drow["granule_ur"]))
        else:
            rec = None
        if rec is None:
            continue
        for col in STAT_COLS:
            ddf.at[idx, col] = rec[col]

    ddf.to_csv(dict_path, sep=sep, index=False)
    print(f"Amended dict:       {dict_path}")


def main() -> None:
    # 0. Authenticate to NASA Earthdata (persisted for streaming reads).
    earthaccess.login(persist=True)
    print("Authenticated to NASA Earthdata.\n")

    # 1. Load the patch-granule dict and pair acquisitions by timestamp.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    rad_urs, rfl_urs = split_products(dict_df, PATCH_ID)
    acquisitions = pair_acquisitions(rad_urs, rfl_urs)
    print(f"Patch {PATCH_ID}: {len(rad_urs)} RAD, {len(rfl_urs)} RFL granule(s) "
          f"-> {len(acquisitions)} acquisition(s)")
    if not acquisitions:
        print("No acquisitions cover this patch. Exiting.")
        sys.exit(1)
    if FILTER_AGG_FLAG:
        print(f"Aggregate-Flag screening ON (min valid frac {MIN_VALID_FRAC:.0%})")
    if CACHE_PIXELS:
        print(f"Pixel-cube caching ON -> {CUBE_DIR}")

    # 2. Patch geometry (reprojected to EPSG:4326 for the pixel mask).
    patch_geom = load_patch_geom_4326(BNDS_PATH, PATCH_ID)

    # 3. Process every acquisition once; bucket outcomes and collect geometry.
    produced, screened, errored, missing = [], [], [], []
    geom_records = []
    for k, acq in enumerate(acquisitions, 1):
        ts = acq["timestamp"]
        print(f"\n[{k}/{len(acquisitions)}] {ts}")
        try:
            status, geom_record = process_acquisition(acq, patch_geom)
        except Exception as ex:
            print(f"  FAILED {ts}: {type(ex).__name__}: {ex}")
            errored.append(ts)
            continue
        {"produced": produced, "screened": screened,
         "missing": missing}[status].append(ts)
        if geom_record is not None:
            geom_records.append(geom_record)

    # 4. Write the per-scene geometry table and amend the dict's SZA/VZA columns.
    if geom_records:
        geom_df = (pd.DataFrame(geom_records)
                   .sort_values("datetime").reset_index(drop=True))
        GEOM_DIR.mkdir(parents=True, exist_ok=True)
        geom_path = GEOM_DIR / f"geometry_{PICS_ID}_{PATCH_ID}.csv"
        geom_df.to_csv(geom_path, index=False)
        print(f"\nGeometry table:     {geom_path}  ({len(geom_df)} scene(s))")
        amend_dict_geometry(DICT_PATH, geom_records)
    else:
        print("\nNo OBS geometry collected; geometry table not written.")

    # 5. Completeness summary — distinguish screening drops from error drops.
    print("\n--- Completeness ---")
    print(f"  acquisitions expected:    {len(acquisitions)}")
    print(f"  produced per-channel:     {len(produced)}")
    print(f"  dropped by screening:     {len(screened)}")
    print(f"  dropped by errors:        {len(errored)}")
    print(f"  no output (missing prod): {len(missing)}")
    if errored:
        print(f"  WARNING: {len(errored)} acquisition(s) dropped by errors "
              f"after retries -- run is INCOMPLETE; consider re-running: "
              f"{errored}")


if __name__ == "__main__":
    main()
