#!/usr/bin/env python
"""Batched cube extraction for EVERY patch of one size across a whole PICS.

The loop-inverted counterpart to ``run_single_patch_cube_extract.py``.  Wrapping
the single-patch runner over every patch re-opens each granule, re-streams its
full-scene lon/lat and rebuilds its geolocation KDTree once *per patch it
covers* — and within a PICS a single EMIT granule covers most patches, so that is
the dominant cost.  This script instead iterates **acquisitions on the outside,
patches on the inside**: each granule is opened once, its lon/lat read once and
its KDTree built once (:func:`emit_pics.PICS_allpatches_cube_extract.open_acquisition_grid`),
then every patch that granule covers is windowed from the already-open cubes.

For each ``(patch, acquisition)`` it writes exactly what the single-patch runner
writes — so all downstream steps (matched pairs, deltas, drift, QC, spectra) see
identical per-patch inputs:

1. per-channel L1B radiance and L2A reflectance summary stats (one CSV each),
2. per-patch per-scene SZA/VZA geometry tables (the matched-pairs input),
3. per-MASK-band summaries, and
4. (optionally) the raw, unscreened in-patch pixel cube to NetCDF.

Only granule opens / lon-lat reads / KDTree builds are saved (they drop from
``O(#granules × #patches)`` to ``O(#granules)``); the windowed spectral reads are
still per-patch, since each patch's pixels differ.

PIXEL-CUBE CACHE (``CACHE_PIXELS``, ON by default): caching is the whole point of
the one full-band stream — it is what lets QC (``run_PICS_allpatches_qc.py`` /
``run_qc_review.py``) and the later BRDF pass run **offline** from local NetCDF
instead of re-streaming every scene.  A stats-only run (``CACHE_PIXELS=False``)
keeps only the reduced per-channel/geometry CSVs and discards the pixels, so any
later QC has to re-download the imagery scene by scene.

SIZING: a 10 km patch cube is ~32 MB per product per acquisition.  Caching
**all** patches × all scenes multiplies fast — estimate ``#patches × #scenes ×
~64 MB`` (~55-70 GB for a 10 km Libya_4).  Turn ``CACHE_PIXELS`` off only when you
are certain you will neither QC nor BRDF this PICS and want to spare the disk.

RESUME (``RESUME``, ON by default): a streaming run can leave a few
``(patch, acquisition)`` units dropped by transient errors — most often a
``403 Forbidden`` when a granule's time-limited pre-signed download URL expires
mid-run.  With RESUME on, any unit whose outputs already exist on disk (its pixel
cube and/or per-channel CSV — see :func:`_unit_outputs_exist`) is skipped, and an
acquisition whose patches are *all* already done is skipped **before** its grid is
opened, so no lon/lat is even streamed for it.  A re-run therefore re-streams only
the stragglers.  The failed units wrote nothing, so they are correctly seen as
not-done and retried; geometry tables are *merged* (not overwritten) so the
skipped scenes' rows survive.  Set ``RESUME = False`` to force a full re-extract.

Edit the user-configurable inputs below, then run::

    python scripts/run_PICS_allpatches_cube_extract.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID    = "Libya_4"      # underscored site id (input + output file paths)
PATCH_SIZE = "10"           # patch-size token (first field of patch_id, e.g. 10 km)

CACHE_PIXELS    = True      # write the raw in-patch pixel cube per (patch, scene).
                            # ON by default: this ONE full-band stream is what makes
                            # QC and BRDF run offline afterwards. Set False only if you
                            # are sure you will not QC/BRDF this PICS and want to spare
                            # the disk (see PIXEL-CUBE CACHE SIZING above: ~55-70 GB for
                            # a 10 km Libya_4). Streaming stats-only leaves nothing on
                            # disk to render from, so QC would re-stream every scene.
WANT_OBS        = True      # stream OBS geometry (SZA/VZA) for the geometry tables

RESUME          = True      # skip (patch, scene) units already written to disk, and
                            # skip opening any acquisition whose patches are ALL done
                            # (no lon/lat stream for it). Lets a re-run recover only the
                            # units a prior run dropped to transient 403s / URL expiry.
                            # Set False to force a full re-extract of every unit.

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
GEOM_DIR  = DATA_ROOT / "geometry"          # per-scene SZA/VZA tables (matched-pairs input)
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
from emit_pics.extract_patch_stats import load_patch_dict, write_perchannel_csv
from emit_pics.single_patch_cube_extract import (
    aggregate_keep,
    channel_stats_from_cube,
    geometry_stats_from_cube,
    mask_summaries_from_cube,
    write_cube_netcdf,
)
from emit_pics.PICS_allpatches_cube_extract import (
    build_acquisition_patch_map,
    close_grid,
    extract_patch_from_grid,
    open_acquisition_grid,
    patch_ids_for_size,
)


def process_patch_cube(cube, patch_id):
    """Write one (patch, acquisition) cube's outputs; return ``(status, geom_record)``.

    Mirrors the output half of the single-patch runner: geometry record, optional
    pixel-cube cache, MASK summary, and the (optionally Aggregate-Flag-screened)
    per-channel stats.  *status* is ``"produced"`` (>=1 per-channel CSV),
    ``"screened"`` (dropped by the valid-fraction guard) or ``"missing"`` (no
    in-patch pixels / no product).  *geom_record* is the per-scene geometry row
    (tagged with ``patch_id``) or ``None`` when no OBS was streamed.
    """
    ts = cube["timestamp"]
    if cube["n_pixels"] == 0:
        return "missing", None

    # --- per-scene geometry (raw; unaffected by screening) ---
    geom = geometry_stats_from_cube(cube)
    geom_record = None
    if geom is not None:
        geom_record = {
            "patch_id": patch_id,
            "scene_id": cube["rad_ur"], "granule_ur": cube["rad_ur"],
            "timestamp": ts, "datetime": cube["datetime"],
            "has_rfl": cube["has_rfl"], **geom,
        }

    # --- raw pixel cube (always unscreened; the cache's whole point) ---
    if CACHE_PIXELS:
        cube_path = CUBE_DIR / f"cube_{PICS_ID}_{patch_id}_{ts}.nc"
        write_cube_netcdf(cube, cube_path)

    # --- MASK-band summaries (raw) ---
    if cube["mask"] is not None:
        mdf = mask_summaries_from_cube(cube)
        write_perchannel_csv(mdf, MASK_DIR / f"{PICS_ID}_{patch_id}_{ts}_mask_summary.csv")

    # --- per-channel stats (the only output gated by screening) ---
    keep = None
    if FILTER_AGG_FLAG:
        keep, vf = aggregate_keep(cube, MIN_VALID_FRAC)
        if keep is None or vf < MIN_VALID_FRAC:
            return "screened", geom_record

    rad_df, rfl_df = channel_stats_from_cube(cube, keep)
    n_written = 0
    if rfl_df is not None:
        write_perchannel_csv(rfl_df, RFL_DIR / f"{PICS_ID}_{patch_id}_{ts}_perchannel_rfl.csv")
        n_written += 1
    if rad_df is not None:
        write_perchannel_csv(rad_df, RAD_DIR / f"{PICS_ID}_{patch_id}_{ts}_perchannel_rad.csv")
        n_written += 1

    return ("produced" if n_written else "missing"), geom_record


def _unit_outputs_exist(patch_id, ts):
    """True if this ``(patch, acquisition)`` already has outputs on disk (resume key).

    A successfully *produced* unit writes its per-channel CSV(s) and — with
    ``CACHE_PIXELS`` on (the default) — its pixel cube; a *screened* unit still
    writes the cube (caching precedes the screening gate in ``process_patch_cube``).
    Any one of those present means the unit completed a prior run and can be
    skipped.  A unit that 403'd mid-stream wrote none of them, so it is not "done"
    and gets retried.  (A genuinely empty "missing" unit also looks not-done and
    gets a quick, network-cheap re-check — it has nothing to stream.)
    """
    cube = CUBE_DIR / f"cube_{PICS_ID}_{patch_id}_{ts}.nc"
    rfl  = RFL_DIR / f"{PICS_ID}_{patch_id}_{ts}_perchannel_rfl.csv"
    rad  = RAD_DIR / f"{PICS_ID}_{patch_id}_{ts}_perchannel_rad.csv"
    return cube.exists() or rfl.exists() or rad.exists()


def write_geometry_tables(geom_by_patch):
    """Write one per-scene geometry table per patch (matched-pairs input).

    Each patch's table is **merged** with any existing table on disk (keyed by
    ``timestamp``, newly-processed rows win) before writing, so a RESUME run —
    which only holds geometry for the scenes it actually reprocessed — augments the
    prior run's tables rather than overwriting them with a subset.
    """
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for pid, recs in sorted(geom_by_patch.items()):
        if not recs:
            continue
        new_df = pd.DataFrame(recs).drop(columns=["patch_id"])
        out_path = GEOM_DIR / f"geometry_{PICS_ID}_{pid}.csv"
        if out_path.exists():
            df = pd.concat([pd.read_csv(out_path), new_df], ignore_index=True)
            df = df.drop_duplicates(subset="timestamp", keep="last")
        else:
            df = new_df
        df = df.sort_values("datetime").reset_index(drop=True)
        df.to_csv(out_path, index=False)
        n += 1
    print(f"\nGeometry tables:    {n} patch table(s) merged/written -> {GEOM_DIR}")


def amend_dict_geometry(dict_path, geom_records):
    """Amend the patch-granule dict with per-(patch, granule) SZA/VZA stats.

    Unlike the single-patch amender, records are scoped by ``(patch_id, granule)``:
    RAD rows match on ``(patch_id, granule_ur)`` and RFL rows on the coincident
    ``(patch_id, timestamp)``, since the in-patch geometry differs per patch.
    """
    sep = sniff_delimiter(dict_path)
    by_patch_ur = {(r["patch_id"], r["granule_ur"]): r for r in geom_records}
    by_patch_ts = {(r["patch_id"], r["timestamp"]): r for r in geom_records}

    ddf = pd.read_csv(dict_path, sep=sep)
    for col in STAT_COLS:
        if col not in ddf.columns:
            ddf[col] = np.nan

    for idx, drow in ddf.iterrows():
        pid = str(drow["patch_id"])
        if drow["product"] == "L1B_RAD":
            rec = by_patch_ur.get((pid, drow["granule_ur"]))
        elif drow["product"] == "L2A_RFL":
            rec = by_patch_ts.get((pid, acq_timestamp(drow["granule_ur"])))
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

    # 1. Load the patch-granule dict and enumerate this size's patches.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    # 2. Patch geometries (EPSG:4326), loaded once. A patch missing from the bnds
    #    gpkg is dropped with a warning rather than failing the whole run.
    patch_geoms = {}
    for pid in patch_ids:
        try:
            patch_geoms[pid] = load_patch_geom_4326(BNDS_PATH, pid)
        except Exception as ex:
            print(f"  WARNING: no geometry for patch {pid} ({ex}); skipped.")
    patch_ids = [p for p in patch_ids if p in patch_geoms]
    if not patch_ids:
        print("No patches with geometry. Exiting.")
        sys.exit(1)

    # 3. Invert the dict into an acquisition-outer schedule.
    acquisitions, patches_by_ts = build_acquisition_patch_map(dict_df, patch_ids)
    naive_opens = sum(len(patches_by_ts[a["timestamp"]]) for a in acquisitions)
    print(f"PICS {PICS_ID}, size {PATCH_SIZE} km: {len(patch_ids)} patch(es), "
          f"{len(acquisitions)} unique acquisition(s).")
    print(f"Granule opens: {len(acquisitions)} (batched) vs {naive_opens} "
          f"(naive per-patch wrapper) -> {naive_opens - len(acquisitions)} redundant "
          f"opens/lon-lat reads/KDTree builds avoided.")
    if FILTER_AGG_FLAG:
        print(f"Aggregate-Flag screening ON (min valid frac {MIN_VALID_FRAC:.0%})")
    if CACHE_PIXELS:
        print(f"Pixel-cube caching ON -> {CUBE_DIR}")
    if RESUME:
        print("RESUME ON -> skipping (patch, scene) units already on disk")

    # 4. One open per acquisition; extract every patch it covers.
    geom_by_patch = {pid: [] for pid in patch_ids}
    n_produced = n_screened = n_missing = n_errored = n_skipped = 0

    for k, acq in enumerate(acquisitions, 1):
        ts = acq["timestamp"]
        pids = patches_by_ts[ts]

        # RESUME: process only the patches this acquisition still owes; a fully
        # complete acquisition is skipped before its grid (lon/lat + KDTree) opens.
        pending = [p for p in pids if not _unit_outputs_exist(p, ts)] if RESUME else pids
        n_skipped += len(pids) - len(pending)
        done_note = (f", {len(pids) - len(pending)} done, {len(pending)} to do"
                     if RESUME and len(pending) != len(pids) else "")
        print(f"\n[{k}/{len(acquisitions)}] {ts}  ({len(pids)} patch(es){done_note})")
        if not pending:
            print(f"  all {len(pids)} patch(es) already complete -> skipped (no stream)")
            continue

        try:
            grid = open_acquisition_grid(
                acq, rfl_short=RFL_SHORT_NAME, rad_short=RAD_SHORT_NAME,
                want_obs=WANT_OBS)
        except Exception as ex:
            print(f"  FAILED to open {ts}: {type(ex).__name__}: {ex} "
                  f"-- {len(pending)} patch(es) dropped for this acquisition")
            n_errored += len(pending)
            continue
        if grid is None:
            print(f"  skipped {ts}: no product available -> {len(pending)} patch(es)")
            n_missing += len(pending)
            continue

        try:
            for pid in pending:
                try:
                    cube = extract_patch_from_grid(grid, patch_geoms[pid], patch_id=pid)
                    status, geom_record = process_patch_cube(cube, pid)
                except Exception as ex:
                    print(f"    FAILED {pid}: {type(ex).__name__}: {ex}")
                    n_errored += 1
                    continue
                if geom_record is not None:
                    geom_by_patch[pid].append(geom_record)
                if status == "produced":
                    n_produced += 1
                elif status == "screened":
                    n_screened += 1
                else:
                    n_missing += 1
            print(f"  done {ts}: {sum(len(geom_by_patch[p]) for p in pending)} "
                  f"geometry row(s) collected this run across its patches")
        finally:
            close_grid(grid)

    # 5. Per-patch geometry tables + dict amendment.
    all_geom = [r for recs in geom_by_patch.values() for r in recs]
    if all_geom:
        write_geometry_tables(geom_by_patch)
        amend_dict_geometry(DICT_PATH, all_geom)
    else:
        print("\nNo OBS geometry collected; geometry tables not written.")

    # 6. Completeness summary (counted in patch × acquisition units).
    total = n_produced + n_screened + n_missing + n_errored + n_skipped
    print("\n--- Completeness (patch x acquisition units) ---")
    print(f"  scheduled:                {total}")
    if RESUME:
        print(f"  skipped (already done):   {n_skipped}")
    print(f"  produced per-channel:     {n_produced}")
    print(f"  dropped by screening:     {n_screened}")
    print(f"  dropped by errors:        {n_errored}")
    print(f"  no output (missing/empty):{n_missing}")
    if n_errored:
        print(f"  WARNING: {n_errored} (patch x acquisition) dropped by errors "
              f"after retries -- run is INCOMPLETE. Re-run to recover them; RESUME "
              f"will skip the {n_skipped + n_produced} unit(s) already on disk.")


if __name__ == "__main__":
    main()
