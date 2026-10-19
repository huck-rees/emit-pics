#!/usr/bin/env python
"""Unified per-acquisition extraction over one patch — open once, mask once.

For streaming **every** patch of a given size across a whole PICS in one pass
(each granule opened once regardless of how many patches it covers), use
``scripts/run_PICS_allpatches_cube_extract.py`` instead — the batched,
loop-inverted counterpart to this single-patch runner.

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

RESUME (``RESUME``, ON by default): mirrors the batched all-patches runner. A
streaming run can drop an acquisition to a transient error (most often a
``403 Forbidden`` when a granule's pre-signed URL expires mid-run); such a unit
writes nothing. With RESUME on, any acquisition whose outputs already exist on
disk (its pixel cube and/or per-channel CSV — see :func:`_unit_outputs_exist`) is
skipped **before** its granules are opened, so a re-run re-streams only the
stragglers.  Skipped scenes' geometry is recovered from their cached cubes and
the geometry table is *merged* (not overwritten), so the previously-written rows
survive.  Set ``RESUME = False`` to force a full re-extract of every acquisition.

PIXEL-CUBE CACHE SIZING (per acquisition, per product): a 10 km patch is
~28k pixels (~32 MB); cache scales with patch area (quadratic in edge length), so
a 40 km patch is ~16x that (~500 MB). Enable CACHE_PIXELS only for the patches the
granule-set-cover selection actually uses; do not cache all nested sizes en masse.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_single_patch_cube_extract.py
"""

import sys

import earthaccess
import numpy as np
import pandas as pd

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.id_matched_pairs import (
    STAT_COLS,
    acq_datetime,
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
from emit_pics.single_patch_cube_extract import (
    aggregate_keep,
    channel_stats_from_cube,
    extract_acquisition,
    geometry_stats_from_cube,
    mask_summaries_from_cube,
    read_cube_netcdf,
    write_cube_netcdf,
)

# INPUTS — set the analysis target in run_config.yaml, not here.
cfg = load_config()
PICS_ID  = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_ID = cfg.patch

CACHE_PIXELS    = cfg.cache_pixels
WANT_OBS        = cfg.want_obs
FILTER_AGG_FLAG = cfg.filter_agg_flag
MIN_VALID_FRAC  = cfg.min_valid_frac
RAD_SHORT_NAME  = cfg.rad_short_name
RFL_SHORT_NAME  = cfg.rfl_short_name

RESUME = True   # script-local toggle: skip acquisitions already written to disk
                # (no granules opened for them). Lets a re-run recover only the
                # acquisition(s) a prior run dropped to transient 403s / URL expiry.
                # Set False to force a full re-extract of every acquisition.

# Repo-relative paths — every name is built in emit_pics.paths.
RAD_DIR   = paths.RAD_DIR
RFL_DIR   = paths.RFL_DIR
MASK_DIR  = paths.MASK_DIR
GEOM_DIR  = paths.GEOM_DIR
CUBE_DIR  = paths.CUBE_DIR
DICT_PATH = paths.patch_dict(PICS_ID)
BNDS_PATH = paths.patch_bnds(PICS_ID)


def _unit_outputs_exist(ts):
    """True if this acquisition already has outputs on disk (RESUME key).

    Mirrors the batched runner's key.  With ``CACHE_PIXELS`` on (the default) the
    pixel **cube** is the key: the per-channel CSVs are not sufficient evidence,
    because an earlier stats-only pass (``CACHE_PIXELS=False``) writes those CSVs
    while writing no cubes, and counting them would falsely mark the unit done and
    skip the very cube this run exists to create.  A cube is written before the
    screening gate, so *screened* acquisitions also have one; a unit that 403'd
    mid-stream has none and is correctly retried.  With ``CACHE_PIXELS`` off there
    is no cube, so fall back to the per-channel CSVs (the only durable artifact).
    """
    if CACHE_PIXELS:
        return paths.pixel_cube(PICS_ID, PATCH_ID, ts).exists()
    rfl = paths.perchannel(PICS_ID, PATCH_ID, ts, "rfl")
    rad = paths.perchannel(PICS_ID, PATCH_ID, ts, "rad")
    return rfl.exists() or rad.exists()


def geom_record_from_cube_file(ts):
    """Rebuild a scene's per-scene geometry record from its cached cube (no stream).

    Lets a RESUME run that SKIPS an already-cached acquisition still contribute
    that scene's geometry row, so the merged geometry table and dict amendment are
    rebuilt over the **full** set of cached scenes — not just the freshly-streamed
    one.  Returns ``None`` if the cube is missing or carries no OBS geometry (e.g.
    cached with ``WANT_OBS=False``); in that case the scene's prior row still
    survives via the table merge in :func:`main`.
    """
    cube_path = paths.pixel_cube(PICS_ID, PATCH_ID, ts)
    if not cube_path.exists():
        return None
    with read_cube_netcdf(cube_path) as ds:
        if "sza" not in ds or "vza" not in ds:
            return None
        proxy = {"sza": ds["sza"].values, "vza": ds["vza"].values,
                 "n_pixels": int(ds.sizes["pixel"])}
        geom = geometry_stats_from_cube(proxy)
        if geom is None:
            return None
        rad_ur = str(ds.attrs.get("rad_ur", ""))
        return {
            "scene_id": rad_ur, "granule_ur": rad_ur,
            "timestamp": str(ds.attrs.get("timestamp", ts)),
            "datetime": acq_datetime(rad_ur),
            "has_rfl": "reflectance" in ds,
            **geom,
        }


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
        cube_path = paths.pixel_cube(PICS_ID, PATCH_ID, ts)
        write_cube_netcdf(cube, cube_path)
        print(f"  cached {cube_path.name} ({cube['n_pixels']} px)")

    # --- MASK-band summaries (raw) ---
    if cube["mask"] is not None:
        mdf = mask_summaries_from_cube(cube)
        mpath = paths.mask_summary(PICS_ID, PATCH_ID, ts)
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
        rfl_path = paths.perchannel(PICS_ID, PATCH_ID, ts, "rfl")
        write_perchannel_csv(rfl_df, rfl_path)
        print(f"  wrote {rfl_path.name}")
        n_written += 1
    if rad_df is not None:
        rad_path = paths.perchannel(PICS_ID, PATCH_ID, ts, "rad")
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
    if RESUME:
        print("RESUME ON -> skipping acquisitions already on disk")

    # 2. Patch geometry (reprojected to EPSG:4326 for the pixel mask).
    patch_geom = load_patch_geom_4326(BNDS_PATH, PATCH_ID)

    # 3. Process every acquisition once; bucket outcomes and collect geometry.
    produced, screened, errored, missing, skipped = [], [], [], [], []
    geom_records = []
    for k, acq in enumerate(acquisitions, 1):
        ts = acq["timestamp"]

        # RESUME: an acquisition already written to disk is skipped before its
        # granules are opened; its geometry is recovered from the cached cube so
        # the merged table/dict still see it.
        if RESUME and _unit_outputs_exist(ts):
            rec = geom_record_from_cube_file(ts)
            if rec is not None:
                geom_records.append(rec)
            skipped.append(ts)
            print(f"\n[{k}/{len(acquisitions)}] {ts}  already done -> skipped "
                  f"(no stream)")
            continue

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
        geom_df = pd.DataFrame(geom_records)
        GEOM_DIR.mkdir(parents=True, exist_ok=True)
        geom_path = paths.geometry(PICS_ID, PATCH_ID)
        # Merge with any existing table (keyed by timestamp, this run's rows win)
        # so a RESUME run — which only holds geometry for the scenes it actually
        # reprocessed plus those recovered from cached cubes — augments the prior
        # table rather than overwriting it with a subset.
        if geom_path.exists():
            geom_df = pd.concat([pd.read_csv(geom_path), geom_df],
                                ignore_index=True)
            geom_df = geom_df.drop_duplicates(subset="timestamp", keep="last")
        # Sort by the canonical "timestamp" ("YYYYMMDDThhmmss"): a homogeneous
        # string in both freshly-built rows and rows read back from CSV, and
        # lexically co-monotonic with "datetime".  Sorting by "datetime" after a
        # merge can mix Timestamp (new) and str (CSV) values, which pandas cannot
        # order ("'<' not supported between Timestamp and str").
        geom_df = geom_df.sort_values("timestamp").reset_index(drop=True)
        geom_df.to_csv(geom_path, index=False)
        provenance.write_sidecar(geom_path, cfg.as_dict())
        print(f"\nGeometry table:     {geom_path}  ({len(geom_df)} scene(s))")
        amend_dict_geometry(DICT_PATH, geom_records)
    else:
        print("\nNo OBS geometry collected; geometry table not written.")

    # 5. Completeness summary — distinguish screening drops from error drops.
    print("\n--- Completeness ---")
    print(f"  acquisitions expected:    {len(acquisitions)}")
    if RESUME:
        print(f"  skipped (already done):   {len(skipped)}")
    print(f"  produced per-channel:     {len(produced)}")
    print(f"  dropped by screening:     {len(screened)}")
    print(f"  dropped by errors:        {len(errored)}")
    print(f"  no output (missing prod): {len(missing)}")
    if errored:
        print(f"  WARNING: {len(errored)} acquisition(s) dropped by errors "
              f"after retries -- run is INCOMPLETE; consider re-running "
              f"(RESUME will skip the {len(skipped) + len(produced)} unit(s) "
              f"already on disk): {errored}")


if __name__ == "__main__":
    main()
