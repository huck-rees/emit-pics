#!/usr/bin/env python
"""Per-scene Landsat 9 TOA extraction over one PICS patch (the EMIT control).

The Landsat analog to ``scripts/run_single_patch_cube_extract.py``. For every
Landsat 9 scene that fully covers the patch in the EMIT window, it windows the 7
reflective bands + the SZA/VZA rasters straight from cloud COGs (no full-tile
downloads), converts to TOA reflectance, and writes — in the SAME schemas the
EMIT side uses, so the sensor-agnostic BRDF fit consumes them unchanged:

1. a per-scene per-band TOA table (``paths.perchannel(..., "toa")``), and
2. a per-scene SZA/VZA geometry table (``paths.ls_geometry``).

QC is NOT done here — it is a separate manual worksheet (opt-out keep/remove),
exactly as on the EMIT side. This runner only writes the raw geometry + stats.

Set the target (site/patch, date window) in run_config.yaml, then run in an env
with pystac-client + rasterio + boto3 (the `ls9-draft` env), with AWS credentials
configured for the requester-pays reads::

    python scripts/run_landsat_cube_extract.py

NOTE (sketch): the STAC/source knobs below are script-local for now; once this
stabilizes they move into a run_config `landsat:` block (see the proposed snippet
in the design notes).
"""

import os
import sys

import pandas as pd
from pystac_client import Client

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.landsat_cube_extract import (
    channel_stats_from_cube,
    extract_scene,
    geometry_record_from_cube,
    load_patch,
    search_scenes,
    write_cube_geotiff,
)

# --- GDAL/AWS knobs for cloud COG reads (requester-pays for the TOA path) ---
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

# --- script-local source knobs (TODO: move to run_config `landsat:` block) ---
SOURCE = "usgs_toa"                                   # "usgs_toa" (TOA) | "mpc_sr"
STAC_URL = "https://landsatlook.usgs.gov/stac-server"
COLLECTION = "landsat-c2l1"
PLATFORM = "landsat-9"
MAX_CLOUD = 5.0
MIN_COVERAGE = 0.99                                   # skip scenes with fill/gaps in-patch
RESUME = True                                         # skip scenes already written to disk

# Pixel-cube cache (QC only). Off by default; when ON the cubes are large, so they
# can be sent to a parallel tree on another volume via EXTERNAL_CUBE_DIR while the
# TOA + geometry CSVs stay repo-relative on the main drive. None = repo-relative.
CACHE_PIXELS = False
EXTERNAL_CUBE_DIR = None                              # e.g. r"D:\JPL\emit-pics\data\landsat\cubes"

cfg = load_config()
PICS_ID = canonical_token(cfg.site)
PATCH_ID = cfg.patch

if EXTERNAL_CUBE_DIR:
    paths.set_ls_cube_dir(EXTERNAL_CUBE_DIR)          # redirect ONLY the cubes off-drive


def _already_done(ts):
    """RESUME key: outputs for this scene already on disk.

    With cube caching ON the cube is the key (a stats-only prior pass writes the
    TOA CSV but no cube, and must be re-done to produce it); otherwise the TOA CSV.
    """
    if CACHE_PIXELS:
        return paths.ls_pixel_cube(PICS_ID, PATCH_ID, ts).exists()
    return paths.perchannel(PICS_ID, PATCH_ID, ts, "toa").exists()


def process_scene(item, patch_bounds, patch_crs):
    """Extract one scene and write its outputs; return ``(status, geom_record)``.

    *status* is ``"produced"`` (TOA CSV written), ``"low_coverage"`` (skipped:
    fill/gaps inside the patch), or ``"missing"``.
    """
    cube = extract_scene(item, patch_bounds, patch_crs, source=SOURCE)
    if cube["n_pixels"] == 0:
        return "missing", None
    if cube["coverage"] < MIN_COVERAGE:
        return "low_coverage", None

    geom_record = geometry_record_from_cube(cube)

    df = channel_stats_from_cube(cube)
    toa_path = paths.perchannel(PICS_ID, PATCH_ID, cube["timestamp"], "toa")
    toa_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(toa_path, index=False)
    provenance.write_sidecar(toa_path, cfg.as_dict())

    # Optional raw pixel cube (off by default) — for offline manual QC only.
    if CACHE_PIXELS:
        write_cube_geotiff(cube, paths.ls_pixel_cube(PICS_ID, PATCH_ID, cube["timestamp"]))

    return "produced", geom_record


def main() -> None:
    patch_bounds, patch_crs, patch_wgs84 = load_patch(paths.patch_bnds(PICS_ID), PATCH_ID)

    catalog = Client.open(STAC_URL)
    scenes = search_scenes(catalog, COLLECTION, patch_wgs84,
                           cfg.date_start, cfg.date_end,
                           platform=PLATFORM, max_cloud=MAX_CLOUD)
    print(f"{PICS_ID} {PATCH_ID}: {len(scenes)} fully-covering L9 scene(s) "
          f"<= {MAX_CLOUD}% cloud in {cfg.date_start}..{cfg.date_end}")
    if not scenes:
        print("No qualifying scenes. Exiting.")
        sys.exit(1)

    produced, low_cov, missing, errored, skipped = [], [], [], [], []
    geom_records = []
    for k, item in enumerate(scenes, 1):
        ts = _ts = pd.to_datetime(item.properties["datetime"]).strftime("%Y%m%dT%H%M%S")
        if RESUME and _already_done(ts):
            skipped.append(ts)
            continue
        print(f"[{k}/{len(scenes)}] {item.id}")
        try:
            status, rec = process_scene(item, patch_bounds, patch_crs)
        except Exception as ex:
            print(f"  FAILED {ts}: {type(ex).__name__}: {ex}")
            errored.append(ts)
            continue
        {"produced": produced, "low_coverage": low_cov,
         "missing": missing}[status].append(ts)
        if rec is not None:
            geom_records.append(rec)

    # Write / merge the per-scene geometry table (parallels the EMIT runner).
    if geom_records:
        geom_df = pd.DataFrame(geom_records)
        geom_path = paths.ls_geometry(PICS_ID, PATCH_ID)
        geom_path.parent.mkdir(parents=True, exist_ok=True)
        if geom_path.exists():
            geom_df = pd.concat([pd.read_csv(geom_path), geom_df], ignore_index=True)
            geom_df = geom_df.drop_duplicates(subset="timestamp", keep="last")
        geom_df = geom_df.sort_values("timestamp").reset_index(drop=True)
        geom_df.to_csv(geom_path, index=False)
        provenance.write_sidecar(geom_path, cfg.as_dict())
        print(f"\nGeometry table: {geom_path}  ({len(geom_df)} scene(s))")

    print("\n--- Completeness ---")
    print(f"  scenes found:        {len(scenes)}")
    if RESUME:
        print(f"  skipped (on disk):   {len(skipped)}")
    print(f"  produced TOA:        {len(produced)}")
    print(f"  dropped low coverage:{len(low_cov)}")
    print(f"  dropped errors:      {len(errored)}")
    if errored:
        print(f"  WARNING: re-run to recover {errored}")


if __name__ == "__main__":
    main()
