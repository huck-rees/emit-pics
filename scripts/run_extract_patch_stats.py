#!/usr/bin/env python
"""Extract per-channel L1B radiance / L2A reflectance stats over one patch.

For every EMIT acquisition that covers the patch, this script:

1. Pairs the L1B_RAD and L2A_RFL granules by acquisition timestamp (from the
   patch–granule dictionary; no live search).
2. Opens each granule's files once, builds the in-patch pixel mask once, and
   extracts per-channel summary statistics for both products.
3. Writes one CSV per product per acquisition (radiance + reflectance).

Optionally screens out pixels flagged in the L2A MASK Aggregate-Flag band,
dropping an acquisition whose valid fraction falls below a threshold.

Edit the user-configurable inputs below, then run::

    python scripts/run_extract_patch_stats.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the dictionary / bnds gpkg

FILTER_AGG_FLAG = False     # True -> screen pixels via L2A MASK Aggregate Flag
MIN_VALID_FRAC  = 0.60      # min in-patch valid fraction (only if filtering)

RAD_SHORT_NAME = "EMITL1BRAD"   # L1B radiance collection
RFL_SHORT_NAME = "EMITL2ARFL"   # L2A reflectance (+ MASK) collection

# Paths (all local; inputs already produced upstream)
DATA_ROOT  = Path(r"C:\Users\huckr\Desktop\JPL\Data")
PATCH_DIR  = DATA_ROOT / "ROIs" / "patches"
RAD_DIR    = DATA_ROOT / "radiance"
RFL_DIR    = DATA_ROOT / "reflectance"

DICT_PATH  = PATCH_DIR / f"patch_EMIT_dict_{PICS_ID}.csv"
BNDS_PATH  = PATCH_DIR / f"patches_{PICS_ID}_bnds.gpkg"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import earthaccess
import numpy as np

from emit_pics.id_matched_pairs import (
    fetch_granule,
    in_patch_mask,
    load_patch_geom_4326,
    sniff_delimiter,
)
from emit_pics.extract_patch_stats import (
    RAD_FILLS,
    RFL_FILLS,
    close_handles,
    load_patch_dict,
    open_l1b,
    open_l2a,
    pair_acquisitions,
    per_channel_stats,
    split_products,
    window_2d,
    window_bounds,
    window_pixels,
    with_retry,
    write_perchannel_csv,
)


def process_acquisition(acq, patch_geom) -> str:
    """Extract and write CSV(s) for one acquisition.

    Returns one of ``"produced"`` (>=1 CSV written), ``"screened"`` (dropped by
    the Aggregate-Flag valid-fraction guard), or ``"missing"`` (no CSV written
    because a required product/asset was absent).  Raises on transient errors
    that survive retries, for the caller to record as an error drop.
    """
    ts = acq["timestamp"]
    have_l2a = acq["rfl_ur"] is not None
    have_l1b = acq["rad_ur"] is not None

    n_written = 0
    keep_sub = None      # shared in-window keep mask (built once from L2A)
    bounds = None        # shared (r0, r1, c0, c1) window

    # --- Reflectance (requires L2A) ---
    if have_l2a:
        gran = fetch_granule(RFL_SHORT_NAME, acq["rfl_ur"])
        l2a = with_retry(lambda: open_l2a(gran, FILTER_AGG_FLAG),
                         label=f"{ts} L2A")
        try:
            mask = in_patch_mask(l2a["lon"], l2a["lat"], patch_geom)
            bounds = window_bounds(mask)
            if bounds is None:
                print(f"  skipped {ts}: no in-patch pixels")
                return "missing"
            r0, r1, c0, c1 = bounds
            in_sub = mask[r0:r1, c0:c1]
            n_in_patch = int(in_sub.sum())

            if FILTER_AGG_FLAG:
                agg_clean = np.isclose(window_2d(l2a["agg"], bounds), 0.0)
                keep_sub = in_sub & agg_clean
                valid_frac = keep_sub.sum() / n_in_patch if n_in_patch else 0.0
                if valid_frac < MIN_VALID_FRAC:
                    print(f"  dropped {ts}: {valid_frac:.0%} valid "
                          f"< {MIN_VALID_FRAC:.0%}")
                    return "screened"
            else:
                keep_sub = in_sub

            rfl_vals = window_pixels(l2a["refl"], bounds, keep_sub)
            rfl_df = per_channel_stats(rfl_vals, l2a["wl"], RFL_FILLS, "rfl")
            rfl_path = RFL_DIR / f"{PICS_ID}_{PATCH_ID}_{ts}_perchannel_rfl.csv"
            write_perchannel_csv(rfl_df, rfl_path)
            print(f"  wrote {rfl_path.name}")
            n_written += 1
        finally:
            close_handles(l2a["_handles"])
    else:
        print(f"  skipped {ts} reflectance: no L2A granule")

    # --- Radiance (requires L1B) ---
    if have_l1b:
        if FILTER_AGG_FLAG and not have_l2a:
            print(f"  skipped {ts} radiance: no L2A MASK available for "
                  f"aggregate-flag screening")
        else:
            gran = fetch_granule(RAD_SHORT_NAME, acq["rad_ur"])
            need_loc = keep_sub is None     # no L2A mask (filtering off, no L2A)
            l1b = with_retry(lambda: open_l1b(gran, want_loc=need_loc),
                             label=f"{ts} L1B")
            try:
                if keep_sub is None:
                    mask = in_patch_mask(l1b["lon"], l1b["lat"], patch_geom)
                    bounds = window_bounds(mask)
                    if bounds is None:
                        print(f"  skipped {ts}: no in-patch pixels")
                        bounds = keep_sub = None
                    else:
                        r0, r1, c0, c1 = bounds
                        keep_sub = mask[r0:r1, c0:c1]

                if bounds is not None and keep_sub is not None:
                    rad_vals = window_pixels(l1b["rad"], bounds, keep_sub)
                    rad_df = per_channel_stats(rad_vals, l1b["wl"], RAD_FILLS, "rad")
                    rad_path = RAD_DIR / f"{PICS_ID}_{PATCH_ID}_{ts}_perchannel_rad.csv"
                    write_perchannel_csv(rad_df, rad_path)
                    print(f"  wrote {rad_path.name}")
                    n_written += 1
            finally:
                close_handles(l1b["_handles"])
    else:
        print(f"  skipped {ts} radiance: no L1B granule")

    return "produced" if n_written else "missing"


def main() -> None:
    # 0. Authenticate to NASA Earthdata (persisted for streaming reads).
    earthaccess.login(persist=True)
    print("Authenticated to NASA Earthdata.\n")

    # 1. Load the patch–granule dict and pair acquisitions.
    #    Auto-detect the delimiter (the dict on disk is comma-separated, though
    #    a tab-separated copy is also supported).
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

    # 2. Patch geometry (reprojected to EPSG:4326 for the pixel mask).
    patch_geom = load_patch_geom_4326(BNDS_PATH, PATCH_ID)

    # 3. Process every acquisition; bucket outcomes for the completeness check.
    produced, screened, errored, missing = [], [], [], []
    for k, acq in enumerate(acquisitions, 1):
        ts = acq["timestamp"]
        print(f"\n[{k}/{len(acquisitions)}] {ts}")
        try:
            status = process_acquisition(acq, patch_geom)
        except Exception as ex:
            print(f"  FAILED {ts}: {type(ex).__name__}: {ex}")
            errored.append(ts)
            continue
        {"produced": produced, "screened": screened,
         "missing": missing}[status].append(ts)

    # 4. Completeness summary — distinguish screening drops from error drops.
    print("\n--- Completeness ---")
    print(f"  acquisitions expected:   {len(acquisitions)}")
    print(f"  produced output:         {len(produced)}")
    print(f"  dropped by screening:    {len(screened)}")
    print(f"  dropped by errors:       {len(errored)}")
    print(f"  no output (missing prod): {len(missing)}")
    if errored:
        print(f"  WARNING: {len(errored)} acquisition(s) dropped by errors "
              f"after retries — run is INCOMPLETE; consider re-running: "
              f"{errored}")


if __name__ == "__main__":
    main()
