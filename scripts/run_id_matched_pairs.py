#!/usr/bin/env python
"""Identify matched pairs of EMIT acquisitions over a single patch.

For one PICS patch, this script:

1. Reads which granules fully cover the patch (RAD + RFL) from the
   patch–granule dictionary, and builds the acquisition list from the RAD
   granules (recording whether each has a coincident RFL).
2. Streams in-patch SZA/VZA geometry from each granule's L1B OBS file.
3. Finds matched pairs (similar in-patch mean SZA *and* VZA).
4. Writes a matched-pairs CSV and figure, and amends the patch attribute
   table and the patch–granule dictionary with the new statistics.

Edit the user-configurable inputs below, then run::

    python scripts/run_id_matched_pairs.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the dictionary / bnds gpkg
SZA_TOL  = 2.0             # solar-zenith match tolerance (degrees)
VZA_TOL  = 5.0             # view-zenith match tolerance (degrees)

RAD_SHORT_NAME = "EMITL1BRAD"   # collection holding the OBS asset (L1B)

# Paths (all local; inputs already produced upstream)
DATA_ROOT    = Path(r"C:\Users\huckr\Desktop\JPL\Data")
PATCH_DIR    = DATA_ROOT / "ROIs" / "patches"
MATCHED_DIR  = DATA_ROOT / "matched_pairs"          # matched-pairs CSV output
FIG_DIR      = Path(r"C:\Users\huckr\Desktop\JPL\Figures")

DICT_PATH  = PATCH_DIR / f"patch_EMIT_dict_{PICS_ID}.csv"
BNDS_PATH  = PATCH_DIR / f"patches_{PICS_ID}_bnds.gpkg"
ATTS_PATH  = PATCH_DIR / f"patches_{PICS_ID}_atts.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import earthaccess
import numpy as np
import pandas as pd

from emit_pics.id_matched_pairs import (
    PAIR_CSV_COLS,
    STAT_COLS,
    acq_timestamp,
    build_acquisitions,
    fetch_granule,
    find_matched_pairs,
    load_patch_dict,
    load_patch_geom_4326,
    plot_matched_pairs,
    scene_geometry_stats,
    sniff_delimiter,
    split_products,
)


def main() -> None:
    # 0. Authenticate to NASA Earthdata (persisted for streaming OBS reads).
    earthaccess.login(persist=True)
    print("Authenticated to NASA Earthdata.\n")

    # 1. Load the patch–granule dict and split into RAD / RFL granule URs.
    dict_sep = sniff_delimiter(DICT_PATH)
    dict_df = load_patch_dict(DICT_PATH, sep=dict_sep)
    rad_urs, rfl_urs = split_products(dict_df, PATCH_ID)
    print(f"Patch {PATCH_ID}: {len(rad_urs)} RAD granule(s), "
          f"{len(rfl_urs)} RFL granule(s)")
    if not rad_urs:
        print("No RAD granules cover this patch. Exiting.")
        sys.exit(1)

    # 2. Build the acquisition list (coincidence check by timestamp).
    acquisitions = build_acquisitions(rad_urs, rfl_urs)
    n_coincident = sum(a["has_rfl"] for a in acquisitions)
    print(f"  {len(acquisitions)} acquisition(s), "
          f"{n_coincident} with a coincident RFL\n")

    # 3. Patch geometry (reprojected to EPSG:4326 for the pixel mask).
    patch_geom = load_patch_geom_4326(BNDS_PATH, PATCH_ID)

    # 4. Stream OBS geometry per granule and assemble per-scene stats.
    #    scene_geometry_stats() retries transient server errors internally;
    #    the per-granule try/except only records failures that survive retries.
    records = []
    failures = []
    for k, acq in enumerate(acquisitions, 1):
        print(f"  [{k}/{len(acquisitions)}] {acq['timestamp']} ...", flush=True)
        try:
            stats = scene_geometry_stats(
                fetch_granule(RAD_SHORT_NAME, acq["granule_ur"]), patch_geom
            )
        except Exception as ex:
            print(f"    FAILED {acq['granule_ur']}: {type(ex).__name__}: {ex}")
            failures.append(acq["granule_ur"])
            continue
        records.append({
            "scene_id": acq["granule_ur"],
            "granule_ur": acq["granule_ur"],
            "timestamp": acq["timestamp"],
            "datetime": acq["datetime"],
            "has_rfl": acq["has_rfl"],
            **stats,
        })

    if not records:
        print("\nNo scenes processed successfully. Exiting.")
        sys.exit(1)

    scenes = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
    scenes["datetime"] = pd.to_datetime(scenes["datetime"])

    # 4b. Completeness check — warn loudly if any scenes were dropped.
    n_expected = len(acquisitions)
    n_processed = len(scenes)
    if n_processed < n_expected:
        print(f"\nWARNING: processed {n_processed}/{n_expected} scenes — "
              f"{n_expected - n_processed} dropped after retries. "
              f"Matched-pair counts may be incomplete; consider re-running.")

    # 5. Matched pairs (geometry-based; identical for both products).
    all_pairs = find_matched_pairs(scenes, SZA_TOL, VZA_TOL)
    matched = all_pairs[all_pairs["matched_pair"]].copy()

    n_rad = int(all_pairs["matched_pair"].sum())
    n_rfl = int((matched["t1_has_rfl"] & matched["t2_has_rfl"]).sum())

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Write the matched-pairs CSV (all pairs, TRUE/FALSE flag).
    out_csv = all_pairs[PAIR_CSV_COLS].copy()
    out_csv["matched_pair"] = all_pairs["matched_pair"].map(
        {True: "TRUE", False: "FALSE"}
    )
    csv_path = MATCHED_DIR / f"{PICS_ID}_{PATCH_ID}_matched_pairs.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out_csv.to_csv(csv_path, index=False)
    print(f"\nMatched-pairs CSV:  {csv_path}")

    # 7. Print matched pairs to stdout.
    print("\n" + "=" * 60)
    print(f"MATCHED PAIRS (|ΔSZA|<={SZA_TOL}°, |ΔVZA|<={VZA_TOL}°): {len(matched)}")
    print("=" * 60)
    for _, r in matched.iterrows():
        d1 = r["t1_scene_dt"].strftime("%Y-%m-%d")
        d2 = r["t2_scene_dt"].strftime("%Y-%m-%d")
        print(f"  {d1} <-> {d2}  ΔSZA={r['delta_SZA']:.2f}  "
              f"ΔVZA={r['delta_VZA']:.2f}  Δt={int(round(r['delta_time']))}d")

    # 8. Matched-pairs figure.
    fig = plot_matched_pairs(scenes, matched, PATCH_ID, SZA_TOL, VZA_TOL)
    fig_path = FIG_DIR / f"{PICS_ID}_{PATCH_ID}_matched_pairs.png"
    fig.savefig(fig_path, dpi=1000, bbox_inches="tight")
    print(f"\nMatched-pairs figure: {fig_path}")

    # 9. Amend the patch attribute table — row-scoped to PATCH_ID.
    atts_sep = sniff_delimiter(ATTS_PATH)
    atts = pd.read_csv(ATTS_PATH, sep=atts_sep)
    row = atts["patch_id"] == PATCH_ID
    for col, val in [("n_matched_pairs_rad", n_rad),
                     ("n_matched_pairs_rfl", n_rfl)]:
        if col not in atts.columns:
            atts[col] = pd.Series(pd.NA, index=atts.index, dtype="Int64")
        else:
            atts[col] = atts[col].astype("Int64")
        atts.loc[row, col] = val
    atts.to_csv(ATTS_PATH, sep=atts_sep, index=False)
    print(f"\nAmended attributes: {ATTS_PATH}")
    print(f"  n_matched_pairs_rad={n_rad}, n_matched_pairs_rfl={n_rfl}")

    # 10. Amend the patch–granule dict with per-granule SZA/VZA stats.
    #     RAD rows get streamed values directly (by granule_ur); RFL rows get
    #     the coincident RAD acquisition's values (matched by timestamp).
    stats_by_ur = {r["granule_ur"]: r for r in records}
    stats_by_ts = {r["timestamp"]: r for r in records}

    ddf = pd.read_csv(DICT_PATH, sep=dict_sep)
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

    ddf.to_csv(DICT_PATH, sep=dict_sep, index=False)
    print(f"Amended dict:       {DICT_PATH}")


if __name__ == "__main__":
    main()
