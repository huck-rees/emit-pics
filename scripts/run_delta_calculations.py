#!/usr/bin/env python
"""Compute per-wavelength % change across matched scene pairs for ONE patch.

Pure **local** post-processor (no Earthdata). For a single PICS/patch it reads:

* the matched-pairs CSV  (``run_id_matched_pairs.py``), and
* the per-acquisition per-channel RAD/RFL stats CSVs (``run_single_patch_cube_extract.py``),

and writes one tidy long CSV — one row per ``(matched pair x wavelength)`` — of
the pairwise t1 -> t2 percent change in reflectance and radiance (median & mean).
No aggregation, no figures: the cross-PICS analysis is a separate future step.

Requires ``run_single_patch_cube_extract.py`` (per-channel CSVs) and
``run_id_matched_pairs.py`` (matched-pairs CSV) to have run first for this patch.

Edit the user-configurable inputs below, then run::

    python scripts/run_delta_calculations.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id to analyze

# Paths (repo-relative; inputs produced by upstream steps)
_REPO_ROOT  = Path(__file__).resolve().parent.parent          # .../emit-pics
DATA_ROOT   = _REPO_ROOT / "data"
MATCHED_DIR = DATA_ROOT / "matched_pairs"
RFL_DIR     = DATA_ROOT / "reflectance"
RAD_DIR     = DATA_ROOT / "radiance"
OUT_DIR     = DATA_ROOT / "delta_r"

MATCHED_PATH = MATCHED_DIR / f"{PICS_ID}_{PATCH_ID}_matched_pairs.csv"
OUT_PATH     = OUT_DIR / f"{PICS_ID}_{PATCH_ID}_delta.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from emit_pics.delta_calculations import (
    DELTA_CSV_COLS,
    WL_MATCH_TOL_NM,
    absorption_band_wavelengths,
    compute_pair_delta,
    load_matched_pairs,
    load_scene_spectrum,
    pair_metadata,
    scene_perchannel_path,
)
from emit_pics.id_matched_pairs import acq_timestamp


def main() -> None:
    # 1. Load matched (TRUE-only) pairs for this patch.
    if not MATCHED_PATH.exists():
        print(f"Matched-pairs CSV not found: {MATCHED_PATH}\n"
              f"Run scripts/run_id_matched_pairs.py for this patch first.")
        sys.exit(1)

    pairs = load_matched_pairs(MATCHED_PATH)
    print(f"Patch {PATCH_ID}: {len(pairs)} matched pair(s) from {MATCHED_PATH.name}")
    if pairs.empty:
        print("No matched pairs to process. Exiting.")
        sys.exit(1)

    # 2. Load each referenced scene's spectrum ONCE (a scene may appear in many
    #    pairs). Radiance anchors the join, so a scene with no radiance CSV is
    #    unusable; record the exact missing file for the failure summary.
    scene_ids = pd.unique(pairs[["t1_scene_id", "t2_scene_id"]].values.ravel())
    specs, missing_rad = {}, {}
    for sid in scene_ids:
        spec = load_scene_spectrum(sid, PICS_ID, PATCH_ID, RFL_DIR, RAD_DIR)
        specs[sid] = spec
        if spec is None:
            missing_rad[sid] = scene_perchannel_path(sid, PICS_ID, PATCH_ID,
                                                     RAD_DIR, "rad")

    # 3. Derive the water-vapor bad-band wavelengths ONCE from the first
    #    reflectance-bearing scene (a fixed instrument property, identical across
    #    scenes). Flagging by wavelength means radiance rows are flagged too. If
    #    no scene has reflectance, absorption_band_flag falls back to the fixed
    #    ABSORPTION_BANDS_NM ranges.
    bad_bands = np.array([], dtype=float)
    for sid, spec in specs.items():
        wls = absorption_band_wavelengths(spec)
        if len(wls):
            bad_bands = wls
            print(f"Absorption bands: {len(wls)} water-vapor band(s) derived from "
                  f"reflectance of scene {acq_timestamp(sid)} "
                  f"(~{wls.min():.0f}-{wls.max():.0f} nm).")
            break
    if not len(bad_bands):
        print("Absorption bands: no reflectance-bearing scene found; flag falls "
              "back to the fixed ABSORPTION_BANDS_NM ranges.")

    # 4. Per pair: compute per-wavelength deltas. A pair can drop out two ways,
    #    both surfaced loudly below: its scene has no radiance CSV (skipped), or
    #    the two scenes' band grids don't correspond, so the band-order join is
    #    rejected and empty (degenerate — an unbridgeable calibration-grid gap).
    frames, skipped, degenerate = [], [], []
    for _, row in pairs.iterrows():
        meta = pair_metadata(row, PICS_ID, PATCH_ID)
        t1, t2 = specs[row["t1_scene_id"]], specs[row["t2_scene_id"]]
        if t1 is None or t2 is None:
            offenders = [(side, sid) for side, sid, spec in
                         (("t1", row["t1_scene_id"], t1),
                          ("t2", row["t2_scene_id"], t2)) if spec is None]
            skipped.append({"pair_id": meta["pair_id"], "offenders": offenders})
            continue
        df = compute_pair_delta(t1, t2, meta, bad_bands)
        if df.empty or int(df["rad_t2_median"].notna().sum()) == 0:
            degenerate.append({"pair_id": meta["pair_id"],
                               "t1": row["t1_scene_id"], "t2": row["t2_scene_id"]})
            continue
        frames.append(df)

    # 5. Write the tidy long table (whatever succeeded).
    if frames:
        table = pd.concat(frames, ignore_index=True)[DELTA_CSV_COLS]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_PATH, index=False)
        n_pairs = table["pair_id"].nunique()
        n_rfl = int(table["delta_rfl_pct_median"].notna().sum())
        n_flagged = int(table["absorption_band_flag"].sum())
        print("\n--- Summary ---")
        print(f"  pairs with deltas:     {n_pairs} of {len(pairs)}")
        print(f"  pairs skipped (no rad):{len(skipped)}")
        print(f"  pairs degenerate:      {len(degenerate)}")
        print(f"  rows (pair x wl):      {len(table)}")
        print(f"  rows with RFL delta:   {n_rfl}  (rest are radiance-only)")
        print(f"  rows flagged absorp.:  {n_flagged}")
        print(f"\nWrote delta table -> {OUT_PATH}")
    else:
        print("\nNo pairs produced deltas (all dropped). No CSV written.")

    # 6. LOUD FAIL — never let a dropped pair pass silently. List every dropped
    #    pair with its cause, then exit non-zero.
    if skipped or degenerate:
        bar = "!" * 70
        print("\n" + bar)
        print(f"!! INCOMPLETE: {len(skipped) + len(degenerate)} of {len(pairs)} "
              f"matched pair(s) DROPPED.")
        print(bar)
        if skipped:
            print(f"\n  {len(skipped)} skipped — missing radiance stats:")
            for s in skipped:
                print(f"    pair {s['pair_id']}:")
                for side, sid in s["offenders"]:
                    print(f"      {side} scene missing radiance CSV: {sid}")
                    print(f"         expected -> {missing_rad[sid]}")
            uniq = sorted({str(p) for p in missing_rad.values()})
            print(f"    {len(uniq)} unique missing file(s):")
            for p in uniq:
                print(f"      - {p}")
            print("    Fix: re-run scripts/run_single_patch_cube_extract.py for this patch, "
                  "then re-run this script.")
        if degenerate:
            print(f"\n  {len(degenerate)} degenerate — scenes' band grids do "
                  f"not correspond:")
            for d in degenerate:
                print(f"    pair {d['pair_id']}:  {d['t1']}  x  {d['t2']}")
            print("    Cause: unequal band count, or a per-band wavelength gap "
                  f"> {WL_MATCH_TOL_NM} nm.\n    Inspect their per-channel "
                  "central_wl columns before changing WL_MATCH_TOL_NM.")
        sys.exit(2)


if __name__ == "__main__":
    main()
