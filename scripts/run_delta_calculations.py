#!/usr/bin/env python
"""Compute per-wavelength % change across matched scene pairs for ONE patch.

Pure **local** post-processor (no Earthdata). For a single PICS/patch it reads:

* the matched-pairs CSV  (``run_id_matched_pairs.py``), and
* the per-acquisition per-channel RAD/RFL stats CSVs (``run_patch_extract.py``),

and writes one tidy long CSV — one row per ``(matched pair x wavelength)`` — of
the pairwise t1 -> t2 percent change in reflectance and radiance (median & mean).
No aggregation, no figures: the cross-PICS analysis is a separate future step.

Requires ``run_patch_extract.py`` (per-channel CSVs) and
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

import pandas as pd

from emit_pics.delta_calculations import (
    DELTA_CSV_COLS,
    compute_pair_delta,
    load_matched_pairs,
    load_scene_spectrum,
    pair_metadata,
)


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

    # 2. Per pair: load both scenes' spectra and compute per-wavelength deltas.
    #    A pair is skipped only if a scene's RADIANCE per-channel CSV is absent
    #    (radiance anchors the join); missing reflectance just yields NaN RFL
    #    deltas for that pair.
    frames, skipped = [], []
    for _, row in pairs.iterrows():
        meta = pair_metadata(row, PICS_ID, PATCH_ID)
        t1 = load_scene_spectrum(row["t1_scene_id"], PICS_ID, PATCH_ID,
                                 RFL_DIR, RAD_DIR)
        t2 = load_scene_spectrum(row["t2_scene_id"], PICS_ID, PATCH_ID,
                                 RFL_DIR, RAD_DIR)
        if t1 is None or t2 is None:
            which = [side for side, spec in (("t1", t1), ("t2", t2)) if spec is None]
            print(f"  skipped pair {meta['pair_id']}: missing radiance CSV "
                  f"for {', '.join(which)}")
            skipped.append(meta["pair_id"])
            continue
        frames.append(compute_pair_delta(t1, t2, meta))

    if not frames:
        print("\nNo pairs produced deltas (all skipped). No CSV written.")
        sys.exit(1)

    # 3. Concatenate and write the tidy long table.
    table = pd.concat(frames, ignore_index=True)[DELTA_CSV_COLS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)

    # 4. Summary.
    n_pairs = table["pair_id"].nunique()
    n_rfl = int(table["delta_rfl_pct_median"].notna().sum())
    print("\n--- Summary ---")
    print(f"  pairs with deltas:   {n_pairs}")
    print(f"  pairs skipped:       {len(skipped)}")
    print(f"  rows (pair x wl):    {len(table)}")
    print(f"  rows with RFL delta: {n_rfl}  (rest are radiance-only)")
    print(f"\nWrote delta table -> {OUT_PATH}")


if __name__ == "__main__":
    main()
