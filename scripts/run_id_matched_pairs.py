#!/usr/bin/env python
"""Identify matched pairs of EMIT acquisitions over a single patch.

Pure **local post-processor** — no Earthdata, no streaming, no re-masking.  All
the geometry it needs (per-scene SZA/VZA means *and* percentiles, plus the
valid-pixel count) was streamed once and cached by ``run_patch_extract.py`` into
the per-scene geometry table; this script just reads that table and:

1. computes every unique scene pair's |ΔSZA| / |ΔVZA| / Δtime and flags the
   matched ones (similar in-patch mean SZA *and* VZA),
2. writes a matched-pairs CSV and the SZA/VZA-over-time figure (the boxplots are
   drawn straight from the cached percentiles), and
3. amends the patch attribute table with the matched-pair counts.

A *matched pair* is two acquisitions with ``|ΔSZA| <= SZA_TOL`` and
``|ΔVZA| <= VZA_TOL``.  Because the geometry comes from the shared OBS file,
matching is identical for the L1B_RAD and L2A_RFL products.

Run ``scripts/run_patch_extract.py`` for this patch first to produce the
geometry table, then edit the inputs below and run::

    python scripts/run_id_matched_pairs.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the geometry table
SZA_TOL  = 2.0             # solar-zenith match tolerance (degrees)
VZA_TOL  = 5.0             # view-zenith match tolerance (degrees)

# Paths (repo-relative; geometry table produced by run_patch_extract.py)
_REPO_ROOT  = Path(__file__).resolve().parent.parent          # .../emit-pics
DATA_ROOT   = _REPO_ROOT / "data"
PATCH_DIR   = DATA_ROOT / "ROIs" / "patches"
GEOM_DIR    = DATA_ROOT / "geometry"
MATCHED_DIR = DATA_ROOT / "matched_pairs"
FIG_DIR     = _REPO_ROOT / "figures"

GEOM_PATH = GEOM_DIR / f"geometry_{PICS_ID}_{PATCH_ID}.csv"
ATTS_PATH = PATCH_DIR / f"patches_{PICS_ID}_atts.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import pandas as pd

from emit_pics.id_matched_pairs import (
    PAIR_CSV_COLS,
    find_matched_pairs,
    plot_matched_pairs,
    sniff_delimiter,
)


def load_scenes(geom_path: Path) -> pd.DataFrame:
    """Load the per-scene geometry table, coercing datetime and the RFL flag."""
    if not geom_path.exists():
        print(f"Geometry table not found: {geom_path}\n"
              f"Run scripts/run_patch_extract.py for this patch first.")
        sys.exit(1)
    scenes = pd.read_csv(geom_path)
    scenes["datetime"] = pd.to_datetime(scenes["datetime"])
    scenes["has_rfl"] = (
        scenes["has_rfl"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return scenes.sort_values("datetime").reset_index(drop=True)


def main() -> None:
    # 1. Load the cached per-scene geometry (no network).
    scenes = load_scenes(GEOM_PATH)
    print(f"Patch {PATCH_ID}: {len(scenes)} scene(s) from {GEOM_PATH.name}")
    if len(scenes) < 2:
        print("Fewer than two scenes; no pairs to form. Exiting.")
        sys.exit(1)

    # 2. Matched pairs (geometry-based; identical for both products).
    all_pairs = find_matched_pairs(scenes, SZA_TOL, VZA_TOL)
    matched = all_pairs[all_pairs["matched_pair"]].copy()

    n_rad = int(all_pairs["matched_pair"].sum())
    n_rfl = int((matched["t1_has_rfl"] & matched["t2_has_rfl"]).sum())

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Write the matched-pairs CSV (all pairs, TRUE/FALSE flag).
    out_csv = all_pairs[PAIR_CSV_COLS].copy()
    out_csv["matched_pair"] = all_pairs["matched_pair"].map(
        {True: "TRUE", False: "FALSE"}
    )
    csv_path = MATCHED_DIR / f"{PICS_ID}_{PATCH_ID}_matched_pairs.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out_csv.to_csv(csv_path, index=False)
    print(f"\nMatched-pairs CSV:  {csv_path}")

    # 4. Print matched pairs to stdout.
    print("\n" + "=" * 60)
    print(f"MATCHED PAIRS (|ΔSZA|<={SZA_TOL}°, |ΔVZA|<={VZA_TOL}°): {len(matched)}")
    print("=" * 60)
    for _, r in matched.iterrows():
        d1 = r["t1_scene_dt"].strftime("%Y-%m-%d")
        d2 = r["t2_scene_dt"].strftime("%Y-%m-%d")
        print(f"  {d1} <-> {d2}  ΔSZA={r['delta_SZA']:.2f}  "
              f"ΔVZA={r['delta_VZA']:.2f}  Δt={int(round(r['delta_time']))}d")

    # 5. Matched-pairs figure (boxplots drawn from the cached percentiles).
    fig = plot_matched_pairs(scenes, matched, PATCH_ID, SZA_TOL, VZA_TOL)
    fig_path = FIG_DIR / f"{PICS_ID}_{PATCH_ID}_matched_pairs.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nMatched-pairs figure: {fig_path}")

    # 6. Amend the patch attribute table — row-scoped to PATCH_ID.
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


if __name__ == "__main__":
    main()
