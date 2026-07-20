#!/usr/bin/env python
"""Identify matched pairs of EMIT acquisitions over a single patch.

Pure **local post-processor** — no Earthdata, no streaming, no re-masking.  All
the geometry it needs (per-scene SZA/VZA means *and* percentiles, plus the
valid-pixel count) was streamed once and cached by ``run_single_patch_cube_extract.py`` into
the per-scene geometry table; this script just reads that table and:

1. computes every unique scene pair's |ΔSZA| / |ΔVZA| / Δtime and flags the
   matched ones (similar in-patch mean SZA *and* VZA),
2. writes a matched-pairs CSV and the SZA/VZA-over-time figure (the boxplots are
   drawn straight from the cached percentiles), and
3. amends the patch attribute table with the matched-pair counts.

A *matched pair* is two acquisitions with ``|ΔSZA| <= SZA_TOL`` and
``|ΔVZA| <= VZA_TOL``.  Because the geometry comes from the shared OBS file,
matching is identical for the L1B_RAD and L2A_RFL products.

Run ``scripts/run_single_patch_cube_extract.py`` for this patch first to produce the
geometry table, then set the target in run_config.yaml and run::

    python scripts/run_id_matched_pairs.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import pandas as pd

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.id_matched_pairs import (
    PAIR_CSV_COLS,
    find_matched_pairs,
    plot_matched_pairs,
    sniff_delimiter,
)
from emit_pics.qc_gate import QCReviewRequired, apply_qc_filter

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PICS_ID  = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_ID = cfg.patch
SZA_TOL  = cfg.sza_tol
VZA_TOL  = cfg.vza_tol
REQUIRE_QC_REVIEW = cfg.require_qc_review

# Repo-relative paths — every name is built in emit_pics.paths.
GEOM_PATH = paths.geometry(PICS_ID, PATCH_ID)
QC_PATH   = paths.qc_sheet(PICS_ID, PATCH_ID)
ATTS_PATH = paths.patch_atts(PICS_ID)


def load_scenes(geom_path: Path) -> pd.DataFrame:
    """Load the per-scene geometry table, coercing datetime and the RFL flag."""
    if not geom_path.exists():
        print(f"Geometry table not found: {geom_path}\n"
              f"Run scripts/run_single_patch_cube_extract.py for this patch first.")
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

    # 1b. Manual QC gate — drop scenes marked keep=FALSE in the reviewed
    #     worksheet (created/refreshed by run_single_patch_qc_products.py).
    try:
        scenes, qc = apply_qc_filter(scenes, QC_PATH,
                                     require_review=REQUIRE_QC_REVIEW)
    except QCReviewRequired as exc:
        print(f"\nQC gate blocked: {exc}")
        sys.exit(1)
    if qc["n_excluded"]:
        print(f"QC gate: excluded {qc['n_excluded']} scene(s) "
              f"({qc['reasons']}); {qc['n_after']} remain.")
    else:
        print("QC gate: no scenes excluded.")

    if len(scenes) < 2:
        print("Fewer than two scenes after QC; no pairs to form. Exiting.")
        sys.exit(1)

    # 2. Matched pairs (geometry-based; identical for both products).
    all_pairs = find_matched_pairs(scenes, SZA_TOL, VZA_TOL)
    matched = all_pairs[all_pairs["matched_pair"]].copy()

    n_rad = int(all_pairs["matched_pair"].sum())
    n_rfl = int((matched["t1_has_rfl"] & matched["t2_has_rfl"]).sum())

    paths.FIGURES.mkdir(parents=True, exist_ok=True)

    # 3. Write the matched-pairs CSV (all pairs, TRUE/FALSE flag).
    out_csv = all_pairs[PAIR_CSV_COLS].copy()
    out_csv["matched_pair"] = all_pairs["matched_pair"].map(
        {True: "TRUE", False: "FALSE"}
    )
    csv_path = paths.matched_pairs(PICS_ID, PATCH_ID)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out_csv.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, cfg.as_dict())
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
    fig_path = paths.matched_pairs_fig(PICS_ID, PATCH_ID)
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
