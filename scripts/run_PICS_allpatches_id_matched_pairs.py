#!/usr/bin/env python
"""Identify matched pairs for EVERY patch of one size across a PICS.

The batched counterpart to ``run_id_matched_pairs.py``: instead of one patch, it
enumerates every patch of ``PATCH_SIZE`` in the PICS (the ``{size}_*`` rows of the
patch-granule dict) and runs the matched-pair detection for each one back-to-back
in a single session. Like the single-patch runner it is a pure **local**
post-processor (no Earthdata): for each patch it reads the cached per-scene
geometry table, applies the identical manual QC gate (drops ``keep=FALSE`` scenes,
refuses a patch whose worksheet is not ``reviewed=TRUE``), flags every unique
scene pair matched in solar/view geometry (``|ΔSZA| <= SZA_TOL`` and
``|ΔVZA| <= VZA_TOL``), and writes that patch's matched-pairs CSV + SZA/VZA-over-
time figure. The patch attribute table is amended **once** at the end with every
patch's matched-pair counts (rather than re-read/re-written per patch).

For each patch it writes exactly what the single-patch runner writes, so the
downstream delta step sees identical inputs:

    data/matched_pairs/{PICS}_{PATCH}_matched_pairs.csv   (all pairs, TRUE/FALSE)
    figures/{PICS}_{PATCH}_matched_pairs.png              (SZA/VZA over time)

and the shared attribute table ``data/ROIs/patches/patches_{PICS}_atts.csv`` gains
``n_matched_pairs_rad`` / ``n_matched_pairs_rfl`` per patch row.

A patch that is missing its geometry table, not yet reviewed, or left with fewer
than two scenes after QC is reported and skipped so the batch continues.

Prerequisite: the cube extraction (per-scene geometry tables) and the manual QC
step (reviewed worksheets) must be complete for the patches. Set the target in
run_config.yaml, then run::

    python scripts/run_PICS_allpatches_id_matched_pairs.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.extract_patch_stats import load_patch_dict
from emit_pics.PICS_allpatches_cube_extract import patch_ids_for_size
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
PICS_ID    = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_SIZE = cfg.patch_size
SZA_TOL    = cfg.sza_tol
VZA_TOL    = cfg.vza_tol
MIN_PAIR_DAYS = cfg.min_pair_days
REQUIRE_QC_REVIEW = cfg.require_qc_review

# Repo-relative paths — every name is built in emit_pics.paths.
DICT_PATH = paths.patch_dict(PICS_ID)
ATTS_PATH = paths.patch_atts(PICS_ID)


def load_scenes(geom_path: Path) -> pd.DataFrame:
    """Load one patch's per-scene geometry table, coercing datetime + RFL flag."""
    scenes = pd.read_csv(geom_path)
    scenes["datetime"] = pd.to_datetime(scenes["datetime"])
    scenes["has_rfl"] = (
        scenes["has_rfl"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return scenes.sort_values("datetime").reset_index(drop=True)


def process_patch(pid: str):
    """Detect matched pairs for one patch; return ``(status, n_rad, n_rfl)``.

    Mirrors ``run_id_matched_pairs.main`` for a single patch but never exits the
    process. *status* is ``"produced"`` (CSV + figure written), ``"blocked"``
    (QC gate — not reviewed), ``"too_few"`` (fewer than two scenes after QC), or
    ``"missing"`` (no geometry table). ``n_rad`` / ``n_rfl`` are the matched-pair
    counts for the attribute-table amendment (``None`` unless produced).
    """
    geom_path = paths.geometry(PICS_ID, pid)
    qc_path   = paths.qc_sheet(PICS_ID, pid)
    if not geom_path.exists():
        print(f"  {pid}: no geometry table ({geom_path.name}); skipped.")
        return "missing", None, None

    scenes = load_scenes(geom_path)

    # Manual QC gate — identical to run_id_matched_pairs: drop keep=FALSE scenes.
    try:
        scenes, qc = apply_qc_filter(scenes, qc_path,
                                     require_review=REQUIRE_QC_REVIEW)
    except QCReviewRequired as exc:
        print(f"  {pid}: QC gate blocked ({exc}); skipped.")
        return "blocked", None, None
    if len(scenes) < 2:
        print(f"  {pid}: {len(scenes)} scene(s) after QC; need >=2 for pairs; "
              f"skipped.")
        return "too_few", None, None

    # Matched pairs (geometry-based; identical for both products).
    all_pairs = find_matched_pairs(scenes, SZA_TOL, VZA_TOL, MIN_PAIR_DAYS)
    matched = all_pairs[all_pairs["matched_pair"]].copy()
    n_rad = int(all_pairs["matched_pair"].sum())
    n_rfl = int((matched["t1_has_rfl"] & matched["t2_has_rfl"]).sum())

    # Matched-pairs CSV (all pairs, TRUE/FALSE flag).
    out_csv = all_pairs[PAIR_CSV_COLS].copy()
    out_csv["matched_pair"] = all_pairs["matched_pair"].map(
        {True: "TRUE", False: "FALSE"})
    paths.MATCHED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = paths.matched_pairs(PICS_ID, pid)
    out_csv.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, cfg.as_dict())

    # SZA/VZA-over-time figure (boxplots drawn from the cached percentiles).
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig = plot_matched_pairs(scenes, matched, pid, SZA_TOL, VZA_TOL)
    fig.savefig(paths.matched_pairs_fig(PICS_ID, pid),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  {pid}: {len(scenes)} scene(s) kept ({qc['n_excluded']} excluded) "
          f"-> {n_rad} matched pair(s) rad / {n_rfl} rfl.")
    return "produced", n_rad, n_rfl


def amend_atts(counts: dict) -> None:
    """Amend the patch attribute table once with every patch's matched counts.

    Reads/writes ``patches_{PICS}_atts.csv`` a single time (vs. per patch in the
    single-patch runner), setting ``n_matched_pairs_rad`` / ``n_matched_pairs_rfl``
    (nullable Int64) on each processed patch's row. A patch absent from the table
    is warned about, not written.
    """
    if not counts:
        return
    if not ATTS_PATH.exists():
        print(f"\nAttribute table not found: {ATTS_PATH}; counts not written.")
        return
    sep = sniff_delimiter(ATTS_PATH)
    atts = pd.read_csv(ATTS_PATH, sep=sep)
    for col in ("n_matched_pairs_rad", "n_matched_pairs_rfl"):
        if col not in atts.columns:
            atts[col] = pd.Series(pd.NA, index=atts.index, dtype="Int64")
        else:
            atts[col] = atts[col].astype("Int64")
    for pid, (n_rad, n_rfl) in counts.items():
        row = atts["patch_id"].astype(str) == str(pid)
        if not row.any():
            print(f"  WARNING: patch {pid} not in attribute table; count not written.")
            continue
        atts.loc[row, "n_matched_pairs_rad"] = n_rad
        atts.loc[row, "n_matched_pairs_rfl"] = n_rfl
    atts.to_csv(ATTS_PATH, sep=sep, index=False)
    print(f"\nAmended attributes ({len(counts)} patch(es)) -> {ATTS_PATH}")


def main() -> None:
    # 1. Enumerate this size's patches from the patch-granule dict.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    dt_note = f", Δt>={MIN_PAIR_DAYS:g}d" if MIN_PAIR_DAYS else ""
    print(f"{PICS_ID}, size {PATCH_SIZE} km: identifying matched pairs "
          f"(|ΔSZA|<={SZA_TOL}°, |ΔVZA|<={VZA_TOL}°{dt_note}) "
          f"for {len(patch_ids)} patch(es).")

    # 2. Detect matched pairs per patch; collect counts for the atts amendment.
    tally = {"produced": 0, "blocked": 0, "too_few": 0, "missing": 0}
    counts = {}
    for pid in patch_ids:
        status, n_rad, n_rfl = process_patch(pid)
        tally[status] += 1
        if status == "produced":
            counts[pid] = (n_rad, n_rfl)

    # 3. Amend the shared attribute table once.
    amend_atts(counts)

    # 4. Summary.
    print("\n--- Summary ---")
    print(f"  patches with matched-pairs CSV: {tally['produced']}")
    print(f"  blocked (QC not reviewed):      {tally['blocked']}")
    print(f"  too few scenes (<2 after QC):   {tally['too_few']}")
    print(f"  missing geometry table:         {tally['missing']}")
    print(f"\nMatched-pairs CSVs -> {paths.MATCHED_DIR}\nFigures -> {paths.FIGURES}")
    if tally["blocked"]:
        print("  Some patches are not marked reviewed. Complete QC "
              "(run_PICS_allpatches_qc_review.py) or set REQUIRE_QC_REVIEW=False.")


if __name__ == "__main__":
    main()
