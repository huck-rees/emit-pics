#!/usr/bin/env python
"""Collapse spatial-replicate patches to one ΔR per acquisition pair (option A).

Batched per-PICS post-processor between the per-patch delta step and the drift
analysis. Pools every per-patch delta CSV for one PICS at one patch size
(``data/delta_r/per_patch/{PICS}_{PATCH}_delta.csv``) and, per EMIT channel,
collapses all spatial-replicate patches of each unique acquisition pair to a
single ΔR (the **median across patches**), so every acquisition pair contributes
exactly one point to the downstream Theil–Sen fit instead of one-per-patch.

This corrects the **pseudo-replication** in the raw drift fit: the same
acquisition pair (same two timestamps, same sensor state and atmosphere) seen by
many patches otherwise enters the regression as many near-identical points,
artificially narrowing the uncertainty. The diagnostic ``n_patches`` and the
across-patch spread are recorded (not used to weight anything here) for a future
option-B inverse-variance scheme.

Output
------
* ``data/delta_r/aggregated/{PICS}_{SIZE}_delta.csv`` — one row per
  ``(acquisition pair × channel)``, preserving the delta-table column
  conventions so it is a drop-in input for
  ``scripts/run_matched_pairs_drift_analysis.py`` (set ``USE_AGGREGATED = True``
  there, its default).

Requires ``scripts/run_delta_calculations.py`` to have produced the per-patch
delta CSVs (under ``data/delta_r/per_patch/``) for this PICS/size first.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_PICS_allpatches_aggregate_pairs.py
"""

import sys

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PICS_ID    = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_SIZE = cfg.patch_size

# Repo-relative paths — every name is built in emit_pics.paths.
PER_PATCH_DIR = paths.DELTA_PER_PATCH_DIR   # step-6 per-patch deltas
AGG_DIR       = paths.DELTA_AGG_DIR         # this stage's output
OUT_PATH      = paths.delta_aggregated(PICS_ID, PATCH_SIZE)

from emit_pics.PICS_allpatches_aggregate_pairs import (
    aggregate_by_pair,
    collapse_summary,
    load_patch_delta_tables,
    write_aggregated_table,
)


def main() -> None:
    # 1. Pool the per-patch delta tables for this PICS + size (loud-fail on an
    #    old-schema CSV; empty if none match).
    try:
        raw = load_patch_delta_tables(PER_PATCH_DIR, PICS_ID, PATCH_SIZE)
    except ValueError as exc:
        print(f"\nInput schema error: {exc}")
        sys.exit(1)
    if raw.empty:
        print(f"No per-patch delta CSVs for {PICS_ID} size {PATCH_SIZE!r} in "
              f"{PER_PATCH_DIR}.\nRun scripts/run_delta_calculations.py for that "
              f"PICS/size first.")
        sys.exit(1)

    print(f"{PICS_ID}, size {PATCH_SIZE} km: {len(raw)} per-patch delta row(s) | "
          f"{raw['patch_id'].nunique()} patch(es) | {raw['pair_id'].nunique()} pair(s) | "
          f"{raw['band_index'].nunique()} channel(s)")

    # 2. Collapse spatial replicates -> one row per (acquisition pair × channel).
    try:
        agg = aggregate_by_pair(raw)
    except ValueError as exc:
        print(f"\nAggregation error: {exc}")
        sys.exit(1)

    # 3. Write the aggregated tidy table.
    out = write_aggregated_table(agg, OUT_PATH)
    provenance.write_sidecar(OUT_PATH, cfg.as_dict())

    # 4. Collapse summary — how much pseudo-replication option A removed.
    s = collapse_summary(raw, agg)
    print("\n--- Collapse summary (pseudo-replication removed) ---")
    print(f"  {s['n_patch_pairs']} patch-pair row(s) -> {s['n_pairs']} unique "
          f"acquisition pair(s); median {s['patches_per_pair_median']:.0f} "
          f"patches/pair, max {s['patches_per_pair_max']}, min {s['patches_per_pair_min']}")
    print(f"  per-channel rows: {s['n_raw_rows']} -> {s['n_agg_rows']} "
          f"(1 row per pair × channel, {s['n_channels']} channel(s))")
    print(f"\nWrote aggregated table -> {out}")
    print("Point run_matched_pairs_drift_analysis.py at it with USE_AGGREGATED = True.")


if __name__ == "__main__":
    main()
