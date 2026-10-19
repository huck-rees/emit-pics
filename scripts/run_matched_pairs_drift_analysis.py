#!/usr/bin/env python
"""Cross-PICS radiometric-drift analysis for one patch size (matched-pairs bootstrap).

Pure **local** post-processor (no Earthdata). Pools the per-patch delta CSVs of one
patch size across all PICS and, per EMIT channel, fits the drift rate as the robust
(Theil-Sen) slope of the pairwise percent change ΔR against the pair's time
separation Δt, with a **PICS-stratified matched-pairs bootstrap** for the
uncertainty (:mod:`emit_pics.matched_pairs_bootstrap`):

* the point estimate is Theil-Sen on one representative ΔR per matched pair (the
  median across its spatial-replicate patches), pooled across PICS;
* the 95% CI and p-value come from resampling the matched pairs with replacement,
  stratified by PICS, drawing one patch's ΔR per resampled pair each iteration.

Consumes the raw **per-patch** delta tables (the bootstrap needs the spatial
replicates and scene ids). Water-vapor absorption bands are excluded.

Outputs
-------
* ``data/drift/{size}_mp_drift.csv`` — per-channel drift stats (RAD + RFL).
* ``figures/matched_pairs/{size}_mp_drift_spectrum.png`` — drift rate vs wavelength,
  with bootstrap 95% CI whiskers and raw-significance colouring, for both products.

Requires the step-6 delta CSVs to carry the SZA_t1/SZA_t2/band_index/scene-id
columns; re-run ``scripts/run_delta_calculations.py`` if they predate that schema.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_matched_pairs_drift_analysis.py
"""

import sys

from emit_pics import paths, provenance
from emit_pics.run_config import load_config

# INPUTS — set the analysis target in run_config.yaml, not here.
cfg = load_config()
paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
PATCH_SIZE  = cfg.patch_size
ALPHA       = cfg.alpha
N_BOOTSTRAP = cfg.drift_n_bootstrap
SEED        = cfg.drift_seed

# Repo-relative paths — every name is built in emit_pics.paths.
PER_PATCH_DIR = paths.DELTA_PER_PATCH_DIR   # raw per-patch delta input
DRIFT_DIR     = paths.DRIFT_DIR
FIG_DIR       = paths.MATCHED_FIG_DIR

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emit_pics.matched_pairs_drift_analysis import load_delta_tables, plot_drift_spectrum
from emit_pics.matched_pairs_bootstrap import compute_bootstrap_drift_spectrum


def main() -> None:
    # 1. Pool the per-patch delta tables for this patch size (all PICS).
    try:
        df = load_delta_tables(PER_PATCH_DIR, PATCH_SIZE)
    except ValueError as exc:
        print(f"\nInput schema error: {exc}")
        sys.exit(1)
    if df.empty:
        print(f"No per-patch delta CSVs for patch size {PATCH_SIZE!r} in "
              f"{PER_PATCH_DIR}.\nRun scripts/run_PICS_allpatches_delta_calculations.py "
              f"(step 6) for the relevant patches first.")
        sys.exit(1)

    # Drop non-invariant PICS (shared with the BRDF fit) so both EMIT drift methods
    # use the same four invariant sites. M1 stays in the raw delta tables and figures.
    excl = {s.replace(" ", "_") for s in cfg.brdf_drift_exclude_sites}
    if excl:
        norm = df["PICS_ID"].astype(str).str.replace(" ", "_", regex=False)
        dropped = sorted(norm[norm.isin(excl)].unique())
        df = df[~norm.isin(excl)].reset_index(drop=True)
        if dropped:
            print(f"Excluded from drift fit (not invariant): {', '.join(dropped)}")

    print(f"Patch size {PATCH_SIZE} km [matched-pairs bootstrap, {N_BOOTSTRAP} iters, "
          f"seed {SEED}]: {len(df)} per-patch row(s) | {df['PICS_ID'].nunique()} PICS "
          f"| {df['patch_id'].nunique()} patch(es) | {df['pair_id'].nunique()} pair(s) "
          f"| {df['band_index'].nunique()} channel(s)")

    # 2. Per-channel bootstrap drift spectra (both products).
    spec_rad = compute_bootstrap_drift_spectrum(df, "rad", n_boot=N_BOOTSTRAP,
                                                alpha=ALPHA, seed=SEED)
    spec_rfl = compute_bootstrap_drift_spectrum(df, "rfl", n_boot=N_BOOTSTRAP,
                                                alpha=ALPHA, seed=SEED)

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = paths.drift(PATCH_SIZE)
    pd.concat([spec_rad, spec_rfl], ignore_index=True).to_csv(out_csv, index=False)
    provenance.write_sidecar(out_csv, cfg.as_dict())
    print(f"\nDrift stats CSV -> {out_csv}")
    for name, spec in (("RAD", spec_rad), ("RFL", spec_rfl)):
        n_fit = int(spec["slope_pct_yr"].notna().sum())
        n_sig = int(spec["raw_significant"].fillna(False).sum())
        edges = spec["n_edges"].replace(0, np.nan).dropna()
        scenes = spec["n_scenes"].replace(0, np.nan).dropna()
        note = (f" | ~{edges.median():.0f} pair(s) across {scenes.median():.0f} scene(s)"
                if len(edges) else "")
        print(f"  {name}: {n_fit} channel(s) fit | {n_sig} with CI excluding 0{note}")

    # 3. Drift-spectrum figure.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_drift_spectrum(spec_rad, spec_rfl, patch_size=PATCH_SIZE, alpha=ALPHA)
    spec_png = paths.drift_spectrum_fig(PATCH_SIZE)
    fig.savefig(spec_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Drift spectrum -> {spec_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()
