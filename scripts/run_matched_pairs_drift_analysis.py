#!/usr/bin/env python
"""Cross-PICS radiometric-drift analysis for one patch size.

Pure **local** post-processor (no Earthdata). Pools the delta CSVs of one patch
size across all PICS and, per EMIT channel, fits the drift rate as the robust
(Theil–Sen) slope of the pairwise percent change ΔR against the pair's time
separation Δt, using the **median** in-patch statistic. Significance is Kendall's
τ, false-discovery controlled across the 285 channels with Benjamini–Hochberg.

By default (``USE_AGGREGATED = True``) it consumes the option-A **aggregated**
tables (``data/delta_r/aggregated/{PICS}_{size}_delta.csv``, one row per
acquisition pair per channel), so each acquisition pair contributes exactly one
point and the fit is not pseudo-replicated across spatial-replicate patches; the
per-channel scatter then sizes each dot by ``n_patches`` (replication).
Set ``USE_AGGREGATED = False`` to fit the raw per-patch deltas
(``data/delta_r/per_patch/{PICS}_{PATCH}_delta.csv``) instead, sized by mean
|SZA| as before.

Outputs
-------
* ``data/drift/{size}_drift.csv`` — per-channel drift stats (RAD + RFL).
* ``figures/matched_pairs/{size}_drift_spectrum.png`` — drift rate vs wavelength,
  with 95% CI whiskers and BH-significance colouring, for both products.
* ``figures/matched_pairs/individual_channels/{size}_{wavelength}.png`` — one
  two-panel ΔR-vs-Δt scatter per channel (RAD left, RFL right; colour/shape =
  PICS, marker size ∝ mean |SZA|). Toggle with ``INDIVIDUAL_CHANNELS``.

Requires the step-6 delta CSVs to carry the SZA_t1/SZA_t2/band_index columns;
re-run ``scripts/run_delta_calculations.py`` if they predate that schema.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_matched_pairs_drift_analysis.py
"""

import sys

from emit_pics import paths, provenance
from emit_pics.run_config import load_config

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PATCH_SIZE = cfg.patch_size
ALPHA = cfg.alpha
USE_AGGREGATED = cfg.use_aggregated
INDIVIDUAL_CHANNELS = cfg.individual_channels
ONLY_BANDS = None           # script-local: e.g. [0, 50, 150] to render a subset for testing; None = all

# Repo-relative paths — every name is built in emit_pics.paths.
AGG_DIR       = paths.DELTA_AGG_DIR         # option-A aggregated input
PER_PATCH_DIR = paths.DELTA_PER_PATCH_DIR   # raw per-patch input
DRIFT_DIR  = paths.DRIFT_DIR
FIG_DIR    = paths.MATCHED_FIG_DIR
CHAN_DIR   = paths.CHANNEL_FIG_DIR

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics.matched_pairs_drift_analysis import (
    absorption_band_indices,
    compute_drift_spectrum,
    load_aggregated_delta_tables,
    load_delta_tables,
    plot_channel_scatter,
    plot_drift_spectrum,
)


def main() -> None:
    # 1. Pool the delta tables for this patch size (aggregated by default).
    delta_dir = AGG_DIR if USE_AGGREGATED else PER_PATCH_DIR
    loader = load_aggregated_delta_tables if USE_AGGREGATED else load_delta_tables
    size_by = "n_patches" if USE_AGGREGATED else "sza"
    try:
        df = loader(delta_dir, PATCH_SIZE)
    except ValueError as exc:
        print(f"\nInput schema error: {exc}")
        sys.exit(1)
    if df.empty:
        upstream = ("run_PICS_allpatches_aggregate_pairs.py" if USE_AGGREGATED
                    else "run_delta_calculations.py")
        print(f"No {'aggregated ' if USE_AGGREGATED else ''}delta CSVs for patch "
              f"size {PATCH_SIZE!r} in {delta_dir}.\n"
              f"Run scripts/{upstream} for the relevant patches first.")
        sys.exit(1)

    src = "aggregated (1 row/pair/channel)" if USE_AGGREGATED else "per-patch (pseudo-replicated)"
    patch_note = (f"{df['patch_id'].nunique()} patch(es) | " if "patch_id" in df.columns
                  else "")
    print(f"Patch size {PATCH_SIZE} km [{src}]: {len(df)} rows | "
          f"{df['PICS_ID'].nunique()} PICS | {patch_note}"
          f"{df['pair_id'].nunique()} pair(s) | {df['band_index'].nunique()} channel(s)")

    # 2. Per-channel drift spectra (both products).
    spec_rad = compute_drift_spectrum(df, "rad", alpha=ALPHA)
    spec_rfl = compute_drift_spectrum(df, "rfl", alpha=ALPHA)

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = paths.drift(PATCH_SIZE)
    pd.concat([spec_rad, spec_rfl], ignore_index=True).to_csv(out_csv, index=False)
    provenance.write_sidecar(out_csv, cfg.as_dict())
    print(f"\nDrift stats CSV -> {out_csv}")
    for name, spec in (("RAD", spec_rad), ("RFL", spec_rfl)):
        n_fit = int(spec["slope_pct_yr"].notna().sum())
        n_raw = int(spec["raw_significant"].fillna(False).sum())
        n_bh = int(spec["bh_significant"].sum())
        print(f"  {name}: {n_fit} channel(s) fit | {n_raw} raw p<{ALPHA} | "
              f"{n_bh} BH-significant (FDR<{ALPHA})")

    # 3. Drift-spectrum figure.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_drift_spectrum(spec_rad, spec_rfl, patch_size=PATCH_SIZE, alpha=ALPHA)
    spec_png = paths.drift_spectrum_fig(PATCH_SIZE)
    fig.savefig(spec_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Drift spectrum  -> {spec_png}")

    # 4. Optional per-channel two-panel scatter plots.
    if INDIVIDUAL_CHANNELS:
        CHAN_DIR.mkdir(parents=True, exist_ok=True)
        all_bands = (sorted(df["band_index"].unique()) if ONLY_BANDS is None
                     else list(ONLY_BANDS))
        abs_bands = set(absorption_band_indices(df))
        bands = [b for b in all_bands if b not in abs_bands]
        spectra = {"rad": spec_rad, "rfl": spec_rfl}
        print(f"\nRendering {len(bands)} per-channel figure(s) "
              f"({len(abs_bands)} water-vapor absorption channel(s) excluded) "
              f"-> {CHAN_DIR}")
        n_written = 0
        for k, bidx in enumerate(bands, 1):
            fig, wl = plot_channel_scatter(df, bidx, spectra=spectra, size_by=size_by)
            if fig is None:
                continue
            path = paths.channel_fig(PATCH_SIZE, wl)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_written += 1
            if k % 50 == 0 or k == len(bands):
                print(f"  [{k}/{len(bands)}] latest: {path.name}")
        print(f"  wrote {n_written} channel figure(s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
