#!/usr/bin/env python
"""Cross-PICS radiometric-drift analysis for one patch size.

Pure **local** post-processor (no Earthdata). Pools every per-patch delta CSV of
one patch size (step 6, ``data/delta_r/{size}_*_delta.csv``) across all PICS and,
per EMIT channel, fits the drift rate as the robust (Theil–Sen) slope of the
pairwise percent change ΔR against the pair's time separation Δt, using the
**median** in-patch statistic. Significance is Kendall's τ, false-discovery
controlled across the 285 channels with Benjamini–Hochberg.

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

Edit the inputs below, then run::

    python scripts/run_matched_pairs_drift_analysis.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PATCH_SIZE = "10"           # patch-size token (first field of patch_id, e.g. 10 km)
ALPHA = 0.05                # Benjamini–Hochberg false-discovery-rate target
INDIVIDUAL_CHANNELS = True  # also render one two-panel scatter per channel (~285 PNGs)
ONLY_BANDS = None           # e.g. [0, 50, 150] to render a subset for testing; None = all

# Paths (repo-relative)
_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics
DATA_ROOT  = _REPO_ROOT / "data"
DELTA_DIR  = DATA_ROOT / "delta_r"
DRIFT_DIR  = DATA_ROOT / "drift"
FIG_DIR    = _REPO_ROOT / "figures" / "matched_pairs"
CHAN_DIR   = FIG_DIR / "individual_channels"

# ============================================================
# Ensure the project root is importable when run directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics.matched_pairs_drift_analysis import (
    absorption_band_indices,
    compute_drift_spectrum,
    load_delta_tables,
    plot_channel_scatter,
    plot_drift_spectrum,
)


def main() -> None:
    # 1. Pool the delta tables for this patch size.
    try:
        df = load_delta_tables(DELTA_DIR, PATCH_SIZE)
    except ValueError as exc:
        print(f"\nInput schema error: {exc}")
        sys.exit(1)
    if df.empty:
        print(f"No delta CSVs for patch size {PATCH_SIZE!r} in {DELTA_DIR}.\n"
              f"Run scripts/run_delta_calculations.py for the relevant patches first.")
        sys.exit(1)

    print(f"Patch size {PATCH_SIZE} km: {len(df)} rows | "
          f"{df['PICS_ID'].nunique()} PICS | {df['patch_id'].nunique()} patch(es) | "
          f"{df['pair_id'].nunique()} pair(s) | {df['band_index'].nunique()} channel(s)")

    # 2. Per-channel drift spectra (both products).
    spec_rad = compute_drift_spectrum(df, "rad", alpha=ALPHA)
    spec_rfl = compute_drift_spectrum(df, "rfl", alpha=ALPHA)

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DRIFT_DIR / f"{PATCH_SIZE}_drift.csv"
    pd.concat([spec_rad, spec_rfl], ignore_index=True).to_csv(out_csv, index=False)
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
    spec_png = FIG_DIR / f"{PATCH_SIZE}_drift_spectrum.png"
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
            fig, wl = plot_channel_scatter(df, bidx, spectra=spectra)
            if fig is None:
                continue
            path = CHAN_DIR / f"{PATCH_SIZE}_{wl:04.0f}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_written += 1
            if k % 50 == 0 or k == len(bands):
                print(f"  [{k}/{len(bands)}] latest: {path.name}")
        print(f"  wrote {n_written} channel figure(s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
