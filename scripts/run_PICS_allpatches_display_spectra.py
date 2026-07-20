#!/usr/bin/env python
"""Display per-scene RAD/RFL spectra for EVERY patch of one size across a PICS.

The batched counterpart to ``run_display_spectra.py``: instead of one patch, it
enumerates every patch of ``PATCH_SIZE`` in the PICS (from the patch-granule
dict) and renders each one's survivor-spectra overlays back-to-back in a single
run. For each patch it applies the identical manual QC gate (drops ``keep=FALSE``
scenes, refuses a patch whose worksheet is not ``reviewed=TRUE``), then draws one
high-resolution overlay figure per product — each surviving scene a separate
series, coloured along a sequential ramp by acquisition date, with the deep
water-vapor absorption bands masked by data availability in *both* products.

This is the quickest way to re-check a whole PICS after a QC pass: flip through
``figures/spectra/`` and any scene that should have been discarded stands out as
a stray curve; a coherent bundle means the survivors are clean. Pure **local**
post-processor — no Earthdata, no streaming; it reads the cached per-channel
CSVs, the per-scene geometry tables, and the reviewed QC worksheets.

Outputs (repo-relative), one pair per patch::

    figures/spectra/reflectance/{PICS}_{PATCH}_{stat}_rfl_spectra.png
    figures/spectra/radiance/{PICS}_{PATCH}_{stat}_rad_spectra.png

Prerequisite: run the cube extraction (per-channel CSVs + geometry tables) and
complete the manual QC step for the patches (any QC runner -> mark each worksheet
reviewed) first. Set the target in run_config.yaml, then run::

    python scripts/run_PICS_allpatches_display_spectra.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics import paths
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.extract_patch_stats import load_patch_dict
from emit_pics.id_matched_pairs import sniff_delimiter
from emit_pics.PICS_allpatches_cube_extract import patch_ids_for_size
from emit_pics.display_spectra import (
    FIG_DPI,
    PRODUCTS,
    derive_bad_bands,
    load_patch_spectra,
    plot_patch_spectra,
)
from emit_pics.qc_gate import QCReviewRequired, apply_qc_filter

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PICS_ID    = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_SIZE = cfg.patch_size
STAT       = cfg.spectra_stat
REQUIRE_QC_REVIEW = cfg.require_qc_review

# Repo-relative paths — every name is built in emit_pics.paths.
DICT_PATH = paths.patch_dict(PICS_ID)


def load_scenes(geom_path: Path) -> pd.DataFrame:
    """Load one patch's per-scene geometry table, coercing datetime + RFL flag."""
    scenes = pd.read_csv(geom_path)
    scenes["datetime"] = pd.to_datetime(scenes["datetime"])
    scenes["has_rfl"] = (
        scenes["has_rfl"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return scenes.sort_values("datetime").reset_index(drop=True)


def process_patch(pid: str) -> str:
    """Render the survivor-spectra overlays for one patch; return a status token.

    Mirrors ``run_display_spectra.main`` for a single patch, but never exits the
    process: a patch that is missing inputs, unreviewed, fully pruned, or lacking
    radiance stats is reported and skipped so the batch continues. Returns one of
    ``"plotted"``, ``"blocked"`` (QC gate), ``"empty"`` (no survivors / no data),
    or ``"missing"`` (no geometry table).
    """
    geom_path = paths.geometry(PICS_ID, pid)
    qc_path   = paths.qc_sheet(PICS_ID, pid)
    if not geom_path.exists():
        print(f"  {pid}: no geometry table ({geom_path.name}); skipped.")
        return "missing"

    scenes = load_scenes(geom_path)

    # Manual QC gate — identical to run_display_spectra: drop keep=FALSE scenes.
    try:
        scenes, qc = apply_qc_filter(scenes, qc_path,
                                     require_review=REQUIRE_QC_REVIEW)
    except QCReviewRequired as exc:
        print(f"  {pid}: QC gate blocked ({exc}); skipped.")
        return "blocked"
    if len(scenes) < 1:
        print(f"  {pid}: no scenes kept after QC; skipped.")
        return "empty"

    # Load each surviving scene's spectrum once (radiance anchors the join).
    loaded, missing = load_patch_spectra(scenes, PICS_ID, pid,
                                         paths.RFL_DIR, paths.RAD_DIR)
    if missing:
        print(f"  {pid}: {len(missing)} scene(s) skipped — no radiance CSV.")
    if not loaded:
        print(f"  {pid}: no scenes have radiance stats; skipped.")
        return "empty"

    # Water-vapor bad bands (fixed instrument property) blanked in both products.
    bad_bands = derive_bad_bands(loaded)

    n_written = 0
    for key, cfg in PRODUCTS.items():
        fig, plotted = plot_patch_spectra(loaded, cfg, STAT, bad_bands,
                                          PICS_ID, pid)
        out_path = paths.spectra_fig(PICS_ID, pid, STAT, key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if plotted:
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            n_written += 1
        plt.close(fig)

    print(f"  {pid}: {qc['n_after']} kept ({qc['n_excluded']} excluded) "
          f"-> {n_written} figure(s).")
    return "plotted" if n_written else "empty"


def main() -> None:
    # 1. Enumerate this size's patches from the patch-granule dict.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    print(f"{PICS_ID}, size {PATCH_SIZE} km: rendering {STAT} survivor spectra for "
          f"{len(patch_ids)} patch(es).")

    # 2. Render each patch; tally outcomes.
    tally = {"plotted": 0, "blocked": 0, "empty": 0, "missing": 0}
    for pid in patch_ids:
        tally[process_patch(pid)] += 1

    # 3. Summary.
    print("\n--- Summary ---")
    print(f"  patches plotted:          {tally['plotted']}")
    print(f"  blocked (QC not reviewed):{tally['blocked']}")
    print(f"  empty (no survivors/data):{tally['empty']}")
    print(f"  missing geometry table:   {tally['missing']}")
    print(f"\nFigures -> {paths.SPECTRA_RFL_DIR.parent}")
    if tally["blocked"]:
        print("  Some patches are not marked reviewed. Complete QC "
              "(run_PICS_allpatches_qc_review.py) or set REQUIRE_QC_REVIEW=False.")


if __name__ == "__main__":
    main()
