#!/usr/bin/env python
"""Display per-scene RAD/RFL spectra for one patch (QC-surviving scenes).

Pure **local** post-processor — no Earthdata, no streaming. Runs *parallel* to
``run_id_matched_pairs.py`` on the **same inputs** (a PICS/patch and its cached
per-scene geometry table): it applies the identical manual QC gate, then for
every surviving scene loads the per-channel RAD/RFL stats CSVs
(``run_single_patch_cube_extract.py``) and draws one high-resolution overlay figure per
product — each scene a separate series, coloured along a sequential ramp by
acquisition date with a date-labelled legend.

The deep water-vapor absorption bands are masked by **data availability** (the
wavelengths where L2A reflectance is fill) in *both* products, so every curve
breaks in the same windows.

Outputs (repo-relative)::

    figures/spectra/reflectance/{PICS}_{PATCH}_{stat}_rfl_spectra.png
    figures/spectra/radiance/{PICS}_{PATCH}_{stat}_rad_spectra.png

Run ``run_single_patch_cube_extract.py`` (per-channel CSVs + geometry table) and complete the
manual QC step (``run_single_patch_qc_products.py`` -> mark the worksheet reviewed) first,
then set the target in run_config.yaml and run::

    python scripts/run_display_spectra.py
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
PICS_ID  = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_ID = cfg.patch
STAT     = cfg.spectra_stat
REQUIRE_QC_REVIEW = cfg.require_qc_review

# Repo-relative paths — every name is built in emit_pics.paths.
GEOM_PATH = paths.geometry(PICS_ID, PATCH_ID)
QC_PATH   = paths.qc_sheet(PICS_ID, PATCH_ID)


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

    # 1b. Manual QC gate — identical to run_id_matched_pairs: drop scenes marked
    #     keep=FALSE in the reviewed worksheet.
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

    if len(scenes) < 1:
        print("No scenes after QC; nothing to plot. Exiting.")
        sys.exit(1)

    # 2. Load each surviving scene's spectrum ONCE. Radiance anchors the join, so
    #    a scene with no radiance CSV cannot be drawn — report it, don't fail.
    loaded, missing = load_patch_spectra(scenes, PICS_ID, PATCH_ID,
                                         paths.RFL_DIR, paths.RAD_DIR,
                                         earth_sun_normalize=cfg.earth_sun_normalize,
                                         e0_method=cfg.e0_method)
    if missing:
        print(f"\n{len(missing)} scene(s) skipped — no radiance per-channel CSV "
              f"(re-run run_single_patch_cube_extract.py to include them):")
        for e in missing:
            print(f"  {e['label']}  ({e['scene_id']})")
    if not loaded:
        print("\nNo scenes have radiance stats; nothing to plot. Exiting.")
        sys.exit(1)

    # 3. Derive the water-vapor bad-band wavelengths ONCE (fixed instrument
    #    property); blank them in BOTH products so every curve breaks identically.
    bad_bands = derive_bad_bands(loaded)
    if len(bad_bands):
        print(f"Absorption mask: {len(bad_bands)} water-vapor band(s) "
              f"(~{bad_bands.min():.0f}-{bad_bands.max():.0f} nm) blanked in "
              f"both products.")
    else:
        print("Absorption mask: no reflectance-bearing scene found; no bands "
              "blanked (radiance shows the full grid).")

    # 4. One overlay figure per product.
    print(f"\n{len(loaded)} scene(s) with data; plotting {STAT} spectra.")
    for key, cfg in PRODUCTS.items():
        fig, plotted = plot_patch_spectra(loaded, cfg, STAT, bad_bands,
                                          PICS_ID, PATCH_ID)
        out_path = paths.spectra_fig(PICS_ID, PATCH_ID, STAT, key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if plotted:
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            print(f"  {cfg['name']} ({plotted} scene(s)) -> {out_path}")
        else:
            print(f"  {cfg['name']}: no scenes carried this product; "
                  f"no figure written.")
        plt.close(fig)


if __name__ == "__main__":
    main()
