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
then edit the inputs below and run::

    python scripts/run_display_spectra.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the geometry table
STAT     = "median"        # in-patch statistic to plot: "median" or "mean"

REQUIRE_QC_REVIEW = True   # require a reviewed QC worksheet before proceeding;
                           # set False to bypass the manual QC gate entirely

# Paths (repo-relative; inputs produced by upstream steps)
_REPO_ROOT = Path(__file__).resolve().parent.parent           # .../emit-pics
DATA_ROOT  = _REPO_ROOT / "data"
GEOM_DIR   = DATA_ROOT / "geometry"
QC_DIR     = DATA_ROOT / "qc"
RFL_DIR    = DATA_ROOT / "reflectance"
RAD_DIR    = DATA_ROOT / "radiance"
FIG_ROOT   = _REPO_ROOT / "figures" / "spectra"

GEOM_PATH = GEOM_DIR / f"geometry_{PICS_ID}_{PATCH_ID}.csv"
QC_PATH   = QC_DIR / f"{PICS_ID}_{PATCH_ID}_qc.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from emit_pics.display_spectra import (
    FIG_DPI,
    PRODUCTS,
    derive_bad_bands,
    load_patch_spectra,
    plot_patch_spectra,
)
from emit_pics.qc_gate import QCReviewRequired, apply_qc_filter


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
                                         RFL_DIR, RAD_DIR)
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
        out_dir = FIG_ROOT / cfg["subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{PICS_ID}_{PATCH_ID}_{STAT}_{key}_spectra.png"
        if plotted:
            fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            print(f"  {cfg['name']} ({plotted} scene(s)) -> {out_path}")
        else:
            print(f"  {cfg['name']}: no scenes carried this product; "
                  f"no figure written.")
        plt.close(fig)


if __name__ == "__main__":
    main()
