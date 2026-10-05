#!/usr/bin/env python
"""Build one Landsat patch's QC products: the snapshot PDF + keep/remove worksheet.

The Landsat analog of ``scripts/run_single_patch_qc_products.py``. Renders one
offline page per cached QC cube (RGB / NIR / SZA / VZA + a TOA-spectrum overlay)
into ``data/landsat/qc/{PICS}_{PATCH}_QC_snapshots.pdf`` and refreshes the
keep/remove worksheet ``data/landsat/qc/{PICS}_{PATCH}_qc.csv``.

Everything is offline (cubes are local), so no credentials are needed. If the
cubes live on an external drive, set EXTERNAL_CUBE_DIR below.

Run in the `ls9-draft` env::

    python scripts/run_landsat_qc_products.py
"""

import matplotlib
matplotlib.use("Agg")            # headless PDF rendering

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.landsat_qc_report import default_qc_config, run_patch_qc

# --- target (script-local for the draft; mirrors run_config site/patch) ---
cfg_run = load_config()
SITE = cfg_run.site
PATCH = cfg_run.patch
MAKE_PDF = True
EXTERNAL_CUBE_DIR = r"D:\JPL\emit-pics\data\landsat\cubes"   # None = repo-relative


def main():
    if EXTERNAL_CUBE_DIR:
        paths.set_ls_cube_dir(EXTERNAL_CUBE_DIR)
    cfg = default_qc_config(SITE, PATCH, make_pdf=MAKE_PDF)
    print(f"Landsat QC products: {cfg.pics_id} {cfg.patch_id}")
    print(f"  cubes: {cfg.cube_dir}")
    summary = run_patch_qc(cfg)
    print(f"\nDone: {summary['n_scenes']} scene(s) in worksheet, "
          f"{summary['produced']} PDF page(s), "
          f"{summary['missing_cube']} scene(s) without a cube.")
    if summary["pdf"]:
        print(f"Inspect {summary['pdf']}, mark bad scenes (keep=FALSE) + reason, "
              f"set reviewed=TRUE in the worksheet top row (or use the interactive "
              f"reviewer: scripts/run_landsat_qc_review.py).")


if __name__ == "__main__":
    main()
