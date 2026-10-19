#!/usr/bin/env python
"""Interactive keep/remove reviewer for one Landsat patch's QC cubes.

The Landsat analog of ``scripts/run_qc_review.py``. Opens a matplotlib viewer
(RGB / NIR / SZA / VZA + TOA-spectrum overlay) with Keep/Remove/Prev/Next/
Reviewed/Quit controls, writing verdicts straight into the per-patch worksheet
``data/landsat/qc/{PICS}_{PATCH}_qc.csv`` (same schema + gate as EMIT).

Needs an interactive matplotlib backend (TkAgg/Qt) and the cubes on disk (set
EXTERNAL_CUBE_DIR if they live on an external drive). Run in the `ls9-draft`
env::

    python scripts/run_landsat_qc_review.py
"""

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.landsat_qc_review import LandsatQCReviewer

cfg_run = load_config()
SITE = cfg_run.site
PATCH = cfg_run.patch
EXTERNAL_CUBE_DIR = None   # None = repo-relative; set an absolute path to read cubes from an external drive


def main():
    if EXTERNAL_CUBE_DIR:
        paths.set_ls_cube_dir(EXTERNAL_CUBE_DIR)
    LandsatQCReviewer(SITE, PATCH).run()


if __name__ == "__main__":
    main()
