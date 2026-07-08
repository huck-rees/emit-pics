#!/usr/bin/env python
"""Render per-scene QA/QC visuals for ONE patch into a single PDF (+ worksheet).

For every EMIT acquisition that covers the patch (read from the patch-granule
dictionary; no live search), this builds one multipanel QA/QC figure — RGB, every
MASK band, SZA, VZA, a context panel (PICS / granule / patch boundaries + nadir
track), and below the maps two spectra panels (reflectance | radiance) overlaying
every scene covering the patch with the current scene bold red — and writes one
page per acquisition to ``data/qc/{PICS}_{PATCH}_QC_snapshots.pdf``.  It then
creates/refreshes the keep/remove worksheet ``data/qc/{PICS}_{PATCH}_qc.csv``.

**Cube-first, lazy login.** Each page renders from a cached pixel cube when one
exists (fully offline); Earthdata authentication is triggered only if a scene
must actually be streamed.  Set ``PREFER_CUBE = False`` to always stream.  The
nadir track needs whole-scene VZA and so is drawn only on the streaming path.

For **every** patch of a size across a PICS in one pass, use
``scripts/run_PICS_allpatches_qc.py`` (this script's batched counterpart); both
delegate to :func:`emit_pics.qc_report.run_patch_qc`.

Edit the user-configurable inputs below, then run::

    python scripts/run_display_qc_visuals.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the dictionary / bnds gpkg

PREFER_CUBE  = True    # render map panels from the cached pixel cube when present
DRAW_NADIR   = False   # draw the nadir track? (streaming path only; whole-scene VZA)
SPECTRA_STAT = "median"  # in-patch statistic for the spectra panels ("median"/"mean")
MAKE_PDF     = True    # write the PDF (False = only refresh the worksheet)

RAD_SHORT_NAME = "EMITL1BRAD"   # L1B radiance (+ bundled OBS) collection
RFL_SHORT_NAME = "EMITL2ARFL"   # L2A reflectance (+ MASK) collection

_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (write straight to PDF)

from emit_pics.qc_report import EarthdataLogin, default_qc_config, run_patch_qc


def main() -> None:
    cfg = default_qc_config(
        _REPO_ROOT, PICS_ID, PATCH_ID,
        rad_short=RAD_SHORT_NAME, rfl_short=RFL_SHORT_NAME,
        prefer_cube=PREFER_CUBE, draw_nadir=DRAW_NADIR,
        spectra_stat=SPECTRA_STAT, make_pdf=MAKE_PDF,
    )
    login = EarthdataLogin()   # lazy — only authenticates if a stream is needed
    summary = run_patch_qc(cfg, login)

    print("\n--- Completeness ---")
    print(f"  acquisitions expected:    {summary['n_acquisitions']}")
    print(f"  pages produced:           {summary['produced']}")
    print(f"  dropped by errors:        {summary['errored']}")
    print(f"  no page (missing prod):   {summary['missing']}")
    if summary["errored"]:
        print(f"  WARNING: {summary['errored']} acquisition(s) dropped by errors "
              f"after retries -- PDF is INCOMPLETE; consider re-running.")
    if summary["qc_path"]:
        print(f"\n  -> Inspect the PDF, set keep=FALSE (with a reason) for bad "
              f"scenes and reviewed=TRUE, or use scripts/run_qc_review.py.")


if __name__ == "__main__":
    main()
