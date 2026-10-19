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
``scripts/run_PICS_allpatches_qc_products.py`` (this script's batched counterpart); both
delegate to :func:`emit_pics.qc_report.run_patch_qc`.

Set the analysis target in run_config.yaml, then run::

    python scripts/run_single_patch_qc_products.py
"""

from emit_pics import paths
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config

# INPUTS — set the analysis target in run_config.yaml, not here.
cfg_run = load_config()
PICS_ID  = canonical_token(cfg_run.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_ID = cfg_run.patch

PREFER_CUBE  = True    # script-local toggle: render map panels from the cached pixel cube when present
DRAW_NADIR   = False   # script-local toggle: draw the nadir track? (streaming path only; whole-scene VZA)
SPECTRA_STAT = cfg_run.spectra_stat  # in-patch statistic for the spectra panels ("median"/"mean")
MAKE_PDF     = True    # script-local toggle: write the PDF (False = only refresh the worksheet)

RAD_SHORT_NAME = cfg_run.rad_short_name   # L1B radiance (+ bundled OBS) collection
RFL_SHORT_NAME = cfg_run.rfl_short_name   # L2A reflectance (+ MASK) collection

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (write straight to PDF)

from emit_pics.qc_report import EarthdataLogin, default_qc_config, run_patch_qc


def main() -> None:
    cfg = default_qc_config(
        paths.REPO_ROOT, PICS_ID, PATCH_ID,
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
