#!/usr/bin/env python
"""Batched QA/QC report for EVERY patch of one size across a whole PICS.

The batched counterpart to ``run_single_patch_qc_products.py``: it enumerates every
patch of ``PATCH_SIZE`` in the PICS (from the patch-granule dict) and, for each,
builds the per-scene QC PDF (optional) and refreshes the keep/remove worksheet —
via the same :func:`emit_pics.qc_report.run_patch_qc` the single-patch runner
uses.

No streaming is involved once the pixel cubes are cached: with ``PREFER_CUBE =
True`` every page renders from the local cubes, so a post-extraction QC pass over
a whole PICS touches **no network** (Earthdata login is lazy — triggered only if
some scene has no cube and must be streamed).

Set ``MAKE_PDF = False`` to skip the PDFs entirely and only (re)build the
worksheets — the lean path when verdicts are entered through the interactive
reviewer ``scripts/run_qc_review.py`` instead of a printed PDF.

Edit the user-configurable inputs below, then run::

    python scripts/run_PICS_allpatches_qc_products.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID    = "Libya_4"      # underscored site id (input + output file paths)
PATCH_SIZE = "10"           # patch-size token (first field of patch_id, e.g. 10 km)

PREFER_CUBE  = True    # render map panels from the cached pixel cubes (offline)
DRAW_NADIR   = False   # draw the nadir track? (streaming path only)
SPECTRA_STAT = "median"  # in-patch statistic for the spectra panels ("median"/"mean")
MAKE_PDF     = True    # write per-patch PDFs (False = only refresh worksheets)

RAD_SHORT_NAME = "EMITL1BRAD"   # L1B radiance (+ bundled OBS) collection
RFL_SHORT_NAME = "EMITL2ARFL"   # L2A reflectance (+ MASK) collection

_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics
DICT_PATH  = _REPO_ROOT / "data" / "ROIs" / "patches" / f"patch_EMIT_dict_{PICS_ID}.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (write straight to PDF)

from emit_pics.id_matched_pairs import sniff_delimiter
from emit_pics.extract_patch_stats import load_patch_dict
from emit_pics.PICS_allpatches_cube_extract import patch_ids_for_size
from emit_pics.qc_report import EarthdataLogin, default_qc_config, run_patch_qc


def main() -> None:
    # 1. Enumerate this size's patches from the (shared) patch-granule dict.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)
    print(f"PICS {PICS_ID}, size {PATCH_SIZE} km: {len(patch_ids)} patch(es). "
          f"{'PDF + worksheet' if MAKE_PDF else 'worksheet only'} per patch.\n")

    # 2. One lazy login shared across all patches (never used if all cubes exist).
    login = EarthdataLogin()
    summaries = []
    for i, pid in enumerate(patch_ids, 1):
        print(f"===== [{i}/{len(patch_ids)}] patch {pid} =====")
        cfg = default_qc_config(
            _REPO_ROOT, PICS_ID, pid,
            rad_short=RAD_SHORT_NAME, rfl_short=RFL_SHORT_NAME,
            prefer_cube=PREFER_CUBE, draw_nadir=DRAW_NADIR,
            spectra_stat=SPECTRA_STAT, make_pdf=MAKE_PDF,
        )
        try:
            summaries.append(run_patch_qc(cfg, login, dict_df=dict_df))
        except Exception as ex:
            print(f"  FAILED patch {pid}: {type(ex).__name__}: {ex}")
            summaries.append({"patch_id": pid, "produced": 0, "errored": -1,
                              "missing": 0, "pdf": None, "n_scenes": 0})
        print()

    # 3. Aggregate completeness across patches.
    n_patch = len(summaries)
    n_pdf = sum(1 for s in summaries if s.get("pdf"))
    n_pages = sum(s.get("produced", 0) for s in summaries)
    n_sheets = sum(1 for s in summaries if s.get("n_scenes", 0))
    n_err = sum(max(s.get("errored", 0), 0) for s in summaries)
    failed = [s["patch_id"] for s in summaries if s.get("errored", 0) < 0]

    print("===== Completeness (across patches) =====")
    print(f"  patches processed:        {n_patch}")
    print(f"  worksheets refreshed:     {n_sheets}")
    if MAKE_PDF:
        print(f"  PDFs written:             {n_pdf}  ({n_pages} page(s) total)")
    if n_err:
        print(f"  pages dropped by errors:  {n_err}")
    if failed:
        print(f"  WARNING: {len(failed)} patch(es) failed outright: {failed}")
    print("\n  -> Enter keep/remove verdicts with scripts/run_qc_review.py, "
          "or edit the worksheets by hand.")


if __name__ == "__main__":
    main()
