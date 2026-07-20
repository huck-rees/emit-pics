#!/usr/bin/env python
"""Interactive keep/remove QC reviewer — click or key your verdicts into qc.csv.

Renders the multipanel QC figure for each scene **from the cached pixel cubes**
(offline) and lets you enter keep/remove verdicts that are written straight to the
per-patch worksheets ``data/qc/{PICS}_{PATCH}_qc.csv`` — no hand-editing.  It is
the interactive alternative to inspecting the PDFs from ``run_PICS_allpatches_qc_products.py``.

Review one patch (set ``PATCH_ID``) or every patch of a size (set
``PATCH_ID = None`` and pick ``PATCH_SIZE``).

Prerequisite: cubes must be cached (extract with ``CACHE_PIXELS = True``) and the
per-patch worksheets must exist (any QC runner; this tool will build a missing
worksheet from the geometry table if present).

Controls (mouse buttons or keyboard):
    k = keep      r = remove (reason picker)   1-4 = remove w/ reason
    left/right = prev/next    v = mark patch reviewed    q = quit

Set the analysis target in run_config.yaml, then run::

    python scripts/run_qc_review.py
"""

import sys

from emit_pics import paths
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config

# ============================================================
# INPUTS — set the analysis target in run_config.yaml, not here.
# ============================================================
cfg = load_config()
PICS_ID    = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_ID   = cfg.patch      # one patch; set to None to review ALL of PATCH_SIZE
PATCH_SIZE = cfg.patch_size # size token used only when PATCH_ID is None
SPECTRA_STAT = cfg.spectra_stat  # in-patch statistic for the spectra panels

DICT_PATH = paths.patch_dict(PICS_ID)

# Select an INTERACTIVE matplotlib backend (unlike the batch scripts, which use
# Agg). Fall back through the common GUI backends; error clearly if none work.
import matplotlib
if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo"):
    for _b in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(_b, force=True)
            break
        except Exception:
            continue
if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo"):
    print("No interactive matplotlib backend available (need Tk or Qt). "
          "Install tkinter/PyQt, or review via the PDFs instead.")
    sys.exit(1)

from emit_pics.id_matched_pairs import sniff_delimiter
from emit_pics.extract_patch_stats import load_patch_dict
from emit_pics.PICS_allpatches_cube_extract import patch_ids_for_size
from emit_pics.qc_review import QCReviewer


def main() -> None:
    if PATCH_ID is not None:
        patch_ids = [PATCH_ID]
    else:
        dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
        patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
        if not patch_ids:
            print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
            sys.exit(1)

    print(f"Reviewing {len(patch_ids)} patch(es) of {PICS_ID} "
          f"(backend: {matplotlib.get_backend()}).")
    reviewer = QCReviewer(paths.REPO_ROOT, PICS_ID, patch_ids, spectra_stat=SPECTRA_STAT)
    reviewer.run()


if __name__ == "__main__":
    main()
