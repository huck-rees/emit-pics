#!/usr/bin/env python
"""Interactive keep/remove QC reviewer — click or key your verdicts into qc.csv.

Renders the multipanel QC figure for each scene **from the cached pixel cubes**
(offline) and lets you enter keep/remove verdicts that are written straight to the
per-patch worksheets ``data/qc/{PICS}_{PATCH}_qc.csv`` — no hand-editing.  It is
the interactive alternative to inspecting the PDFs from ``run_PICS_allpatches_qc.py``.

Review one patch (set ``PATCH_ID``) or every patch of a size (set
``PATCH_ID = None`` and pick ``PATCH_SIZE``).

Prerequisite: cubes must be cached (extract with ``CACHE_PIXELS = True``) and the
per-patch worksheets must exist (any QC runner; this tool will build a missing
worksheet from the geometry table if present).

Controls (mouse buttons or keyboard):
    k = keep      r = remove (reason picker)   1-4 = remove w/ reason
    left/right = prev/next    v = mark patch reviewed    q = quit

Edit the inputs below, then run::

    python scripts/run_qc_review.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID    = "Libya_4"      # underscored site id
PATCH_ID   = "10_7_1"       # one patch; set to None to review ALL of PATCH_SIZE
PATCH_SIZE = "10"           # size token used only when PATCH_ID is None
SPECTRA_STAT = "median"     # in-patch statistic for the spectra panels

_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics
DICT_PATH  = _REPO_ROOT / "data" / "ROIs" / "patches" / f"patch_EMIT_dict_{PICS_ID}.csv"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

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
    reviewer = QCReviewer(_REPO_ROOT, PICS_ID, patch_ids, spectra_stat=SPECTRA_STAT)
    reviewer.run()


if __name__ == "__main__":
    main()
