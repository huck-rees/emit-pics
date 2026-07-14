#!/usr/bin/env python
"""Interactive keep/remove QC review for EVERY patch of one size across a PICS.

The batched counterpart to ``run_qc_review.py``: instead of one patch, it
enumerates every patch of ``PATCH_SIZE`` in the PICS (from the patch-granule
dict) and reviews them back-to-back in one session, rendering each scene from the
cached pixel cubes (offline) and writing keep/remove verdicts straight to the
per-patch worksheets ``data/qc/{PICS}_{PATCH}_qc.csv`` — no hand-editing.

Because every worksheet (and PDF) is patch-specific, the reviewer guards each
patch boundary: moving from the last scene of a patch that is **not yet marked
reviewed** into the next patch arms a one-key confirm gate, so a distracted
reviewer cannot silently roll into the following patch having forgotten to mark
this one.  Press ``v`` to mark reviewed and continue, the forward key again to
skip anyway, or the back key to return; quitting with any patch still unreviewed
asks for the same confirmation.

Prerequisite: cubes must be cached (extract with ``CACHE_PIXELS = True``) and the
per-patch worksheets must exist (any QC runner; this tool builds a missing
worksheet from the geometry table if present).

Controls (mouse buttons or keyboard):
    k = keep      r = remove (reason picker)   1-4 = remove w/ reason
    left/right = prev/next    v = mark patch reviewed    q = quit

Edit the inputs below, then run::

    python scripts/run_PICS_allpatches_qc_review.py
"""

import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID    = "Libya_4"      # underscored site id
PATCH_SIZE = "10"           # patch-size token (first field of patch_id, e.g. 10 km)
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
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    print(f"Reviewing ALL {len(patch_ids)} patch(es) of size {PATCH_SIZE} km "
          f"in {PICS_ID} (backend: {matplotlib.get_backend()}).")
    reviewer = QCReviewer(_REPO_ROOT, PICS_ID, patch_ids, spectra_stat=SPECTRA_STAT)
    reviewer.run()


if __name__ == "__main__":
    main()
