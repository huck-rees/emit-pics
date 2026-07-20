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

Set the analysis target in run_config.yaml, then run::

    python scripts/run_PICS_allpatches_qc_review.py
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
PATCH_SIZE = cfg.patch_size # patch-size token (first field of patch_id, e.g. 10 km)
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
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    print(f"Reviewing ALL {len(patch_ids)} patch(es) of size {PATCH_SIZE} km "
          f"in {PICS_ID} (backend: {matplotlib.get_backend()}).")
    reviewer = QCReviewer(paths.REPO_ROOT, PICS_ID, patch_ids, spectra_stat=SPECTRA_STAT)
    reviewer.run()


if __name__ == "__main__":
    main()
