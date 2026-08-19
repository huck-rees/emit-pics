"""Within-site correlation of solar geometry (cos theta) with time.

The pooled quasi-BRDF drift model assumes that, *within* a site, the solar
geometry ``cos(SZA)`` does not covary with acquisition time ``t``: a systematic
cos(SZA)-vs-time trend would be partly absorbed by the shared drift term and bias
the reported drift. This module quantifies that assumption per PICS as the
**Spearman rank correlation** between ``cos(SZA)`` and ``t`` over the site's
QC-surviving scenes — the same per-scene design table the BRDF drift fit uses
(:func:`emit_pics.brdf_drift.assemble_design`, which supplies ``g = cos(SZA)``
and ``t`` in years since the common origin).

Spearman (rank-based) rather than Pearson so the check is robust to the
non-uniform seasonal sampling of the acquisitions and needs no linearity
assumption. The companion figure is
:func:`emit_pics.viz.figures.cos_sza_time_corr`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Columns of the tidy per-site correlation table.
COS_SZA_CORR_COLS = ["site", "n", "spearman_rho", "p_value"]


def site_cos_sza_time_corr(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-site Spearman ``corr(cos SZA, t)`` from a pooled design table.

    Parameters
    ----------
    obs : pandas.DataFrame
        Pooled per-scene table with ``site``, ``g`` (= ``cos(SZA)``), and ``t``
        (years). Exactly the frame :func:`emit_pics.brdf_drift.assemble_design`
        returns.

    Returns
    -------
    pandas.DataFrame
        One row per site (sorted), columns :data:`COS_SZA_CORR_COLS`. A site with
        fewer than three scenes, or with no spread in ``g`` or ``t``, yields NaN
        ``spearman_rho``/``p_value`` (the rank correlation is undefined there).
    """
    rows = []
    for site, sub in obs.groupby("site"):
        g = sub["g"].to_numpy(float)
        t = sub["t"].to_numpy(float)
        n = len(sub)
        if n >= 3 and np.ptp(g) > 0 and np.ptp(t) > 0:
            rho, p = spearmanr(g, t)
            rho, p = float(rho), float(p)
        else:
            rho, p = np.nan, np.nan
        rows.append({"site": site, "n": int(n), "spearman_rho": rho, "p_value": p})
    return (pd.DataFrame(rows, columns=COS_SZA_CORR_COLS)
            .sort_values("site").reset_index(drop=True))
