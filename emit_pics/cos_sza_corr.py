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


#: Column order for the two-sensor cos(SZA)-vs-time figure (Mauritania 1 last so
#: its blank EMIT cell falls in the final column).
DUAL_SITE_ORDER = ["Libya_1", "Libya_4", "Algeria_3", "Algeria_5", "Mauritania_1"]

_YR_SEC = 365.25 * 86400.0


def dual_sensor_design(cfg):
    """Assemble EMIT and Landsat 9 per-scene cos(SZA)/t designs + per-site corr.

    EMIT uses the BRDF design with the non-invariant sites dropped
    (:func:`emit_pics.brdf_drift.apply_drift_exclusion`); Landsat keeps all PICS.
    Both ``t`` columns are recomputed on one shared origin (the earlier of the two
    first-scene dates) so the panels line up; the Spearman statistic is
    origin-invariant. Returns ``None`` if there is no usable EMIT design.

    Returns
    -------
    dict or None
        ``{emit_obs, emit_corr, ls_obs, ls_corr, order, T0}`` where ``order`` is the
        canonical column order (:data:`DUAL_SITE_ORDER`) restricted to sites present.
    """
    from emit_pics import brdf_drift, landsat_brdf
    from emit_pics.config import canonical_token

    roster, _, _ = brdf_drift.resolve_roster(cfg)
    if not roster:
        return None
    emit_obs, _, T0e, _ = brdf_drift.assemble_design(roster, cfg)
    if emit_obs is None or emit_obs.empty:
        return None
    emit_obs, _ = brdf_drift.apply_drift_exclusion(emit_obs, cfg)

    try:
        ls_obs, _, _, _ = landsat_brdf.assemble_design_landsat(roster, cfg)
    except Exception:
        ls_obs = None

    # Normalise both clocks to tz-naive UTC (Landsat is tz-aware, EMIT naive) so a
    # shared origin can be taken; the Spearman statistic is origin-invariant anyway.
    emit_obs = emit_obs.copy()
    emit_obs["datetime"] = pd.to_datetime(emit_obs["datetime"], utc=True).dt.tz_localize(None)
    if ls_obs is not None and not ls_obs.empty:
        ls_obs = ls_obs.copy()
        ls_obs["datetime"] = pd.to_datetime(ls_obs["datetime"], utc=True).dt.tz_localize(None)
    else:
        ls_obs = None

    T0 = emit_obs["datetime"].min()
    if ls_obs is not None:
        T0 = min(T0, ls_obs["datetime"].min())
    emit_obs["t"] = (emit_obs["datetime"] - T0).dt.total_seconds() / _YR_SEC
    emit_corr = site_cos_sza_time_corr(emit_obs)
    if ls_obs is not None:
        ls_obs["t"] = (ls_obs["datetime"] - T0).dt.total_seconds() / _YR_SEC
        ls_corr = site_cos_sza_time_corr(ls_obs)
    else:
        ls_corr = None

    present = (set(emit_obs["site"].unique())
               | (set(ls_obs["site"].unique()) if ls_obs is not None else set()))
    order = [canonical_token(s) for s in DUAL_SITE_ORDER]
    order = [s for s in order if s in present]
    return {"emit_obs": emit_obs, "emit_corr": emit_corr,
            "ls_obs": ls_obs, "ls_corr": ls_corr, "order": order, "T0": T0}
