"""Quasi-BRDF pooled multi-site drift — the model, the fits, the bootstraps.

Port of ``notebooks/BRDF_drift_workflow_allPICS.ipynb`` into the library. Every
PICS contributes **one representative patch = one "site"**; all sites share a
**single drift rate** ``gamma`` while each keeps its own brightness/BRDF terms.
For ``S`` sites the state vector is ``2S + 1``::

    rho_hat_s = (alpha_s + beta_s * cos(theta)) * (1 + gamma * t)
    x = [alpha_1..alpha_S, beta_1..beta_S, gamma]

``alpha_s`` / ``beta_s`` are per-site nuisances (each pinned by its own scenes);
``gamma`` is the one shared instrument drift, constrained by all sites jointly.
``t`` is years since the earliest scene across all sites (a common clock), and
``g = cos(SZA)``.

This module has two halves:

* **model & fitting** (this file, below) — pure NumPy/SciPy, deterministic given
  the fixed bootstrap seeds, so results reproduce the notebook exactly and the
  fits are unit-testable with synthetic data.
* **assembly / IO** (appended below) — roster resolution + per-scene table
  assembly from geometry/QC/spectra, reusing the existing pipeline loaders.

Nothing here calls ``sys.exit`` or prints; the executable scripts own I/O and
messaging. The shared-drift **consistency check** (:func:`consistency_check`)
computes a verdict for the runner to warn on — it never blocks.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from emit_pics import paths
from emit_pics.config import load_sites
from emit_pics.delta_calculations import _perchannel_path, load_scene_spectrum
from emit_pics.earth_sun import normalize_radiance_spectrum
from emit_pics.qc_gate import QCReviewRequired, apply_qc_filter

#: Compact acquisition-timestamp format carried in the design table (``ts``).
_TS_FMT = "%Y%m%dT%H%M%S"

#: Seconds per Julian year — the common-clock scale for ``t`` (years).
_YR_SEC = 365.25 * 86400


# ============================================================
# MODEL
# ============================================================
def _resid(x, S, sidx, g, t, rho):
    """Residual of ``(alpha_s + beta_s*g)(1 + gamma*t) - rho`` for a ``2S+1`` vector.

    ``x = [alpha_1..alpha_S, beta_1..beta_S, gamma]``; ``sidx`` maps each scene to
    its site index (so the per-scene alpha/beta are gathered by ``sidx``).
    """
    a = x[:S][sidx]
    b = x[S:2 * S][sidx]
    ga = x[2 * S]
    return (a + b * g) * (1 + ga * t) - rho


# ============================================================
# PER-SITE STANDALONE FIT (§4b — shared-drift consistency check)
# ============================================================
def fit_site(g, t, rho):
    """Standalone 3-parameter fit ``[alpha, beta, gamma]`` for one site/channel.

    Warm-started from the OLS of ``rho`` on ``[1, g]`` with ``gamma0 = 0``. Non-
    finite rows are dropped. Returns ``{alpha, beta, gamma, n}``.
    """
    g = np.asarray(g, float); t = np.asarray(t, float); rho = np.asarray(rho, float)
    ok = np.isfinite(g) & np.isfinite(t) & np.isfinite(rho)
    g, t, rho = g[ok], t[ok], rho[ok]
    a0, b0 = np.linalg.lstsq(np.column_stack([np.ones(g.size), g]), rho, rcond=None)[0]
    res = least_squares(lambda x: (x[0] + x[1] * g) * (1 + x[2] * t) - rho, [a0, b0, 0.0])
    return {"alpha": res.x[0], "beta": res.x[1], "gamma": res.x[2], "n": int(g.size)}


def boot_site_gamma(g, t, rho, n_boot, seed=1):
    """One site's standalone ``gamma`` + a bootstrap 95% CI (percentile).

    Case-resamples the site's scenes ``n_boot`` times (fixed ``seed`` for
    reproducibility) and refits. Returns ``(gamma_point, (ci_lo, ci_hi))``.
    """
    g = np.asarray(g, float); t = np.asarray(t, float); rho = np.asarray(rho, float)
    ok = np.isfinite(g) & np.isfinite(t) & np.isfinite(rho)
    g, t, rho = g[ok], t[ok], rho[ok]
    base = fit_site(g, t, rho)
    x0 = [base["alpha"], base["beta"], base["gamma"]]
    rng = np.random.default_rng(seed)
    n, draws = g.size, np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        r = least_squares(
            lambda x: (x[0] + x[1] * g[idx]) * (1 + x[2] * t[idx]) - rho[idx], x0)
        draws[k] = r.x[2]
    return base["gamma"], np.percentile(draws, [2.5, 97.5])


def consistency_check(persite, *, marginal_ratio=0.25):
    """Do the per-site standalone ``gamma`` CIs support one shared drift? (§4b).

    The machine version of the eyeball forest-plot test: for each product, the
    sites share a common drift only if **all** their CIs overlap — i.e. the
    intersection ``[max(ci_lo), min(ci_hi)]`` is non-empty. A wide overlap is
    reassuring; an edge-touching sliver is not.

    Parameters
    ----------
    persite : pandas.DataFrame
        Columns ``product``, ``site``, ``gamma_pct_yr``, ``ci_lo``, ``ci_hi``
        (the §4b table). Rows with non-finite CI bounds are ignored.
    marginal_ratio : float
        Overlap width below this fraction of the median CI width is flagged
        ``"marginal"`` (a thin/edge-touching overlap) rather than ``"pass"``.

    Returns
    -------
    dict
        ``{"overall": verdict, "per_product": {prod: {...}}}`` where each product
        carries ``verdict`` (``"pass"`` | ``"marginal"`` | ``"fail"`` | ``"n/a"``),
        ``overlap_lo``, ``overlap_hi``, ``overlap_width``, ``median_ci_width``,
        ``n_sites``. ``overall`` is the worst product verdict
        (``fail`` > ``marginal`` > ``pass``; ``n/a`` if no product could be tested).
    """
    rank = {"fail": 0, "marginal": 1, "pass": 2}
    per_product = {}
    for prod, sub in persite.groupby("product"):
        lo = np.asarray(sub["ci_lo"], float)
        hi = np.asarray(sub["ci_hi"], float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        lo, hi = lo[ok], hi[ok]
        if lo.size == 0:
            per_product[str(prod)] = {"verdict": "n/a", "overlap_lo": np.nan,
                                      "overlap_hi": np.nan, "overlap_width": np.nan,
                                      "median_ci_width": np.nan, "n_sites": 0}
            continue
        overlap_lo, overlap_hi = float(lo.max()), float(hi.min())
        overlap_width = overlap_hi - overlap_lo
        median_ci_width = float(np.median(hi - lo))
        if overlap_width < 0:
            verdict = "fail"
        elif median_ci_width > 0 and overlap_width < marginal_ratio * median_ci_width:
            verdict = "marginal"
        else:
            verdict = "pass"
        per_product[str(prod)] = {
            "verdict": verdict, "overlap_lo": overlap_lo, "overlap_hi": overlap_hi,
            "overlap_width": overlap_width, "median_ci_width": median_ci_width,
            "n_sites": int(lo.size),
        }

    testable = [v["verdict"] for v in per_product.values() if v["verdict"] != "n/a"]
    overall = min(testable, key=lambda v: rank[v]) if testable else "n/a"
    return {"overall": overall, "per_product": per_product}


# ============================================================
# JOINT POOLED FIT (§5 — per-site alpha/beta, one shared gamma)
# ============================================================
def _build(frame, product):
    """Extract ``(sites, sidx, g, t, rho)`` for *product* from a design frame."""
    sites = sorted(frame["site"].unique())
    lut = {s: i for i, s in enumerate(sites)}
    sidx = frame["site"].map(lut).to_numpy()
    g = frame["g"].to_numpy(float)
    t = frame["t"].to_numpy(float)
    rho = frame[f"rho_{product}"].to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(t) & np.isfinite(rho)
    return sites, sidx[ok], g[ok], t[ok], rho[ok]


def fit_joint(frame, product):
    """Pooled ``2S+1`` fit: per-site ``alpha``/``beta`` with one shared ``gamma``.

    Warm-started per site from the OLS of ``rho`` on ``[1, g]`` (``gamma0 = 0``).
    Returns the fit plus the aligned arrays used, so downstream (correction,
    bootstrap) needs no re-extraction.
    """
    sites, sidx, g, t, rho = _build(frame, product)
    S = len(sites)
    a0, b0 = np.empty(S), np.empty(S)
    for i in range(S):
        m = sidx == i
        a0[i], b0[i] = np.linalg.lstsq(
            np.column_stack([np.ones(m.sum()), g[m]]), rho[m], rcond=None)[0]
    x0 = np.concatenate([a0, b0, [0.0]])
    res = least_squares(_resid, x0, args=(S, sidx, g, t, rho))
    return {"sites": sites, "alpha": res.x[:S], "beta": res.x[S:2 * S],
            "gamma": float(res.x[2 * S]), "success": bool(res.success),
            "sidx": sidx, "labels": np.array(sites)[sidx],
            "g": g, "t": t, "rho": rho, "n": int(g.size)}


# ============================================================
# POOLED GAMMA BOOTSTRAP (§7 — stratified within-site)
# ============================================================
def bootstrap_joint_gamma(frame, product, n_boot, seed=42):
    """Bootstrap the pooled shared ``gamma`` by stratified within-site resampling.

    Every site keeps its scene count (stratified), so the multi-site design is
    preserved and only between-scene scatter is captured. Returns
    ``{gamma, se, ci: [lo, hi], draws}`` (percentile CI).
    """
    base = fit_joint(frame, product)
    S, sidx = len(base["sites"]), base["sidx"]
    g, t, rho = base["g"], base["t"], base["rho"]
    x0 = np.concatenate([base["alpha"], base["beta"], [base["gamma"]]])
    pos = [np.where(sidx == i)[0] for i in range(S)]     # stratified by site
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        pick = np.concatenate([rng.choice(p, size=p.size, replace=True) for p in pos])
        r = least_squares(_resid, x0, args=(S, sidx[pick], g[pick], t[pick], rho[pick]))
        draws[k] = r.x[2 * S]
    return {"gamma": base["gamma"], "se": float(draws.std()),
            "ci": np.percentile(draws, [2.5, 97.5]), "draws": draws}


# ============================================================
# PER-CHANNEL GAMMA (§8 — the gamma(lambda) drift spectrum)
# ============================================================
def _warm(S, sidx, g, t, rho):
    """Warm-start ``2S+1`` vector: per-site OLS alpha/beta (single-scene -> mean)."""
    a0, b0 = np.zeros(S), np.zeros(S)
    for i in range(S):
        m = sidx == i
        if m.sum() >= 2:
            a0[i], b0[i] = np.linalg.lstsq(
                np.column_stack([np.ones(m.sum()), g[m]]), rho[m], rcond=None)[0]
        elif m.sum() == 1:
            a0[i] = rho[m].mean()
    return np.concatenate([a0, b0, [0.0]])


def channel_gamma(rho, g_all, t_all, site_all, n_boot, rng):
    """Pooled shared ``gamma`` at ONE channel + bootstrap CI and two-sided boot p.

    ``rho`` is the per-scene reflectance/radiance at one band (aligned with
    ``g_all`` / ``t_all`` / ``site_all``, the pooled design). Only sites actually
    present at this channel are fit (band-order alignment can leave a scene out).
    Returns ``(gamma, (ci_lo, ci_hi), p_value, n)``; a degenerate channel
    (``n <= 2S``) returns NaNs. Pass a shared ``rng`` so channels advance one
    reproducible stream (as in the notebook).
    """
    ok = np.isfinite(rho) & np.isfinite(g_all) & np.isfinite(t_all)
    g, t, r = g_all[ok], t_all[ok], rho[ok]
    present, si = np.unique(site_all[ok], return_inverse=True)   # remap present sites
    S = len(present)
    if S == 0 or r.size <= 2 * S:
        return np.nan, (np.nan, np.nan), np.nan, int(r.size)
    base = least_squares(_resid, _warm(S, si, g, t, r), args=(S, si, g, t, r))
    gam = base.x[2 * S]
    pos = [np.where(si == i)[0] for i in range(S)]
    draws = np.empty(n_boot)
    for k in range(n_boot):
        pick = np.concatenate([rng.choice(p, size=p.size, replace=True) for p in pos])
        res = least_squares(_resid, base.x, args=(S, si[pick], g[pick], t[pick], r[pick]))
        draws[k] = res.x[2 * S]
    ci = np.percentile(draws, [2.5, 97.5])
    npos, nneg = int((draws > 0).sum()), int((draws < 0).sum())
    pval = min(1.0, 2 * min((npos + 1) / (n_boot + 1), (nneg + 1) / (n_boot + 1)))
    return gam, (ci[0], ci[1]), pval, int(r.size)


# ============================================================
# ASSEMBLY / IO  (reuses the pipeline's geometry / QC / spectra loaders)
# ============================================================
def resolve_roster(cfg):
    """Map the configured roster to valid, non-excluded PICS (§1).

    ``cfg.brdf_site_patches`` already dropped null-patch sites (see the config
    loader). Here we drop any site named in ``cfg.brdf_exclude_sites`` and any
    site not present in ``config/sites.yaml`` (canonical token), so the run only
    proceeds on real, intended sites.

    Returns
    -------
    (dict, list, list)
        ``roster`` (``{pics_id: patch}``, source order), ``excluded`` (dropped
        by config), ``unknown`` (not found in sites.yaml).
    """
    valid = {info["site_name"].replace(" ", "_") for info in load_sites().values()}
    roster, excluded, unknown = {}, [], []
    for site, patch in cfg.brdf_site_patches.items():
        if site in cfg.brdf_exclude_sites:
            excluded.append(site)
        elif site not in valid:
            unknown.append(site)
        else:
            roster[site] = patch
    return roster, excluded, unknown


def scene_perchannel(site, patch, ts, product, *, earth_sun_normalize=True,
                     e0_method="spencer"):
    """One scene's per-channel stats CSV as a DataFrame, or None if absent.

    When *product* is ``"rad"`` and *earth_sun_normalize* is set (default), the
    radiance stat columns are divided by the scene's Earth–Sun eccentricity
    factor ``E0`` — the same 1-AU normalization applied in
    :func:`load_scene_spectrum`; *e0_method* selects the ``E0`` provider
    (``"spencer"`` | ``"table"``). Reflectance is returned unchanged.
    """
    d = paths.RAD_DIR if product == "rad" else paths.RFL_DIR
    path = _perchannel_path(d, site, patch, str(ts), product)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if earth_sun_normalize and product == "rad":
        df = normalize_radiance_spectrum(df, pd.to_datetime(str(ts), format=_TS_FMT),
                                         method=e0_method)
    return df


def load_site_obs(pics_id, patch, target_wl, *, require_qc_review,
                  earth_sun_normalize=True, e0_method="spencer"):
    """Per-scene obs for one site: QC-surviving, ``has_rfl``, median rho at the band (§2).

    Mirrors the single-patch recipe: read the geometry table, apply the QC gate,
    keep only reflectance-bearing survivors, then take each scene's median rho at
    the channel nearest *target_wl* (band-order aligned to the first survivor).
    ``rho_rad`` is Earth–Sun (``E0``) normalized when *earth_sun_normalize* is set
    (default), via :func:`load_scene_spectrum`; ``rho_rfl`` is untouched.

    Returns
    -------
    (pandas.DataFrame | None, float | None, str)
        The per-scene table (site, patch, timestamp, datetime, sza_mean,
        scene_id, rho_rad, rho_rfl), the wavelength actually used, and a status
        note. ``(None, None, note)`` when the site yields no usable scenes.
    """
    geom_path = paths.geometry(pics_id, patch)
    qc_path = paths.qc_sheet(pics_id, patch)
    if not geom_path.exists():
        return None, None, f"missing geometry {geom_path.name}"

    geom = pd.read_csv(geom_path)
    geom["datetime"] = pd.to_datetime(geom["datetime"])
    geom["has_rfl"] = (geom["has_rfl"].astype(str).str.strip().str.lower()
                       .isin(["true", "1"]))
    geom = geom.sort_values("datetime").reset_index(drop=True)
    n_all = len(geom)

    try:
        geom, _ = apply_qc_filter(geom, qc_path, require_review=require_qc_review)
    except QCReviewRequired as exc:
        return None, None, f"QC not reviewed ({exc})"
    geom = geom[geom["has_rfl"]].reset_index(drop=True)
    if geom.empty:
        return None, None, "no QC-surviving scenes with reflectance"

    ref = load_scene_spectrum(geom.loc[0, "scene_id"], pics_id, patch,
                              paths.RFL_DIR, paths.RAD_DIR,
                              earth_sun_normalize=earth_sun_normalize,
                              e0_method=e0_method)
    if ref is None:
        return None, None, "first surviving scene has no radiance CSV"
    bidx = int((ref["central_wl"] - target_wl).abs().idxmin())
    wl_used = float(ref.loc[bidx, "central_wl"])

    recs = []
    for _, s in geom.iterrows():
        spec = load_scene_spectrum(s["scene_id"], pics_id, patch,
                                   paths.RFL_DIR, paths.RAD_DIR,
                                   earth_sun_normalize=earth_sun_normalize,
                                   e0_method=e0_method)
        if spec is None or len(spec) != len(ref):
            continue
        recs.append({"site": pics_id, "patch": patch,
                     "timestamp": s["timestamp"], "datetime": s["datetime"],
                     "sza_mean": s["sza_mean"], "scene_id": s["scene_id"],
                     "rho_rad": spec.loc[bidx, "median_rad"],
                     "rho_rfl": spec.loc[bidx, "median_rfl"]})
    return pd.DataFrame(recs), wl_used, f"{n_all} scenes -> {len(recs)} usable"


def assemble_design(roster, cfg):
    """Stack every site into the pooled design table with a common clock (§2).

    Sets ``t`` = years since the earliest scene across *all* sites (so ``gamma``
    is one shared drift) and ``g = cos(SZA)``.

    Returns
    -------
    (pandas.DataFrame | None, dict, Timestamp | None, list)
        ``obs`` (pooled per-scene table, or None if no site was usable),
        ``wl_by_site`` (``{pics_id: wl_used}``), ``T0`` (common time origin), and
        ``notes`` (one dict per roster site: site, patch, wl_used, note, kept).
    """
    frames, wl_by_site, notes = [], {}, []
    for pics_id, patch in roster.items():
        df, wl_used, note = load_site_obs(
            pics_id, patch, cfg.brdf_target_wl,
            require_qc_review=cfg.require_qc_review,
            earth_sun_normalize=cfg.earth_sun_normalize,
            e0_method=cfg.e0_method)
        kept = df is not None and not df.empty
        notes.append({"site": pics_id, "patch": patch, "wl_used": wl_used,
                      "note": note, "kept": kept})
        if not kept:
            continue
        wl_by_site[pics_id] = wl_used
        frames.append(df)

    if not frames:
        return None, wl_by_site, None, notes

    obs = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    T0 = obs["datetime"].min()
    obs["t"] = (obs["datetime"] - T0).dt.total_seconds() / _YR_SEC
    obs["g"] = np.cos(np.radians(obs["sza_mean"]))
    return obs, wl_by_site, T0, notes


def leverage_summary(obs):
    """Per-site scene count, date/SZA/time spread, and geometry confounding (§3).

    ``corr_g_t`` = ``corr(cos SZA, t)`` per site; near ±1 means that site's own
    ``beta``/``gamma`` are geometry-confounded even if its scene count is fine.
    """
    rows = []
    for site, sub in obs.groupby("site"):
        rows.append({"site": site, "n": len(sub),
                     "first": sub["datetime"].min().date(),
                     "last": sub["datetime"].max().date(),
                     "sza_lo": round(sub["sza_mean"].min(), 1),
                     "sza_hi": round(sub["sza_mean"].max(), 1),
                     "cosSZA_span": round(sub["g"].max() - sub["g"].min(), 3),
                     "t_span_yr": round(sub["t"].max() - sub["t"].min(), 2),
                     "corr_g_t": (round(np.corrcoef(sub["g"], sub["t"])[0, 1], 2)
                                  if len(sub) > 2 else np.nan)})
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def apply_min_scenes_gate(obs, summary, min_scenes):
    """Drop sites with fewer than *min_scenes* post-QC scenes (§3).

    A site needs more scenes than its 2 local parameters to inform ``gamma`` at
    all. Returns ``(obs, summary, dropped_sites)`` with the small sites removed
    from both tables.
    """
    too_small = summary.loc[summary["n"] < min_scenes, "site"].tolist()
    if too_small:
        obs = obs[~obs["site"].isin(too_small)].reset_index(drop=True)
        summary = summary[~summary["site"].isin(too_small)].reset_index(drop=True)
    return obs, summary, too_small


def water_vapor_mask(ref_rfl):
    """Fixed water-vapor exclusion mask from one reflectance per-channel table (§8).

    EMIT L2A reflectance is fill (-> NaN) in the deep H2O windows — a fixed
    instrument property — so a single reflectance-bearing scene defines the
    excluded bands for BOTH products.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        ``wlgrid`` (band centres), ``excl`` (bool mask of excluded bands),
        ``fit_bands`` (indices of the bands to fit).
    """
    wlgrid = ref_rfl["central_wl"].to_numpy(float)
    excl = ~np.isfinite(ref_rfl["median_rfl"].to_numpy(float))
    fit_bands = np.where(~excl)[0]
    return wlgrid, excl, fit_bands


def channel_matrix(obs, product, nband, *, earth_sun_normalize=True,
                   e0_method="spencer"):
    """Scenes x bands median-rho matrix for *product*, band-order aligned (§8).

    Row ``i`` is ``obs`` scene ``i``'s per-channel median; a scene whose CSV is
    missing or has a different band count is left as NaN (and drops out per
    channel in :func:`channel_gamma`). Radiance rows are Earth–Sun (``E0``)
    normalized when *earth_sun_normalize* is set (default), via *e0_method*;
    reflectance is not.
    """
    M = np.full((len(obs), nband), np.nan)
    for i, (_, row) in enumerate(obs.iterrows()):
        s = scene_perchannel(row["site"], row["patch"], row["timestamp"], product,
                             earth_sun_normalize=earth_sun_normalize,
                             e0_method=e0_method)
        if s is not None and len(s) == nband:
            M[i] = s[f"median_{product}"].to_numpy(float)
    return M
