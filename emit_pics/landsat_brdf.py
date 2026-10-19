"""Landsat 9 control — pooled quasi-BRDF design assembly (reuses the EMIT fit).

The Landsat analog of the *assembly* half of :mod:`emit_pics.brdf_drift`. It builds
the pooled per-scene design table (``rho_toa``, ``g = cos SZA``, common-clock ``t``)
from the Landsat geometry + TOA per-band tables, applying the SAME manual QC gate
(:mod:`emit_pics.qc_gate`) the EMIT side uses. The *model/fit* half of
``brdf_drift`` — :func:`~emit_pics.brdf_drift.fit_joint`,
:func:`~emit_pics.brdf_drift.channel_gamma`,
:func:`~emit_pics.brdf_drift.bootstrap_joint_gamma`,
:func:`~emit_pics.brdf_drift.consistency_check` — is sensor-agnostic and reused
unchanged; only this IO differs.

Product token is ``"toa"`` (there is no rad/rfl split for Landsat, and no
Earth-Sun normalization — TOA reflectance already folds it in). Everything here is
pure pandas/numpy, so the Landsat drift analysis runs in the main ``emit-pics``
env off the CSVs on disk — the geospatial stack is only needed for extraction/QC.
"""

import numpy as np
import pandas as pd

from emit_pics import paths
from emit_pics.qc_gate import QCReviewRequired, apply_qc_filter
from emit_pics.brdf_drift import channel_matrix, channel_gamma, _stack_design

#: Landsat product token (routes to paths.LS_TOA_DIR via paths._PRODUCT_DIR).
PRODUCT = "toa"


def _load_toa_scene(pics_id, patch, ts):
    """One scene's 7-band TOA per-channel table (band-center sorted), or None."""
    path = paths.perchannel(pics_id, patch, str(ts), PRODUCT)
    if not path.exists():
        return None
    return pd.read_csv(path).sort_values("central_wl").reset_index(drop=True)


def load_site_obs_landsat(pics_id, patch, target_wl, *, require_qc_review=True):
    """Per-scene obs for one Landsat site: QC-surviving, median TOA at the band.

    Mirrors :func:`emit_pics.brdf_drift.load_site_obs`: read the geometry table,
    **apply the QC gate** (drop ``keep=FALSE`` scenes; refuse if the worksheet is
    unreviewed and *require_qc_review*), then take each scene's median TOA at the
    channel nearest *target_wl* (band-order aligned to the first survivor).

    Returns ``(DataFrame | None, wl_used | None, note)``. The frame carries
    ``site, patch, timestamp, datetime, sza_mean, scene_id, rho_toa``.
    """
    geom_path = paths.ls_geometry(pics_id, patch)
    qc_path = paths.ls_qc_sheet(pics_id, patch)
    if not geom_path.exists():
        return None, None, f"missing geometry {geom_path.name}"

    geom = pd.read_csv(geom_path)
    geom["datetime"] = pd.to_datetime(geom["datetime"])
    geom = geom.sort_values("datetime").reset_index(drop=True)
    n_all = len(geom)

    try:
        geom, _ = apply_qc_filter(geom, qc_path, require_review=require_qc_review,
                                  timestamp_col="timestamp")
    except QCReviewRequired as exc:
        return None, None, f"QC not reviewed ({exc})"
    if geom.empty:
        return None, None, "no QC-surviving scenes"

    ref = _load_toa_scene(pics_id, patch, geom.loc[0, "timestamp"])
    if ref is None:
        return None, None, "first surviving scene has no TOA CSV"
    bidx = int((ref["central_wl"] - target_wl).abs().idxmin())
    wl_used = float(ref.loc[bidx, "central_wl"])

    recs = []
    for _, s in geom.iterrows():
        spec = _load_toa_scene(pics_id, patch, s["timestamp"])
        if spec is None or len(spec) != len(ref):
            continue
        recs.append({"site": pics_id, "patch": patch,
                     "timestamp": s["timestamp"], "datetime": s["datetime"],
                     "sza_mean": s["sza_mean"], "scene_id": s["scene_id"],
                     "rho_toa": spec.loc[bidx, "median_toa"]})
    return pd.DataFrame(recs), wl_used, f"{n_all} scenes -> {len(recs)} usable"


def assemble_design_landsat(roster, cfg):
    """Stack every Landsat site into the pooled design table with a common clock.

    The Landsat analog of :func:`emit_pics.brdf_drift.assemble_design`. Sets ``t``
    = years since the earliest QC-surviving scene across *all* sites (so ``gamma``
    is one shared drift) and ``g = cos(SZA)``. Uses ``cfg.brdf_target_wl`` and
    ``cfg.require_qc_review`` — the same knobs the EMIT BRDF branch reads.

    Returns
    -------
    (DataFrame | None, dict, Timestamp | None, list)
        ``obs`` (pooled per-scene table with ``rho_toa``, ``t``, ``g``; or None if
        no site was usable), ``wl_by_site``, ``T0`` (common time origin), and
        ``notes`` (one dict per roster site: site, patch, wl_used, note, kept).
    """
    def load_one(pics_id, patch):
        return load_site_obs_landsat(pics_id, patch, cfg.brdf_target_wl,
                                     require_qc_review=cfg.require_qc_review)
    return _stack_design(roster, load_one)


def landsat_drift_spectrum(obs, *, n_boot=300, seed=0):
    """Pooled shared-γ at every OLI-2 band — the Landsat analog of the EMIT drift spectrum.

    Builds the scenes×7 median-TOA matrix (:func:`emit_pics.brdf_drift.channel_matrix`)
    and, per band, fits the pooled shared γ with a stratified within-site bootstrap
    CI and two-sided p (:func:`emit_pics.brdf_drift.channel_gamma`) — the identical
    estimator the EMIT spectrum uses, so the two are directly comparable. No
    water-vapor masking (the 7 reflective bands avoid the deep absorptions).

    Returns a tidy frame matching the EMIT ``brdf_drift_spectrum`` schema:
    ``product, band_index, wavelength, gamma_pct_yr, ci_lo_pct_yr, ci_hi_pct_yr,
    p_value, n, sig_raw`` (``sig_raw`` = 95% CI excludes zero).
    """
    from emit_pics.landsat_cube_extract import OLI2_BANDS  # lazy (pulls rasterio)
    bands = sorted(OLI2_BANDS, key=lambda x: x["center_nm"])
    wlgrid = np.array([b["center_nm"] for b in bands], float)
    nband = len(wlgrid)
    M = channel_matrix(obs, PRODUCT, nband)                 # scenes x 7 median TOA
    g = obs["g"].to_numpy(float)
    t = obs["t"].to_numpy(float)
    site = obs["site"].to_numpy()

    rng = np.random.default_rng(seed)
    rows = []
    for b in range(nband):
        gam, (lo, hi), p, n = channel_gamma(M[:, b], g, t, site, n_boot, rng)
        rows.append({"product": PRODUCT, "band_index": b,
                     "wavelength": float(wlgrid[b]),
                     "gamma_pct_yr": 100.0 * gam,
                     "ci_lo_pct_yr": 100.0 * lo, "ci_hi_pct_yr": 100.0 * hi,
                     "p_value": p, "n": n,
                     "sig_raw": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (lo > 0 or hi < 0))})
    return pd.DataFrame(rows)


def landsat_persite_drift_spectrum(obs, *, n_boot=300, seed=0):
    """Per-PICS standalone gamma at every OLI-2 band (each site fit on its own).

    The per-site analog of :func:`landsat_drift_spectrum`: instead of one pooled
    shared gamma, fit each PICS's OWN gamma at every band with a within-site
    bootstrap CI (:func:`emit_pics.brdf_drift.boot_site_gamma`). This is the direct
    invariance test across the spectrum — a site that departs from flat gamma (e.g.
    Mauritania 1) is not temporally invariant. No water-vapor masking (the 7
    reflective bands avoid the deep absorptions).

    Returns a tidy frame: ``site, band_index, wavelength, gamma_pct_yr,
    ci_lo_pct_yr, ci_hi_pct_yr, n, sig_raw`` (``sig_raw`` = 95% CI excludes zero).
    """
    from emit_pics.landsat_cube_extract import OLI2_BANDS  # lazy (pulls rasterio)
    from emit_pics.brdf_drift import boot_site_gamma
    bands = sorted(OLI2_BANDS, key=lambda x: x["center_nm"])
    wlgrid = np.array([b["center_nm"] for b in bands], float)
    nband = len(wlgrid)
    M = channel_matrix(obs, PRODUCT, nband)                 # scenes x 7 median TOA
    g = obs["g"].to_numpy(float)
    t = obs["t"].to_numpy(float)
    site = obs["site"].to_numpy()
    sites = sorted(pd.unique(site))

    rows = []
    for i, s in enumerate(sites):
        m = site == s
        for b in range(nband):
            gam, (lo, hi) = boot_site_gamma(g[m], t[m], M[m, b], n_boot,
                                            seed=seed + 100 * i + b)
            rows.append({"site": s, "band_index": b,
                         "wavelength": float(wlgrid[b]),
                         "gamma_pct_yr": 100.0 * gam,
                         "ci_lo_pct_yr": 100.0 * lo, "ci_hi_pct_yr": 100.0 * hi,
                         "n": int(m.sum()),
                         "sig_raw": bool(np.isfinite(lo) and np.isfinite(hi)
                                         and (lo > 0 or hi < 0))})
    return pd.DataFrame(rows)


def landsat_site_mean_spectra(roster, cfg):
    """Per-PICS mean TOA reflectance spectrum (7 OLI bands) over QC-surviving scenes.

    The Landsat analog of :func:`emit_pics.brdf_drift.site_mean_spectra`. For each
    roster site it reads the geometry table, applies the QC gate, and averages each
    QC-surviving scene's 7-band median TOA reflectance band by band. The band
    wavelengths come from the per-scene tables themselves, so this needs no
    geospatial dependency and runs in the main env.

    Returns ``({pics_id: {"wl", "mean_toa", "n", "patch"}}, notes)`` where notes is
    one dict per roster site (site, patch, n, note, kept).
    """
    spectra, notes = {}, []
    for pics_id, patch in roster.items():
        geom_path = paths.ls_geometry(pics_id, patch)
        qc_path = paths.ls_qc_sheet(pics_id, patch)
        if not geom_path.exists():
            notes.append({"site": pics_id, "patch": patch, "n": 0,
                          "note": f"missing geometry {geom_path.name}", "kept": False})
            continue
        geom = pd.read_csv(geom_path)
        geom["datetime"] = pd.to_datetime(geom["datetime"])
        geom = geom.sort_values("datetime").reset_index(drop=True)
        try:
            geom, _ = apply_qc_filter(geom, qc_path,
                                      require_review=cfg.require_qc_review,
                                      timestamp_col="timestamp")
        except QCReviewRequired as exc:
            notes.append({"site": pics_id, "patch": patch, "n": 0,
                          "note": f"QC not reviewed ({exc})", "kept": False})
            continue

        rows, wl = [], None
        for ts in geom["timestamp"]:
            spec = _load_toa_scene(pics_id, patch, ts)
            if spec is None:
                continue
            if wl is None:
                wl = spec["central_wl"].to_numpy(float)
            if len(spec) == len(wl):
                rows.append(spec["median_toa"].to_numpy(float))
        if not rows:
            notes.append({"site": pics_id, "patch": patch, "n": 0,
                          "note": "no usable TOA scenes", "kept": False})
            continue
        spectra[pics_id] = {"wl": wl,
                            "mean_toa": np.nanmean(np.vstack(rows), axis=0),
                            "n": len(rows), "patch": str(patch)}
        notes.append({"site": pics_id, "patch": patch, "n": len(rows),
                      "note": f"{len(rows)} scenes", "kept": True})
    return spectra, notes
