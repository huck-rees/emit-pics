#!/usr/bin/env python
"""EMIT-vs-Landsat drift comparison, synthesized into the 7 OLI-2 reflective bands.

For each OLI-2 band, synthesize an EMIT-equivalent per-scene value by weighting each
EMIT channel by the band's relative spectral response (RSR) and integrating, then fit
the pooled shared-gamma quasi-BRDF drift over the four invariant PICS (the same
estimator as the per-channel drift spectrum, with a stratified within-site bootstrap
CI). Done for EMIT TOA radiance and surface reflectance. The measured Landsat band
drift (over the same four PICS) is computed alongside for direct comparison.

**Channel selection.** Only EMIT channels that pass the drift-spectrum summary-stat
screen are used in the synthesis: outside the deep water-vapor features AND
well-constrained (per-channel 95% bootstrap CI width < 3 %/yr). Applied per product,
so radiance and reflectance use their own well-constrained sets. A band with no
surviving channel for a product yields NaN (e.g. coastal-aerosol reflectance, which
falls entirely on EMIT's noisy blue edge).

Pure local post-processor (runs in the ``emit-pics`` env off the cached CSVs).
Requires ``data/brdf/brdf_design.csv``, ``data/brdf/brdf_drift_spectrum.csv``, the
Landsat TOA tables, and ``data/landsat/oli2_rsr/L9_OLI2_RSR_tidy.csv``.

Output
------
* ``data/emit_oli2_compare/band_drift_comparison.csv`` -- one row per (band, series)
  with gamma_pct_yr, ci_lo/hi_pct_yr, n_scenes, n_inband, n_used (+ provenance sidecar).

Run::

    python scripts/run_emit_oli2_band_comparison.py
"""

import os
import numpy as np
import pandas as pd

from emit_pics import brdf_drift, landsat_brdf, paths, provenance
from emit_pics.run_config import load_config
from emit_pics.landsat_cube_extract import OLI2_BANDS

CI_WIDTH_MAX = 3.0                      # %/yr; "well-constrained" screen (matches reporting)


def main() -> None:
    cfg = load_config()
    n_boot = cfg.brdf_n_bootstrap
    roster, _, _ = brdf_drift.resolve_roster(cfg)

    # --- EMIT pooled design over the four invariant PICS, full per-channel matrices ---
    obs_all, _, _, _ = brdf_drift.assemble_design(roster, cfg)
    obs, dropped = brdf_drift.apply_drift_exclusion(obs_all, cfg)
    print(f"EMIT: {len(obs)} scenes over {obs['site'].nunique()} PICS (dropped {dropped})")
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    wlgrid, _excl, _fit = brdf_drift.water_vapor_mask(ref)
    nband = len(wlgrid)
    wlr = np.round(wlgrid, 2)
    M = {p: brdf_drift.channel_matrix(obs, p, nband,
                                      earth_sun_normalize=cfg.earth_sun_normalize,
                                      e0_method=cfg.e0_method) for p in ("rad", "rfl")}
    g = obs["g"].to_numpy(float); t = obs["t"].to_numpy(float); site = obs["site"].to_numpy()

    # --- well-constrained channel wavelengths (CI width < 3 %/yr), per product ---
    drift = pd.read_csv(paths.brdf_drift_spectrum())
    drift["w"] = drift["ci_hi_pct_yr"] - drift["ci_lo_pct_yr"]
    wc = {p: set(np.round(drift[(drift["product"] == p) & (drift["w"] < CI_WIDTH_MAX)]["wavelength"], 2))
          for p in ("rad", "rfl")}
    print(f"well-constrained channels: rad {len(wc['rad'])}, rfl {len(wc['rfl'])}")

    # --- RSR (7 reflective bands) ---
    rsr_all = pd.read_csv(paths.DATA / "landsat" / "oli2_rsr" / "L9_OLI2_RSR_tidy.csv")
    bands = (rsr_all[["band", "band_n", "center_nm"]].drop_duplicates()
             .sort_values("band_n").reset_index(drop=True))

    rng = np.random.default_rng(cfg.drift_seed)
    rows = []
    for _, bnd in bands.iterrows():
        rr = rsr_all[rsr_all["band"] == bnd.band].sort_values("wavelength_nm")
        w_full = np.interp(wlgrid, rr["wavelength_nm"], rr["rsr"], left=0.0, right=0.0)
        inband = w_full > 1e-4
        for p in ("rad", "rfl"):
            keep = inband & np.array([w in wc[p] for w in wlr])
            base = {"band": bnd.band, "band_n": int(bnd.band_n), "center_nm": float(bnd.center_nm),
                    "series": f"emit_{p}", "n_scenes": len(obs),
                    "n_inband": int(inband.sum()), "n_used": int(keep.sum())}
            if keep.sum() == 0:
                rows.append({**base, "gamma_pct_yr": np.nan,
                             "ci_lo_pct_yr": np.nan, "ci_hi_pct_yr": np.nan})
                continue
            weights = np.where(keep, w_full, 0.0)
            num = np.nansum(M[p] * weights, axis=1)
            den = np.nansum((~np.isnan(M[p])) * weights, axis=1)
            synth = num / den
            gam, (lo, hi), _p, _n = brdf_drift.channel_gamma(synth, g, t, site, n_boot, rng)
            rows.append({**base, "gamma_pct_yr": 100 * gam,
                         "ci_lo_pct_yr": 100 * lo, "ci_hi_pct_yr": 100 * hi})

    # --- measured Landsat per-band drift over the same four PICS ---
    ls_all, _, _, _ = landsat_brdf.assemble_design_landsat(roster, cfg)
    ls, _ = brdf_drift.apply_drift_exclusion(ls_all, cfg)
    ls_bands = sorted(OLI2_BANDS, key=lambda x: x["center_nm"])
    Mls = brdf_drift.channel_matrix(ls, "toa", len(ls_bands))
    gl = ls["g"].to_numpy(float); tl = ls["t"].to_numpy(float); sl = ls["site"].to_numpy()
    for bi, b in enumerate(ls_bands):
        gam, (lo, hi), _p, _n = brdf_drift.channel_gamma(Mls[:, bi], gl, tl, sl, n_boot, rng)
        rows.append({"band": b["asset"], "band_n": int(b["band_n"]),
                     "center_nm": float(b["center_nm"]), "series": "landsat",
                     "gamma_pct_yr": 100 * gam, "ci_lo_pct_yr": 100 * lo,
                     "ci_hi_pct_yr": 100 * hi, "n_scenes": len(ls),
                     "n_inband": 1, "n_used": 1})

    res = pd.DataFrame(rows).sort_values(["band_n", "series"]).reset_index(drop=True)
    outdir = paths.DATA / "emit_oli2_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "band_drift_comparison.csv"
    res.to_csv(out, index=False)
    provenance.write_sidecar(out, cfg.as_dict())

    pd.set_option("display.width", 200)
    print("\n" + res.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
