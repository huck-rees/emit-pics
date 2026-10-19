#!/usr/bin/env python
"""Landsat control robustness: per-site consistency + leave-one-out drift spectra.

Two checks, both mirroring the EMIT BRDF branch and reusing its (sensor-agnostic)
estimators:

1. **Per-site consistency (EMIT analog).** At the target band (nearest
   ``brdf.target_wl``), fit each PICS's *standalone* γ with a within-site bootstrap
   CI (:func:`emit_pics.brdf_drift.boot_site_gamma`) and test whether all five site
   CIs overlap (:func:`emit_pics.brdf_drift.consistency_check`) — i.e. is one shared
   drift supportable, or does a site disagree? Writes a table + forest plot.

2. **Leave-one-out (EMIT Appendix C analog).** Refit the pooled shared-γ spectrum
   five times, each dropping one PICS (:func:`emit_pics.brdf_loo.leave_one_out_drift_spectra`),
   to check no single site drives the pooled drift. Writes a tidy spectrum CSV +
   summary + overlay figure.

Pure analysis (CSVs on disk) — runs in the main ``emit-pics`` env::

    python scripts/run_landsat_brdf_robustness.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.brdf_drift import (
    resolve_roster, channel_matrix, channel_gamma,
    boot_site_gamma, consistency_check,
)
from emit_pics.brdf_loo import (
    leave_one_out_drift_spectra, combo_summary, abbrev_site,
)
from emit_pics.landsat_brdf import assemble_design_landsat, landsat_persite_drift_spectrum
from emit_pics.landsat_cube_extract import OLI2_BANDS

PRODUCT = "toa"


def main():
    cfg = load_config()
    roster, _, _ = resolve_roster(cfg)
    obs, wl_by_site, T0, notes = assemble_design_landsat(roster, cfg)
    if obs is None:
        raise SystemExit("No usable Landsat data.")

    bands = sorted(OLI2_BANDS, key=lambda x: x["center_nm"])
    wlgrid = np.array([b["center_nm"] for b in bands], float)
    fit_bands = np.arange(len(wlgrid))
    M = channel_matrix(obs, PRODUCT, len(wlgrid))          # scenes x 7
    sites = sorted(obs["site"].unique())
    out_dir = paths.LS_BRDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = paths.FIGURES / "landsat"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ================= 1. PER-SITE CONSISTENCY (target band) =================
    bidx = int(np.abs(wlgrid - cfg.brdf_target_wl).argmin())
    wl_used = float(wlgrid[bidx])
    g_all = obs["g"].to_numpy(float)
    t_all = obs["t"].to_numpy(float)
    site_all = obs["site"].to_numpy()

    rows = []
    for i, s in enumerate(sites):
        m = site_all == s
        gam, (lo, hi) = boot_site_gamma(g_all[m], t_all[m], M[m, bidx],
                                        n_boot=cfg.brdf_per_site_boot,
                                        seed=cfg.drift_seed + i)
        rows.append({"product": PRODUCT, "site": s, "n": int(m.sum()),
                     "gamma_pct_yr": 100 * gam, "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    persite = pd.DataFrame(rows)
    # Pooled shared-gamma at the same band (reference line).
    pg, (plo, phi), pp, pn = channel_gamma(M[:, bidx], g_all, t_all, site_all,
                                           cfg.brdf_per_site_boot,
                                           np.random.default_rng(cfg.drift_seed))
    verdict = consistency_check(persite)

    persite_csv = out_dir / f"landsat_persite_gamma_{wl_used:.0f}nm.csv"
    persite.to_csv(persite_csv, index=False)

    # Per-site standalone gamma SPECTRUM (all 7 bands) for the publication figure.
    persite_spec = landsat_persite_drift_spectrum(
        obs, n_boot=cfg.brdf_per_site_boot, seed=cfg.drift_seed)
    persite_spec_csv = out_dir / "landsat_persite_drift_spectrum.csv"
    persite_spec.to_csv(persite_spec_csv, index=False)

    print(f"=== Per-site standalone gamma @ {wl_used:.0f} nm (TOA) ===")
    for _, r in persite.iterrows():
        print(f"  {r['site']:14s} n={int(r['n']):3d}  gamma={r['gamma_pct_yr']:+.3f} "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] %/yr")
    print(f"  {'POOLED':14s} n={pn:3d}  gamma={100*pg:+.3f} "
          f"[{100*plo:+.3f}, {100*phi:+.3f}] %/yr")
    vp = verdict["per_product"][PRODUCT]
    print(f"  consistency verdict: {verdict['overall'].upper()}  "
          f"(overlap [{vp['overlap_lo']:+.3f}, {vp['overlap_hi']:+.3f}], "
          f"width {vp['overlap_width']:+.3f}, median CI width {vp['median_ci_width']:.3f})")

    # forest plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(persite))
    ax.errorbar(persite["gamma_pct_yr"], y,
                xerr=[persite["gamma_pct_yr"] - persite["ci_lo"],
                      persite["ci_hi"] - persite["gamma_pct_yr"]],
                fmt="o", color="#d62728", capsize=4, lw=1.5, ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (n={int(n)})" for s, n in zip(persite["site"], persite["n"])])
    ax.axvspan(100 * plo, 100 * phi, color="#1f77b4", alpha=0.15, label="pooled 95% CI")
    ax.axvline(100 * pg, color="#1f77b4", lw=1.5, label=f"pooled γ = {100*pg:+.2f}%/yr")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standalone drift γ (%/yr)")
    ax.set_title(f"Landsat 9 per-site consistency @ {wl_used:.0f} nm  "
                 f"[verdict: {verdict['overall']}]")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    forest_png = fig_dir / f"landsat_persite_gamma_{wl_used:.0f}nm.png"
    fig.savefig(forest_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 2. LEAVE-ONE-OUT DRIFT SPECTRA =================
    loo = leave_one_out_drift_spectra(
        obs, {PRODUCT: M}, wlgrid, fit_bands,
        n_boot=cfg.brdf_spectrum_n_boot, gamma_cap=cfg.brdf_gamma_cap,
        alpha=cfg.alpha, seed=cfg.drift_seed)
    loo_csv = out_dir / "landsat_loo_drift_spectrum.csv"
    loo.to_csv(loo_csv, index=False)
    summ = combo_summary(loo)

    print(f"\n=== Leave-one-out (drop 1 PICS, pool remaining 4) ===")
    for _, r in summ.iterrows():
        print(f"  drop {r['left_out']:14s} (keep {r['combo']:14s})  "
              f"median γ={r['median_gamma_pct_yr']:+.3f} %/yr  "
              f"sig {int(r['n_sig_raw'])}/{int(r['n_stable'])} bands")
    gspan = summ["median_gamma_pct_yr"]
    print(f"  spread of median γ across drops: "
          f"[{gspan.min():+.3f}, {gspan.max():+.3f}] %/yr")

    # overlay figure: 5 LOO spectra
    fig, ax = plt.subplots(figsize=(11, 6))
    for combo, gsub in loo.groupby("combo", sort=False):
        gsub = gsub.sort_values("wavelength")
        ax.plot(gsub["wavelength"], gsub["gamma_pct_yr"], "o-", ms=6, lw=1.2,
                label=f"keep {combo}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Pooled drift γ (%/yr)")
    ax.set_title("Landsat 9 leave-one-out drift spectra (each drops one PICS)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    loo_png = fig_dir / "landsat_loo_drift_spectrum.png"
    fig.savefig(loo_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {persite_csv}\n      {forest_png}\n      {loo_csv}\n      {loo_png}")


if __name__ == "__main__":
    main()
