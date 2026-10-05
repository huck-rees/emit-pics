#!/usr/bin/env python
"""Pooled Landsat 9 TOA drift spectrum (7 OLI bands) + overlay vs. the EMIT spectrum.

The payoff step of the Landsat control: assemble the QC-gated pooled design table,
fit the shared drift γ at every OLI-2 band (same estimator as the EMIT BRDF
spectrum), write the tidy CSV, and plot the 7 Landsat γ points (with bootstrap CI)
over EMIT's radiance + reflectance drift spectra. If Landsat reproduces EMIT's
spectral *shape*, the trend is (partly) surface/atmosphere; if Landsat is flat
where EMIT dips, EMIT's drift is instrumental.

Pure analysis (CSVs on disk) — runs in the main ``emit-pics`` env::

    python scripts/run_landsat_brdf_drift_spectrum.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.brdf_drift import resolve_roster
from emit_pics.landsat_brdf import assemble_design_landsat, landsat_drift_spectrum


def main():
    cfg = load_config()
    roster, _, _ = resolve_roster(cfg)
    obs, wl_by_site, T0, notes = assemble_design_landsat(roster, cfg)
    if obs is None:
        raise SystemExit("No usable Landsat data (check QC review / extraction).")
    print(f"Pooled {len(obs)} QC-surviving scenes across {obs['site'].nunique()} sites.")

    spec = landsat_drift_spectrum(obs, n_boot=cfg.brdf_spectrum_n_boot,
                                  seed=cfg.drift_seed)
    out_csv = paths.ls_brdf_drift_spectrum()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    spec.to_csv(out_csv, index=False)

    # --- print Landsat vs EMIT (nearest wavelength) side by side ---
    emit_path = paths.brdf_drift_spectrum()
    emit = pd.read_csv(emit_path) if emit_path.exists() else None
    print("\n  wl(nm)  Landsat γ (95% CI)         sig   EMIT rfl γ   EMIT rad γ")
    for _, r in spec.iterrows():
        e_rfl = e_rad = np.nan
        if emit is not None:
            for prod, tgt in (("rfl", "e_rfl"), ("rad", "e_rad")):
                ep = emit[emit["product"] == prod]
                if len(ep):
                    j = (ep["wavelength"] - r.wavelength).abs().idxmin()
                    val = ep.loc[j, "gamma_pct_yr"]
                    if prod == "rfl":
                        e_rfl = val
                    else:
                        e_rad = val
        print(f"  {r.wavelength:6.0f}  {r.gamma_pct_yr:+6.3f} "
              f"[{r.ci_lo_pct_yr:+6.3f},{r.ci_hi_pct_yr:+6.3f}]  "
              f"{'YES' if r.sig_raw else ' - ':>3s}   {e_rfl:+7.3f}    {e_rad:+7.3f}")

    # --- figure: Landsat points over EMIT spectra ---
    fig, ax = plt.subplots(figsize=(11, 6))
    if emit is not None:
        e = emit[~emit["unstable"]] if "unstable" in emit.columns else emit
        for prod, color, lbl in (("rfl", "#1f77b4", "EMIT reflectance γ"),
                                 ("rad", "#2ca02c", "EMIT radiance γ")):
            ep = e[e["product"] == prod].sort_values("wavelength")
            if len(ep):
                ax.plot(ep["wavelength"], ep["gamma_pct_yr"], color=color, lw=1,
                        alpha=0.8, label=lbl)
                ax.fill_between(ep["wavelength"], ep["ci_lo_pct_yr"],
                                ep["ci_hi_pct_yr"], color=color, alpha=0.12)

    yerr = [spec["gamma_pct_yr"] - spec["ci_lo_pct_yr"],
            spec["ci_hi_pct_yr"] - spec["gamma_pct_yr"]]
    ax.errorbar(spec["wavelength"], spec["gamma_pct_yr"], yerr=yerr, fmt="o",
                color="#d62728", ms=8, capsize=4, lw=1.5, zorder=5,
                label="Landsat 9 TOA γ (7 OLI bands)")
    for _, r in spec.iterrows():
        ax.annotate(f"{r.wavelength:.0f}", (r.wavelength, r.gamma_pct_yr),
                    fontsize=7, xytext=(0, 9), textcoords="offset points", ha="center")

    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Drift γ (%/yr)")
    ax.set_title("Landsat 9 TOA drift spectrum vs. EMIT BRDF drift spectrum")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    out_png = paths.FIGURES / "landsat" / "landsat_brdf_drift_spectrum.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
