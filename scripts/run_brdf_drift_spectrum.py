#!/usr/bin/env python
"""Pooled quasi-BRDF drift spectrum gamma(lambda) across every EMIT channel.

Third and final stage of the BRDF workflow (port of
``notebooks/BRDF_drift_workflow_allPICS.ipynb`` section 8). Repeats the pooled
joint fit — per-site ``alpha``/``beta`` with one shared ``gamma`` — at **every**
EMIT channel (water-vapor absorption bands excluded, for both products), and
reports the shared-``gamma`` best estimate with its bootstrap 95% CI per
wavelength. This is the multi-site analog of the matched-pairs drift spectrum.

Significance is the uncorrected per-channel p-value: each channel's p is a
two-sided **bootstrap** p (twice the smaller tail fraction of its
gamma draws about 0). Channels whose multiplicative baseline
``(alpha + beta cos theta)`` approaches zero (deep-SWIR / low-signal radiance)
make ``gamma`` explode — a fit degeneracy, not drift; they are flagged
``unstable`` (``|gamma|`` or CI half-width > ``brdf.gamma_cap``) and kept off the
plot scale, never read as significant.

Pure **local** post-processor. Requires ``data/brdf/brdf_design.csv`` (run
``scripts/run_brdf_assemble.py`` first) and the per-channel RAD/RFL stats CSVs for
every design scene.

Runtime note: this fits ``2 x (#channels) x (spectrum_n_boot + 1)`` small models.
Lower ``brdf.spectrum_n_boot`` for a quick look, raise it for publication.

Outputs
-------
* ``data/brdf/brdf_drift_spectrum.csv`` — per-channel shared gamma, CI, p, flags.
* ``figures/brdf/brdf_drift_spectrum.png`` — drift spectrum with raw per-channel
  significance colouring.

Set the roster + knobs in run_config.yaml (``brdf:`` block), then run::

    python scripts/run_brdf_drift_spectrum.py
"""

import sys

import numpy as np

from emit_pics import brdf_drift, paths, provenance
from emit_pics.run_config import load_config

# ============================================================
# INPUTS — set the roster + knobs in run_config.yaml (brdf:), not here.
# ============================================================
cfg = load_config()
paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
paths.set_brdf_label(cfg.brdf_label)    # extra BRDF-only sub-namespace (e.g. patch-size sweep)
TARGET_WL       = cfg.brdf_target_wl
SPECTRUM_N_BOOT = cfg.brdf_spectrum_n_boot   # per-channel resamples (heavier than §7)
ALPHA           = cfg.alpha                  # per-channel (raw) significance level
GAMMA_CAP       = cfg.brdf_gamma_cap         # %/yr; degenerate-fit flag

#: Script-local: manual y-limits (%/yr) per panel, or None to autoscale.
RAD_YLIM = None    # e.g. (-60, 60)
RFL_YLIM = None    # e.g. (-20, 15)

PRODUCTS = ("rad", "rfl")

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def _runs(mask):
    """Contiguous True-runs of *mask* as (start, end) index pairs (for shading)."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    return runs


def _plot_drift_spectrum(spectrum, sig_col, sig_label, title_tag, *,
                         wlgrid, excl, n_sites):
    """Two-panel drift spectrum (rad top, rfl bottom); red = *sig_col* True."""
    wv_runs = _runs(excl)
    nfit = int((spectrum["product"] == "rad").sum())
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, prod, ylab in zip(
            axes, ["rad", "rfl"],
            ["Radiance (L1B)\ndrift (%/yr)", "Reflectance (L2A)\ndrift (%/yr)"]):
        for k, (a, b) in enumerate(wv_runs):
            ax.axvspan(wlgrid[a], wlgrid[b], color="0.85", zorder=0,
                       label="water-vapor band (excluded)" if k == 0 else None)
        d = spectrum[(spectrum["product"] == prod)
                     & (~spectrum["unstable"])].sort_values("wavelength")
        yerr = np.vstack([d["gamma_pct_yr"] - d["ci_lo_pct_yr"],
                          d["ci_hi_pct_yr"] - d["gamma_pct_yr"]])
        ax.errorbar(d["wavelength"], d["gamma_pct_yr"], yerr=yerr, fmt="none",
                    ecolor="0.6", elinewidth=0.8, capsize=1.5, zorder=1, label="95% CI")
        ns, sg = d[~d[sig_col]], d[d[sig_col]]
        ax.scatter(ns["wavelength"], ns["gamma_pct_yr"], s=16, facecolors="none",
                   edgecolors="0.35", linewidths=0.8, zorder=3,
                   label="drift rate - not significant")
        ax.scatter(sg["wavelength"], sg["gamma_pct_yr"], s=18, color="red", zorder=4,
                   label=f"drift rate - {sig_label}")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_ylabel(ylab)
        ylim = RAD_YLIM if prod == "rad" else RFL_YLIM
        if ylim is not None:
            ax.set_ylim(*ylim)
        nhidden = int(((spectrum["product"] == prod) & spectrum["unstable"]).sum())
        if nhidden:
            ax.text(0.99, 0.03, f"{nhidden} unstable channel(s) hidden "
                    "(near-zero baseline)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color="0.4")
        ax.legend(fontsize=8, loc="best")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].set_title(
        f"Radiometric drift spectrum - all PICS (pooled, {n_sites} sites) - {title_tag}   "
        f"(marker = pooled shared-gamma, whisker = bootstrap 95% CI; {nfit} channels fit)",
        fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    # 0. Load the pooled design table.
    design_path = paths.brdf_design()
    if not design_path.exists():
        print(f"Design table not found: {design_path}\n"
              "Run scripts/run_brdf_assemble.py first.")
        sys.exit(1)
    obs = pd.read_csv(design_path, parse_dates=["datetime"])
    sites = sorted(obs["site"].unique())
    print(f"Design: {len(obs)} scene(s) across {len(sites)} site(s).")

    # 1. Reference grid + water-vapor exclusion, from one reflectance-bearing scene.
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    if ref is None:
        print(f"First design scene has no reflectance per-channel CSV; cannot set "
              f"the band grid. Re-run the extract for the roster patches.")
        sys.exit(1)
    wlgrid, excl, fit_bands = brdf_drift.water_vapor_mask(ref)
    nband = len(wlgrid)
    print(f"{nband} channels | {int(excl.sum())} water-vapor band(s) excluded | "
          f"{len(fit_bands)} fit per product")

    # 2. Scenes x bands matrices, and the pooled design arrays.
    mat = {p: brdf_drift.channel_matrix(obs, p, nband,
                                        earth_sun_normalize=cfg.earth_sun_normalize,
                                        e0_method=cfg.e0_method)
           for p in PRODUCTS}
    g_all = obs["g"].to_numpy(float)
    t_all = obs["t"].to_numpy(float)
    site_all = obs["site"].to_numpy()

    # 3. Pooled shared-gamma per fit channel, per product (single shared RNG stream).
    rows, rng = [], np.random.default_rng(2024)
    for prod in PRODUCTS:
        for j, b in enumerate(fit_bands):
            gam, (lo, hi), pval, n = brdf_drift.channel_gamma(
                mat[prod][:, b], g_all, t_all, site_all, SPECTRUM_N_BOOT, rng)
            rows.append({"product": prod, "band_index": int(b),
                         "wavelength": float(wlgrid[b]), "gamma_pct_yr": 100 * gam,
                         "ci_lo_pct_yr": 100 * lo, "ci_hi_pct_yr": 100 * hi,
                         "p_value": pval, "n": n})
            if j % 40 == 0:
                print(f"  {prod}: channel {j + 1}/{len(fit_bands)} ({wlgrid[b]:.0f} nm)")
    spectrum = pd.DataFrame(rows)

    # 4. Significance: uncorrected per-channel bootstrap p-value.
    spectrum["sig_raw"] = spectrum["p_value"] <= ALPHA
    for prod in PRODUCTS:
        m = (spectrum["product"] == prod).to_numpy()
        print(f"{prod}: significant p<{ALPHA}: "
              f"{int(spectrum.loc[m, 'sig_raw'].sum())}   (of {int(m.sum())} channels)")

    # 5. Flag numerically unstable channels (near-zero baseline -> gamma explodes).
    _ci_hw = (spectrum["ci_hi_pct_yr"] - spectrum["ci_lo_pct_yr"]).abs() / 2
    spectrum["unstable"] = (~np.isfinite(spectrum["gamma_pct_yr"])
                            | (spectrum["gamma_pct_yr"].abs() > GAMMA_CAP)
                            | (_ci_hw > GAMMA_CAP))
    for prod in PRODUCTS:
        nu = int(spectrum.loc[spectrum["product"] == prod, "unstable"].sum())
        if nu:
            print(f"{prod}: {nu} unstable channel(s) (near-zero baseline, |gamma| or "
                  f"CI half-width > {GAMMA_CAP:.0f} %/yr) hidden from the plot scale")

    # 6. Write the spectrum table (with provenance sidecar).
    paths.BRDF_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = paths.brdf_drift_spectrum()
    spectrum.to_csv(out_csv, index=False)
    provenance.write_sidecar(out_csv, cfg.as_dict())
    print(f"\nDrift spectrum CSV -> {out_csv}")

    # 7. Figure: per-channel drift spectrum with raw-significance colouring.
    paths.BRDF_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = _plot_drift_spectrum(spectrum, "sig_raw",
                               f"significant (p < {ALPHA})",
                               "pooled radiometric drift spectrum",
                               wlgrid=wlgrid, excl=excl, n_sites=len(sites))
    fig.savefig(paths.brdf_drift_spectrum_fig(), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Drift spectrum figure -> {paths.BRDF_FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
