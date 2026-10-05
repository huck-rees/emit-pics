#!/usr/bin/env python
"""Single-channel quasi-BRDF pooled-drift fit at ``brdf.target_wl``.

Second stage of the BRDF workflow (port of
``notebooks/BRDF_drift_workflow_allPICS.ipynb`` sections 4b, 5, 6, 7). Reads the
pooled design table from ``run_brdf_assemble.py`` and, at the single channel
nearest ``brdf.target_wl``:

* **§4b — shared-drift consistency check.** Fit each site's standalone gamma with
  a bootstrap CI, then test whether all sites' CIs overlap (one common drift).
  The verdict is computed and **warned loudly if not ``pass``**, but the run
  proceeds (a ``fail`` means the pooled gamma blends differing drifts — a red
  flag; ``marginal`` means caution). The verdict is recorded in the per-site
  table's provenance sidecar.
* **§5 — joint pooled fit.** One nonlinear fit of the ``2S+1`` vector: per-site
  ``alpha``/``beta`` with a single shared ``gamma``.
* **§7 — bootstrap the shared gamma** by stratified within-site resampling.
* **§6 — quasi-BRDF correction** figure: divide each scene by its own
  ``(alpha_s + beta_s cos theta)`` so all sites collapse onto ``1 + gamma t``.

Pure **local** post-processor. Requires ``data/brdf/brdf_design.csv`` (run
``scripts/run_brdf_assemble.py`` first).

Outputs (``{wl}`` = target wavelength, rounded)
-----------------------------------------------
* ``data/brdf/brdf_persite_gamma_{wl}nm.csv`` — §4b per-site standalone gammas + CIs.
* ``data/brdf/brdf_joint_fit_{wl}nm.csv`` — §5 per-site alpha/beta + shared gamma.
* ``data/brdf/brdf_bootstrap_gamma_{wl}nm.csv`` — §7 pooled gamma, SE, 95% CI.
* ``figures/brdf/brdf_persite_gamma_{wl}nm.png`` — §4b forest plot.
* ``figures/brdf/brdf_persite_gamma_spectrum.{pdf,svg}`` — §4b per-site standalone
  gamma(lambda) across all channels (one point series per PICS, best fit, no
  bootstrap), rendered to publication (manuscript) specs.
* ``figures/brdf/brdf_corrected_{wl}nm.png`` — §6 corrected collapse.
* ``figures/brdf/brdf_bootstrap_{wl}nm.png`` — §7 bootstrap histogram.

Set the roster + knobs in run_config.yaml (``brdf:`` block), then run::

    python scripts/run_brdf_single_channel.py
"""

import sys

import numpy as np

from emit_pics import brdf_drift, paths, provenance
from emit_pics.run_config import load_config
from emit_pics.viz import figures as vizfigures, style

# ============================================================
# INPUTS — set the roster + knobs in run_config.yaml (brdf:), not here.
# ============================================================
cfg = load_config()
paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
paths.set_brdf_label(cfg.brdf_label)    # extra BRDF-only sub-namespace (e.g. patch-size sweep)
TARGET_WL     = cfg.brdf_target_wl
N_BOOTSTRAP   = cfg.brdf_n_bootstrap     # pooled shared-gamma resamples (§7)
PER_SITE_BOOT = cfg.brdf_per_site_boot   # per-site standalone-gamma resamples (§4b)
GAMMA_CAP     = cfg.brdf_gamma_cap       # %/yr; degenerate per-site fits hidden from the spectrum

#: Product -> plot colour (matches the notebook).
PRODUCTS = {"rad": "#1f77b4", "rfl": "#2ca02c"}

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm


def _f(x):
    """JSON-friendly float (NaN/None -> None)."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _clean_verdict(v):
    """Cast the consistency verdict to plain JSON types for the sidecar."""
    pp = {p: {"verdict": d["verdict"], "n_sites": int(d["n_sites"]),
              "overlap_lo": _f(d["overlap_lo"]), "overlap_hi": _f(d["overlap_hi"]),
              "overlap_width": _f(d["overlap_width"]),
              "median_ci_width": _f(d["median_ci_width"])}
          for p, d in v["per_product"].items()}
    return {"overall": v["overall"], "per_product": pp}


def _report_consistency(v):
    """Print the §4b verdict; warn loudly (banner) if it is not a clean pass."""
    print("\nShared-drift consistency check (step 4b):")
    for prod, d in v["per_product"].items():
        if d["verdict"] == "n/a":
            print(f"  {prod.upper()}: n/a (no site with a finite CI)")
        else:
            print(f"  {prod.upper()}: {d['verdict'].upper()} - common overlap "
                  f"[{d['overlap_lo']:+.3f}, {d['overlap_hi']:+.3f}] %/yr "
                  f"(width {d['overlap_width']:+.3f}, median CI {d['median_ci_width']:.3f})")
    if v["overall"] != "pass":
        bar = "!" * 70
        print("\n" + bar)
        print(f"!! SHARED-GAMMA CHECK: {v['overall'].upper()} — the per-site drift "
              "CIs do not cleanly agree.")
        print("!! The pooled gamma below is trustworthy ONLY if the sites share one "
              "common drift.")
        print("!! Read 'fail' as a red flag (pooled value blends differing drifts); "
              "'marginal' as caution.")
        print(bar)


def _plot_persite(persite, sites):
    """§4b forest plot: per-site standalone gamma with bootstrap CI, per product."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prod in zip(axes, PRODUCTS):
        d = persite[persite["product"] == prod].reset_index(drop=True)
        ypos = np.arange(len(d))
        ax.errorbar(d["gamma_pct_yr"], ypos,
                    xerr=[d["gamma_pct_yr"] - d["ci_lo"], d["ci_hi"] - d["gamma_pct_yr"]],
                    fmt="o", color=PRODUCTS[prod], ecolor="0.5", capsize=3)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(ypos)
        ax.set_yticklabels(d["site"])
        ax.set_xlabel("per-site gamma (%/yr)")
        ax.set_title(prod.upper())
    fig.suptitle("Per-site standalone drift (shared-gamma check) — "
                 "overlapping CIs support one common gamma", fontsize=11)
    fig.tight_layout()
    return fig


def _plot_corrected(joint, obs, site_colors, T0):
    """§6: divide out each site's (alpha + beta cos theta); collapse onto 1 + gamma t."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    tg = np.linspace(float(obs["t"].min()), float(obs["t"].max()), 60)
    for ax, prod in zip(axes, PRODUCTS):
        r = joint[prod]
        a = r["alpha"][r["sidx"]]
        b = r["beta"][r["sidx"]]
        corrected = r["rho"] / (a + b * r["g"])
        for s in r["sites"]:
            m = r["labels"] == s
            ax.scatter(r["t"][m], corrected[m], s=24, color=site_colors.get(s, "gray"),
                       edgecolors="k", linewidths=0.3, label=s, zorder=3)
        ax.plot(tg, 1 + r["gamma"] * tg, "k-", lw=1.8, zorder=4,
                label=f"1 + gamma*t ({100 * r['gamma']:+.3f} %/yr)")
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{prod.upper()} — corrected")
        ax.set_xlabel(f"t (yr since {T0.date()})")
    axes[0].set_ylabel(r"$\rho\,/\,(\alpha_s + \beta_s\,\cos\theta)$")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"All-PICS quasi-BRDF correction — band ~{TARGET_WL:.0f} nm: "
                 "per-site factor divided out, shared drift retained", fontsize=12)
    fig.tight_layout()
    return fig


def _plot_bootstrap(boots, joint):
    """§7: bootstrap histogram of the pooled shared gamma, both products."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for p, color in PRODUCTS.items():
        ax.hist(100 * boots[p]["draws"], bins=40, color=color, alpha=0.55, label=p)
        ax.axvline(100 * joint[p]["gamma"], color=color, lw=1.6)
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("shared gamma (%/yr)")
    ax.set_ylabel("bootstrap count")
    ax.set_title(f"All-PICS pooled drift bootstrap — band ~{TARGET_WL:.0f} nm "
                 "(stratified within-site)")
    ax.legend()
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
    # Drop non-invariant sites from the FIT only (they stay in the design/figures).
    obs, dropped = brdf_drift.apply_drift_exclusion(obs, cfg)
    if dropped:
        print(f"Excluded from drift fit (not invariant): {', '.join(dropped)}")
    sites = sorted(obs["site"].unique())
    T0 = obs["datetime"].min()
    site_colors = {s: cm.tab10(i % 10) for i, s in enumerate(sites)}
    print(f"Design: {len(obs)} scene(s) across {len(sites)} site(s) | "
          f"band ~{TARGET_WL:.0f} nm | t0 = {T0:%Y-%m-%d}")

    paths.BRDF_DIR.mkdir(parents=True, exist_ok=True)
    paths.BRDF_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. §4b — per-site standalone gamma + bootstrap CI, then the consistency check.
    rows = []
    for prod in PRODUCTS:
        for s in sites:
            sub = obs[obs["site"] == s]
            g0, ci = brdf_drift.boot_site_gamma(sub["g"], sub["t"], sub[f"rho_{prod}"],
                                                PER_SITE_BOOT)
            rows.append({"product": prod, "site": s, "n": len(sub),
                         "gamma_pct_yr": 100 * g0,
                         "ci_lo": 100 * ci[0], "ci_hi": 100 * ci[1]})
    persite = pd.DataFrame(rows)
    verdict = brdf_drift.consistency_check(persite)
    _report_consistency(verdict)

    persite_path = paths.brdf_persite_gamma(TARGET_WL)
    persite.to_csv(persite_path, index=False)
    provenance.write_sidecar(persite_path, cfg.as_dict(),
                             extra={"brdf_consistency": _clean_verdict(verdict)})
    print(f"\nPer-site gammas -> {persite_path}")

    fig = _plot_persite(persite, sites)
    fig.savefig(paths.brdf_persite_gamma_fig(TARGET_WL), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1b. §4b spectral stress test — each PICS's standalone gamma across ALL channels.
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    if ref is None:
        print("First design scene has no reflectance per-channel CSV; skipping the "
              "per-site gamma spectrum.")
    else:
        wlgrid, _excl, fit_bands = brdf_drift.water_vapor_mask(ref)
        spec_frames = [brdf_drift.persite_gamma_spectrum(
                           obs, prod, fit_bands, wlgrid,
                           earth_sun_normalize=cfg.earth_sun_normalize,
                           e0_method=cfg.e0_method)
                       for prod in PRODUCTS]
        persite_spec = pd.concat(spec_frames, ignore_index=True)
        # Hide degenerate per-site fits (near-zero baseline -> gamma explodes), as
        # the pooled spectrum does, so the points stay on a readable scale.
        persite_spec.loc[persite_spec["gamma_pct_yr"].abs() > GAMMA_CAP,
                         "gamma_pct_yr"] = np.nan
        # Publication specs: manuscript theme (Arial, Elsevier width, vector PDF+SVG,
        # no on-figure title), dots (no interpolation across the masked bands).
        with style.theme("manuscript"):
            fig = vizfigures.persite_gamma_spectrum(persite_spec, ylim=(-3.0, 3.0))
            written = style.save(fig, "brdf_persite_gamma_spectrum",
                                 out_dir=paths.BRDF_FIG_DIR)
        for p in written:
            print(f"Per-site gamma spectrum -> {p}")

    # 2. §5 — joint pooled fit (per-site alpha/beta, one shared gamma).
    joint = {p: brdf_drift.fit_joint(obs, p) for p in PRODUCTS}
    jrows = []
    print()
    for prod, r in joint.items():
        print(f"== {prod.upper()} ==  shared gamma = {100 * r['gamma']:+.3f} %/yr   "
              f"(n={r['n']}, {len(r['sites'])} site(s), success={r['success']})")
        for s, a, b in zip(r["sites"], r["alpha"], r["beta"]):
            print(f"    {s:<14} alpha={a:>12.4f}  beta={b:>12.4f}")
            jrows.append({"product": prod, "site": s, "alpha": a, "beta": b,
                          "gamma_pct_yr": 100 * r["gamma"], "success": r["success"],
                          "n": r["n"]})
    joint_path = paths.brdf_joint_fit(TARGET_WL)
    pd.DataFrame(jrows).to_csv(joint_path, index=False)
    provenance.write_sidecar(joint_path, cfg.as_dict())
    print(f"\nJoint fit       -> {joint_path}")

    fig = _plot_corrected(joint, obs, site_colors, T0)
    fig.savefig(paths.brdf_corrected_fig(TARGET_WL), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. §7 — bootstrap the pooled shared gamma (stratified within-site).
    boots = {p: brdf_drift.bootstrap_joint_gamma(obs, p, N_BOOTSTRAP) for p in PRODUCTS}
    brows = []
    print(f"\n{'':<8}{'gamma %/yr':>12}{'boot SE':>10}{'95% CI (%/yr)':>24}"
          f"{'n':>6}{'sites':>7}")
    for p, b in boots.items():
        lo, hi = 100 * b["ci"]
        print(f"{p:<8}{100 * b['gamma']:>+12.3f}{100 * b['se']:>10.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>24}{joint[p]['n']:>6}"
              f"{len(joint[p]['sites']):>7}")
        brows.append({"product": p, "gamma_pct_yr": 100 * b["gamma"],
                      "boot_se": 100 * b["se"], "ci_lo": lo, "ci_hi": hi,
                      "n": joint[p]["n"], "n_sites": len(joint[p]["sites"])})
    boot_path = paths.brdf_bootstrap_gamma(TARGET_WL)
    pd.DataFrame(brows).to_csv(boot_path, index=False)
    provenance.write_sidecar(boot_path, cfg.as_dict())
    print(f"\nBootstrap gamma -> {boot_path}")

    fig = _plot_bootstrap(boots, joint)
    fig.savefig(paths.brdf_bootstrap_fig(TARGET_WL), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures -> {paths.BRDF_FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
