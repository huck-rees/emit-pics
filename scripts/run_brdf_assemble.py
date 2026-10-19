#!/usr/bin/env python
"""Assemble the pooled multi-site design table for the BRDF drift analysis.

First stage of the quasi-BRDF pooled-drift workflow (port of
``notebooks/BRDF_drift_workflow_allPICS.ipynb``, sections 1-3 + 4a). Every PICS in
the ``brdf.site_patches`` roster contributes ONE representative patch (one patch =
one "site"). This script:

1. resolves the roster against ``config/sites.yaml`` (drops excluded / unknown),
2. builds the pooled per-scene design table — per site: QC-surviving,
   reflectance-bearing scenes with their median rho at the channel nearest
   ``brdf.target_wl`` — on a single common clock ``t`` (years since the earliest
   scene across all sites) with ``g = cos(SZA)``,
3. applies the ``brdf.min_scenes`` per-site floor (**warns** on drops; a site with
   too few scenes cannot constrain the shared drift beyond its own alpha/beta),
4. renders the §4a leverage / confounding scatter and reports pooled
   ``corr(cos SZA, t)``.

Pure **local** post-processor (no Earthdata). Needs each roster patch's geometry
table + reviewed QC worksheet + per-channel RAD/RFL stats to already exist (run
the extract + QC stages for those patches first).

Outputs
-------
* ``data/brdf/brdf_design.csv`` — pooled per-scene design table (the gated fit set).
* ``data/brdf/brdf_leverage.csv`` — per-site leverage summary (``kept`` flags the
  gate survivors).
* ``figures/brdf/brdf_leverage_scatter_{wl}nm.png`` — rho vs t and rho vs cos(SZA),
  coloured by site (§4a).

Set the roster + knobs in run_config.yaml (``brdf:`` block), then run::

    python scripts/run_brdf_assemble.py
"""

import sys

from emit_pics import brdf_drift, paths, provenance
from emit_pics.run_config import load_config

# INPUTS — set the roster + knobs in run_config.yaml (brdf:), not here.
cfg = load_config()
paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
paths.set_brdf_label(cfg.brdf_label)    # extra BRDF-only sub-namespace (e.g. patch-size sweep)
TARGET_WL  = cfg.brdf_target_wl
MIN_SCENES = cfg.brdf_min_scenes

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _plot_leverage(obs, sites, T0):
    """§4a: rho vs t (left) and rho vs cos(SZA) (right), rad top / rfl bottom."""
    site_colors = {s: cm.tab10(i % 10) for i, s in enumerate(sites)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, prod in enumerate(("rad", "rfl")):
        y = obs[f"rho_{prod}"]
        for s in sites:
            m = obs["site"] == s
            axes[i, 0].scatter(obs.loc[m, "t"], y[m], s=26, color=site_colors[s],
                               edgecolors="k", linewidths=0.3, label=s)
            axes[i, 1].scatter(obs.loc[m, "g"], y[m], s=26, color=site_colors[s],
                               edgecolors="k", linewidths=0.3)
        axes[i, 0].set_xlabel(f"t (yr since {T0.date()})")
        axes[i, 0].set_ylabel(f"{prod} rho (median)")
        axes[i, 1].set_xlabel("cos(SZA)")
        axes[i, 1].set_ylabel(f"{prod} rho (median)")
    axes[0, 0].set_title("rho vs time")
    axes[0, 1].set_title("rho vs cos(SZA)")
    axes[0, 0].legend(fontsize=7, ncol=2, title="site")
    fig.suptitle(f"All-PICS leverage check — band ~{TARGET_WL:.0f} nm", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    # 1. Resolve the roster against sites.yaml.
    roster, excluded, unknown = brdf_drift.resolve_roster(cfg)
    print(f"Roster: {len(roster)} site(s) configured with a patch.")
    for s, p in roster.items():
        print(f"  {s:<14} patch {p}")
    if excluded:
        print(f"Excluded by config:  {', '.join(sorted(excluded))}")
    for s in unknown:
        print(f"[skip] {s}: not a site in config/sites.yaml.")
    if not roster:
        print("\nNo sites mapped. Fill brdf.site_patches in run_config.yaml with "
              "at least one patch id.")
        sys.exit(1)

    # 2. Assemble the pooled design table (per-site QC-surviving scenes).
    obs, wl_by_site, T0, notes = brdf_drift.assemble_design(roster, cfg)
    print("\nPer-site assembly:")
    for n in notes:
        tag = "ok " if n["kept"] else "DROP"
        wl = f"  (band ~{n['wl_used']:.1f} nm)" if n["wl_used"] else ""
        print(f"  [{tag}] {n['site']:<14} {n['note']}{wl}")
    if obs is None:
        print("\nNo usable site data assembled (no site produced QC-surviving "
              "reflectance scenes). Check the extract + QC stages for the roster "
              "patches.")
        sys.exit(1)

    wl_set = sorted({round(w, 1) for w in wl_by_site.values()})
    print(f"\nPooled: {len(obs)} scenes across {obs['site'].nunique()} site(s) | "
          f"t0 = {T0:%Y-%m-%d} | bands {wl_set} nm")

    # 3. Leverage summary + MIN_SCENES gate (warn loudly on drops).
    summary = brdf_drift.leverage_summary(obs)
    obs, _, dropped = brdf_drift.apply_min_scenes_gate(obs, summary, MIN_SCENES)
    summary["kept"] = ~summary["site"].isin(dropped)
    for s in dropped:
        n = int(summary.loc[summary.site == s, "n"].iloc[0])
        print(f"[warn] MIN_SCENES: dropping {s} ({n} scene(s) < {MIN_SCENES}); "
              "too few to constrain drift beyond its own alpha/beta.")
    if obs.empty:
        print(f"\nAll sites fell below MIN_SCENES={MIN_SCENES}. Nothing to fit.")
        sys.exit(1)

    sites = sorted(obs["site"].unique())
    print(f"\nFinal fit set: {len(sites)} site(s), {len(obs)} scene(s).")

    # 4. Write the design table + leverage summary (with provenance sidecars).
    paths.BRDF_DIR.mkdir(parents=True, exist_ok=True)
    design_path = paths.brdf_design()
    obs.to_csv(design_path, index=False)
    provenance.write_sidecar(design_path, cfg.as_dict())
    print(f"\nDesign table    -> {design_path}")

    lev_path = paths.brdf_leverage()
    summary.to_csv(lev_path, index=False)
    provenance.write_sidecar(lev_path, cfg.as_dict())
    print(f"Leverage table  -> {lev_path}")

    # 5. §4a leverage / confounding scatter.
    paths.BRDF_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = _plot_leverage(obs, sites, T0)
    fig_path = paths.brdf_leverage_fig(TARGET_WL)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Leverage figure -> {fig_path}")

    corr = np.corrcoef(obs["g"], obs["t"])[0, 1]
    print(f"\nPooled corr(cos SZA, t) = {corr:+.2f} "
          "(near 0 -> drift well separated from geometry across the pooled set)")
    print("\nDone.")


if __name__ == "__main__":
    main()
