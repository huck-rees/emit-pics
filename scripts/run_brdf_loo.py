#!/usr/bin/env python
"""Leave-one-out (jackknife) quasi-BRDF drift spectra — appendix robustness check.

Inverts the standalone per-site drift analysis. Rather than fitting each PICS on
its own (few scenes → wide uncertainty), this drops one PICS at a time and fits
the pooled shared-``gamma`` drift on the remaining invariant sites, at every EMIT
channel, with the same stratified within-site bootstrap as the primary drift
spectrum (:func:`emit_pics.brdf_drift.channel_gamma`). One leave-one-out combo per
invariant site gives well-constrained ``gamma(lambda)`` spectra whose agreement
shows no one site drives the pooled drift. Sites in ``brdf.drift_exclude_sites``
are not fit (same invariant set as the primary drift spectrum).

Pure local post-processor. Requires the per-site geometry tables + QC sheets and
the cached per-scene per-channel CSVs (the same inputs as the BRDF workflow); set
the roster + knobs in run_config.yaml (``brdf:`` block).

Outputs
-------
* ``data/brdf_loo/loo_drift_spectrum.csv`` — tidy per-channel γ, CI, p, flags for
  every combo × product (+ provenance sidecar).
* ``data/brdf_loo/loo_combos.csv`` — the five combos (left-out site, kept sites, n).
* ``data/brdf_loo/loo_summary.csv`` — per-combo/product rollup (median γ, CI, n sig).
* ``figures/publication/manuscript/brdf_loo_spectrum_<combo>.{pdf,svg}`` — five
  drift spectra (rad = A, rfl = B), each tagged with its kept-site combo.
* ``figures/publication/manuscript/brdf_loo_gamma_overlay.{pdf,svg}`` — all five
  combos' γ(λ) overlaid.

Run::

    python scripts/run_brdf_loo.py
"""

import sys

import pandas as pd

from emit_pics import brdf_drift, brdf_loo, paths, provenance
from emit_pics.run_config import load_config
from emit_pics.viz import figures, style

PRODUCTS = ("rad", "rfl")


def _slug(label: str) -> str:
    """Filename-safe combo tag: ``"A3, A5, L1, L4"`` -> ``"A3-A5-L1-L4"``."""
    return label.replace(", ", "-").replace(" ", "")


def main() -> int:
    cfg = load_config()
    paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
    paths.set_brdf_label(cfg.brdf_label)    # read BRDF products from the labelled sub-namespace, if any

    # 1. Roster -> pooled per-scene design table (all sites).
    roster, excluded, unknown = brdf_drift.resolve_roster(cfg)
    if not roster:
        print("No BRDF roster (brdf.site_patches empty in run_config.yaml).")
        return 1
    if excluded:
        print(f"Excluded by config: {', '.join(excluded)}")
    if unknown:
        print(f"Unknown (not in sites.yaml), skipped: {', '.join(unknown)}")

    obs, _, T0, notes = brdf_drift.assemble_design(roster, cfg)
    for n in notes:
        print(f"  [{'kept' if n['kept'] else 'DROP'}] {n['site']:<14} {n['note']}")
    if obs is None or obs.empty:
        print("No usable scenes across the roster; nothing to fit.")
        return 1
    # Drop non-invariant sites from the FIT only (they stay in the design/figures).
    obs, dropped = brdf_drift.apply_drift_exclusion(obs, cfg)
    if dropped:
        print(f"Excluded from drift fit (not invariant): {', '.join(dropped)}")
    sites = sorted(obs["site"].unique())
    if len(sites) < 3:
        print(f"Only {len(sites)} site(s) usable; leave-one-out needs >= 3.")
        return 1
    print(f"Design: {len(obs)} scenes across {len(sites)} sites | "
          f"{len(sites)} leave-one-out combos | origin {T0.date()}")

    # 2. Reference grid + water-vapor exclusion, from one reflectance-bearing scene.
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    if ref is None:
        print("First design scene has no reflectance per-channel CSV; cannot set "
              "the band grid. Re-run the extract for the roster patches.")
        return 1
    wlgrid, excl, fit_bands = brdf_drift.water_vapor_mask(ref)
    nband = len(wlgrid)
    print(f"{nband} channels | {int(excl.sum())} water-vapor band(s) excluded | "
          f"{len(fit_bands)} fit per product")

    # 3. Scene×band matrices for the full design (subset per combo inside the lib).
    mats = {p: brdf_drift.channel_matrix(obs, p, nband,
                                         earth_sun_normalize=cfg.earth_sun_normalize,
                                         e0_method=cfg.e0_method)
            for p in PRODUCTS}

    # 4. Leave-one-out per-channel drift spectra (both products, all combos).
    print(f"Fitting {len(sites)} leave-one-out combos x {len(PRODUCTS)} products x "
          f"{len(fit_bands)} channels (n_boot={cfg.brdf_spectrum_n_boot}) ...")
    spectra = brdf_loo.leave_one_out_drift_spectra(
        obs, mats, wlgrid, fit_bands,
        n_boot=cfg.brdf_spectrum_n_boot, gamma_cap=cfg.brdf_gamma_cap,
        alpha=cfg.alpha)

    combos = brdf_loo.leave_one_out_combos(sites)
    counts = obs["site"].value_counts()
    combos_tbl = pd.DataFrame([{
        "combo": c["label"], "left_out": c["left_out"],
        "kept_sites": ", ".join(c["included"]),
        "n_scenes": int(sum(int(counts.get(s, 0)) for s in c["included"])),
    } for c in combos])
    summary = brdf_loo.combo_summary(spectra)

    # 5. Data sheets (each with a provenance sidecar).
    out_dir = paths.DATA / "brdf_loo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tbl in (("loo_drift_spectrum", spectra),
                      ("loo_combos", combos_tbl),
                      ("loo_summary", summary)):
        path = out_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        provenance.write_sidecar(path, cfg.as_dict())
        print(f"  data -> {path}")

    print("\nPer-combo summary:")
    for _, r in summary.iterrows():
        print(f"  {r['combo']:<16} {r['product']}  median gamma={r['median_gamma_pct_yr']:+.3f} "
              f"%/yr  median CI={r['median_ci_width_pct_yr']:.3f}  "
              f"n_sig={int(r['n_sig_raw'])}/{int(r['n_stable'])}")

    # 6. Publication figures (manuscript theme -> figures/publication/manuscript/).
    with style.theme("manuscript"):
        for c in combos:                                 # five per-combo spectra
            dsub = spectra[spectra["combo"] == c["label"]]
            fig = figures.drift_spectrum(
                dsub, sig_col="sig_raw",
                sig_label="significant (95% CI excludes 0)",
                corner_label=c["label"])
            for p in style.save(fig, f"brdf_loo_spectrum_{_slug(c['label'])}"):
                print(f"  figure -> {p}")
        fig = figures.loo_gamma_overlay(spectra)          # the overlay
        for p in style.save(fig, "brdf_loo_gamma_overlay"):
            print(f"  figure -> {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
