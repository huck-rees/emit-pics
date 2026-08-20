#!/usr/bin/env python
"""Temporal delete-one-block jackknife of the pooled quasi-BRDF drift (Appendix D).

The temporal counterpart to the leave-one-PICS-out check. The pooled acquisition
timeline is split into contiguous blocks; each block is dropped in turn and the
pooled shared-gamma per-channel drift spectrum is refit on the rest, with the same
stratified within-site bootstrap as the primary fit. Because time is the regressor,
dropping an end block shifts gamma by construction (leverage), so the test is not
"did gamma change" but "does each block-deleted gamma stay inside the FULL-data
fit's 95% bootstrap CI?" A block-deleted estimate escaping that band flags an
influential epoch.

Pure local post-processor. Requires the per-site geometry/QC + cached per-scene
CSVs (as the BRDF workflow), and the full-fit reference spectrum
``data/brdf/brdf_drift_spectrum.csv`` (run scripts/run_brdf_drift_spectrum.py
first). Set the roster in run_config.yaml (brdf:) and the blocks in (jackknife:).

Outputs
-------
* ``data/brdf_jackknife/jackknife_drift_spectrum.csv`` -- per block-out x product x
  channel: gamma, CI, p, flags, and outside_full_ci (+ sidecar).
* ``data/brdf_jackknife/jackknife_blocks.csv`` -- the block roster (dates, n).
* ``data/brdf_jackknife/jackknife_summary.csv`` -- per block/product rollup incl.
  CI-escape counts.
* ``data/brdf_jackknife/jackknife_full_fit.csv`` -- the full-data reference spectrum.
* ``figures/publication/manuscript/jackknife_gamma_overlay.{pdf,svg}`` -- headline
  overlay (full-fit CI band + block-deleted gamma).
* ``figures/publication/manuscript/jackknife_spectrum_without_<block>.{pdf,svg}`` --
  one per block left out.

Run::

    python scripts/run_brdf_jackknife.py
"""

import sys

import pandas as pd

from emit_pics import brdf_drift, brdf_jackknife, paths, provenance
from emit_pics.run_config import load_config
from emit_pics.viz import figures, style

PRODUCTS = ("rad", "rfl")


def main() -> int:
    cfg = load_config()
    paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
    paths.set_brdf_label(cfg.brdf_label)    # read BRDF products from the labelled sub-namespace, if any

    # 1. Roster -> pooled per-scene design table (all sites, all scenes).
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
    print(f"Design: {len(obs)} scenes across {obs['site'].nunique()} sites | "
          f"origin {T0.date()}")

    # 2. Full-data reference spectrum (the band the block-deleted fits are judged against).
    full_path = paths.brdf_drift_spectrum()
    if not full_path.exists():
        print(f"Full-fit reference {full_path} not found; run "
              "scripts/run_brdf_drift_spectrum.py first.")
        return 1
    full_df = pd.read_csv(full_path)

    # 3. Reference grid + water-vapor exclusion, from one reflectance-bearing scene.
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    if ref is None:
        print("First design scene has no reflectance per-channel CSV; cannot set "
              "the band grid. Re-run the extract for the roster patches.")
        return 1
    wlgrid, excl, fit_bands = brdf_drift.water_vapor_mask(ref)
    print(f"{len(wlgrid)} channels | {int(excl.sum())} water-vapor band(s) excluded | "
          f"{len(fit_bands)} fit per product")

    # 4. Scene x band matrices (aligned to obs rows).
    mats = {p: brdf_drift.channel_matrix(obs, p, len(wlgrid),
                                         earth_sun_normalize=cfg.earth_sun_normalize,
                                         e0_method=cfg.e0_method)
            for p in PRODUCTS}

    # 5. Temporal blocks on the pooled timeline (block col aligns with mats rows).
    obs, blocks = brdf_jackknife.make_time_blocks(
        obs, scheme=cfg.jackknife_block_scheme, n_blocks=cfg.jackknife_n_blocks)
    print(f"\nBlock scheme '{cfg.jackknife_block_scheme}' -> {len(blocks)} blocks:")
    for _, b in blocks.iterrows():
        print(f"  {b['short']:<5} {b['start']} .. {b['end']}  (n={int(b['n_scenes'])})")

    # 6. Leave-one-block-out per-channel drift spectra, then the CI-escape test.
    print(f"\nFitting {len(blocks)} block-deletions x {len(PRODUCTS)} products x "
          f"{len(fit_bands)} channels (n_boot={cfg.brdf_spectrum_n_boot}) ...")
    jack = brdf_jackknife.leave_one_block_out_drift_spectra(
        obs, mats, wlgrid, fit_bands, blocks,
        n_boot=cfg.brdf_spectrum_n_boot, gamma_cap=cfg.brdf_gamma_cap,
        alpha=cfg.alpha)
    jack = brdf_jackknife.flag_ci_escapes(jack, full_df)
    summary = brdf_jackknife.block_summary(jack)

    # 7. Data sheets (each with a provenance sidecar).
    out_dir = paths.DATA / "brdf_jackknife"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tbl in (("jackknife_drift_spectrum", jack),
                      ("jackknife_blocks", blocks),
                      ("jackknife_summary", summary),
                      ("jackknife_full_fit", full_df)):
        path = out_dir / f"{name}.csv"
        tbl.to_csv(path, index=False)
        provenance.write_sidecar(path, cfg.as_dict())
        print(f"  data -> {path}")

    print("\nPer-block summary (channels escaping the full-fit 95% CI):")
    for _, r in summary.iterrows():
        print(f"  {r['block_out']:<22} {r['product']}  median gamma="
              f"{r['median_gamma_pct_yr']:+.3f} %/yr  "
              f"outside_full_CI={int(r['n_outside_full_ci'])}/{int(r['n_stable'])} "
              f"({r['pct_outside_full_ci']}%)")

    # 8. Publication figures (manuscript theme -> figures/publication/manuscript/).
    with style.theme("manuscript"):
        for _, b in blocks.iterrows():                 # one spectrum per block left out
            dsub = jack[jack["block_out"] == b["label"]]
            fig = figures.drift_spectrum(
                dsub, sig_col="sig_raw",
                sig_label="significant (95% CI excludes 0)",
                corner_label=f"without {b['label']}")
            for p in style.save(fig, f"jackknife_spectrum_without_{b['short']}"):
                print(f"  figure -> {p}")
        fig = figures.jackknife_gamma_overlay(jack, full_df)      # headline overlay
        for p in style.save(fig, "jackknife_gamma_overlay"):
            print(f"  figure -> {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
