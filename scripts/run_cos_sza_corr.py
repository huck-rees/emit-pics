#!/usr/bin/env python
"""Within-site correlation of solar geometry (cos theta) with time (Appendix B).

The pooled quasi-BRDF drift model assumes that, within a site, ``cos(SZA)`` does
not covary with acquisition time ``t`` — otherwise a geometry-vs-time trend would
be partly absorbed by the shared drift term. This runner assembles the same
pooled per-scene design table the BRDF drift fit uses
(:func:`emit_pics.brdf_drift.assemble_design`), reports the per-PICS **Spearman**
rank correlation between ``cos(SZA)`` and ``t``, and renders the five-panel
publication scatter.

Pure local post-processor. Requires the per-site geometry tables + QC sheets and
the cached per-scene per-channel CSVs (the same inputs as the BRDF workflow).

Outputs
-------
* ``data/cos_sza_corr/cos_sza_time_corr.csv`` — per-site n, Spearman rho, p (+ sidecar).
* ``figures/publication/manuscript/cos_sza_time_corr.{pdf,svg}`` — the scatter figure.

Set the roster in run_config.yaml (``brdf:`` block), then run::

    python scripts/run_cos_sza_corr.py
"""

import sys

from emit_pics import brdf_drift, cos_sza_corr, paths, provenance
from emit_pics.run_config import load_config
from emit_pics.viz import figures, style

#: Figure output stem (theme decides the extensions + directory).
FIG_NAME = "cos_sza_time_corr"


def main() -> int:
    cfg = load_config()
    paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
    paths.set_brdf_label(cfg.brdf_label)    # read BRDF products from the labelled sub-namespace, if any

    # 1. Assemble both sensors' per-scene cos(SZA)/t designs on a shared origin:
    #    Landsat 9 over all PICS, EMIT over the invariant PICS used in the drift fit.
    data = cos_sza_corr.dual_sensor_design(cfg)
    if data is None:
        print("No usable BRDF design (roster empty or no EMIT scenes).")
        return 1
    T0 = data["T0"]

    for label, corr in (("EMIT", data["emit_corr"]), ("Landsat 9", data["ls_corr"])):
        if corr is None:
            continue
        print(f"\nWithin-site Spearman corr(cos SZA, t) [{label}], origin {T0.date()}:")
        for _, r in corr.iterrows():
            print(f"  {r['site']:<14} n={int(r['n']):>2}  rho={r['spearman_rho']:+.3f}  "
                  f"p={r['p_value']:.3f}")

    # 2. Write the per-sensor results tables (with provenance sidecars).
    out_dir = paths.DATA / "cos_sza_corr"
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_csv = out_dir / "cos_sza_time_corr.csv"
    data["emit_corr"].to_csv(emit_csv, index=False)
    provenance.write_sidecar(emit_csv, cfg.as_dict())
    print(f"\nEMIT results CSV -> {emit_csv}")
    if data["ls_corr"] is not None:
        ls_csv = out_dir / "cos_sza_time_corr_landsat.csv"
        data["ls_corr"].to_csv(ls_csv, index=False)
        provenance.write_sidecar(ls_csv, cfg.as_dict())
        print(f"Landsat results CSV -> {ls_csv}")

    # 3. Publication figure (manuscript theme -> figures/publication/manuscript/).
    with style.theme("manuscript"):
        fig = figures.cos_sza_time_corr(
            data["emit_obs"], data["emit_corr"],
            ls_obs=data["ls_obs"], ls_corr=data["ls_corr"],
            order=data["order"], T0=T0)
        written = style.save(fig, FIG_NAME)
    for p in written:
        print(f"Figure -> {p}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
