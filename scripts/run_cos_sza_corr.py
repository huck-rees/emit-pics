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

    # 1. Resolve the roster and assemble the pooled per-scene design table.
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
        flag = "kept" if n["kept"] else "DROP"
        print(f"  [{flag}] {n['site']:<14} {n['note']}")
    if obs is None or obs.empty:
        print("No usable scenes across the roster; nothing to correlate.")
        return 1

    # 2. Per-site Spearman correlation between cos(SZA) and t.
    corr = cos_sza_corr.site_cos_sza_time_corr(obs)
    print(f"\nWithin-site Spearman corr(cos SZA, t), origin {T0.date()}:")
    for _, r in corr.iterrows():
        print(f"  {r['site']:<14} n={int(r['n']):>2}  rho={r['spearman_rho']:+.3f}  "
              f"p={r['p_value']:.3f}")

    # 3. Write the results table (with a provenance sidecar).
    out_dir = paths.DATA / "cos_sza_corr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cos_sza_time_corr.csv"
    corr.to_csv(out_csv, index=False)
    provenance.write_sidecar(out_csv, cfg.as_dict())
    print(f"\nResults CSV -> {out_csv}")

    # 4. Publication figure (manuscript theme -> figures/publication/manuscript/).
    with style.theme("manuscript"):
        fig = figures.cos_sza_time_corr(obs, corr, T0=T0)
        written = style.save(fig, FIG_NAME)
    for p in written:
        print(f"Figure -> {p}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
