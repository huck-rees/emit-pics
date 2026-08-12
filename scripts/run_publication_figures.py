#!/usr/bin/env python
"""Generate publication / PPT figures from the pipeline's CSV outputs.

One arg-driven runner for every publication figure (keeps `scripts/` from growing
a file per figure). Each figure is a pure builder in `emit_pics.viz.figures` that
reads a CSV the pipeline already wrote — no recompute, so regeneration is a few
seconds. Every figure is emitted in the requested theme(s): `manuscript` (Elsevier
vector PDF + SVG) and/or `ppt` (SVG + PNG for slides).

    python scripts/run_publication_figures.py [--figures KEY ...]
                                              [--theme manuscript|ppt|both]
                                              [--out DIR]

`--out` writes the files **flat** into DIR (e.g. your local Overleaf clone's
`figures/`), so you regenerate straight into the manuscript repo and commit there;
omit it to write under `figures/publication/<theme>/`. `--figures` defaults to all
known figures; run with `--list` to see them.

Requires the relevant pipeline CSVs to exist (run the pipeline first).
"""

import argparse
import inspect
import sys

import pandas as pd

from emit_pics import paths
from emit_pics.run_config import load_config
from emit_pics.viz import figures, style


# --------------------------------------------------------------------- loaders
# Each loader reads already-generated pipeline outputs and returns
# ``(build_kwargs, source_note)``, or ``(None, reason)`` if an input is missing.
def _load_mp_drift_spectrum(cfg):
    """Matched-pairs cross-PICS drift spectrum, adapted to the BRDF builder schema.

    Reads whichever matched-pairs drift CSV the configured estimator wrote —
    ``{size}_drift.csv`` (Theil-Sen, the default) or ``{size}_drift_bootstrap.csv``
    (``drift.method: bootstrap``) — and renames its columns into the names
    ``figures.drift_spectrum`` expects, so the matched-pairs figure is rendered by
    the same builder with the same specs (incl. the +/-2.5 %/yr y-limits). Both CSVs
    share the same schema (``slope_pct_yr``/``ci_*_pct_yr``/``raw_significant``), so
    for the bootstrap file the marker becomes the bootstrap median and the whiskers
    its percentile CI. Water-vapor absorption bands and unfit (NaN-slope) channels
    are marked ``unstable`` so the builder drops them, exactly as it drops BRDF's
    numerically unstable channels.
    """
    if cfg.drift_method == "bootstrap":
        csv = paths.drift_bootstrap(cfg.patch_size)
        hint = "scripts/run_matched_pairs_drift_analysis.py with drift.method: bootstrap"
    else:
        csv = paths.drift(cfg.patch_size)
        hint = "scripts/run_matched_pairs_drift_analysis.py"
    if not csv.exists():
        return None, f"{csv.name} not found — run {hint} first."
    df = pd.read_csv(csv).rename(columns={
        "slope_pct_yr": "gamma_pct_yr",
        "ci_low_pct_yr": "ci_lo_pct_yr",
        "ci_high_pct_yr": "ci_hi_pct_yr",
        "raw_significant": "sig_raw",
    })
    df["unstable"] = df["absorption_band"].astype(bool) | df["gamma_pct_yr"].isna()
    return {"df": df}, csv.name


def _load_brdf_cos_sza(cfg, *, highlight="Libya 4"):
    """Fig. 2 inputs: the BRDF design table (per-scene cos(SZA) + median rho).

    Restricted to one PICS (*highlight*, default Libya 4) so the panels are a clean
    single-surface fit; the builder accepts ``site=None`` to pool all sites.
    """
    from emit_pics.config import canonical_token
    csv = paths.brdf_design()
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_brdf_assemble.py first."
    return ({"design": pd.read_csv(csv), "site": canonical_token(highlight),
             "target_wl": cfg.brdf_target_wl}, f"{csv.name} ({highlight})")


def _load_brdf_single_channel(cfg):
    """Fig. 4 inputs: the single-channel corrected collapse + pooled-γ bootstrap.

    Reads the pooled BRDF design table (``data/brdf/brdf_design.csv``) and, at the
    channel nearest ``brdf.target_wl``, recomputes the §5 joint fit and the §7
    stratified bootstrap for both products via :mod:`emit_pics.brdf_drift`. The
    bootstrap *draws* are not persisted by the BRDF step (only their summary CSV),
    and the fits use fixed seeds, so recomputing here is both necessary and exactly
    reproduces the BRDF step — the same on-the-fly pattern the other builders use.
    """
    from emit_pics import brdf_drift
    csv = paths.brdf_design()
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_brdf_assemble.py first."
    obs = pd.read_csv(csv, parse_dates=["datetime"])
    joint = {p: brdf_drift.fit_joint(obs, p) for p in ("rad", "rfl")}
    boots = {p: brdf_drift.bootstrap_joint_gamma(obs, p, cfg.brdf_n_bootstrap)
             for p in ("rad", "rfl")}
    T0 = obs["datetime"].min()
    return ({"joint": joint, "boots": boots, "T0": T0,
             "target_wl": cfg.brdf_target_wl},
            f"{csv.name} (band ≈ {cfg.brdf_target_wl:.0f} nm, "
            f"{obs['site'].nunique()} sites)")


def _study_sites(cfg):
    """The study PICS as a GeoDataFrame of ROI boxes, plus a provenance note.

    The site set is the BRDF roster (the sites actually in the study, e.g. the
    five North-African PICS) — NOT every box in sites.yaml — resolved through
    ``pics_geodataframe`` so the bbox geometry comes from the one source of truth.
    """
    from emit_pics.config import pics_geodataframe
    roster = list(cfg.brdf_site_patches) or None      # tokens, e.g. "Libya_4"
    if roster:
        sites = pd.concat([pics_geodataframe(s) for s in roster], ignore_index=True)
        return sites, f"{len(sites)} study PICS (BRDF roster)"
    sites = pics_geodataframe(None)                    # fall back to all sites.yaml boxes
    return sites, f"{len(sites)} PICS (all of sites.yaml — brdf.site_patches empty)"


def _load_site_map(cfg, *, highlight="Libya 4"):
    """Fig. 1A inputs: the study PICS roster as boxes for the locator map."""
    sites, note = _study_sites(cfg)
    return {"sites_gdf": sites, "highlight": highlight}, note


def _load_africa_inset(cfg):
    """Fig. 1B inputs: the study PICS (only their extent is used for the box)."""
    sites, note = _study_sites(cfg)
    return {"sites_gdf": sites}, note


def _load_coverage_single(cfg, *, highlight="Libya 4", coverage_product="rfl"):
    """Fig. 1C inputs: the highlighted site's coverage-tagged patch grid + boundary."""
    from emit_pics.id_granules import load_patches
    from emit_pics.config import pics_geodataframe
    bnds = paths.patch_bnds(highlight)
    if not bnds.exists():
        return None, f"{bnds.name} not found — run scripts/run_id_granules.py first."
    patches = load_patches(bnds, cfg.search_patch_size_km)
    count_col = f"n_gran_ovlp_{coverage_product}"
    count_label = "L1B radiance" if coverage_product == "rad" else "L2A reflectance"
    return ({"patches": patches, "boundary": pics_geodataframe(highlight),
             "count_col": count_col, "count_label": count_label},
            f"{highlight} patches from {bnds.name}")


#: figure key -> {name (output stem; ``{wl}`` filled from brdf.target_wl), loader,
#: builder}. No manuscript numbers — those churn as the paper is edited, so each
#: figure carries a stable descriptive name instead.
FIGURES = {
    "site_map": {"name": "studyoverview_panelA_PICS_locs", "load": _load_site_map,
                 "build": figures.site_overview_map},
    "africa_inset": {"name": "studyoverview_panelB_Africainset",
                     "load": _load_africa_inset, "build": figures.africa_locator},
    "coverage_map": {"name": "studyoverview_panelD_patchgrid",
                     "load": _load_coverage_single,
                     "build": figures.patch_coverage_single},
    "brdf_cos_sza": {"name": "solargeometry_rad_rfl", "load": _load_brdf_cos_sza,
                     "build": figures.brdf_cos_sza},
    "mp_drift_spectrum": {"name": "matchedpairs_drift_spectrum",
                          "load": _load_mp_drift_spectrum,
                          "build": figures.drift_spectrum,
                          "kwargs": {"sig_col": "sig_raw",
                                     "sig_label": "significant (p < 0.05)",
                                     "ylim": (-5.0, 5.0),
                                     "title": "EMIT matched-pairs radiometric drift spectrum"}},
    "brdf_single_channel": {"name": "brdf_singlechannel_{wl}nm",
                            "load": _load_brdf_single_channel,
                            "build": figures.brdf_single_channel},
}


def _name(stem: str, theme: str) -> str:
    """Output stem: the bare name for the manuscript, `_ppt`-suffixed for slides."""
    return stem if theme == "manuscript" else f"{stem}_ppt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate publication / PPT figures.")
    ap.add_argument("--figures", nargs="+", choices=list(FIGURES),
                    default=list(FIGURES), metavar="KEY",
                    help="which figures to build (default: all).")
    ap.add_argument("--theme", choices=["manuscript", "ppt", "both"], default="both")
    ap.add_argument("--out", default=None,
                    help="write flat into this dir (e.g. the manuscript repo's figures/).")
    ap.add_argument("--highlight", default="Libya 4",
                    help="site bolded in the site_map, and whose grid the coverage_map "
                         "draws (default: Libya 4).")
    ap.add_argument("--coverage-product", choices=["rad", "rfl"], default="rfl",
                    help="which product's per-patch scene count colours the coverage_map "
                         "(default: rfl / L2A).")
    ap.add_argument("--annotate", choices=["both", "count", "none"], default=None,
                    help="per-cell label for map figures: 'both' (id+count), "
                         "'count' (cleaner), or 'none'. Applied only to figures "
                         "that support it.")
    ap.add_argument("--list", action="store_true", help="list known figures and exit.")
    args = ap.parse_args(argv)

    if args.list:
        for key, spec in FIGURES.items():
            print(f"  {spec['name']:<32} {key}")
        return 0

    cfg = load_config()
    paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
    paths.set_brdf_label(cfg.brdf_label)    # read BRDF products from the labelled sub-namespace, if any
    themes = ["manuscript", "ppt"] if args.theme == "both" else [args.theme]
    wl_tag = f"{cfg.brdf_target_wl:.0f}"     # fills the ``{wl}`` slot in wavelength-tagged names
    n_written = 0
    for key in args.figures:
        spec = FIGURES[key]
        # Pass each optional CLI knob only to loaders whose signature accepts it.
        load_params = inspect.signature(spec["load"]).parameters
        _load_cli = {"highlight": args.highlight,
                     "coverage_product": args.coverage_product}
        load_kw = {k: v for k, v in _load_cli.items() if k in load_params}
        kwargs, note = spec["load"](cfg, **load_kw)
        if kwargs is None:
            print(f"[skip] {key}: {note}")
            continue
        # Static per-figure build kwargs (e.g. which product), then CLI overrides
        # applied only to builders whose signature accepts them.
        kwargs.update(spec.get("kwargs", {}))
        params = inspect.signature(spec["build"]).parameters
        if args.annotate is not None and "annotate" in params:
            kwargs["annotate"] = args.annotate
        stem = spec["name"].format(wl=wl_tag)
        for th in themes:
            with style.theme(th):
                fig = spec["build"](**kwargs)
                written = style.save(fig, _name(stem, th), out_dir=args.out)
            for p in written:
                print(f"  {key} [{th}] -> {p}")
                n_written += 1

    print(f"\nWrote {n_written} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
