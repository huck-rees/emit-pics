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
def _load_drift_spectrum(cfg):
    csv = paths.brdf_drift_spectrum()
    if not csv.exists():
        return None, f"{csv.name} not found — run the BRDF pipeline first."
    return {"df": pd.read_csv(csv)}, csv.name


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


def _load_matched_pairs(cfg, *, patch="10_7_1"):
    csv = paths.matched_pairs(cfg.site, patch)
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_id_matched_pairs.py first."
    return ({"pairs": pd.read_csv(csv), "site": cfg.site, "patch": patch,
             "sza_tol": cfg.sza_tol}, csv.name)


def _load_channel_drift(cfg):
    """Original cross-PICS pool: ONE representative patch per PICS.

    Reads the per-patch delta tables (SZA-sized, as in the original analysis) —
    NOT the ``use_aggregated`` / all-patches source, which collapses one PICS's
    many patches and would show a single PICS. Reduces to one patch per PICS
    (config roster + the configured site, auto for single-patch PICS), then
    computes the Theil-Sen stats on the fly from that same pool so the fit
    line and stats box always match the points shown.
    """
    from emit_pics.matched_pairs_drift_analysis import (
        load_delta_tables, compute_drift_spectrum, select_one_patch_per_pics)
    try:
        pooled = load_delta_tables(paths.DELTA_PER_PATCH_DIR, cfg.patch_size)
    except ValueError as exc:
        return None, f"per-patch delta tables have an outdated schema: {exc}"
    if pooled.empty:
        return None, ("no per-patch delta CSVs in data/delta_r/per_patch — run "
                      "scripts/run_delta_calculations.py per PICS first.")

    # Pin the configured site to its patch; roster (if any) resolves the rest.
    roster = dict(cfg.drift_cross_pics_patches)
    roster.setdefault(str(cfg.site), str(cfg.patch))
    pooled, ambiguous = select_one_patch_per_pics(pooled, roster)
    if ambiguous:
        lines = "; ".join(f"{p} -> {ps}" for p, ps in ambiguous)
        return None, ("could not reduce to one patch per PICS "
                      f"({lines}); set drift.cross_pics_patches in run_config.yaml.")

    drift = pd.concat([compute_drift_spectrum(pooled, "rad", alpha=cfg.alpha),
                       compute_drift_spectrum(pooled, "rfl", alpha=cfg.alpha)],
                      ignore_index=True)
    return ({"deltas": pooled, "drift": drift},
            f"per-patch pool: 1 patch/PICS across {pooled['PICS_ID'].nunique()} PICS")


def _load_patch_coverage(cfg):
    from emit_pics.id_granules import load_patches
    from emit_pics.config import pics_geodataframe
    bnds = paths.patch_bnds(cfg.site)
    if not bnds.exists():
        return None, f"{bnds.name} not found — run scripts/run_id_granules.py first."
    patches = load_patches(bnds, cfg.search_patch_size_km)
    return {"patches": patches, "site": pics_geodataframe(cfg.site)}, bnds.name


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


#: figure key -> {number (manuscript sequence), loader, builder}.
FIGURES = {
    "site_map": {"number": "fig01a", "load": _load_site_map,
                 "build": figures.site_overview_map},
    "africa_inset": {"number": "fig01b", "load": _load_africa_inset,
                     "build": figures.africa_locator},
    "coverage_map": {"number": "fig01c", "load": _load_coverage_single,
                     "build": figures.patch_coverage_single},
    "patch_coverage": {"number": "fig01d", "load": _load_patch_coverage,
                       "build": figures.patch_coverage},
    "brdf_cos_sza": {"number": "fig02", "load": _load_brdf_cos_sza,
                     "build": figures.brdf_cos_sza},
    "matched_pairs": {"number": "fig02b", "load": _load_matched_pairs,
                      "build": figures.matched_pairs_geometry},
    "drift_spectrum": {"number": "fig03", "load": _load_drift_spectrum,
                       "build": figures.drift_spectrum,
                       "kwargs": {"sig_col": "sig_raw", "sig_label": "significant (raw p < 0.05)"}},
    "mp_drift_spectrum": {"number": "fig05", "load": _load_mp_drift_spectrum,
                          "build": figures.drift_spectrum,
                          "kwargs": {"sig_col": "sig_raw",
                                     "sig_label": "significant (raw p < 0.05)",
                                     "ylim": (-5.0, 5.0),
                                     "title": "EMIT matched-pairs radiometric drift spectrum"}},
    "channel_drift_rad": {"number": "fig04a", "load": _load_channel_drift,
                          "build": figures.channel_drift, "kwargs": {"product": "rad"}},
    "channel_drift_rfl": {"number": "fig04b", "load": _load_channel_drift,
                          "build": figures.channel_drift, "kwargs": {"product": "rfl"}},
}


def _name(key: str, number: str, theme: str) -> str:
    """Output stem: numbered for the manuscript, `_ppt`-suffixed for slides."""
    base = f"{number}_{key}"
    return base if theme == "manuscript" else f"{base}_ppt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate publication / PPT figures.")
    ap.add_argument("--figures", nargs="+", choices=list(FIGURES),
                    default=list(FIGURES), metavar="KEY",
                    help="which figures to build (default: all).")
    ap.add_argument("--theme", choices=["manuscript", "ppt", "both"], default="both")
    ap.add_argument("--out", default=None,
                    help="write flat into this dir (e.g. the manuscript repo's figures/).")
    ap.add_argument("--patch", default="10_7_1",
                    help="patch id for per-patch figures (default: 10_7_1, Libya 4).")
    ap.add_argument("--highlight", default="Libya 4",
                    help="site bolded in the site_map, and whose grid the coverage_map "
                         "draws (default: Libya 4).")
    ap.add_argument("--coverage-product", choices=["rad", "rfl"], default="rfl",
                    help="which product's per-patch scene count colours the coverage_map "
                         "(default: rfl / L2A).")
    ap.add_argument("--wavelength", type=float, default=1030.0,
                    help="target wavelength (nm) for per-channel figures; the "
                         "nearest non-absorption channel is drawn (default: 1030).")
    ap.add_argument("--annotate", choices=["both", "count", "none"], default=None,
                    help="per-cell label for map figures: 'both' (id+count), "
                         "'count' (cleaner), or 'none'. Applied only to figures "
                         "that support it.")
    ap.add_argument("--list", action="store_true", help="list known figures and exit.")
    args = ap.parse_args(argv)

    if args.list:
        for key, spec in FIGURES.items():
            print(f"  {spec['number']:<7} {key}")
        return 0

    cfg = load_config()
    paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
    paths.set_brdf_label(cfg.brdf_label)    # read BRDF products from the labelled sub-namespace, if any
    themes = ["manuscript", "ppt"] if args.theme == "both" else [args.theme]
    n_written = 0
    for key in args.figures:
        spec = FIGURES[key]
        # Pass each optional CLI knob only to loaders whose signature accepts it.
        load_params = inspect.signature(spec["load"]).parameters
        _load_cli = {"patch": args.patch, "highlight": args.highlight,
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
        if "target_nm" in params:
            kwargs["target_nm"] = args.wavelength
        for th in themes:
            with style.theme(th):
                fig = spec["build"](**kwargs)
                written = style.save(fig, _name(key, spec["number"], th), out_dir=args.out)
            for p in written:
                print(f"  {key} [{th}] -> {p}")
                n_written += 1

    print(f"\nWrote {n_written} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
