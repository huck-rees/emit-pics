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
    """Pooled BRDF drift spectrum gamma(lambda), read straight from the BRDF CSV.

    ``data/brdf/brdf_drift_spectrum.csv`` already carries the exact columns the
    ``figures.drift_spectrum`` builder expects (``product``, ``wavelength``,
    ``gamma_pct_yr``, ``ci_{lo,hi}_pct_yr``, ``sig_raw``, ``unstable``), so no
    adaptation is needed — unlike the matched-pairs variant below.
    """
    csv = paths.brdf_drift_spectrum()
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_brdf_drift_spectrum.py first."
    return {"df": pd.read_csv(csv)}, csv.name


def _load_mp_drift_spectrum(cfg):
    """Matched-pairs cross-PICS drift spectrum, adapted to the BRDF builder schema.

    Reads the matched-pairs bootstrap drift CSV (``{size}_mp_drift.csv``) and renames
    its columns into the names ``figures.drift_spectrum`` expects, so the matched-
    pairs figure is rendered by the same builder as the pooled BRDF spectrum (same
    specs). The marker is the per-channel drift point estimate and the whiskers its
    bootstrap 95% CI. Water-vapor absorption bands and unfit (NaN-slope) channels are
    marked ``unstable`` so the builder drops them, exactly as it drops BRDF's
    numerically unstable channels.
    """
    csv = paths.drift(cfg.patch_size)
    if not csv.exists():
        return None, (f"{csv.name} not found — run "
                      "scripts/run_matched_pairs_drift_analysis.py first.")
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
    from emit_pics import brdf_drift, landsat_brdf
    csv = paths.brdf_design()
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_brdf_assemble.py first."
    site_tok = canonical_token(highlight)

    # Landsat cos(SZA) design for the same site: per-scene cos(SZA) + spatial-median
    # TOA reflectance at the NIR band nearest brdf.target_wl (865 nm for OLI-2).
    ls_design = None
    try:
        roster, _, _ = brdf_drift.resolve_roster(cfg)
        ls_obs, _, _, _ = landsat_brdf.assemble_design_landsat(roster, cfg)
        if ls_obs is not None:
            sub = ls_obs[ls_obs["site"] == site_tok]
            ls_design = sub if not sub.empty else None
    except Exception as exc:  # Landsat is an optional overlay; EMIT panels still render.
        print(f"  [brdf_cos_sza] Landsat panel skipped: {exc}")

    return ({"design": pd.read_csv(csv), "landsat_design": ls_design,
             "site": site_tok, "target_wl": cfg.brdf_target_wl},
            f"{csv.name} ({highlight})"
            + ("; +Landsat" if ls_design is not None else "; EMIT-only"))


def _load_brdf_single_channel(cfg):
    """Fig. 4 inputs: the single-channel corrected collapse + pooled-γ bootstrap.

    Reads the pooled BRDF design table (``data/brdf/brdf_design.csv``) and, at the
    channel nearest ``brdf.target_wl``, recomputes the §5 joint fit and the §7
    stratified bootstrap for both products via :mod:`emit_pics.brdf_drift`. The
    bootstrap *draws* are not persisted by the BRDF step (only their summary CSV),
    and the fits use fixed seeds, so recomputing here is both necessary and exactly
    reproduces the BRDF step — the same on-the-fly pattern the other builders use.
    """
    import numpy as np
    from emit_pics import brdf_drift
    csv = paths.brdf_design()
    if not csv.exists():
        return None, f"{csv.name} not found — run scripts/run_brdf_assemble.py first."
    obs = pd.read_csv(csv, parse_dates=["datetime"])
    # Drop non-invariant sites from the FIT only (they stay in the design/figures).
    obs, _ = brdf_drift.apply_drift_exclusion(obs, cfg)

    # Display channel: brdf.single_channel_wl if set (e.g. the SWIR2 band nearest
    # Landsat OLI-2), else the NIR target_wl. The design table carries rho only at
    # target_wl, so recompute per-scene rho at the chosen band from the cached CSVs.
    wl_want = cfg.brdf_single_channel_wl or cfg.brdf_target_wl
    ref = brdf_drift.scene_perchannel(obs.iloc[0]["site"], obs.iloc[0]["patch"],
                                      obs.iloc[0]["timestamp"], "rfl")
    if ref is None:
        return None, "first design scene has no reflectance CSV; cannot set band grid."
    wlgrid, _excl, _fb = brdf_drift.water_vapor_mask(ref)
    bidx = int(np.abs(wlgrid - wl_want).argmin())
    wl_used = float(wlgrid[bidx])

    joint, boots = {}, {}
    for p in ("rad", "rfl"):
        M = brdf_drift.channel_matrix(obs, p, len(wlgrid),
                                      earth_sun_normalize=cfg.earth_sun_normalize,
                                      e0_method=cfg.e0_method)
        frame = obs.copy()
        frame[f"rho_{p}"] = M[:, bidx]
        frame = frame[np.isfinite(frame[f"rho_{p}"])]
        joint[p] = brdf_drift.fit_joint(frame, p)
        boots[p] = brdf_drift.bootstrap_joint_gamma(frame, p, cfg.brdf_n_bootstrap)
    T0 = obs["datetime"].min()
    return ({"joint": joint, "boots": boots, "T0": T0, "target_wl": wl_used},
            f"{csv.name} (band ≈ {wl_used:.0f} nm, {obs['site'].nunique()} sites)")


def _load_brdf_site_spectra(cfg):
    """Mean rad/rfl spectra, one line per PICS over its 10 km BRDF patch's QC scenes.

    Resolves the BRDF roster (the study PICS + their representative patches) and
    averages each patch's QC-surviving per-scene spectra band-by-band via
    :func:`emit_pics.brdf_drift.site_mean_spectra` — the same scene set and loaders
    the pooled drift fit uses, so this figure is a faithful picture of the surfaces
    that went into the analysis. Reads the cached per-scene per-channel CSVs.
    """
    from emit_pics import brdf_drift, landsat_brdf
    roster, _, _ = brdf_drift.resolve_roster(cfg)
    if not roster:
        return None, "no BRDF roster (brdf.site_patches empty)."
    spectra, notes = brdf_drift.site_mean_spectra(roster, cfg)
    if not spectra:
        why = "; ".join(f"{n['site']}: {n['note']}" for n in notes)
        return None, f"no usable PICS ({why})."
    # Landsat 9 mean TOA spectra (panel a); same roster and QC gate.
    ls_spectra, _ = landsat_brdf.landsat_site_mean_spectra(roster, cfg)
    dropped = [n["site"] for n in notes if not n["kept"]]
    tail = f"; dropped {', '.join(dropped)}" if dropped else ""
    ls_tail = f"; Landsat {len(ls_spectra)} PICS" if ls_spectra else "; Landsat none"
    return ({"spectra": spectra, "landsat_spectra": ls_spectra or None},
            f"{len(spectra)} PICS mean spectra{tail}{ls_tail}")


def _load_cos_sza_corr(cfg):
    """Two-sensor cos(theta)/t designs + per-site Spearman corr for figureB1.

    Builds the Landsat 9 (all PICS) and EMIT (invariant PICS only) per-scene
    designs on one shared time origin via
    :func:`emit_pics.cos_sza_corr.dual_sensor_design`.
    """
    from emit_pics import cos_sza_corr
    data = cos_sza_corr.dual_sensor_design(cfg)
    if data is None:
        return None, "no usable BRDF design (roster empty or no EMIT scenes)."
    n_ls = 0 if data["ls_corr"] is None else len(data["ls_corr"])
    return data, f"LS9 {n_ls} PICS + EMIT {len(data['emit_corr'])} PICS cos-SZA/time"


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


def _load_temporal_coverage(cfg):
    """Per-PICS temporal coverage: matched-pair SZA-vs-time + BRDF scene dates.

    For each study PICS (fixed display order) assembles its **unique** matched pairs
    from the aggregated delta table (``data/delta_r/aggregated/{site}_{size}_delta.csv``
    — one row per unique acquisition pair per band, deduped to one row per pair) with
    each pair's two scene dates and in-patch mean SZAs, plus the QC-retained BRDF
    joint-fit scene dates from ``brdf_design.csv``. No recompute — both are read
    straight from pipeline outputs.
    """
    from emit_pics.config import canonical_token
    order = ["Libya_1", "Libya_4", "Algeria_3", "Algeria_5", "Mauritania_1"]
    # Sites dropped from the EMIT drift analyses keep only their OLI-2 column.
    excluded = {canonical_token(x) for x in cfg.brdf_drift_exclude_sites}

    des_path = paths.brdf_design()
    if not des_path.exists():
        return None, f"{des_path.name} not found — run scripts/run_brdf_assemble.py first."
    des = pd.read_csv(des_path, parse_dates=["datetime"])

    # Landsat BRDF scenes (same best patch per site, QC-surviving) for the 3rd column.
    from emit_pics import brdf_drift, landsat_brdf
    ls_obs = None
    try:
        roster, _, _ = brdf_drift.resolve_roster(cfg)
        ls_obs, _, _, _ = landsat_brdf.assemble_design_landsat(roster, cfg)
    except Exception as exc:  # Landsat is an optional column; EMIT columns still render.
        print(f"  [temporal_coverage] Landsat column skipped: {exc}")

    sites_data = []
    for s in order:
        tok = canonical_token(s)
        agg = paths.delta_aggregated(s, cfg.patch_size)
        if not agg.exists():
            return None, (f"{agg.name} not found — run the matched-pairs analysis "
                          "(aggregate step) first.")
        d = pd.read_csv(agg, parse_dates=["t1_dt", "t2_dt"])
        pairs = d.drop_duplicates("pair_id")[["t1_dt", "t2_dt", "SZA_t1", "SZA_t2"]]
        brdf = (des.loc[des["site"] == tok, ["datetime", "sza_mean"]]
                .sort_values("datetime"))
        if ls_obs is not None:
            ls = (ls_obs.loc[ls_obs["site"] == tok, ["datetime", "sza_mean"]]
                  .sort_values("datetime"))
        else:
            ls = brdf.iloc[0:0]
        sites_data.append({
            "name": s.replace("_", " "),
            "site": tok,
            "pairs": pairs.reset_index(drop=True),
            "dates": brdf["datetime"].to_numpy(),
            "sza": brdf["sza_mean"].to_numpy(float),
            "ls_dates": pd.to_datetime(ls["datetime"], utc=True)
                          .dt.tz_localize(None).to_numpy(),
            "ls_sza": ls["sza_mean"].to_numpy(float),
            "n_pairs": int(len(pairs)),
            "n_brdf": int(len(brdf)),
            "n_landsat": int(len(ls)),
            "show_emit": tok not in excluded,
        })
    return ({"sites_data": sites_data},
            f"{len(sites_data)} PICS temporal coverage"
            + ("" if ls_obs is not None else " (EMIT only)"))


def _load_landsat_invariance(cfg):
    """Landsat invariance inputs: per-site + leave-one-out drift spectra (all 7 bands).

    Reads the tidy tables written by ``scripts/run_landsat_brdf_robustness.py``
    (``landsat_persite_drift_spectrum.csv`` and ``landsat_loo_drift_spectrum.csv``
    under ``paths.LS_BRDF_DIR``) — no recompute.
    """
    ps = paths.LS_BRDF_DIR / "landsat_persite_drift_spectrum.csv"
    loo = paths.LS_BRDF_DIR / "landsat_loo_drift_spectrum.csv"
    for p in (ps, loo):
        if not p.exists():
            return None, (f"{p.name} not found — run "
                          "scripts/run_landsat_brdf_robustness.py first.")
    return ({"persite": pd.read_csv(ps), "loo": pd.read_csv(loo)},
            f"{ps.name} + {loo.name}")


def _load_emit_drift_overlay(cfg):
    """EMIT drift spectrum overlay: the 4-site matched-pairs + BRDF per-channel tables.

    Reads ``data/drift/{size}_mp_drift.csv`` and ``data/brdf/brdf_drift_spectrum.csv``.
    """
    brdf_csv = paths.brdf_drift_spectrum()
    mp_csv = paths.drift(cfg.patch_size)
    for c in (mp_csv, brdf_csv):
        if not c.exists():
            return None, f"{c.name} not found — run the matched-pairs/BRDF drift first."
    return ({"mp": pd.read_csv(mp_csv), "brdf": pd.read_csv(brdf_csv)},
            f"{mp_csv.name} + {brdf_csv.name}")


def _load_emit_oli2_bands(cfg):
    """OLI-2 band comparison: ``data/emit_oli2_compare/band_drift_comparison.csv``
    (from ``scripts/run_emit_oli2_band_comparison.py``). No recompute.
    """
    bands_csv = paths.DATA / "emit_oli2_compare" / "band_drift_comparison.csv"
    if not bands_csv.exists():
        return None, (f"{bands_csv.name} not found — run "
                      "scripts/run_emit_oli2_band_comparison.py first.")
    return {"bands": pd.read_csv(bands_csv)}, bands_csv.name


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
    "brdf_drift_spectrum": {"name": "brdf_drift_spectrum",
                            "load": _load_drift_spectrum,
                            "build": figures.drift_spectrum},
    "mp_drift_spectrum": {"name": "mp_drift_spectrum",
                          "load": _load_mp_drift_spectrum,
                          "build": figures.drift_spectrum,
                          "kwargs": {"sig_col": "sig_raw",
                                     "sig_label": "significant (95% CI excludes 0)",
                                     "ylim": (-5.0, 5.0),
                                     "title": "EMIT matched-pairs radiometric drift spectrum"}},
    "brdf_single_channel": {"name": "brdf_singlechannel_{wl}nm",
                            "load": _load_brdf_single_channel,
                            "build": figures.brdf_single_channel},
    "site_mean_spectra": {"name": "meanspectra_byPICS",
                          "load": _load_brdf_site_spectra,
                          "build": figures.brdf_site_mean_spectra},
    "temporal_coverage": {"name": "temporal_coverage",
                          "load": _load_temporal_coverage,
                          "build": figures.temporal_coverage},
    "landsat_invariance": {"name": "landsat_invariance_spectra",
                           "load": _load_landsat_invariance,
                           "build": figures.landsat_invariance_spectra},
    "emit_drift_overlay": {"name": "emit_drift_spectrum_overlay",
                           "load": _load_emit_drift_overlay,
                           "build": figures.emit_drift_spectrum_overlay},
    "emit_oli2_bands": {"name": "emit_oli2_band_comparison",
                        "load": _load_emit_oli2_bands,
                        "build": figures.emit_oli2_band_comparison},
    "cos_sza_time_corr": {"name": "cos_sza_time_corr",
                          "load": _load_cos_sza_corr,
                          "build": figures.cos_sza_time_corr},
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
    # fills the ``{wl}`` slot; the only wavelength-tagged figure is the single-channel
    # one, which displays brdf.single_channel_wl (falls back to the NIR target_wl).
    wl_tag = f"{(cfg.brdf_single_channel_wl or cfg.brdf_target_wl):.0f}"
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
