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


def _load_patch_coverage(cfg):
    from emit_pics.id_granules import load_patches
    from emit_pics.config import pics_geodataframe
    bnds = paths.patch_bnds(cfg.site)
    if not bnds.exists():
        return None, f"{bnds.name} not found — run scripts/run_id_granules.py first."
    patches = load_patches(bnds, cfg.search_patch_size_km)
    return {"patches": patches, "site": pics_geodataframe(cfg.site)}, bnds.name


#: figure key -> {number (manuscript sequence), loader, builder}.
FIGURES = {
    "patch_coverage": {"number": "fig01", "load": _load_patch_coverage,
                       "build": figures.patch_coverage},
    "drift_spectrum": {"number": "fig03", "load": _load_drift_spectrum,
                       "build": figures.drift_spectrum},
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
    themes = ["manuscript", "ppt"] if args.theme == "both" else [args.theme]
    n_written = 0
    for key in args.figures:
        spec = FIGURES[key]
        kwargs, note = spec["load"](cfg)
        if kwargs is None:
            print(f"[skip] {key}: {note}")
            continue
        # Pass CLI overrides only to builders that accept them.
        params = inspect.signature(spec["build"]).parameters
        if args.annotate is not None and "annotate" in params:
            kwargs["annotate"] = args.annotate
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
