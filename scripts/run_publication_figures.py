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
import sys

import pandas as pd

from emit_pics import paths
from emit_pics.viz import figures, style

#: figure key -> {number (manuscript sequence), csv path fn, builder, width class}.
FIGURES = {
    "drift_spectrum": {
        "number": "fig03",
        "csv": paths.brdf_drift_spectrum,
        "build": figures.drift_spectrum,
        "size": "double",
    },
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
    ap.add_argument("--list", action="store_true", help="list known figures and exit.")
    args = ap.parse_args(argv)

    if args.list:
        for key, spec in FIGURES.items():
            print(f"  {spec['number']:<7} {key:<18} <- {spec['csv']().name}")
        return 0

    themes = ["manuscript", "ppt"] if args.theme == "both" else [args.theme]
    n_written = 0
    for key in args.figures:
        spec = FIGURES[key]
        csv = spec["csv"]()
        if not csv.exists():
            print(f"[skip] {key}: {csv} not found — run the pipeline first.")
            continue
        df = pd.read_csv(csv)
        for th in themes:
            with style.theme(th):
                fig = spec["build"](df, size=spec["size"])
                written = style.save(fig, _name(key, spec["number"], th), out_dir=args.out)
            for p in written:
                print(f"  {key} [{th}] -> {p}")
                n_written += 1

    print(f"\nWrote {n_written} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
