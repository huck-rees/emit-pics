#!/usr/bin/env python
"""Diagnose 10 km vs 20 km representative-patch size for the BRDF drift spectrum.

Supplementary, **read-only** analysis: it compares two BRDF configurations that
were already computed by the ordinary workflow — the main flat ``data/brdf/`` run
(10 km) and a labelled sweep ``data/brdf/<label>/`` (20 km, ``label: "20km"``) —
and never recomputes a drift fit. If the 20 km outputs are absent it exits with a
message rather than doing anything, so the 20 km results must be generated first::

    # 20 km, into its own folder (edit run_config.yaml: 20 km roster, label: "20km")
    python scripts/run_brdf_assemble.py
    python scripts/run_brdf_drift_spectrum.py
    # 10 km, the main/flat run (roster back to 10 km, label unset)
    python scripts/run_brdf_assemble.py
    python scripts/run_brdf_drift_spectrum.py
    # then this diagnostic
    python scripts/run_patch_size_sensitivity.py

Outputs (adjacent to ``data/brdf`` and ``figures/``, NOT variant-nested)
------------------------------------------------------------------------
* ``data/20_km_patch_sensitivity/drift_spectrum_comparison.csv`` — per channel:
  median + 95 % CI drift for both sizes and ``gamma_20 - gamma_10``.
* ``data/20_km_patch_sensitivity/summary_diagnostics.csv`` — per product: median
  CI width (precision), RMS / max difference, significant-channel counts.
* ``data/20_km_patch_sensitivity/leverage_comparison.csv`` — per-site scene counts
  and ``corr(cos SZA, t)`` at each size.
* ``figures/20_km_patch_sensitivity/drift_spectrum_10km_vs_20km.png`` (+ ``.pdf``) —
  two panels (radiance / reflectance): rolling-median-smoothed median drift with a
  semi-transparent 95 % CI band, 10 km vs 20 km overlaid.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (set before viz imports pyplot)
import matplotlib.pyplot as plt

from emit_pics import paths, patch_size_sensitivity as pss
from emit_pics.run_config import load_config
from emit_pics.viz import figures as vizfigures, style

#: Overlay colours (Wong palette): 10 km blue, 20 km orange.
_COLORS = {"10km": "#0072B2", "20km": "#E69F00"}


def _read_config_outputs(label):
    """(drift-spectrum df, leverage df, note) for one BRDF label, or (None, None, why)."""
    paths.set_brdf_label(label)
    spec_path, lev_path = paths.brdf_drift_spectrum(), paths.brdf_leverage()
    if not spec_path.exists():
        return None, None, f"{spec_path} not found"
    lev = pd.read_csv(lev_path) if lev_path.exists() else None
    return pd.read_csv(spec_path), lev, str(spec_path.parent)


def _plot(cmp, *, tag10, tag20, window, target_wl):
    """Two-panel smoothed drift comparison (rad top / rfl bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, prod, ylab in zip(
            axes, ("rad", "rfl"),
            ("Radiance (L1B)\ndrift (%/yr)", "Reflectance (L2A)\ndrift (%/yr)")):
        d = cmp[cmp["product"] == prod].sort_values("wavelength")
        wl = d["wavelength"].to_numpy(float)
        segs = pss.spectrum_segments(wl)   # break at the omitted water-vapor gaps
        lo_all, hi_all = [], []
        for tag in (tag10, tag20):
            # Mask this config's own unstable channels before smoothing.
            ok = ~d[f"unstable_{tag}"].to_numpy(bool)
            gam = np.where(ok, d[f"gamma_pct_yr_{tag}"], np.nan)
            lo = np.where(ok, d[f"ci_lo_pct_yr_{tag}"], np.nan)
            hi = np.where(ok, d[f"ci_hi_pct_yr_{tag}"], np.nan)
            c = _COLORS.get(tag, None)
            labeled = False
            # Smooth/draw each contiguous run on its own so nothing bridges a gap.
            for sl in segs:
                w = wl[sl]
                gam_s = pss.smooth_series(w, gam[sl], window=window)
                lo_s = pss.smooth_series(w, lo[sl], window=window)
                hi_s = pss.smooth_series(w, hi[sl], window=window)
                ax.fill_between(w, lo_s, hi_s, color=c, alpha=0.22, linewidth=0,
                                zorder=2, label=(f"{tag} 95% CI" if not labeled else None))
                ax.plot(w, gam_s, color=c, lw=1.8, alpha=0.95, zorder=3,
                        label=(f"{tag} drift fit" if not labeled else None))
                labeled = True
                lo_all.append(lo_s); hi_all.append(hi_s)
        ax.axhline(0.0, color="k", lw=0.8, zorder=1)
        ax.set_ylabel(ylab)
        # Robust y-limits: trim the wide-CI SWIR tails so the trend stays readable.
        lo_c, hi_c = np.concatenate(lo_all), np.concatenate(hi_all)
        if np.isfinite(lo_c).any():
            ax.set_ylim(np.nanpercentile(lo_c, 2), np.nanpercentile(hi_c, 98))
        ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].set_title(
        f"BRDF drift spectrum — patch-size sensitivity ({tag10} vs {tag20})   "
        f"(rolling-median smoothed, window {window}; band ~{target_wl:.0f} nm fit)",
        fontsize=12)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="10 km vs 20 km BRDF drift-spectrum diagnostic.")
    ap.add_argument("--label-10km", default="", help="BRDF label of the main/flat (10 km) run.")
    ap.add_argument("--label-20km", default="20km", help="BRDF label of the 20 km run.")
    ap.add_argument("--window", type=int, default=11, help="rolling-median smoothing window (channels).")
    ap.add_argument("--alpha", type=float, default=None,
                    help="significance level for the counts (default: drift.alpha from config).")
    args = ap.parse_args(argv)

    cfg = load_config()
    paths.set_variant(cfg.output_variant)
    alpha = cfg.alpha if args.alpha is None else args.alpha
    tag10, tag20 = args.label_10km or "10km", args.label_20km

    spec10, lev10, note10 = _read_config_outputs(args.label_10km)
    if spec10 is None:
        print(f"[abort] 10 km (main/flat) drift spectrum missing: {note10}\n"
              "Run scripts/run_brdf_assemble.py + run_brdf_drift_spectrum.py first.")
        return 1
    spec20, lev20, note20 = _read_config_outputs(args.label_20km)
    if spec20 is None:
        print(f"[abort] 20 km drift spectrum missing: {note20}\n"
              f"Generate it first (set brdf.label: \"{args.label_20km}\" + the 20 km "
              "roster, then run assemble + drift_spectrum). Nothing written.")
        return 1
    print(f"10 km <- {note10}\n20 km <- {note20}")

    cmp = pss.compare_spectra(spec10, spec20, tag10=tag10, tag20=tag20)
    summary = pss.summarize(cmp, tag10=tag10, tag20=tag20, alpha=alpha)
    leverage = (pss.compare_leverage(lev10, lev20, tag10=tag10, tag20=tag20)
                if lev10 is not None and lev20 is not None else None)

    paths.PATCH_SENS_DIR.mkdir(parents=True, exist_ok=True)
    cmp_path = paths.PATCH_SENS_DIR / "drift_spectrum_comparison.csv"
    sum_path = paths.PATCH_SENS_DIR / "summary_diagnostics.csv"
    cmp.to_csv(cmp_path, index=False)
    summary.to_csv(sum_path, index=False)
    print(f"\nPer-channel comparison -> {cmp_path}")
    print(f"Summary diagnostics    -> {sum_path}")
    if leverage is not None:
        lev_path = paths.PATCH_SENS_DIR / "leverage_comparison.csv"
        leverage.to_csv(lev_path, index=False)
        print(f"Leverage comparison    -> {lev_path}")
    else:
        print("[warn] a leverage table was missing; skipped leverage_comparison.csv.")

    print("\nSummary (over channels stable in both configs):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))

    paths.PATCH_SENS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Diagnostic figure: autoscaled, titled, full range — a PNG for internal review
    # (the publication figure below owns the vector PDF/SVG deliverable).
    fig = _plot(cmp, tag10=tag10, tag20=tag20, window=args.window,
                target_wl=cfg.brdf_target_wl)
    fig.savefig(paths.PATCH_SENS_FIG_DIR / "drift_spectrum_10km_vs_20km.png",
                dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic figure  -> {paths.PATCH_SENS_FIG_DIR / 'drift_spectrum_10km_vs_20km.png'}")

    # Publication figure: manuscript theme, no title, +/-3 %/yr, water-vapor masked.
    pub_stem = paths.PATCH_SENS_FIG_DIR / "drift_spectrum_10km_vs_20km_publication"
    with style.theme("manuscript"):
        pubfig = vizfigures.patch_size_drift_spectrum(
            cmp, window=args.window, tag10=tag10, tag20=tag20)
        for ext in ("pdf", "svg", "png"):
            pubfig.savefig(f"{pub_stem}.{ext}", dpi=500)
    plt.close(pubfig)
    print(f"Publication figure -> {pub_stem}.pdf (+ .svg, .png)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
