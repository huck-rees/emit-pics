"""Publication figure builders — pure ``(data) -> Figure``.

Each function reads already-generated pipeline outputs and returns a Matplotlib
figure. Builders run under an active :mod:`emit_pics.viz.style` theme (which sets
the font family + output format); size-sensitive builders also read the theme's
width presets and scale, while a dense reference figure may fix its own generous
size so all its labels stay legible.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from emit_pics.viz import style


def drift_spectrum(df, *, size="double", aspect=0.60, sig_col="sig_raw",
                   sig_label="significant (95% CI excludes 0)", corner_label=None,
                   ylim=(-2.5, 2.5), title="EMIT pooled radiometric drift spectrum"):
    """BRDF pooled drift spectrum gamma(lambda): rad (top) / rfl (bottom) vs wavelength.

    Reads ``data/brdf/brdf_drift_spectrum.csv``. Numerically unstable channels are
    dropped; the bootstrap 95% CI is drawn as whiskers, and channels significant
    by *sig_col* (the 95% CI excludes zero — i.e. the drawn whisker clears zero)
    are filled in the significance colour while the rest are open grey.

    Parameters
    ----------
    df : pandas.DataFrame
        The drift-spectrum table (columns per the data dictionary).
    size : str
        Width class resolved by the active theme (``single`` / ``onehalf`` / ``double``).
    aspect : float
        Height / width of the whole figure.
    sig_col : str
        Significance column to colour by (``sig_raw`` for the pooled BRDF table,
        ``raw_significant`` for the matched-pairs table).
    sig_label : str
        Legend label for the filled significant points (match it to *sig_col*).
    corner_label : str or None
        If given, drawn bold in the top-left of the radiance (top) panel — used by
        the leave-one-out appendix to tag each panel with its site combo (e.g.
        ``"A3, A5, L1, L4"``). ``None`` (the default) leaves the primary figure
        untouched.
    ylim : tuple or None
        Shared y-axis limits (% yr$^{-1}$) for both panels; ``None`` autoscales. Points
        or CI whiskers outside the range are clipped to the fixed scale.
    title : str
        On-figure title (slide theme only; the manuscript caption is in LaTeX).
    """
    df = df[~df["unstable"]].copy()
    sc = style.scale()
    fig, axes = plt.subplots(2, 1, figsize=style.figsize(size, aspect),
                             sharex=True, layout="constrained")

    for ax, prod, ylab in zip(
            axes, ("rad", "rfl"),
            ("Radiance drift (% yr$^{-1}$)", "Reflectance drift (% yr$^{-1}$)")):
        d = df[df["product"] == prod].sort_values("wavelength")
        yerr = np.vstack([d["gamma_pct_yr"] - d["ci_lo_pct_yr"],
                          d["ci_hi_pct_yr"] - d["gamma_pct_yr"]])
        ax.errorbar(d["wavelength"], d["gamma_pct_yr"], yerr=yerr, fmt="none",
                    ecolor="0.7", elinewidth=0.6 * sc, capsize=1.2 * sc, zorder=1)
        # Same significance styling in BOTH panels so the single legend matches:
        # not-significant = open grey circle; significant = filled SIG_COLOR circle.
        ns, sg = d[~d[sig_col]], d[d[sig_col]]
        ax.scatter(ns["wavelength"], ns["gamma_pct_yr"], s=9 * sc, marker="o",
                   facecolors="none", edgecolors=style.NS_EDGE, linewidths=0.5 * sc,
                   zorder=3)
        ax.scatter(sg["wavelength"], sg["gamma_pct_yr"], s=11 * sc, marker="o",
                   color=style.SIG_COLOR, zorder=4)
        ax.axhline(0.0, color="k", lw=0.6 * sc, zorder=2)
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[1].set_xlabel("Wavelength (nm)")
    handles = [
        Line2D([0], [0], marker="o", color=style.NS_EDGE, ls="none", mfc="none",
               label="not significant"),
        Line2D([0], [0], marker="o", color=style.SIG_COLOR, ls="none",
               label=sig_label),
        Line2D([0], [0], marker="|", color="0.7", ls="none",
               markersize=7 * sc, markeredgewidth=0.9 * sc, label="95% CI"),
    ]
    # Semi-transparent box, top-left of the reflectance (bottom) panel.
    axes[1].legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.8, edgecolor="0.7")
    if corner_label:
        axes[0].text(0.02, 0.94, corner_label, transform=axes[0].transAxes,
                     va="top", ha="left", fontsize=8 * sc, fontweight="bold",
                     zorder=6)
    if style.show_title():
        axes[0].set_title(title)
    return fig


def patch_size_drift_spectrum(cmp, *, window=11, tag10="10km", tag20="20km",
                              size="double", aspect=0.62, ylim=(-3.0, 3.0),
                              colors=None):
    """Publication 10 km vs 20 km BRDF drift-spectrum comparison (rad top / rfl bottom).

    Reads the per-channel comparison table from
    ``data/20_km_patch_sensitivity/drift_spectrum_comparison.csv`` (written by
    ``scripts/run_patch_size_sensitivity.py``). This is the ordinary pooled
    :func:`drift_spectrum` rendered for two configurations at once, with the
    dot-and-whisker markers turned into a **line** (the per-channel drift fit
    ``gamma_pct_yr`` — the original least-squares point estimate, identical to the
    primary figure's markers, only smoothed across wavelength by a rolling median
    for legibility) and a semi-transparent 95 % CI **band** (the bootstrap
    percentile CI, likewise smoothed), one colour per size. The central line is
    NOT a bootstrap median. **No title** is drawn; the y-axis is fixed to *ylim*.

    The excluded water-vapor channels are already absent from the source drift
    spectrum (they are simply not fit), so they appear as wide gaps in the ~7 nm
    grid. The series is split at those gaps (:func:`patch_size_sensitivity.
    spectrum_segments`) and each contiguous run is smoothed and drawn on its own,
    so the line and CI band break exactly where channels are missing — the data's
    own missingness is the exclusion window; nothing is shaded or hand-set.

    Parameters
    ----------
    cmp : pandas.DataFrame
        The comparison table: needs ``product``, ``wavelength`` and, per config,
        ``gamma_pct_yr_<tag>``, ``ci_{lo,hi}_pct_yr_<tag>``, ``unstable_<tag>``.
    window : int
        Rolling-median smoothing window (channels); matches the diagnostic figure.
    tag10, tag20 : str
        Column suffixes / legend labels of the two configs (``"10km"``/``"20km"``).
    size, aspect : str, float
        Theme width class and height/width.
    ylim : tuple or None
        Fixed y-limits (% yr$^{-1}$) for both panels (default +/-3).
    colors : dict or None
        ``{tag: colour}`` override; defaults to Wong blue (10 km) / orange (20 km).
    """
    from emit_pics.patch_size_sensitivity import smooth_series, spectrum_segments
    sc = style.scale()
    colors = colors or {tag10: "#0072B2", tag20: "#E69F00"}

    fig, axes = plt.subplots(2, 1, figsize=style.figsize(size, aspect),
                             sharex=True, layout="constrained")
    for ax, prod, ylab in zip(
            axes, ("rad", "rfl"),
            ("Radiance drift (% yr$^{-1}$)", "Reflectance drift (% yr$^{-1}$)")):
        d = cmp[cmp["product"] == prod].sort_values("wavelength")
        wl = d["wavelength"].to_numpy(float)
        segs = spectrum_segments(wl)   # break at the omitted water-vapor gaps
        for tag in (tag10, tag20):
            c = colors[tag]
            ok = ~d[f"unstable_{tag}"].to_numpy(bool)
            gam = np.where(ok, d[f"gamma_pct_yr_{tag}"], np.nan)
            lo_ = np.where(ok, d[f"ci_lo_pct_yr_{tag}"], np.nan)
            hi_ = np.where(ok, d[f"ci_hi_pct_yr_{tag}"], np.nan)
            labeled = False
            # Smooth and draw each contiguous run separately so neither the line
            # nor the CI band bridges a missing (absorption) region.
            for sl in segs:
                w = wl[sl]
                gs = smooth_series(w, gam[sl], window=window)
                ls = smooth_series(w, lo_[sl], window=window)
                hs = smooth_series(w, hi_[sl], window=window)
                ax.fill_between(w, ls, hs, color=c, alpha=0.22, linewidth=0, zorder=2)
                ax.plot(w, gs, color=c, lw=1.4 * sc, alpha=0.95, zorder=3,
                        label=(None if labeled else
                               f"{tag.replace('km', ' km')} (drift fit + 95% CI)"))
                labeled = True
        ax.axhline(0.0, color="k", lw=0.6 * sc, zorder=1)
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[1].set_xlabel("Wavelength (nm)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="0.7")
    return fig


#: Line styles cycled with the site colours so the 5 PICS stay distinguishable in
#: greyscale / colour-blind viewing (redundant encoding, per the RSE spec).
_SPECTRA_LS = ["-", "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1))]


def brdf_site_mean_spectra(spectra, *, landsat_spectra=None, size="double",
                           aspect=None, figsize=None,
                           rad_label="EMIT mean TOA radiance,\n$L$ (µW/cm²/sr/nm)",
                           rfl_label="EMIT mean surface\nreflectance, $\\rho$ (unitless)",
                           toa_label="Landsat 9 OLI-2 mean TOA\nreflectance, $\\rho$ (unitless)",
                           title="Mean spectra by PICS (10 km BRDF patch)"):
    """Stacked mean-spectra panels, one line per PICS, in stable per-site colours.

    Reads the assembled EMIT per-site mean spectra from
    :func:`emit_pics.brdf_drift.site_mean_spectra`: ``{pics_id: {"wl", "mean_rad",
    "mean_rfl", "n", "patch"}}``. Each PICS is a single line (the mean over that
    representative patch's QC-surviving scenes) in its stable Wong colour
    (:func:`style.site_styles`) plus a cycled line style for greyscale safety. The
    EMIT lines break across the deep water-vapor bands so the panels exclude the
    SAME channels the drift figures do.

    If *landsat_spectra* is given (same structure, with ``mean_toa`` over the seven
    OLI-2 bands, from :func:`emit_pics.landsat_brdf.landsat_site_mean_spectra`), a
    Landsat TOA-reflectance panel is added on top, giving three panels: (a) Landsat
    9 TOA reflectance, (b) EMIT radiance, (c) EMIT surface reflectance. Landsat is
    drawn as markers (seven discrete bands) with a thin guide line, so a reader does
    not read it as continuous. Without *landsat_spectra* the original two-panel
    EMIT figure is produced.

    Parameters
    ----------
    spectra : dict
        EMIT ``{pics_id: {"wl", "mean_rad", "mean_rfl": array, "n": int, "patch"}}``.
    landsat_spectra : dict or None
        Landsat ``{pics_id: {"wl", "mean_toa": array, "n": int, "patch"}}`` or None.
    size, aspect : str, float or None
        Theme width class and height/width (``None`` picks a taller value for the
        three-panel layout).
    rad_label, rfl_label, toa_label : str
        Y-axis labels.
    title : str
        On-figure title (slide theme only; the manuscript caption is in LaTeX).
    """
    from emit_pics.delta_calculations import ABSORPTION_BANDS_NM
    sc = style.scale()
    sites = sorted(spectra)
    styles = style.site_styles(sites)

    # Panels top-to-bottom: optional Landsat TOA, then EMIT radiance, EMIT reflectance.
    # Each tuple: (kind, data-key, y-label, data-dict, legend-location).
    panels = []
    if landsat_spectra is not None:
        panels.append(("toa", "mean_toa", toa_label, landsat_spectra, "lower right"))
    panels.append(("emit", "mean_rad", rad_label, spectra, "upper right"))
    panels.append(("emit", "mean_rfl", rfl_label, spectra, "lower right"))
    nrows = len(panels)
    if figsize is None:
        if nrows == 3:
            figsize = (6.5, 7.0)   # manuscript: 6.5 in wide, fits page height with caption
        else:
            figsize = style.figsize(size, 0.9 if aspect is None else aspect)

    fig, axes = plt.subplots(nrows, 1, figsize=figsize,
                             sharex=True, layout="constrained")
    if nrows == 1:
        axes = [axes]
    for ax, (kind, key, ylab, data, legend_loc) in zip(axes, panels):
        for i, s in enumerate(sites):
            if s not in data:
                continue
            d = data[s]
            wl = np.asarray(d["wl"], float)
            y = np.asarray(d[key], float).copy()
            if kind == "emit":
                # Break across the deep water-vapor windows (L2A is fill there;
                # radiance matched so both panels exclude the same channels).
                for lo, hi in ABSORPTION_BANDS_NM:
                    y[(wl >= lo) & (wl <= hi)] = np.nan
                ax.plot(wl, y, color=styles[s]["color"],
                        ls=_SPECTRA_LS[i % len(_SPECTRA_LS)], lw=1.0 * sc, alpha=0.9)
            else:
                # Landsat: seven discrete bands as markers plus a thin guide line.
                ax.plot(wl, y, color=styles[s]["color"],
                        ls=_SPECTRA_LS[i % len(_SPECTRA_LS)], lw=0.7 * sc, alpha=0.5,
                        marker="o", ms=4.0 * sc, mec="none")
        ax.set_ylabel(ylab)
        ax.margins(x=0.01)
        # Per-panel legend with that panel's own scene counts (n differs by sensor).
        handles = [Line2D([0], [0], color=styles[s]["color"],
                          ls=_SPECTRA_LS[i % len(_SPECTRA_LS)], lw=1.3 * sc,
                          marker=("o" if kind == "toa" else None),
                          label=f"{s.replace('_', ' ')} (n = {data[s]['n']})")
                   for i, s in enumerate(sites) if s in data]
        ax.legend(handles=handles, loc=legend_loc, frameon=True,
                  framealpha=0.85, edgecolor="0.7")

    axes[-1].set_xlabel("Wavelength (nm)")
    if style.show_title():
        axes[0].set_title(title)
    return fig


def persite_gamma_spectrum(persite_spec, *, size="double", aspect=0.75, ylim=None,
                           rad_ylab="Radiance per-site drift (% yr$^{-1}$)",
                           rfl_ylab="Reflectance per-site drift (% yr$^{-1}$)",
                           title="Per-site standalone drift spectrum"):
    """Per-site standalone ``gamma(lambda)`` as points — one marker per PICS per channel.

    Reads the tidy per-site gamma spectrum from
    :func:`emit_pics.brdf_drift.persite_gamma_spectrum` (columns ``product``,
    ``site``, ``wavelength``, ``gamma_pct_yr``). Each PICS is drawn as **point
    markers only** (no connecting line), so nothing interpolates across the excluded
    water-vapor bands — those channels simply have no points. Colour **and** marker
    encode the PICS (redundant, greyscale/colour-blind safe). Radiance (top) /
    reflectance (bottom) share **one** legend; no on-figure title in the manuscript
    theme (the caption is in LaTeX).

    Parameters
    ----------
    persite_spec : pandas.DataFrame
        Long per-site gamma spectrum (``product``, ``site``, ``wavelength``,
        ``gamma_pct_yr``); degenerate fits should already be NaN.
    size, aspect : str, float
        Theme width class and height/width.
    ylim : tuple or None
        Fixed y-limits (% yr$^{-1}$) for both panels; ``None`` uses a robust 1–99
        percentile window per panel so a few SWIR outliers don't swamp the scale.
    """
    sc = style.scale()
    sites = sorted(persite_spec["site"].unique())
    styles = style.site_styles(sites)

    fig, axes = plt.subplots(2, 1, figsize=style.figsize(size, aspect),
                             sharex=True, layout="constrained")
    for ax, prod, ylab in zip(axes, ("rad", "rfl"), (rad_ylab, rfl_ylab)):
        d = persite_spec[persite_spec["product"] == prod]
        for s in sites:
            ds = d[d["site"] == s]
            ax.scatter(ds["wavelength"], ds["gamma_pct_yr"], s=8 * sc,
                       marker=styles[s]["marker"], color=styles[s]["color"],
                       edgecolors="none", alpha=0.8, zorder=3)
        ax.axhline(0.0, color="k", lw=0.6 * sc, zorder=2)
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            v = d["gamma_pct_yr"].to_numpy(float)
            if np.isfinite(v).any():
                lo, hi = np.nanpercentile(v, [1, 99])
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                ax.set_ylim(lo - pad, hi + pad)

    axes[1].set_xlabel("Wavelength (nm)")
    handles = [Line2D([0], [0], marker=styles[s]["marker"], ls="none",
                      color=styles[s]["color"], markersize=5 * sc,
                      label=s.replace("_", " ")) for s in sites]
    axes[0].legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.85, edgecolor="0.7", ncol=len(sites),
                   columnspacing=1.0, handletextpad=0.3)
    if style.show_title():
        axes[0].set_title(title)
    return fig


def loo_gamma_overlay(df, *, size="double", aspect=0.75, ylim=(-2.5, 2.5),
                      rad_ylab="Radiance drift (% yr$^{-1}$)",
                      rfl_ylab="Reflectance drift (% yr$^{-1}$)",
                      title="Leave-one-out drift spectra (all combos)"):
    """All leave-one-out combos' ``gamma(lambda)`` overlaid — rad (top) / rfl (bottom).

    Reads the tidy leave-one-out table from
    :func:`emit_pics.brdf_loo.leave_one_out_drift_spectra` (columns ``combo``,
    ``product``, ``wavelength``, ``gamma_pct_yr``, ``unstable``). Each combo is
    drawn as **point markers only** (no connecting line, so nothing interpolates
    across the excluded water-vapor bands) in a unique colour **and** marker
    (:func:`style.site_styles` over the combo labels — Wong palette, redundant
    colour/marker encoding). Degenerate (``unstable``) channels are dropped. No CI
    is drawn (the per-combo figures carry the uncertainty); this figure is the
    side-by-side comparison of the five point estimates.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy leave-one-out spectrum table (all combos, both products).
    size, aspect : str, float
        Theme width class and height/width.
    ylim : tuple or None
        Shared y-limits (% yr$^{-1}$) for both panels; ``None`` uses a robust 1–99
        percentile window per panel.
    rad_ylab, rfl_ylab : str
        Y-axis labels for the two panels.
    title : str
        On-figure title (slide theme only; the manuscript caption is in LaTeX).
    """
    sc = style.scale()
    d = df[~df["unstable"]]
    combos = sorted(d["combo"].unique())
    styles = style.site_styles(combos)

    fig, axes = plt.subplots(2, 1, figsize=style.figsize(size, aspect),
                             sharex=True, layout="constrained")
    for ax, prod, ylab in zip(axes, ("rad", "rfl"), (rad_ylab, rfl_ylab)):
        dp = d[d["product"] == prod]
        for c in combos:
            dc = dp[dp["combo"] == c].sort_values("wavelength")
            ax.scatter(dc["wavelength"], dc["gamma_pct_yr"], s=8 * sc,
                       marker=styles[c]["marker"], color=styles[c]["color"],
                       edgecolors="none", alpha=0.8, zorder=3)
        ax.axhline(0.0, color="k", lw=0.6 * sc, zorder=2)
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            v = dp["gamma_pct_yr"].to_numpy(float)
            if np.isfinite(v).any():
                lo, hi = np.nanpercentile(v, [1, 99])
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                ax.set_ylim(lo - pad, hi + pad)

    axes[1].set_xlabel("Wavelength (nm)")
    handles = [Line2D([0], [0], marker=styles[c]["marker"], ls="none",
                      color=styles[c]["color"], markersize=5 * sc, label=c)
               for c in combos]
    axes[0].legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.85, edgecolor="0.7", ncol=1,
                   title="combo (sites kept)", title_fontsize=6.5 * sc)
    if style.show_title():
        axes[0].set_title(title)
    return fig


def cos_sza_time_corr(emit_obs, emit_corr, ls_obs=None, ls_corr=None, *,
                      order=None, T0=None, fig_w=6.5, fig_h=3.5,
                      title="Solar geometry vs time by PICS"):
    """Two-row ``cos(theta)``-vs-``t`` scatter: Landsat 9 over EMIT, by PICS.

    Columns are PICS in a fixed order. The **top** row is the Landsat 9 OLI-2 BRDF
    scenes (all PICS); the **bottom** row is the EMIT BRDF scenes (only the invariant
    PICS used in the drift fit, so an excluded site's EMIT cell is left blank). Each
    panel scatters ``cos(SZA)`` against ``t`` in that site's stable Wong colour and
    marker (:func:`style.site_styles`, edged black), prints the Spearman rank
    correlation ``rho`` and its ``p``-value (no box), and carries a bold panel letter
    top-left. No trend line is drawn (the statistic is rank-based). All panels share
    one ``t`` axis and one ``cos(theta)`` axis so the sites read directly.

    Parameters
    ----------
    emit_obs, ls_obs : pandas.DataFrame
        Per-scene design tables with ``site``, ``g`` (= ``cos(SZA)``), ``t`` (years
        since *T0*). ``ls_obs`` may be None (then the top row is blank).
    emit_corr, ls_corr : pandas.DataFrame
        Per-site Spearman tables (columns ``site``, ``spearman_rho``, ``p_value``).
    order : list of str or None
        Column (PICS) order; defaults to the sorted union of sites present.
    T0 : pandas.Timestamp or None
        Common time origin, for the x-axis label.
    fig_w, fig_h : float
        Figure size in inches (publication spec).
    title : str
        On-figure suptitle (slide theme only; the manuscript caption is in LaTeX).
    """
    from matplotlib.gridspec import GridSpec
    sc = style.scale()

    allsites = set(emit_obs["site"].unique())
    ls_sites = set(ls_obs["site"].unique()) if ls_obs is not None else set()
    allsites |= ls_sites
    order = [s for s in (order or sorted(allsites)) if s in allsites]
    ncol = len(order)
    styles = style.site_styles(order)
    emit_sites = set(emit_obs["site"].unique())

    estats = {r["site"]: (r["spearman_rho"], r["p_value"]) for _, r in emit_corr.iterrows()}
    lstats = ({r["site"]: (r["spearman_rho"], r["p_value"]) for _, r in ls_corr.iterrows()}
              if ls_corr is not None else {})

    # Shared t / cos(theta) ranges across both sensors.
    frames = ([ls_obs] if ls_obs is not None else []) + [emit_obs]
    allt = np.concatenate([f["t"].to_numpy(float) for f in frames])
    allg = np.concatenate([f["g"].to_numpy(float) for f in frames])
    tpad = 0.03 * (np.ptp(allt) or 1.0); gpad = 0.06 * (np.ptp(allg) or 1.0)
    xlim = (allt.min() - tpad, allt.max() + tpad)
    ylim = (allg.min() - gpad, allg.max() + gpad)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, ncol, figure=fig, left=0.12, right=0.995, top=0.86, bottom=0.145,
                  wspace=0.09, hspace=0.26)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(2)]

    # (row tag, per-site obs, per-site stats, left-edge row label).
    rows = [(ls_obs, ls_sites, lstats, "Landsat 9"),
            (emit_obs, emit_sites, estats, "EMIT")]
    filled = [[ (ls_obs is not None and s in ls_sites) for s in order],
              [ s in emit_sites for s in order]]

    letter = 0
    for ri, (dfobs, rsites, stats, _rowlab) in enumerate(rows):
        for c, s in enumerate(order):
            ax = axes[ri][c]
            if not filled[ri][c]:
                ax.set_axis_off()
                continue
            sub = dfobs[dfobs["site"] == s]
            st = styles[s]
            ax.scatter(sub["t"], sub["g"], s=13 * sc, color=st["color"],
                       marker=st["marker"], edgecolors="k", linewidths=0.3 * sc,
                       alpha=0.85, zorder=3)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            rho, p = stats.get(s, (np.nan, np.nan))
            ax.text(0.95, 0.05, f"$\\rho$={rho:+.2f}\n$p$={p:.2f}",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=6.0 * sc, family="monospace", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.55, edgecolor="none",
                              boxstyle="round,pad=0.15"))
            ax.text(0.05, 0.95, f"{chr(ord('a') + letter)})", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8 * sc, fontweight="bold", zorder=6)
            letter += 1
            if ri == 0:
                ax.set_title(str(s).replace("_", " "), fontsize=7.5 * sc, pad=3 * sc)

    # y-label on the left column; x-label on the lowest filled panel of each column.
    xlab = f"t (yr since {pd.Timestamp(T0).date()})" if T0 is not None else "t (yr)"
    for c in range(ncol):
        br = 1 if filled[1][c] else 0
        for r in (0, 1):
            if not filled[r][c]:
                continue
            ax = axes[r][c]
            ax.tick_params(labelsize=6.5 * sc, length=2 * sc)
            ax.tick_params(labelleft=(c == 0), labelbottom=(r == br))
            if c == 0:
                ax.set_ylabel(r"$\cos\,\theta$", fontsize=8 * sc)
        axes[br][c].set_xlabel(xlab, fontsize=7.5 * sc)

    # Bold sensor labels down the far left, one per row.
    fig.text(0.022, 0.70, "Landsat 9", rotation=90, va="center", ha="center",
             fontsize=8 * sc, fontweight="bold")
    fig.text(0.022, 0.30, "EMIT", rotation=90, va="center", ha="center",
             fontsize=8 * sc, fontweight="bold")

    if style.show_title():
        fig.suptitle(title)
    return fig


def _norm_site(s):
    """Normalize a site id for matching (case/space/underscore-insensitive)."""
    return str(s).strip().lower().replace(" ", "_")


def brdf_cos_sza(design, *, landsat_design=None, site="Libya_4", fig_width=6.5,
                 aspect=0.42, target_wl=None, suptitle=None):
    """Fig. 3: in-patch spatial median vs cos(SZA), per product (Landsat + EMIT).

    Motivates the quasi-BRDF correction: at a fixed target band, both products vary
    with solar-illumination angle (radiance because incident flux scales with
    ``cos(SZA)``, reflectance through residual surface BRDF). Reads the pooled BRDF
    design table (``data/brdf/brdf_design.csv``: per-scene ``g = cos(SZA)`` and the
    in-patch median ``rho_rad`` / ``rho_rfl``). A single OLS line is fit per panel,
    with its statistics ($R^2$, slope $m$, $p$, $n$) printed top-left.

    By default the figure is restricted to **one PICS** (*site*, default Libya 4) so
    the relationship is a clean single-surface fit; pass ``site=None`` to pool every
    site (coloured by PICS) instead. Total width is fixed at *fig_width* inches
    (default 6.5) to meet the publication spec; fonts follow the active theme.

    Parameters
    ----------
    design : pandas.DataFrame
        BRDF design table with ``g``, ``rho_rad``, ``rho_rfl`` (and ``site``).
    site : str or None
        Restrict to this PICS (matched case/space/underscore-insensitively).
        ``None`` pools all sites and colours points by PICS.
    fig_width : float
        Total figure width in inches (default 6.5).
    aspect : float
        Height / width of the whole figure.
    target_wl : float or None
        Target band (nm) for the on-figure title (slide theme only).
    suptitle : str or None
        Title (drawn only under a title-showing theme).
    """
    from scipy.stats import linregress

    sc = style.scale()
    pooled = site is None or "site" not in design.columns
    if not pooled:
        design = design[design["site"].map(_norm_site) == _norm_site(site)]
        site_label = str(site).replace("_", " ")
    styles = style.site_styles(design["site"].unique()) if pooled else None

    # Panels left-to-right: optional Landsat TOA (a), then EMIT radiance, reflectance.
    panels = []
    if landsat_design is not None and not pooled:
        panels.append(("toa", landsat_design, "rho_toa",
                       "Landsat 9 OLI-2 median TOA\nreflectance, $\\rho$ (unitless)"))
    panels.append(("rad", design, "rho_rad",
                   "EMIT median TOA radiance,\n$L$ (µW/cm²/sr/nm)"))
    panels.append(("rfl", design, "rho_rfl",
                   "EMIT median surface\nreflectance, $\\rho$ (unitless)"))
    ncols = len(panels)

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_width * aspect),
                             layout="constrained")
    if ncols == 1:
        axes = [axes]

    for ax, (key, dsrc, col, ylab) in zip(axes, panels):
        d = dsrc[np.isfinite(dsrc["g"]) & np.isfinite(dsrc[col])]
        x = d["g"].to_numpy(dtype=float)
        y = d[col].to_numpy(dtype=float)

        if pooled:
            for s, gg in d.groupby("site"):
                st = styles[s]
                ax.scatter(gg["g"], gg[col], s=11 * sc, color=st["color"],
                           marker=st["marker"], edgecolors="k", linewidths=0.3 * sc,
                           alpha=0.85, zorder=3, label=str(s))
        else:
            # Landsat (toa) panel: the reddish-purple triangles used in the band
            # comparison figure, so it is not confused with EMIT reflectance.
            pc = ({"color": "#CC79A7", "marker": "^"} if key == "toa"
                  else style.PRODUCT_STYLE[key])
            ax.scatter(x, y, s=13 * sc, color=pc["color"], marker=pc["marker"],
                       edgecolors="k", linewidths=0.3 * sc, alpha=0.85, zorder=3)

        # One OLS line per panel + a top-left stats box (R², slope, p, n).
        # Site name is given in the caption, so it is omitted here as redundant.
        header = ""
        if x.size >= 3 and np.ptp(x) > 0:
            lr = linregress(x, y)
            xr = np.array([x.min(), x.max()])
            ax.plot(xr, lr.intercept + lr.slope * xr, color="k", lw=1.4 * sc, zorder=4)
            pstr = "p < 0.001" if lr.pvalue < 1e-3 else f"p = {lr.pvalue:.3f}"
            txt = (f"{header}$R^2$ = {lr.rvalue**2:.3f}\n"
                   f"m = {lr.slope:.3g}\n"
                   f"{pstr}\n"
                   f"n = {x.size}")
        else:
            txt = f"{header}n = {x.size}\n(insufficient for fit)"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7 * sc, family="monospace", zorder=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        ax.set_xlabel("cos(SZA)")
        ax.set_ylabel(ylab)

    if pooled:
        axes[0].legend(loc="lower right", fontsize=6.5 * sc, framealpha=0.9,
                       edgecolor="0.7", title="PICS", title_fontsize=6.5 * sc)
    if style.show_title():
        subj = "all PICS" if pooled else site_label
        t = suptitle or (f"Illumination dependence — {subj}"
                         + (f", band ≈ {target_wl:.0f} nm" if target_wl else ""))
        fig.suptitle(t, fontsize=12 * sc, fontweight="bold")
    return fig


def brdf_single_channel(joint, boots, T0, *, target_wl=None,
                        size="double", aspect=0.74):
    """Single-channel quasi-BRDF result: corrected collapse + pooled-γ bootstrap.

    A publication-ready recomposition of the two diagnostics the BRDF drift step
    emits (``brdf_corrected_{wl}nm.png`` and ``brdf_bootstrap_{wl}nm.png``) into
    one three-panel figure at the channel nearest ``brdf.target_wl``:

    * **top row** — per-site *corrected* collapse for radiance (A) and reflectance
      (B): each scene's ``rho`` divided by its own ``(alpha_s + beta_s cos theta)``
      so every site falls onto the shared ``1 + gamma t`` line (drawn black); the
      per-panel γ is printed bottom-left.
    * **bottom** — the pooled shared-γ bootstrap histogram (rad + rfl overlaid),
      with each product's point estimate and the zero line marked.

    The fit inputs (``joint``, ``boots``) come precomputed from the runner's loader
    (``emit_pics.brdf_drift`` on the pooled design table, fixed seeds — identical to
    the BRDF step), so this builder stays a pure ``(data) -> Figure``.

    Parameters
    ----------
    joint : dict
        ``{"rad": fit, "rfl": fit}`` from :func:`emit_pics.brdf_drift.fit_joint`;
        each ``fit`` carries ``sites, alpha, beta, gamma, sidx, labels, g, t, rho``.
    boots : dict
        ``{"rad": b, "rfl": b}`` from :func:`emit_pics.brdf_drift.bootstrap_joint_gamma`;
        each ``b`` carries the shared-γ ``draws`` array (fraction/yr).
    T0 : pandas.Timestamp
        Common time origin (earliest scene) — labels the corrected panels' x-axis.
    target_wl : float or None
        Target band (nm); shown in the slide-theme title only.
    size : str
        Width class resolved by the active theme (default ``double``).
    aspect : float
        Height / width of the whole figure.
    """
    sc = style.scale()
    all_sites = sorted(set(joint["rad"]["sites"]) | set(joint["rfl"]["sites"]))
    styles = style.site_styles(all_sites)

    fig, axd = plt.subplot_mosaic([["rad", "rfl"], ["boot", "boot"]],
                                  figsize=style.figsize(size, aspect),
                                  layout="constrained")

    # Product identifiers (in-panel, top-right) and y-axis labels: radiance uses the
    # radiance symbol L, reflectance uses rho. Reused for the bootstrap legend below.
    PROD_LABEL = {"rad": r"Radiance, $L$", "rfl": r"Reflectance, $\rho$"}
    YLAB = {"rad": r"$L\,/\,(\alpha_s + \beta_s\,\cos\theta)$",
            "rfl": r"$\rho\,/\,(\alpha_s + \beta_s\,\cos\theta)$"}

    # Top row: per-site corrected collapse onto the shared drift line.
    for key in ("rad", "rfl"):
        ax = axd[key]
        r = joint[key]
        a = r["alpha"][r["sidx"]]
        b = r["beta"][r["sidx"]]
        corrected = r["rho"] / (a + b * r["g"])
        for s in r["sites"]:
            m = r["labels"] == s
            st = styles[s]
            ax.scatter(r["t"][m], corrected[m], s=16 * sc, color=st["color"],
                       marker=st["marker"], edgecolors="k", linewidths=0.3 * sc,
                       alpha=0.85, zorder=3)
        tg = np.linspace(float(r["t"].min()), float(r["t"].max()), 60)
        ax.plot(tg, 1 + r["gamma"] * tg, color="k", lw=1.6 * sc, zorder=4,
                label=f"$1+\\gamma t$ ({100 * r['gamma']:+.3f} % yr$^{{-1}}$)")
        ax.axhline(1.0, color="k", lw=0.6 * sc, ls=":", zorder=2)
        ax.text(0.03, 0.05, f"$\\gamma$ = {100 * r['gamma']:+.3f} % yr$^{{-1}}$",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=7 * sc,
                family="monospace", zorder=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlabel(f"t (yr since {T0.date()})")
        ax.set_ylabel(YLAB[key])
        # In-panel product label at top-right, replacing the old axes title.
        ax.text(0.97, 0.97, PROD_LABEL[key], transform=ax.transAxes, va="top",
                ha="right", fontsize=8 * sc, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # One shared PICS colour/marker legend, in the radiance panel. Kept at upper-left
    # so it clears the new top-right product label.
    site_handles = [Line2D([], [], marker=styles[s]["marker"], color=styles[s]["color"],
                           ls="", markeredgecolor="k", markersize=5 * sc,
                           label=str(s).replace("_", " ")) for s in all_sites]
    axd["rad"].legend(handles=site_handles, loc="upper left", ncol=1,
                      fontsize=6.5 * sc, frameon=True, framealpha=0.9, edgecolor="0.7")

    # Bottom row: pooled shared-γ bootstrap, both products overlaid. Colour is
    # colour-blind-safe (Wong blue vs green); a distinct hatch + black outline per
    # product adds a redundant, greyscale-safe cue where the two histograms overlap.
    axb = axd["boot"]
    hatches = {"rad": "///", "rfl": "\\\\"}
    for p in ("rad", "rfl"):
        pc = style.PRODUCT_STYLE[p]
        axb.hist(100 * boots[p]["draws"], bins=40, color=pc["color"], alpha=0.5,
                 edgecolor="k", linewidth=0.5 * sc, hatch=hatches[p],
                 label=PROD_LABEL[p])
        axb.axvline(100 * joint[p]["gamma"], color=pc["color"], lw=1.4 * sc, zorder=4)
    axb.axvline(0.0, color="k", lw=0.8 * sc, ls="--", zorder=3)
    axb.set_xlabel("Shared drift γ (% yr$^{-1}$)")
    axb.set_ylabel("Bootstrap count")
    axb.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.7")

    if style.show_title():
        band = f" — band ≈ {target_wl:.0f} nm" if target_wl else ""
        fig.suptitle(f"Quasi-BRDF corrected collapse & pooled-γ bootstrap{band}",
                     fontsize=12 * sc, fontweight="bold")
    return fig


# ============================================================
# STUDY-OVERVIEW PIECES (Fig. 1) — emitted as SEPARATE figures so each can be
# placed independently in Affinity (the Libya-4 RGB is a third piece, added by
# hand). No panel letters are drawn; label them in the layout tool.
#   site_overview_map     — North-African PICS locator over faded satellite imagery
#   patch_coverage_single — one PICS's patch grid shaded by per-patch coverage (n)
# ============================================================

#: A light halo so on-map text stays legible over imagery and colour fills.
def _halo(lw=1.6, fg="white"):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def _fmt_lon(d):
    """Format a longitude tick, e.g. -10 -> ``10°W``, 20 -> ``20°E``."""
    return "0°" if d == 0 else f"{abs(int(d))}°{'E' if d > 0 else 'W'}"


def _fmt_lat(d):
    """Format a latitude tick, e.g. 20 -> ``20°N``."""
    return "0°" if d == 0 else f"{abs(int(d))}°{'N' if d > 0 else 'S'}"


def _north_arrow(ax, *, x=0.93, y=0.85, length=0.10, sc=1.0, color="k"):
    """A simple north arrow in *ax* axes-fraction coords (north = up).

    Sits entirely inside the axes: the arrow head is at *y* and the ``N`` label
    just above it, so keep ``y`` (plus a little) below 1.0.
    """
    ax.annotate("", xy=(x, y), xytext=(x, y - length), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3 * sc),
                zorder=8)
    ax.text(x, y + 0.008, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontweight="bold", fontsize=9 * sc, color=color, zorder=8,
            path_effects=_halo(1.6 * sc))


def _scale_bar(ax, length_data, label, *, loc=(0.06, 0.08), sc=1.0, color="k"):
    """Draw a scale bar *length_data* wide (in the axes' own data units).

    Placement is in axes fraction (*loc* = left end); the bar length is converted
    from data units to a fraction of the current x-span, so the same helper serves
    the geographic panel (Web-Mercator metres) and the UTM patch panel (metres).
    *label* is the human distance (e.g. ``"1000 km"``).
    """
    x0, y0 = loc
    xmin, xmax = ax.get_xlim()
    frac = float(length_data) / (xmax - xmin)
    lw = 2.6 * sc
    ax.plot([x0, x0 + frac], [y0, y0], transform=ax.transAxes, color=color,
            lw=lw, solid_capstyle="butt", zorder=8, path_effects=_halo(lw + 1.2))
    for xx in (x0, x0 + frac):
        ax.plot([xx, xx], [y0, y0 + 0.014], transform=ax.transAxes, color=color,
                lw=lw * 0.6, zorder=8, path_effects=_halo(lw))
    ax.text(x0 + frac / 2, y0 + 0.022, label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5 * sc, color=color, zorder=8,
            path_effects=_halo(1.6 * sc))


def _draw_site_map(ax, sites_gdf, *, highlight=None, scale_km=1000, sc=1.0,
                   basemap=True, basemap_alpha=0.9, fade=0.35, grid_step=10,
                   margin_frac=0.10):
    """Draw the PICS locator into a plain (EPSG:3857) *ax*.

    A faded Esri World-Imagery basemap backs the map; each PICS is drawn as its
    actual ROI box (soft red fill + red border) with a label; a graticule is drawn
    every *grid_step* degrees; a ground-distance scale bar (bottom-right) and a
    north arrow (inside the frame) complete it. The basemap is best-effort — a
    fully offline run falls back to a plain desert-toned background.

    *sites_gdf* is the PICS GeoDataFrame (any CRS; reprojected to Web Mercator)
    with ``site_name`` and a bbox ``geometry``.
    """
    from pyproj import Transformer

    sites = sites_gdf.to_crs(3857)
    minx, miny, maxx, maxy = sites.total_bounds
    m = margin_frac * max(maxx - minx, maxy - miny)
    x0, x1, y0, y1 = minx - m, maxx + m, miny - m, maxy + m
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery,
                            alpha=basemap_alpha, attribution=False, zorder=0)
        except Exception as exc:                  # pragma: no cover - network dependent
            print(f"[site_overview_map] satellite basemap skipped ({exc}); "
                  "plain background.")
            ax.set_facecolor("#e7e0d2")
        ax.set_xlim(x0, x1)                        # add_basemap can nudge the limits
        ax.set_ylim(y0, y1)
    else:
        ax.set_facecolor("#e7e0d2")

    # Semi-transparent white veil so the imagery reads as a soft backdrop.
    if fade:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white",
                               alpha=fade, edgecolor="none", zorder=1))

    # National borders as grey lines (Natural Earth admin-0; cartopy is the data
    # source). Best-effort — skipped if the shapefile can't be reached.
    try:
        import geopandas as gpd
        import cartopy.io.shapereader as shpreader
        bshp = shpreader.natural_earth(resolution="50m", category="cultural",
                                       name="admin_0_boundary_lines_land")
        borders = gpd.read_file(bshp).to_crs(3857).cx[x0:x1, y0:y1]
        borders.plot(ax=ax, color="0.5", linewidth=0.5 * sc, zorder=2)
    except Exception as exc:                      # pragma: no cover - network dependent
        print(f"[site_overview_map] national borders skipped ({exc}).")

    # Actual PICS ROI boxes: soft red fill + crisp red border (no stars).
    sites.plot(ax=ax, facecolor="#D7191C", alpha=0.28, edgecolor="none", zorder=3)
    sites.boundary.plot(ax=ax, color="#B2182B", linewidth=1.3 * sc, zorder=4)

    pad = 0.012 * (maxx - minx)
    for _, r in sites.iterrows():
        name = str(r["site_name"])
        bx0, by0, bx1, by1 = r.geometry.bounds
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        is_hi = highlight is not None and name == highlight
        right_edge = cx > x0 + 0.72 * (x1 - x0)    # flip label off the box's near side
        lx = (bx0 - pad) if right_edge else (bx1 + pad)
        ax.text(lx, cy, name, ha=("right" if right_edge else "left"), va="center",
                fontsize=7.5 * sc, fontweight=("bold" if is_hi else "normal"),
                color="k", zorder=7, clip_on=False, path_effects=_halo(1.9 * sc))

    # Graticule every grid_step degrees, as labelled ticks -> grid lines. In
    # Web Mercator meridians are vertical and parallels horizontal, so ticks at
    # these positions draw a true lon/lat graticule.
    tf = Transformer.from_crs(4326, 3857, always_xy=True)
    inv = Transformer.from_crs(3857, 4326, always_xy=True)
    lon_lo, lat_lo = inv.transform(x0, y0)
    lon_hi, lat_hi = inv.transform(x1, y1)
    lons = [d for d in range(-180, 181, grid_step) if lon_lo <= d <= lon_hi]
    lats = [d for d in range(-90, 91, grid_step) if lat_lo <= d <= lat_hi]
    ax.set_xticks([tf.transform(d, 0)[0] for d in lons])
    ax.set_xticklabels([_fmt_lon(d) for d in lons])
    ax.set_yticks([tf.transform(0, d)[1] for d in lats])
    ax.set_yticklabels([_fmt_lat(d) for d in lats])
    ax.tick_params(labelsize=7 * sc, length=2.5 * sc)
    ax.grid(True, color="white", alpha=0.55, linewidth=0.5 * sc, linestyle=":")

    # Ground-distance scale bar, bottom-right (Web-Mercator stretches E-W by
    # 1/cos(lat), so convert ground km at the bar's latitude to map units).
    if scale_km:
        import math
        lat_ref = inv.transform(0.5 * (x0 + x1), y0 + 0.08 * (y1 - y0))[1]
        length_merc = (scale_km * 1000.0) / math.cos(math.radians(lat_ref))
        frac = length_merc / (x1 - x0)
        _scale_bar(ax, length_merc, f"{scale_km:g} km", sc=sc,
                   loc=(0.955 - frac, 0.05))
    _north_arrow(ax, x=0.93, y=0.85, length=0.10, sc=sc)


def site_overview_map(sites_gdf, *, highlight="Libya 4", scale_km=1000,
                      panel_in=5.6, basemap=True, basemap_alpha=0.9, fade=0.35,
                      grid_step=10, suptitle=None):
    """Fig. 1A: locator map of the study PICS over faded satellite imagery.

    Each PICS is its actual ROI box (soft red fill + red border), labelled; a
    graticule is drawn every *grid_step* degrees; a *scale_km* ground-distance bar
    sits bottom-right and a north arrow inside the frame. The figure is sized to the
    data extent (Web Mercator) so there's no wasted whitespace. No panel letter is
    drawn — add it in the layout tool.

    Parameters
    ----------
    sites_gdf : geopandas.GeoDataFrame
        Study PICS with ``site_name`` and bbox ``geometry`` (any CRS).
    highlight : str
        Site whose label is bolded (subject of the companion RGB / coverage pieces).
    scale_km : float
        Scale-bar length in ground km (default 1000).
    panel_in : float
        Figure width in inches; height follows the map's aspect.
    basemap, basemap_alpha, fade : bool, float, float
        Satellite basemap on/off, its opacity, and the white-veil strength that
        fades it into a backdrop.
    grid_step : int
        Graticule spacing in degrees (default 10).
    suptitle : str or None
        Title (drawn only under a title-showing theme, e.g. ``ppt``).
    """
    sc = style.scale()
    s3857 = sites_gdf.to_crs(3857)
    minx, miny, maxx, maxy = s3857.total_bounds
    m = 0.10 * max(maxx - minx, maxy - miny)
    aspect = ((maxy - miny) + 2 * m) / ((maxx - minx) + 2 * m)
    fig, ax = plt.subplots(figsize=(panel_in, panel_in * aspect), layout="constrained")

    _draw_site_map(ax, sites_gdf, highlight=highlight, scale_km=scale_km, sc=sc,
                   basemap=basemap, basemap_alpha=basemap_alpha, fade=fade,
                   grid_step=grid_step, margin_frac=0.10)

    if style.show_title():
        ax.set_title(suptitle or "Study PICS — North African calibration sites",
                     fontsize=12 * sc, fontweight="bold")
    return fig


def _site_map_extent_lonlat(sites_gdf, *, margin_frac=0.10):
    """The lon/lat (min_lon, min_lat, max_lon, max_lat) box of the site map.

    Computed exactly as :func:`_draw_site_map` frames itself — Web-Mercator bounds
    plus a *margin_frac* margin — then inverse-transformed to degrees, so the inset
    rectangle matches the locator map's extent.
    """
    from pyproj import Transformer
    s = sites_gdf.to_crs(3857)
    minx, miny, maxx, maxy = s.total_bounds
    m = margin_frac * max(maxx - minx, maxy - miny)
    inv = Transformer.from_crs(3857, 4326, always_xy=True)
    lon0, lat0 = inv.transform(minx - m, miny - m)
    lon1, lat1 = inv.transform(maxx + m, maxy + m)
    return lon0, lat0, lon1, lat1


def africa_locator(sites_gdf, *, panel_in=3.0, africa_extent=(-20, -37, 55, 40),
                   rect_color="k", suptitle=None):
    """Simple Africa inset: the continent with the site-map extent as a black box.

    A minimal locator for the corner of Fig. 1 — Africa's countries in light grey
    (Natural Earth admin-0 via cartopy) with the :func:`site_overview_map` extent
    drawn as a rectangle, so a reader sees where the study region sits. Frame/ticks
    are stripped; no panel letter is drawn.

    Parameters
    ----------
    sites_gdf : geopandas.GeoDataFrame
        Study PICS (any CRS) — used only to reproduce the site-map extent box.
    panel_in : float
        Figure width in inches (small — this is an inset).
    africa_extent : tuple
        ``(min_lon, min_lat, max_lon, max_lat)`` view window (default whole continent).
    rect_color : str
        Colour of the study-region rectangle (default black).
    suptitle : str or None
        Title (drawn only under a title-showing theme).
    """
    import geopandas as gpd
    import cartopy.io.shapereader as shpreader

    sc = style.scale()
    lon0, lat0, lon1, lat1 = _site_map_extent_lonlat(sites_gdf)
    x0, y0, x1, y1 = africa_extent

    fig, ax = plt.subplots(figsize=(panel_in, panel_in * 1.12), layout="constrained")
    try:
        shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                      name="admin_0_countries")
        world = gpd.read_file(shp)
        africa = world[world["CONTINENT"] == "Africa"] if "CONTINENT" in world.columns else world
        africa.plot(ax=ax, facecolor="#e9e4d8", edgecolor="0.6", linewidth=0.3 * sc, zorder=1)
    except Exception as exc:                      # pragma: no cover - network dependent
        print(f"[africa_locator] Africa outline skipped ({exc}); drawing box only.")
        ax.set_facecolor("white")

    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False,
                           edgecolor=rect_color, linewidth=1.5 * sc, zorder=5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    if style.show_title():
        ax.set_title(suptitle or "Africa", fontsize=11 * sc, fontweight="bold")
    return fig


def _grid_on_ax(ax, patches, boundary, valcol, norm, cmap, sc):
    """Draw one PICS's patch grid (coloured by *valcol*) + boundary onto *ax*."""
    site_proj = boundary.to_crs(patches.crs)
    patches.plot(ax=ax, column=valcol, cmap=cmap, norm=norm, edgecolor="white",
                 linewidth=0.15 * sc, legend=False, zorder=2)
    site_proj.boundary.plot(ax=ax, color="black", linewidth=0.6 * sc, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def temporal_coverage(sites_data, *, fig_w=6.5, fig_h=6.0,
                      headers=("OLI-2 BRDF\ntime series", "EMIT matched\npairs coverage",
                               "EMIT BRDF\ntime series")):
    """Per-PICS temporal coverage: OLI-2 scenes + EMIT matched pairs + EMIT BRDF scenes.

    One row per PICS, three columns on a shared time and SZA axis:

    * **left** — the QC-retained Landsat 9 OLI-2 BRDF scenes, one dot per scene at its
      acquisition date (x) and in-patch mean SZA (y). This column carries the SZA tick
      labels and the row name for the whole row.
    * **middle** — the site's unique EMIT matched pairs. Each pair is two scenes plotted
      at their acquisition date (x) and in-patch mean SZA (y), joined by a near-horizontal
      segment (the pair is SZA-matched, so the connector is nearly flat but spans the
      time gap that carries the drift signal).
    * **right** — the QC-retained EMIT BRDF joint-fit scenes, one dot per scene at its
      acquisition date and in-patch mean SZA.

    Sites excluded from the EMIT drift analyses (``show_emit`` False, e.g. Mauritania 1)
    keep only their OLI-2 panel; their two EMIT cells are left blank because that data is
    never used. Every PICS is drawn in its stable Wong colour (:func:`style.site_styles`),
    so the rows match the other publication figures. All columns share one time range (a
    scene lines up vertically across columns) and one SZA range.

    Parameters
    ----------
    sites_data : list of dict
        One per PICS, in display order, each with: ``name`` (str display label),
        ``site`` (canonical token, for the colour map), ``pairs`` (DataFrame with
        ``t1_dt``, ``t2_dt``, ``SZA_t1``, ``SZA_t2`` — one row per unique matched
        pair), ``dates`` / ``sza`` (EMIT BRDF scene dates and mean SZAs), ``ls_dates`` /
        ``ls_sza`` (OLI-2 scenes), the counts ``n_pairs`` / ``n_brdf`` / ``n_landsat``,
        and ``show_emit`` (whether to draw the two EMIT columns for this site).
    fig_w, fig_h : float
        Figure size in inches (fixed to the publication spec; default 6.5 × 6.0).
    headers : tuple
        The three column titles, drawn over the top row (OLI-2, matched pairs, EMIT BRDF).
    """
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    sc = style.scale()
    n = len(sites_data)
    styles = style.site_styles([s["site"] for s in sites_data])

    # Shared time range across ALL columns (pair endpoints + EMIT and OLI-2 scene dates),
    # so a scene aligns vertically between the panels of every row.
    all_dates = []
    for s in sites_data:
        p = s["pairs"]
        all_dates += list(pd.to_datetime(p["t1_dt"])) + list(pd.to_datetime(p["t2_dt"]))
        all_dates += list(pd.to_datetime(s["dates"]))
        all_dates += list(pd.to_datetime(s.get("ls_dates", [])))
    d0, d1 = min(all_dates), max(all_dates)
    dpad = (d1 - d0) * 0.03
    d0, d1 = d0 - dpad, d1 + dpad

    # Shared SZA range across ALL columns, so the OLI-2 dots, the matched-pair
    # connectors, and the EMIT BRDF dots read on one common SZA axis.
    all_sza = np.concatenate([
        np.r_[s["pairs"]["SZA_t1"].to_numpy(float),
              s["pairs"]["SZA_t2"].to_numpy(float),
              np.asarray(s["sza"], float),
              np.asarray(s.get("ls_sza", []), float)]
        for s in sites_data])
    slo, shi = float(np.nanmin(all_sza)), float(np.nanmax(all_sza))
    spad = 0.10 * (shi - slo) if shi > slo else 2.0
    slo, shi = slo - spad, shi + spad

    # Bottom-most visible row per column for the year tick labels: the OLI-2 column is
    # always full, the two EMIT columns end at the last site not excluded from EMIT.
    last_emit_row = max((i for i, s in enumerate(sites_data) if s.get("show_emit", True)),
                        default=n - 1)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(n, 3, figure=fig, left=0.115, right=0.99, top=0.925, bottom=0.055,
                  wspace=0.07, hspace=0.16, width_ratios=[1.0, 1.0, 1.0])
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(n)]

    for i, s in enumerate(sites_data):
        c = styles[s["site"]]["color"]
        ax_ls, ax_mp, ax_ts = axes[i]   # columns: OLI-2 | EMIT matched pairs | EMIT BRDF
        show_emit = s.get("show_emit", True)

        # OLI-2 panel (left): one dot per retained scene; this column carries the SZA
        # tick labels and the row name for all three columns.
        ax_ls.scatter(pd.to_datetime(s.get("ls_dates", [])),
                      np.asarray(s.get("ls_sza", []), float),
                      s=9 * sc, color=c, edgecolors="white", linewidths=0.2 * sc,
                      alpha=0.85, zorder=3)
        ax_ls.set_ylim(slo, shi)
        ax_ls.yaxis.set_major_locator(MaxNLocator(3))
        ax_ls.tick_params(axis="y", labelsize=7 * sc, length=2 * sc)
        ax_ls.text(0.985, 0.94, f"n = {s.get('n_landsat', 0)}", transform=ax_ls.transAxes,
                   ha="right", va="top", fontsize=6.2 * sc, color="0.35",
                   path_effects=_halo(1.4 * sc))
        # Rotated PICS name identifying the row (black; only the data is coloured).
        ax_ls.annotate(s["name"], xy=(-0.15, 0.5), xycoords="axes fraction",
                       rotation=90, ha="center", va="center",
                       fontsize=8 * sc, fontweight="bold", color="black")

        # EMIT matched-pairs (middle) and EMIT BRDF (right) panels: drawn only for sites
        # used in the EMIT drift analyses; excluded sites keep blank cells.
        if show_emit:
            p = s["pairs"]
            t1 = pd.to_datetime(p["t1_dt"]); t2 = pd.to_datetime(p["t2_dt"])
            y1 = p["SZA_t1"].to_numpy(float); y2 = p["SZA_t2"].to_numpy(float)
            # Matched-pairs panel: a connector per pair + a marker at each endpoint.
            for a, b, ya, yb in zip(t1, t2, y1, y2):
                ax_mp.plot([a, b], [ya, yb], color=c, lw=0.7 * sc, alpha=0.55,
                           solid_capstyle="round", zorder=2)
            ax_mp.scatter(np.r_[t1.values, t2.values], np.r_[y1, y2], s=18 * sc, color=c,
                          edgecolors="white", linewidths=0.3 * sc, zorder=3)
            ax_mp.set_ylim(slo, shi)
            ax_mp.tick_params(axis="y", left=False, labelleft=False)
            ax_mp.text(0.015, 0.06, f"{s['n_pairs']} pairs", transform=ax_mp.transAxes,
                       ha="left", va="bottom", fontsize=6.2 * sc, color="0.35",
                       path_effects=_halo(1.4 * sc))

            # EMIT BRDF panel: one dot per retained scene at its date and mean SZA.
            ax_ts.scatter(pd.to_datetime(s["dates"]), np.asarray(s["sza"], float),
                          s=18 * sc, color=c, edgecolors="white", linewidths=0.3 * sc,
                          zorder=3)
            ax_ts.set_ylim(slo, shi)
            ax_ts.tick_params(axis="y", left=False, labelleft=False)
            ax_ts.text(0.985, 0.94, f"n = {s['n_brdf']}", transform=ax_ts.transAxes,
                       ha="right", va="top", fontsize=6.2 * sc, color="0.35",
                       path_effects=_halo(1.4 * sc))
        else:
            ax_mp.axis("off")
            ax_ts.axis("off")

        # Shared time axis; year labels on the bottom-most VISIBLE panel of each column.
        ax_ls.set_xlim(d0, d1)
        ax_ls.xaxis.set_major_locator(mdates.YearLocator())
        ax_ls.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax_ls.tick_params(axis="x", labelbottom=(i == n - 1),
                          labelsize=6.5 * sc, length=2 * sc)
        if show_emit:
            for ax in (ax_mp, ax_ts):
                ax.set_xlim(d0, d1)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
                ax.tick_params(axis="x", labelbottom=(i == last_emit_row),
                               labelsize=6.5 * sc, length=2 * sc)

    axes[0][0].set_title(headers[0], fontsize=7.5 * sc, fontweight="bold", pad=4 * sc)
    axes[0][1].set_title(headers[1], fontsize=7.5 * sc, fontweight="bold", pad=4 * sc)
    axes[0][2].set_title(headers[2], fontsize=7.5 * sc, fontweight="bold", pad=4 * sc)
    # One shared SZA label at the far-left edge (the row names sit between it and the
    # ticks). The time axis needs no label — years on the bottom row make it obvious.
    fig.supylabel("Mean SZA (°)", fontsize=8 * sc, x=0.008)
    return fig


def landsat_invariance_spectra(persite, loo, *, fig_w=6.5, fig_h=7.2,
                               site_order=("Libya_1", "Libya_4", "Algeria_3",
                                           "Algeria_5", "Mauritania_1")):
    """Landsat 9 OLI-2 invariance as a 3x5 grid of forest plots (dot + 95% CI whisker).

    No interpolation between bands — each estimate is an independent dot with a
    horizontal 95% bootstrap CI whisker, drift gamma on the x-axis. Every panel is
    one OLI-2 band and holds five rows coloured by PICS (the Fig.~4 colour scheme).
    In reading order:

    * **panels 1-7** — per-site standalone gamma at each band (row = that PICS).
    * **panels 8-14** — leave-one-out pooled gamma at each band (row = the combination
      with that PICS withheld, coloured by the *missing* PICS).
    * **panel 15** — a legend decoding the PICS abbreviations and colours.

    Per-site panels share one gamma x-axis (wide enough for Mauritania~1); the
    leave-one-out panels share a second, narrower one. ``persite`` and ``loo`` are
    the tables written by ``run_landsat_brdf_robustness.py``.
    """
    from matplotlib.lines import Line2D
    sc = style.scale()
    sites = [s for s in site_order if s in set(persite["site"])]
    styles = style.site_styles(sites)
    abbr = {s: f"{s.split('_')[0][0]}{s.split('_')[-1]}" for s in sites}
    wls = sorted(persite["wavelength"].unique())
    ncol, nrow = 3, 5
    panels = ([("ps", wl) for wl in wls] + [("loo", wl) for wl in wls]
              + [("legend", None)])

    # Two shared x-ranges: per-site (spans the large Mauritania 1 excursion) and
    # leave-one-out (narrower). Each group is directly comparable within itself.
    def _xlim(df, pad=0.08):
        lo = float(df["ci_lo_pct_yr"].min()); hi = float(df["ci_hi_pct_yr"].max())
        m = pad * (hi - lo)
        return lo - m, hi + m
    xlim = {"ps": _xlim(persite), "loo": _xlim(loo)}

    def _rec(df, mask):
        r = df[mask]
        return None if r.empty else (float(r["gamma_pct_yr"].iloc[0]),
                                     float(r["ci_lo_pct_yr"].iloc[0]),
                                     float(r["ci_hi_pct_yr"].iloc[0]))

    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), layout="constrained")
    axflat = axes.flatten()
    n = len(sites)

    for k, ax in enumerate(axflat):
        if k >= len(panels):
            ax.axis("off")
            continue
        kind, wl = panels[k]
        if kind == "legend":
            ax.axis("off")
            handles = [Line2D([0], [0], marker="o", ls="none", ms=5 * sc,
                              mfc=styles[s]["color"], mec="white", mew=0.4 * sc)
                       for s in sites]
            labels = [f"{abbr[s]} - {s.replace('_', ' ')}" for s in sites]
            ax.legend(handles, labels, loc="center", fontsize=7 * sc, frameon=True,
                      edgecolor="0.7", framealpha=0.95, handletextpad=0.5,
                      borderpad=0.9, labelspacing=0.8, title="PICS",
                      title_fontsize=7.5 * sc)
            continue

        # Each row is a PICS (per-site) or the drop-that-PICS combo (LOO), same order.
        for i, s in enumerate(sites):
            if kind == "ps":
                v = _rec(persite, (persite["site"] == s)
                         & np.isclose(persite["wavelength"], wl))
            else:
                v = _rec(loo, (loo["left_out"] == s)
                         & np.isclose(loo["wavelength"], wl))
            if v is None:
                continue
            g, lo, hi = v
            ax.errorbar(g, n - 1 - i, xerr=[[g - lo], [hi - g]], fmt="o",
                        ms=4.0 * sc, color=styles[s]["color"],
                        ecolor=styles[s]["color"], elinewidth=1.0 * sc,
                        capsize=2.0 * sc, mec="white", mew=0.3 * sc, zorder=3)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_xlim(*xlim[kind])
        ax.axvline(0, color="k", lw=0.8, ls="--", zorder=1)
        ax.grid(True, color="0.92", lw=0.4, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{wl:.0f} nm" + (" (single PICS)" if kind == "ps" else " (LOO)"),
                     fontsize=7 * sc, pad=2 * sc)
        ax.text(0.04, 0.97, f"{chr(ord('a') + k)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold",
                fontfamily="Arial", zorder=6)

        below = panels[k + ncol] if k + ncol < len(panels) else None
        ax.tick_params(axis="x", labelsize=6 * sc, length=2 * sc,
                       labelbottom=(below is None or below[0] != kind))
        if k % ncol == 0:
            ax.set_yticklabels([abbr[s] for s in sites], fontsize=6 * sc)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel("Drift $\\gamma$ (% yr$^{-1}$)", fontsize=8 * sc)
    return fig


def emit_drift_spectrum_overlay(mp, brdf, *, fig_w=6.5, fig_h=4.8, ylim=(-3.0, 3.0)):
    """EMIT per-channel drift spectrum: matched pairs overlaid by the BRDF fit.

    Matched pairs as a grey 95% CI band, the BRDF fit as points with 95% bootstrap CI
    whiskers, for EMIT TOA radiance (a) and surface reflectance (b). The shared drift
    spectrum the BRDF and matched-pairs results sections both reference. ``mp`` /
    ``brdf`` are the 4-site per-channel drift tables (``10_mp_drift.csv`` /
    ``brdf_drift_spectrum.csv``).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from emit_pics.patch_size_sensitivity import spectrum_segments
    sc = style.scale()
    U = "% yr$^{-1}$"

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.975, bottom=0.10, hspace=0.10)

    def panel(ax, prod, ylab, letter, legkw):
        m = mp[(mp["product"] == prod) & (~mp["absorption_band"].astype(bool))].sort_values("wavelength")
        wl = m["wavelength"].to_numpy(float)
        lo = m["ci_low_pct_yr"].to_numpy(float); hi = m["ci_high_pct_yr"].to_numpy(float)
        for seg in spectrum_segments(wl):
            ax.fill_between(wl[seg], lo[seg], hi[seg], color="0.78", alpha=0.4, lw=0, zorder=1)
        b = brdf[brdf["product"] == prod].copy()
        if "unstable" in b.columns:
            b = b[~b["unstable"].astype(bool)]
        b = b.sort_values("wavelength")
        bw = b["wavelength"].to_numpy(float); bg = b["gamma_pct_yr"].to_numpy(float)
        blo = b["ci_lo_pct_yr"].to_numpy(float); bhi = b["ci_hi_pct_yr"].to_numpy(float)
        sig = b["sig_raw"].astype(bool).to_numpy()
        col = style.PRODUCT_STYLE[prod]["color"]
        # not significant: hollow dot + normal whisker; significant: filled + thicker.
        if (~sig).any():
            ax.errorbar(bw[~sig], bg[~sig], yerr=[bg[~sig] - blo[~sig], bhi[~sig] - bg[~sig]],
                        fmt="o", ms=2.2 * sc, mfc="white", mec=col, ecolor=col,
                        elinewidth=0.45 * sc, capsize=0, zorder=3)
        if sig.any():
            ax.errorbar(bw[sig], bg[sig], yerr=[bg[sig] - blo[sig], bhi[sig] - bg[sig]],
                        fmt="o", ms=2.8 * sc, mfc=col, mec=col, ecolor=col,
                        elinewidth=0.85 * sc, capsize=0, zorder=4)
        ax.axhline(0, color="k", lw=0.7, zorder=2)
        ax.set_ylabel(ylab, fontsize=8 * sc)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7 * sc)
        ax.text(0.006, 0.95, letter, transform=ax.transAxes, va="top", ha="left",
                fontsize=12, fontweight="bold", fontfamily="Arial", zorder=6)
        # legend in a white box near the top, ~25% in from the left to clear the data.
        ax.legend(handles=[
            Line2D([0], [0], marker="o", mfc="white", mec=col, color=col,
                   ms=4 * sc, lw=0.8 * sc, label="BRDF, not significant"),
            Line2D([0], [0], marker="o", mfc=col, mec=col, color=col,
                   ms=4 * sc, lw=1.1 * sc, label="BRDF, 95% CI excludes 0"),
            Patch(facecolor="0.78", alpha=0.4, label="Matched pairs, 95% CI")],
            fontsize=5.6 * sc, framealpha=1.0, facecolor="white", edgecolor="0.6",
            handletextpad=0.5, labelspacing=0.3, borderpad=0.5, **legkw)

    panel(ax_a, "rad", f"EMIT TOA radiance drift,\n$\\gamma$ ({U})", "a)",
          dict(loc="lower left"))
    panel(ax_b, "rfl", f"EMIT surface reflectance drift,\n$\\gamma$ ({U})", "b)",
          dict(loc="upper left", bbox_to_anchor=(0.25, 0.98)))
    ax_b.set_xlabel("Wavelength (nm)", fontsize=8 * sc)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    return fig


def emit_oli2_band_comparison(bands, *, fig_w=6.5, fig_h=3.1):
    """EMIT-vs-Landsat drift synthesised into the 7 OLI-2 reflective bands (Fig.~5 style).

    One panel per band; in each, a dot + 95% bootstrap CI whisker for the measured
    Landsat TOA reflectance drift, the EMIT-synthesised TOA radiance drift, and the
    EMIT-synthesised surface reflectance drift (shared drift axis). ``bands`` is
    ``band_drift_comparison.csv``.
    """
    from matplotlib.lines import Line2D
    sc = style.scale()
    U = "% yr$^{-1}$"
    SER = [("landsat", "Landsat 9 OLI-2 TOA reflectance, $\\rho$", "#CC79A7"),
           ("emit_rad", "EMIT TOA radiance, $L$", style.PRODUCT_STYLE["rad"]["color"]),
           ("emit_rfl", "EMIT surface reflectance, $\\rho$", style.PRODUCT_STYLE["rfl"]["color"])]

    fig, axes = plt.subplots(1, 7, figsize=(fig_w, fig_h), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.17, wspace=0.28)
    bb = bands.dropna(subset=["gamma_pct_yr"])
    ylo = min(float(bb["ci_lo_pct_yr"].min()), 0.0)
    yhi = max(float(bb["ci_hi_pct_yr"].max()), 0.0)
    pad = 0.10 * (yhi - ylo)
    for j, (ax, bn) in enumerate(zip(axes, sorted(bands["band_n"].unique()))):
        sub = bands[bands["band_n"] == bn]
        center = float(sub["center_nm"].iloc[0])
        for xi, (ser, _lab, c) in enumerate(SER):
            r = sub[sub["series"] == ser]
            if r.empty or not np.isfinite(r["gamma_pct_yr"].iloc[0]):
                continue
            gm = float(r["gamma_pct_yr"].iloc[0])
            lo = float(r["ci_lo_pct_yr"].iloc[0]); hi = float(r["ci_hi_pct_yr"].iloc[0])
            ax.errorbar(xi, gm, yerr=[[gm - lo], [hi - gm]], fmt="o", ms=4 * sc,
                        color=c, ecolor=c, elinewidth=1.0 * sc, capsize=2 * sc,
                        mec="white", mew=0.3 * sc, zorder=3)
        ax.axhline(0, color="k", lw=0.7, ls="--", zorder=1)
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks([])
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_title(f"{center:.0f} nm", fontsize=7 * sc, pad=2 * sc)
        ax.grid(axis="y", color="0.92", lw=0.4, zorder=0)
        ax.tick_params(labelsize=6.5 * sc)
        ax.text(0.09, 0.97, f"{chr(ord('a') + j)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold", fontfamily="Arial")
    axes[0].set_ylabel(f"Band drift, $\\gamma$ ({U})", fontsize=8 * sc)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="none", color=c, ms=5 * sc, label=lab)
                        for _s, lab, c in SER],
               loc="lower center", ncol=3, fontsize=6.8 * sc, framealpha=0.9,
               edgecolor="0.7", bbox_to_anchor=(0.5, 0.05),
               columnspacing=1.2, handletextpad=0.4)
    return fig


def patch_coverage_single(patches, boundary, *, count_col="n_gran_ovlp_rfl",
                          count_label="L2A reflectance", annotate="both",
                          cmap="viridis_r", panel_in=6.0, scale_km=20,
                          suptitle=None):
    """Fig. 1C: one PICS's patch grid shaded by per-patch scene coverage (n).

    Patches (UTM CRS) are shaded by *count_col* (the fully-covering-scene count),
    the PICS boundary is outlined, each patch is annotated (id and/or n), and a
    single colourbar and a *scale_km* bar are drawn. Only the boundary + grid show —
    the axes frame/ticks are stripped, and no panel letter is drawn. A missing count
    column degrades to n=0 (with a note) rather than raising.

    Parameters
    ----------
    patches : geopandas.GeoDataFrame
        Patch grid (UTM CRS) with ``patch_id`` and the *count_col* column.
    boundary : geopandas.GeoDataFrame
        Single-row PICS boundary (reprojected internally) — the outline.
    count_col, count_label : str
        Per-patch scene-count column and its human label (default L2A reflectance,
        ``n_gran_ovlp_rfl``; pass ``n_gran_ovlp_rad`` for L1B).
    annotate : str
        Per-cell label: ``"both"`` (id + n), ``"count"`` (n), or ``"none"``.
    panel_in : float
        Panel side length in inches — keep generous so labels read.
    scale_km : float
        Scale-bar length in km.
    suptitle : str or None
        Title (drawn only under a title-showing theme).
    """
    sc = style.scale()
    fig, ax = plt.subplots(figsize=(panel_in * 1.14, panel_in), layout="constrained")

    site_proj = boundary.to_crs(patches.crs)
    gmap = patches.copy()
    if count_col in gmap.columns:
        gmap["_n"] = gmap[count_col].fillna(0).astype(int)
    else:
        print(f"[patch_coverage_single] '{count_col}' not in patch table; showing n=0. "
              "Run scripts/run_id_granules.py to tag coverage counts.")
        gmap["_n"] = 0
    vmax = max(1, int(gmap["_n"].max()))
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    site_proj.boundary.plot(ax=ax, color="black", linewidth=1.4 * sc, zorder=3)
    gmap.plot(ax=ax, column="_n", cmap=cmap, norm=norm, edgecolor="white",
              linewidth=0.5 * sc, legend=False, zorder=2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, fraction=0.05)
    cb.set_label(f"{count_label} coverage (n scenes)", fontsize=8 * sc)
    cb.ax.tick_params(labelsize=7 * sc)

    if annotate != "none":
        for _, prow in gmap.iterrows():
            cpt = prow.geometry.centroid
            lbl = (f"{prow['patch_id']}\n{prow['_n']}" if annotate == "both"
                   else str(prow["_n"]))
            ax.annotate(lbl, (cpt.x, cpt.y), ha="center", va="center",
                        fontsize=6.0 * sc, color="white", fontweight="bold",
                        zorder=5, path_effects=_halo(0.9 * sc, fg="0.25"))

    if scale_km:
        _scale_bar(ax, scale_km * 1000.0, f"{scale_km:g} km", sc=sc, loc=(0.16, 0.04))

    ax.set_aspect("equal")
    ax.axis("off")                                 # boundary + grid only, no frame
    if style.show_title():
        nm = str(boundary.iloc[0]["site_name"]) if "site_name" in boundary.columns else "PICS"
        ax.set_title(suptitle or f"{nm} — patch coverage (n scenes)",
                     fontsize=12 * sc, fontweight="bold")
    return fig
