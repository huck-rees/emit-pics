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
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from emit_pics.viz import style


def drift_spectrum(df, *, size="double", aspect=0.60, sig_col="sig_raw",
                   sig_label="significant (raw p < 0.05)",
                   ylim=(-2.5, 2.5), title="EMIT pooled radiometric drift spectrum"):
    """BRDF pooled drift spectrum gamma(lambda): rad (top) / rfl (bottom) vs wavelength.

    Reads ``data/brdf/brdf_drift_spectrum.csv``. Numerically unstable channels are
    dropped; the bootstrap 95% CI is drawn as whiskers, and channels significant
    by *sig_col* (the uncorrected per-channel p-value) are filled in the
    significance colour while the rest are open grey.

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
    ylim : tuple or None
        Shared y-axis limits (%/yr) for both panels; ``None`` autoscales. Points
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
            ("Radiance drift (%/yr)", "Reflectance drift (%/yr)")):
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
    if style.show_title():
        axes[0].set_title(title)
    return fig


def matched_pairs_geometry(pairs, *, site="Libya 4", patch="10_7_1",
                           size="onehalf", aspect=0.62, sza_tol=None):
    """Matched-pair geometry for one patch: acquisition date (x) vs in-patch SZA (y).

    Reads the already-generated ``{site}_{patch}_matched_pairs.csv`` — one row per
    *unordered* scene pair, so every scene appears in the ``t1_*``/``t2_*`` columns.
    Each surviving scene is drawn once at ``(acquisition date, mean in-patch SZA)``;
    a thin line links the two endpoints of every *matched* pair. Scenes that take
    part in at least one match are filled in the accent colour, unmatched scenes are
    open grey — so the reader sees both the geometry sampling and which acquisitions
    have a same-geometry partner for the ΔR drift analysis.

    Parameters
    ----------
    pairs : pandas.DataFrame
        The matched-pairs table (``PAIR_CSV_COLS`` from ``id_matched_pairs``): needs
        ``t1_scene_id/dt/SZA``, ``t2_scene_id/dt/SZA`` and the ``matched_pair`` flag.
    site, patch : str
        Labels for the (slide-only) title.
    size : str
        Width class resolved by the active theme (default ``onehalf``).
    aspect : float
        Height / width of the figure.
    sza_tol : float or None
        SZA matching tolerance (deg); shown in the slide title when given.
    """
    sc = style.scale()
    accent = style.PRODUCT_STYLE["rad"]["color"]

    # Recover the per-scene table from both halves of every pair row, then dedupe.
    halves = [pairs[[f"{p}_scene_id", f"{p}_scene_dt", f"{p}_scene_SZA"]]
              .set_axis(["scene_id", "dt", "sza"], axis=1) for p in ("t1", "t2")]
    scenes = (pd.concat(halves, ignore_index=True)
              .dropna(subset=["scene_id"])
              .drop_duplicates(subset="scene_id"))
    scenes["dt"] = pd.to_datetime(scenes["dt"])

    matched = pairs[pairs["matched_pair"].astype(bool)].copy()
    matched_ids = set(matched["t1_scene_id"]) | set(matched["t2_scene_id"])
    scenes["in_pair"] = scenes["scene_id"].isin(matched_ids)

    fig, ax = plt.subplots(figsize=style.figsize(size, aspect), layout="constrained")

    # Matched-pair links first (behind the points).
    for _, r in matched.iterrows():
        ax.plot([pd.to_datetime(r["t1_scene_dt"]), pd.to_datetime(r["t2_scene_dt"])],
                [r["t1_scene_SZA"], r["t2_scene_SZA"]],
                color="0.6", lw=0.7 * sc, alpha=0.7, zorder=1)

    iso, part = scenes[~scenes["in_pair"]], scenes[scenes["in_pair"]]
    ax.scatter(iso["dt"], iso["sza"], s=14 * sc, marker="o", facecolors="none",
               edgecolors=style.NS_EDGE, linewidths=0.6 * sc, zorder=3)
    ax.scatter(part["dt"], part["sza"], s=16 * sc, marker="o", color=accent, zorder=4)

    ax.set_ylabel("In-patch mean SZA (degrees)")
    ax.set_xlabel("Acquisition date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    handles = [
        Line2D([0], [0], marker="o", color=accent, ls="none", label="scene (matched)"),
        Line2D([0], [0], marker="o", color=style.NS_EDGE, ls="none", mfc="none",
               label="scene (unmatched)"),
        Line2D([0], [0], color="0.6", lw=0.9 * sc, label="matched pair"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.8, edgecolor="0.7")

    if style.show_title():
        tol = f" (|dSZA| <= {sza_tol:g} deg)" if sza_tol is not None else ""
        ax.set_title(f"{site} {patch} - matched-pair geometry{tol}")
    return fig


#: Median % change column + human label per product (matches the drift analysis).
_DELTA_COL = {"rad": "delta_rad_pct_median", "rfl": "delta_rfl_pct_median"}
_PRODUCT_LABEL = {"rad": "Radiance (L1B)", "rfl": "Reflectance (L2A)"}


def _draw_channel_fit(ax, stats, x, sc):
    """Overlay the Theil-Sen line + monospace stats box for one channel/product.

    *stats* is a row from the drift spectrum CSV (same values as the published
    drift spectrum, so line and spectrum always agree); *x* bounds the line.
    """
    from emit_pics.matched_pairs_drift_analysis import DAYS_PER_YEAR
    x = np.asarray(x, dtype=float)
    slope = stats.get("slope_pct_yr", np.nan) if stats is not None else np.nan
    n = int(stats.get("n", 0) or 0) if stats is not None else 0
    if stats is not None and x.size and np.isfinite(slope):
        xr = np.array([np.nanmin(x), np.nanmax(x)])
        yline = stats["intercept_pct"] + (slope / DAYS_PER_YEAR) * xr
        ax.plot(xr, yline, color="k", lw=1.4 * sc, zorder=5)
        txt = (f"drift = {slope:+.3f} %/yr\n"
               f"95% CI [{stats['ci_low_pct_yr']:+.2f}, {stats['ci_high_pct_yr']:+.2f}]\n"
               f"intercept = {stats['intercept_pct']:+.2f} %\n"
               f"Kendall p = {stats['p_value']:.2g}   n = {n}")
        if pd.notna(stats.get("raw_significant", np.nan)):
            txt += "\n" + ("significant (p < 0.05)" if bool(stats["raw_significant"])
                           else "not significant")
    else:
        txt = f"no drift fit (n = {n})"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=7 * sc,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85), zorder=6)


def _channel_site_legend(ax, styles, sub, sc):
    """PICS colour/shape legend + a mean-|SZA| marker-size reference."""
    from emit_pics.matched_pairs_drift_analysis import sza_marker_size
    handles = [Line2D([], [], color=styles[p]["color"], marker=styles[p]["marker"],
                      ls="", markeredgecolor="k", markersize=7 * sc, label=str(p))
               for p in sorted(sub["PICS_ID"].unique())]
    handles.append(Line2D([], [], ls="", label=" "))          # spacer
    for sza in (20, 40, 60):
        handles.append(Line2D([], [], color="0.5", marker="o", ls="",
                              markeredgecolor="k",
                              markersize=np.sqrt(sza_marker_size(sza) * sc),
                              label=f"SZA {sza}°"))
    ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.9,
              edgecolor="0.7")


def channel_drift(deltas, drift, *, product="rad", target_nm=1030.0,
                  size="onehalf", aspect=0.78, stat_label="median", ylim=None):
    """Single-product per-channel drift scatter: DeltaR (%) vs Delta-t (days).

    Reproduces the pipeline's per-channel scatter (``plot_channel_scatter``) as a
    one-product publication figure. The nearest non-absorption EMIT channel to
    *target_nm* is chosen; every pooled matched pair is a marker coloured/shaped
    by PICS (Wong palette via :func:`style.site_styles`) and sized by the pair's
    mean ``|SZA|``. A dashed zero line marks no-change; the black Theil-Sen line
    and the monospace stats box use the **precomputed** drift-spectrum values, so
    they match the published spectrum exactly.

    Parameters
    ----------
    deltas : pandas.DataFrame
        Pooled delta table (``load_(aggregated_)delta_tables``): needs
        ``PICS_ID, dt_days, band_index, wavelength, sza_size``, the
        ``delta_{rad,rfl}_pct_median`` columns and ``absorption_band_flag``.
    drift : pandas.DataFrame
        The drift spectrum CSV (``paths.drift``) — precomputed per-channel stats.
    product : str
        ``"rad"`` or ``"rfl"`` — one product per figure.
    target_nm : float
        Target wavelength (nm); the nearest non-absorption channel is drawn.
    size, aspect, ylim :
        Theme width class, height/width, and optional shared y-limits (%).
    """
    from emit_pics.matched_pairs_drift_analysis import (
        absorption_band_indices, sza_marker_size)
    sc = style.scale()
    dcol = _DELTA_COL[product]

    # Nearest non-absorption channel to the target wavelength.
    abs_bands = set(absorption_band_indices(deltas))
    wl_by_band = (deltas[~deltas["band_index"].isin(abs_bands)]
                  .groupby("band_index")["wavelength"].median())
    band = int((wl_by_band - target_nm).abs().idxmin())
    wl = float(wl_by_band.loc[band])
    sub = deltas[deltas["band_index"] == band]
    styles = style.site_styles(sub["PICS_ID"].unique())

    fig, ax = plt.subplots(figsize=style.figsize(size, aspect), layout="constrained")
    ax.axhline(0.0, color="gray", lw=0.8 * sc, ls="--", zorder=1)
    for pics, g in sub.groupby("PICS_ID"):
        st = styles[pics]
        ax.scatter(g["dt_days"], g[dcol], s=sza_marker_size(g["sza_size"]) * sc,
                   c=st["color"], marker=st["marker"], edgecolors="k",
                   linewidths=0.4 * sc, alpha=0.85, zorder=3)

    row = drift[(drift["product"] == product) & (drift["band_index"] == band)]
    _draw_channel_fit(ax, row.iloc[0].to_dict() if not row.empty else None,
                      sub["dt_days"].to_numpy(), sc)

    ax.set_xlabel("Δt — days between pair")
    ax.set_ylabel(f"ΔR — % change ({stat_label})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _channel_site_legend(ax, styles, sub, sc)

    if style.show_title():
        ax.set_title(f"{_PRODUCT_LABEL[product]} — channel {band} ({wl:.1f} nm)")
    return fig


def brdf_cos_sza(design, *, fig_width=6.5, aspect=0.52, color_by_site=True,
                 target_wl=None, suptitle=None):
    """Fig. 2: in-patch median vs cos(SZA) — radiance (A) and reflectance (B).

    Motivates the quasi-BRDF correction: at a fixed target band, both products vary
    with solar-illumination angle (radiance because incident flux scales with
    ``cos(SZA)``, reflectance through residual surface BRDF). Reads the pooled BRDF
    design table (``data/brdf/brdf_design.csv``: per-scene ``g = cos(SZA)`` and the
    in-patch median ``rho_rad`` / ``rho_rfl``). Points are coloured by PICS (the
    per-site baselines the BRDF model fits); one ordinary-least-squares line is fit
    per panel, and its statistics ($R^2$, slope $m$, $p$, $n$) are printed top-left.

    Total width is fixed at *fig_width* inches (default 6.5) to meet the publication
    spec; fonts/markers still follow the active theme.

    Parameters
    ----------
    design : pandas.DataFrame
        BRDF design table with ``g``, ``rho_rad``, ``rho_rfl`` (and ``site``).
    fig_width : float
        Total figure width in inches (default 6.5).
    aspect : float
        Height / width of the whole figure.
    color_by_site : bool
        Colour/shape points by PICS (needs a ``site`` column); else single product colour.
    target_wl : float or None
        Target band (nm) for the on-figure title (slide theme only).
    suptitle : str or None
        Title (drawn only under a title-showing theme).
    """
    from scipy.stats import linregress

    sc = style.scale()
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width * aspect),
                             layout="constrained")
    prods = (("rad", "rho_rad", "Median radiance (µW/cm²/sr/nm)"),
             ("rfl", "rho_rfl", "Median reflectance (unitless)"))
    use_site = color_by_site and "site" in design.columns
    styles = style.site_styles(design["site"].unique()) if use_site else None

    for ax, (key, col, ylab) in zip(axes, prods):
        d = design[np.isfinite(design["g"]) & np.isfinite(design[col])]
        x = d["g"].to_numpy(dtype=float)
        y = d[col].to_numpy(dtype=float)

        if use_site:
            for s, gg in d.groupby("site"):
                st = styles[s]
                ax.scatter(gg["g"], gg[col], s=11 * sc, color=st["color"],
                           marker=st["marker"], edgecolors="k", linewidths=0.3 * sc,
                           alpha=0.85, zorder=3, label=str(s))
        else:
            pc = style.PRODUCT_STYLE[key]
            ax.scatter(x, y, s=11 * sc, color=pc["color"], marker=pc["marker"],
                       edgecolors="k", linewidths=0.3 * sc, alpha=0.85, zorder=3)

        # One OLS line per panel + a top-left stats box (R², slope, p, n).
        if x.size >= 3 and np.ptp(x) > 0:
            lr = linregress(x, y)
            xr = np.array([x.min(), x.max()])
            ax.plot(xr, lr.intercept + lr.slope * xr, color="k", lw=1.4 * sc, zorder=4)
            pstr = "p < 0.001" if lr.pvalue < 1e-3 else f"p = {lr.pvalue:.3f}"
            txt = (f"$R^2$ = {lr.rvalue**2:.3f}\n"
                   f"m = {lr.slope:.3g}\n"
                   f"{pstr}\n"
                   f"n = {x.size}")
        else:
            txt = f"n = {x.size}\n(insufficient for fit)"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7 * sc, family="monospace", zorder=6,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

        ax.set_xlabel("cos(SZA)")
        ax.set_ylabel(ylab)

    if use_site:
        axes[0].legend(loc="lower right", fontsize=6.5 * sc, framealpha=0.9,
                       edgecolor="0.7", title="PICS", title_fontsize=6.5 * sc)
    if style.show_title():
        t = suptitle or ("Illumination dependence of radiance and reflectance"
                         + (f" — band ≈ {target_wl:.0f} nm" if target_wl else ""))
        fig.suptitle(t, fontsize=12 * sc, fontweight="bold")
    return fig


def patch_coverage(patches, site, *, annotate="both", cmap="viridis_r",
                   count_cols=(("n_gran_ovlp_rad", "L1B Radiance"),
                               ("n_gran_ovlp_rfl", "L2A Reflectance")),
                   panel_in=8.0, suptitle=None):
    """One map per product: patches shaded by fully-covering granule count.

    Faithful reproduction of the original coverage figure — a generously sized
    grid (``panel_in`` inches per square panel) so every per-patch label stays
    legible. Each panel shades the patches by that product's granule count
    (``viridis_r``; darker = more), outlines the PICS boundary, carries its own
    colourbar, and annotates every patch with its id and count. Reads the
    already-generated ``patches_{site}_bnds.gpkg`` (geometry + ``n_gran_ovlp_*``
    columns) and the PICS boundary — no Earthdata re-search.

    Parameters
    ----------
    patches, site : geopandas.GeoDataFrame
        Patch polygons (projected CRS, with ``patch_id`` + the count columns) and
        the single-row PICS boundary (reprojected internally).
    annotate : str
        Per-cell label: ``"both"`` (id + count), ``"count"`` (count only), or
        ``"none"``.
    cmap : str
        Colormap; ``viridis_r`` so darker = more granules (colour-blind-safe).
    panel_in : float
        Side length (inches) of each square panel — keep generous so labels read.
    suptitle : str or None
        Figure title; defaults to "<site> — patches by granule count (darker = more)".
    """
    site_proj = site.to_crs(patches.crs)
    n = len(count_cols)
    fig, axes = plt.subplots(1, n, figsize=(panel_in * n, panel_in))
    axes = np.atleast_1d(axes)

    for ax, (col, title) in zip(axes, count_cols):
        gmap = patches.copy()
        gmap["_n"] = gmap[col].fillna(0).astype(int)

        site_proj.boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=3)
        gmap.plot(ax=ax, column="_n", cmap=cmap, edgecolor="white", linewidth=0.5,
                  legend=True, legend_kwds={"label": f"{title} granules", "shrink": 0.7})

        if annotate != "none":
            for _, prow in gmap.iterrows():
                cpt = prow.geometry.centroid
                lbl = (f"{prow['patch_id']}\n{prow['_n']}" if annotate == "both"
                       else str(prow["_n"]))
                ax.annotate(lbl, (cpt.x, cpt.y), ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold", zorder=5)

        ax.set_title(f"{title} — fully-covering scenes per patch", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    if suptitle is None:
        nm = str(site.iloc[0]["site_name"]) if "site_name" in site.columns else "PICS"
        suptitle = f"{nm} — patches by granule count (darker = more)"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
