"""Publication figure builders — pure ``(data) -> Figure``.

Each function reads already-generated pipeline outputs and returns a Matplotlib
figure. Builders run under an active :mod:`emit_pics.viz.style` theme (which sets
the font family + output format); size-sensitive builders also read the theme's
width presets and scale, while a dense reference figure may fix its own generous
size so all its labels stay legible.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from emit_pics.viz import style


def drift_spectrum(df, *, size="double", aspect=0.60, sig_col="sig_bh",
                   ylim=(-2.5, 2.5)):
    """BRDF pooled drift spectrum gamma(lambda): rad (top) / rfl (bottom) vs wavelength.

    Reads ``data/brdf/brdf_drift_spectrum.csv``. Numerically unstable channels are
    dropped; the bootstrap 95% CI is drawn as whiskers, and channels significant
    by *sig_col* (default BH-FDR) are filled in the significance colour while the
    rest are open grey.

    Parameters
    ----------
    df : pandas.DataFrame
        The drift-spectrum table (columns per the data dictionary).
    size : str
        Width class resolved by the active theme (``single`` / ``onehalf`` / ``double``).
    aspect : float
        Height / width of the whole figure.
    sig_col : str
        Significance column to colour by (``sig_bh`` or ``sig_raw``).
    ylim : tuple or None
        Shared y-axis limits (%/yr) for both panels; ``None`` autoscales. Points
        or CI whiskers outside the range are clipped to the fixed scale.
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
               label="BH-significant"),
        Line2D([0], [0], marker="|", color="0.7", ls="none",
               markersize=7 * sc, markeredgewidth=0.9 * sc, label="95% CI"),
    ]
    # Semi-transparent box, top-left of the reflectance (bottom) panel.
    axes[1].legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.8, edgecolor="0.7")
    if style.show_title():
        axes[0].set_title("EMIT pooled radiometric drift spectrum")
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
        if pd.notna(stats.get("bh_significant", np.nan)):
            txt += "\n" + ("BH-significant" if bool(stats["bh_significant"])
                           else "not BH-significant")
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
