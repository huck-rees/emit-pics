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
# STUDY-OVERVIEW COMPOSITE (Fig. 1)
#   A  North-African PICS locator map (all study sites, red stars)
#   B  blank, correctly proportioned slot for a Libya-4 RGB (placed in Affinity)
#   C  Libya-4 patch grid shaded by per-patch scene coverage (n)
# The three panels are assembled here as ONE figure so the layout, panel tags,
# and shared styling are fixed in code; only the RGB is dropped into panel B by
# hand afterwards. Cartopy is imported lazily so the other builders don't need it.
# ============================================================

#: A light halo so on-map text stays legible over land/ocean fills and colour.
def _halo(lw=1.6, fg="white"):
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def _north_arrow(ax, *, x=0.93, y=0.95, length=0.11, sc=1.0):
    """A simple north arrow in *ax* axes-fraction coords (north = up)."""
    ax.annotate("", xy=(x, y), xytext=(x, y - length), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3 * sc),
                zorder=8)
    ax.text(x, y + 0.005, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontweight="bold", fontsize=9 * sc, zorder=8,
            path_effects=_halo(1.4 * sc))


def _scale_bar(ax, length_data, label, *, loc=(0.06, 0.08), sc=1.0, color="k"):
    """Draw a scale bar *length_data* wide (in the axes' own data units).

    Placement is in axes fraction (*loc* = left end); the bar length is converted
    from data units to a fraction of the current x-span, so the same helper serves
    the geographic panel (degrees) and the UTM patch panel (metres). *label* is the
    human distance (e.g. ``"200 km"``).
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
            ha="center", va="bottom", fontsize=7.5 * sc, zorder=8,
            path_effects=_halo(1.4 * sc))


def _panel_tag(ax, tag, *, sc=1.0):
    """Bold A/B/C tag at the panel's top-left corner."""
    ax.text(0.012, 0.985, tag, transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=12 * sc, zorder=10,
            path_effects=_halo(2.0 * sc))


def _draw_site_map(ax, sites_gdf, *, highlight=None, scale_km=200, sc=1.0,
                   margin_deg=2.5, label_dx=0.35, label_dy=0.35):
    """Panel A: locator map of the study PICS as labelled red stars.

    *ax* must be a Cartopy ``GeoAxes`` in ``PlateCarree``. *sites_gdf* is the
    PICS GeoDataFrame (EPSG:4326) with ``site_name``, ``center_lon``,
    ``center_lat`` and a bbox ``geometry``. Land/ocean/coast/border features are
    added best-effort (skipped if Natural Earth data can't be reached), then each
    site's ROI box is outlined and its centre marked with a red star and a label.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pc = ccrs.PlateCarree()
    minx, miny, maxx, maxy = sites_gdf.total_bounds
    ax.set_extent([minx - margin_deg, maxx + margin_deg,
                   miny - margin_deg, maxy + margin_deg], crs=pc)

    # Context features — never fatal (a fully offline run still gets the stars/grid).
    try:
        ax.add_feature(cfeature.LAND, facecolor="#f2efe9", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f2", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5 * sc, edgecolor="0.4", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35 * sc, edgecolor="0.55",
                       linestyle="-", zorder=1)
    except Exception as exc:                      # pragma: no cover - network dependent
        print(f"[study_overview] basemap features skipped ({exc}); "
              "drawing stars + grid only.")

    gl = ax.gridlines(draw_labels=True, linewidth=0.4 * sc, color="0.6",
                      alpha=0.6, linestyle=":", zorder=2)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7 * sc}

    x0, x1, _, _ = ax.get_extent(crs=pc)
    for _, r in sites_gdf.iterrows():
        name = str(r["site_name"])
        lon, lat = float(r["center_lon"]), float(r["center_lat"])
        is_hi = highlight is not None and name == highlight
        # ROI box (thin) so the star's footprint is honest, then the star itself.
        ax.add_geometries([r.geometry], crs=pc, facecolor="none",
                          edgecolor="#B22222", linewidth=(1.1 if is_hi else 0.7) * sc,
                          zorder=4)
        ax.scatter([lon], [lat], transform=pc, marker="*",
                   s=(190 if is_hi else 130) * sc, color="#D7191C",
                   edgecolor="k", linewidth=0.5 * sc, zorder=6, clip_on=False)
        # Flip the label to the left of the star when it sits near the right edge,
        # so an easternmost site (e.g. Libya 4) isn't clipped by the panel border.
        right_edge = lon > x0 + 0.72 * (x1 - x0)
        ax.text(lon + (-label_dx if right_edge else label_dx), lat + label_dy, name,
                transform=pc, ha=("right" if right_edge else "left"), va="bottom",
                fontsize=7.5 * sc, fontweight=("bold" if is_hi else "normal"),
                zorder=7, clip_on=False, path_effects=_halo(1.8 * sc))

    if scale_km:
        lat_ref = np.deg2rad(0.5 * (miny + maxy))
        deg = scale_km / (111.320 * np.cos(lat_ref))     # km -> degrees lon at mid-lat
        _scale_bar(ax, deg, f"{scale_km:g} km", sc=sc)
    _north_arrow(ax, sc=sc)


def _draw_blank_panel(ax, label, *, sc=1.0):
    """Panel B: an empty, correctly proportioned slot for the Libya-4 RGB.

    Drawn as a dashed placeholder box with a centred note; the RGB is dropped in
    here manually (Affinity). Kept square so it matches a scene chip's aspect.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, linestyle=(0, (5, 4)),
                           edgecolor="0.5", linewidth=1.0 * sc))
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5 * sc, color="0.4", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_coverage_panel(ax, patches, boundary, *, count_col, count_label,
                         annotate="both", cmap="viridis_r", scale_km=None, sc=1.0):
    """Panel C: Libya-4 patch grid shaded by per-patch scene coverage (n).

    Reuses the coverage-map convention of :func:`patch_coverage` for a SINGLE
    product: patches (UTM CRS) shaded by *count_col* (the fully-covering-scene
    count), the PICS boundary outlined, each patch annotated with its id and n,
    and one colourbar. A missing count column degrades to n=0 (with a note) rather
    than raising, so the panel still renders if the gpkg hasn't been coverage-tagged.
    """
    site_proj = boundary.to_crs(patches.crs)
    gmap = patches.copy()
    if count_col in gmap.columns:
        gmap["_n"] = gmap[count_col].fillna(0).astype(int)
    else:
        print(f"[study_overview] '{count_col}' not in patch table; panel C shows n=0. "
              "Run scripts/run_id_granules.py to tag coverage counts.")
        gmap["_n"] = 0
    vmax = max(1, int(gmap["_n"].max()))
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    site_proj.boundary.plot(ax=ax, color="black", linewidth=1.4 * sc, zorder=3)
    gmap.plot(ax=ax, column="_n", cmap=cmap, norm=norm, edgecolor="white",
              linewidth=0.5 * sc, legend=False, zorder=2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, fraction=0.05)
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
        _scale_bar(ax, scale_km * 1000.0, f"{scale_km:g} km", sc=sc)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def study_overview(sites_gdf, patches, boundary, *, highlight="Libya 4",
                   count_col="n_gran_ovlp_rfl", count_label="L2A reflectance",
                   annotate="both", cmap="viridis_r", panel_in=5.2,
                   scale_km_a=200, scale_km_c=20,
                   blank_label="RGB — Libya 4\n(insert in Affinity)",
                   suptitle=None):
    """Three-panel study-overview figure: A locator map | B blank RGB slot | C coverage.

    A generously sized fixed canvas (``panel_in`` per panel) — like
    :func:`patch_coverage` — so every site label and per-patch count stays legible;
    the panels are meant to be finalised (RGB dropped into B, minor nudges) in
    Affinity, so exact cross-panel alignment is deliberately left loose.

    Parameters
    ----------
    sites_gdf : geopandas.GeoDataFrame
        The study PICS (EPSG:4326) with ``site_name``, ``center_lon/lat`` and a
        bbox ``geometry`` — panel A. Order/content decides which stars are drawn.
    patches : geopandas.GeoDataFrame
        Libya-4 patch grid (UTM CRS) with ``patch_id`` and the *count_col* column
        — panel C.
    boundary : geopandas.GeoDataFrame
        Single-row Libya-4 PICS boundary (reprojected internally) — panel C outline.
    highlight : str
        Site emphasised in panel A (bold label + heavier star); the subject of B/C.
    count_col, count_label : str
        Per-patch scene-count column and its human label (default L2A reflectance,
        ``n_gran_ovlp_rfl``; pass ``n_gran_ovlp_rad`` for L1B).
    annotate : str
        Panel-C per-cell label: ``"both"`` (id + n), ``"count"`` (n), or ``"none"``.
    panel_in : float
        Side length (inches) of each panel — keep generous so labels read.
    scale_km_a, scale_km_c : float
        Scale-bar lengths (km) for panels A and C.
    blank_label : str
        Placeholder note drawn in the empty panel B.
    suptitle : str or None
        Figure title (drawn only under a title-showing theme, e.g. ``ppt``).
    """
    import cartopy.crs as ccrs

    sc = style.scale()
    fig = plt.figure(figsize=(3.05 * panel_in, 1.12 * panel_in))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.16,
                          left=0.04, right=0.98, bottom=0.06, top=0.92)
    axA = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    _draw_site_map(axA, sites_gdf, highlight=highlight, scale_km=scale_km_a, sc=sc)
    _draw_blank_panel(axB, blank_label, sc=sc)
    _draw_coverage_panel(axC, patches, boundary, count_col=count_col,
                         count_label=count_label, annotate=annotate, cmap=cmap,
                         scale_km=scale_km_c, sc=sc)

    for ax, tag in ((axA, "A"), (axB, "B"), (axC, "C")):
        _panel_tag(ax, tag, sc=sc)

    if suptitle is None:
        suptitle = f"Study PICS overview — locator, {highlight} RGB, patch coverage"
    if style.show_title():
        fig.suptitle(suptitle, fontsize=13 * sc, fontweight="bold")
    return fig
