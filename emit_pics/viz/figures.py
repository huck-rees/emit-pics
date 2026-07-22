"""Publication figure builders — pure ``(DataFrame) -> Figure``.

Each function reads a pipeline CSV (already on disk) and returns a styled
Matplotlib figure. Builders are **theme-agnostic**: they pull fonts/weights from
the active :mod:`emit_pics.viz.style` theme's rcParams, sizes from
:func:`style.figsize`, and the few non-rc sizes (scatter ``s``, capsize) from
:func:`style.scale` — so the same builder renders both the manuscript and the PPT
variant. Titles are added only when the active theme asks (slides), never for the
manuscript (its caption lives in LaTeX).
"""

import warnings

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from emit_pics.viz import style


def drift_spectrum(df, *, size="double", aspect=0.60, sig_col="sig_bh",
                   ylim=(-2.5, 2.5)):
    """BRDF pooled drift spectrum gamma(lambda): rad (top) / rfl (bottom) vs wavelength.

    Reads ``data/brdf/brdf_drift_spectrum.csv``. Numerically unstable channels are
    dropped; the bootstrap 95% CI is drawn as whiskers, and channels significant
    by *sig_col* (default BH-FDR) are filled in the product colour while the rest
    are open grey — redundant with the per-product marker shape for greyscale.

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


def _add_basemap(ax, crs, alpha):
    """Faded Esri satellite basemap behind the map (needs contextily + network).

    Degrades gracefully: a missing package or a failed tile fetch (offline) warns
    and leaves the figure otherwise intact. On-figure attribution is suppressed —
    credit Esri / World Imagery in the caption.
    """
    try:
        import contextily as cx
    except ImportError:
        warnings.warn("contextily not installed; skipping satellite basemap "
                      "(conda install -c conda-forge contextily).")
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()   # basemap must not change extent
    try:
        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldImagery,
                       alpha=alpha, attribution="", zorder=-5)
    except Exception as exc:                     # network / tile errors are non-fatal
        warnings.warn(f"basemap fetch failed ({exc}); rendering without it.")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _add_scalebar(ax, length_m, label, sc):
    """A framed km scale bar (data-unit length) in the lower-left corner."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.font_manager import FontProperties
    bar = AnchoredSizeBar(ax.transData, length_m, label, loc="lower left",
                          pad=0.3, borderpad=0.5, sep=3, frameon=True,
                          color="black", size_vertical=0,
                          fontproperties=FontProperties(size=7 * sc))
    bar.patch.set(facecolor="white", edgecolor="0.6", alpha=0.75)
    ax.add_artist(bar)


def _add_north(ax, sc):
    """North arrow (grid north = up in the projected CRS) in the top-right corner."""
    for lw, color, z in ((3.4 * sc, "white", 5), (1.3 * sc, "black", 6)):
        ax.annotate("", xy=(0.96, 0.97), xytext=(0.96, 0.87), zorder=z,
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    ax.text(0.96, 0.985, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8 * sc, fontweight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8 * sc, foreground="white")])


def _add_boundary_label(ax, site_proj, label, sc):
    """Label the PICS boundary just inside its top edge."""
    minx, miny, maxx, maxy = site_proj.total_bounds
    ax.text((minx + maxx) / 2, maxy - 0.02 * (maxy - miny), label,
            ha="center", va="top", zorder=6, fontsize=8 * sc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.8))


def patch_coverage(patches, site, *, size="double", aspect=0.62,
                   count_cols=(("n_gran_ovlp_rad", "L1B radiance"),
                               ("n_gran_ovlp_rfl", "L2A reflectance")),
                   annotate="both", cmap="viridis_r",
                   basemap=True, basemap_alpha=0.5, scalebar_km=20,
                   north=True, boundary_label=None):
    """Per-product maps of patches shaded by fully-covering granule count.

    One panel per product (``count_cols``), each drawing the patch grid in its
    native (projected) CRS shaded by the granule count, over a faded satellite
    basemap, with the labelled PICS boundary, a km scale bar, a north arrow, and a
    shared colour scale + colourbar. Reads the already-generated
    ``patches_{site}_bnds.gpkg`` (geometry + ``n_gran_ovlp_*`` columns amended by
    ``run_id_granules.py``) and the PICS boundary — no Earthdata re-search.

    Parameters
    ----------
    patches : geopandas.GeoDataFrame
        Patches for one size, with ``patch_id``, ``geometry`` (projected CRS), and
        the count columns named in *count_cols*.
    site : geopandas.GeoDataFrame
        Single-row PICS boundary (any CRS; reprojected to *patches*' CRS). Its
        ``site_name`` labels the boundary unless *boundary_label* overrides it.
    annotate : str
        Per-cell label: ``"both"`` (patch id + count), ``"count"`` (count only,
        cleaner/larger), or ``"none"``.
    cmap : str
        Colormap; ``viridis_r`` so darker = more granules (colour-blind-safe).
    basemap : bool
        Draw a faded Esri satellite basemap (needs ``contextily`` + network;
        skipped with a warning if unavailable).
    basemap_alpha : float
        Basemap opacity (lower = more faded).
    scalebar_km : int or None
        Scale-bar length in km (``None`` to omit).
    north : bool
        Draw a north arrow.
    boundary_label : str or None
        Text for the PICS boundary label (default: the site's ``site_name``).
    """
    cols = [c for c, _ in count_cols]
    vals = np.concatenate([patches[c].fillna(0).to_numpy(dtype=float) for c in cols])
    vmin, vmax = float(vals.min()), float(vals.max())
    norm, cmap_obj = Normalize(vmin=vmin, vmax=vmax), plt.get_cmap(cmap)
    sc = style.scale()
    annot_fs = (7.0 if annotate == "both" else 9.0) * sc

    site_proj = site.to_crs(patches.crs)
    if boundary_label is None:
        boundary_label = (str(site.iloc[0]["site_name"])
                          if "site_name" in site.columns else "PICS boundary")

    fig, axes = plt.subplots(1, len(count_cols), figsize=style.figsize(size, aspect),
                             layout="constrained")
    axes = np.atleast_1d(axes)

    for ax, (col, title) in zip(axes, count_cols):
        gmap = patches.copy()
        gmap["_n"] = gmap[col].fillna(0).astype(float)
        gmap.plot(ax=ax, column="_n", cmap=cmap, vmin=vmin, vmax=vmax,
                  edgecolor="white", linewidth=0.4 * sc, legend=False, zorder=1)
        site_proj.boundary.plot(ax=ax, color="black", linewidth=0.8 * sc, zorder=3)

        if basemap:
            _add_basemap(ax, patches.crs, basemap_alpha)

        if annotate != "none":
            for _, prow in gmap.iterrows():
                n = int(prow["_n"])
                r, g, b, _ = cmap_obj(norm(prow["_n"]))
                tc = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
                lbl = f"{prow['patch_id']}\n{n}" if annotate == "both" else str(n)
                cpt = prow.geometry.centroid
                ax.annotate(lbl, (cpt.x, cpt.y), ha="center", va="center",
                            fontsize=annot_fs, color=tc, zorder=5,
                            path_effects=[pe.withStroke(
                                linewidth=0.6 * sc,
                                foreground="black" if tc == "white" else "white")])

        _add_boundary_label(ax, site_proj, boundary_label, sc)
        if scalebar_km:
            _add_scalebar(ax, scalebar_km * 1000.0, f"{scalebar_km} km", sc)
        if north:
            _add_north(ax, sc)

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), shrink=0.7,
                        label="Fully-covering scenes per patch")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if style.show_title():
        fig.suptitle("EMIT granule coverage per patch")
    return fig
