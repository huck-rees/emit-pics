"""Publication figure builders — pure ``(data) -> Figure``.

Each function reads already-generated pipeline outputs and returns a Matplotlib
figure. Builders run under an active :mod:`emit_pics.viz.style` theme (which sets
the font family + output format); size-sensitive builders also read the theme's
width presets and scale, while a dense reference figure may fix its own generous
size so all its labels stay legible.
"""

import numpy as np
import matplotlib.pyplot as plt
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
