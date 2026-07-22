"""Publication figure builders — pure ``(DataFrame) -> Figure``.

Each function reads a pipeline CSV (already on disk) and returns a styled
Matplotlib figure. Builders are **theme-agnostic**: they pull fonts/weights from
the active :mod:`emit_pics.viz.style` theme's rcParams, sizes from
:func:`style.figsize`, and the few non-rc sizes (scatter ``s``, capsize) from
:func:`style.scale` — so the same builder renders both the manuscript and the PPT
variant. Titles are added only when the active theme asks (slides), never for the
manuscript (its caption lives in LaTeX).
"""

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from emit_pics.viz import style


def drift_spectrum(df, *, size="double", aspect=0.60, sig_col="sig_bh"):
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
    """
    df = df[~df["unstable"]].copy()
    sc = style.scale()
    fig, axes = plt.subplots(2, 1, figsize=style.figsize(size, aspect),
                             sharex=True, layout="constrained")

    for ax, prod, ylab in zip(
            axes, ("rad", "rfl"),
            ("Radiance drift (%/yr)", "Reflectance drift (%/yr)")):
        d = df[df["product"] == prod].sort_values("wavelength")
        st = style.PRODUCT_STYLE[prod]
        yerr = np.vstack([d["gamma_pct_yr"] - d["ci_lo_pct_yr"],
                          d["ci_hi_pct_yr"] - d["gamma_pct_yr"]])
        ax.errorbar(d["wavelength"], d["gamma_pct_yr"], yerr=yerr, fmt="none",
                    ecolor="0.7", elinewidth=0.6 * sc, capsize=1.2 * sc, zorder=1)
        ns, sg = d[~d[sig_col]], d[d[sig_col]]
        ax.scatter(ns["wavelength"], ns["gamma_pct_yr"], s=9 * sc, marker=st["marker"],
                   facecolors="none", edgecolors="0.5", linewidths=0.5 * sc, zorder=3)
        ax.scatter(sg["wavelength"], sg["gamma_pct_yr"], s=11 * sc, marker=st["marker"],
                   color=st["color"], zorder=4)
        ax.axhline(0.0, color="k", lw=0.6 * sc, zorder=2)
        ax.set_ylabel(ylab)

    axes[1].set_xlabel("Wavelength (nm)")
    handles = [
        Line2D([0], [0], marker="o", color="0.5", ls="none", mfc="none",
               label="not significant"),
        Line2D([0], [0], marker="o", color="k", ls="none", label="BH-significant"),
    ]
    axes[0].legend(handles=handles, loc="best")
    if style.show_title():
        axes[0].set_title("EMIT pooled radiometric drift spectrum")
    return fig
