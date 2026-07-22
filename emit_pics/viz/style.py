"""Figure theme registry: one style applied once, two output targets.

A publication figure's *content* (what is plotted, from the data) is separated
from its *style* (fonts, sizes, weights, output format). Figure builders in
:mod:`emit_pics.viz.figures` never hardcode any of that — they read the **active
theme** set by :func:`theme`, so the same builder renders both a journal figure
and a slide figure.

Two themes ship:

* ``manuscript`` — Elsevier / RSE artwork spec: Arial, >=7 pt lettering, >=0.25 pt
  lines, sized to the fixed column widths (90 / 140 / 185 mm), vector **PDF**
  (fonts embedded, text kept editable) + **SVG** master for Affinity.
* ``ppt`` — the same figures scaled up for slides: larger fonts/weights, 16:9
  slide-region sizes, **SVG** (native + editable in PowerPoint 365) + 300 dpi PNG.

Both themes share **one** colour palette and marker/line-style mapping, so a
figure is recognisably the same in the paper and the deck — only scale, size,
output format, and on-figure titles differ. Add a third target (e.g. a poster)
by adding one entry to the registries below; no figure code changes.

Usage::

    with style.theme("manuscript"):
        fig = figures.drift_spectrum(df, size="double")
        style.save(fig, "fig03_drift_spectrum")          # -> figures/publication/manuscript/
"""

import contextlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from emit_pics import paths

#: mm -> inches (figures are built at final size, never scaled afterwards).
MM = 1.0 / 25.4

#: Available theme names.
THEMES = ("manuscript", "ppt")

# ------------------------------------------------------------------ colour identity
# Wong colour-blind-safe palette; every series also carries a distinct MARKER and
# LINE STYLE so figures survive greyscale / colour-blind viewing (redundant
# encoding), as the RSE spec requires.
PRODUCT_STYLE = {
    "rad": {"color": "#0072B2", "marker": "o", "ls": "-"},    # blue,  circle, solid
    "rfl": {"color": "#009E73", "marker": "s", "ls": "--"},   # green, square, dashed
}

_SITE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
_SITE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def site_styles(sites):
    """Stable ``{site: {color, marker}}`` mapping (sorted, colour-blind-safe)."""
    return {s: {"color": _SITE_COLORS[i % len(_SITE_COLORS)],
                "marker": _SITE_MARKERS[i % len(_SITE_MARKERS)]}
            for i, s in enumerate(sorted(sites))}

# --------------------------------------------------------------------- rc + sizes
#: rcParams shared by every theme (fonts embedded & kept as editable text).
_BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,          # embed TrueType; text stays editable (Affinity)
    "ps.fonttype": 42,
    "svg.fonttype": "none",      # SVG keeps live text
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
}

#: Per-theme rcParams overrides (fonts >=7 pt; lines well above the 0.25 pt floor).
_RC = {
    "manuscript": {
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "lines.linewidth": 0.9, "lines.markersize": 3.0,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    },
    "ppt": {
        "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 16,
        "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "lines.linewidth": 1.8, "lines.markersize": 6.0,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    },
}

#: Semantic width class -> width (inches), per theme. Manuscript = Elsevier column
#: widths; ppt = 16:9 slide regions.
_SIZES = {
    "manuscript": {"single": 90 * MM, "onehalf": 140 * MM, "double": 185 * MM},
    "ppt":        {"single": 6.5,     "onehalf": 8.5,       "double": 11.0},
}

#: Extra multiplier for the few sizes that don't flow through rcParams (scatter
#: ``s``, errorbar ``capsize``, explicit offsets), so they scale with the theme.
_SCALE = {"manuscript": 1.0, "ppt": 1.9}

#: Whether builders draw an on-figure title (slides yes; manuscript caption is in LaTeX).
_SHOW_TITLE = {"manuscript": False, "ppt": True}

#: Output formats + raster dpi + background, per theme.
_SAVE = {
    "manuscript": {"formats": ("pdf", "svg"), "dpi": 500, "transparent": False},
    "ppt":        {"formats": ("svg", "png"), "dpi": 300, "transparent": False},
}

_ACTIVE = None


# ------------------------------------------------------------------ theme control
@contextlib.contextmanager
def theme(name):
    """Activate a theme: apply its rcParams (scoped) and mark it active.

    rcParams are set via :func:`matplotlib.rc_context`, so they never leak past
    the ``with`` block. Raises ``ValueError`` for an unknown theme.
    """
    global _ACTIVE
    if name not in _RC:
        raise ValueError(f"Unknown theme '{name}'. Choose from {list(_RC)}.")
    prev = _ACTIVE
    _ACTIVE = name
    try:
        with mpl.rc_context({**_BASE_RC, **_RC[name]}):
            yield name
    finally:
        _ACTIVE = prev


def active():
    """The active theme name; raises if called outside a :func:`theme` block."""
    if _ACTIVE is None:
        raise RuntimeError("No active figure theme; call within `with style.theme(...)`.")
    return _ACTIVE


# ------------------------------------------------------------------ size / scale helpers
def figwidth(size_class):
    """Width (inches) of *size_class* for the active theme."""
    sizes = _SIZES[active()]
    if size_class not in sizes:
        raise ValueError(f"Unknown size '{size_class}'. Choose from {list(sizes)}.")
    return sizes[size_class]


def figsize(size_class, aspect):
    """``(width, height)`` in inches for *size_class* at height/width = *aspect*."""
    w = figwidth(size_class)
    return (w, w * aspect)


def scale():
    """Theme scale multiplier for non-rcParams sizes (scatter ``s``, capsize, ...)."""
    return _SCALE[active()]


def show_title():
    """Whether the active theme wants an on-figure title."""
    return _SHOW_TITLE[active()]


# ------------------------------------------------------------------------- saving
def save(fig, name, *, out_dir=None):
    """Write *fig* in the active theme's formats; return the paths written.

    Files land in ``figures/publication/<theme>/`` unless *out_dir* is given (e.g.
    the manuscript repo's ``figures/`` — the runner's ``--out``), in which case
    they are written flat into that directory. The figure is saved at its exact
    built size (no bbox cropping), so the final width matches the spec.
    """
    t = active()
    cfg = _SAVE[t]
    base = Path(out_dir) if out_dir is not None else (paths.FIGURES / "publication" / t)
    base.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in cfg["formats"]:
        p = base / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=cfg["dpi"], transparent=cfg["transparent"])
        written.append(p)
    plt.close(fig)
    return written
