"""Per-scene spectra plots for one PICS/patch (QC-surviving scenes).

Pure function library — no top-level execution, no hardcoded paths, no side
effects on import. Given a QC-filtered per-scene table (the same geometry table
``id_matched_pairs`` consumes) and the per-acquisition per-channel RAD/RFL stats
CSVs (from ``run_patch_extract``), this loads each surviving scene's spectrum and
draws one overlay figure per product — every scene a separate series.

Design mirrors the delta/drift steps:

* Scenes are the ones that **passed the manual QC gate** (the runner applies
  :func:`emit_pics.qc_gate.apply_qc_filter` exactly as ``run_id_matched_pairs``
  does), so this runs *parallel* to matched-pair detection on the same inputs.
* The deep **water-vapor absorption bands** are defined by data availability —
  the wavelengths where L2A reflectance is fill (:func:`absorption_band_wavelengths`)
  — and are blanked in **both** products (reflectance is already NaN there;
  radiance is atmospherically contaminated), so every curve breaks in the same
  windows.
* Series are coloured along a **sequential ramp by acquisition date** (oldest ->
  newest) with a cycling line style and a date-labelled legend.

Typical usage from a runner::

    from emit_pics.display_spectra import (
        PRODUCTS, derive_bad_bands, load_patch_spectra, plot_patch_spectra,
    )
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emit_pics.delta_calculations import (
    absorption_band_wavelengths,
    load_scene_spectrum,
)
from emit_pics.id_matched_pairs import acq_timestamp

# ============================================================
# CONSTANTS
# ============================================================
#: High-resolution export DPI for the PNGs.
FIG_DPI = 1000

#: Line styles cycled across scenes (in addition to the sequential colour ramp),
#: so that scenes close in date — hence close in colour — are still separable.
LINE_STYLES = ["-", "--", "-.", ":"]

#: Sequential colormap sampled oldest -> newest, so temporal order reads off hue.
DATE_CMAP = "viridis"

#: nm tolerance for matching a channel to a derived bad-band centre wavelength.
ABS_TOL_NM = 0.5

#: Per-product display config. The statistic (``median``/``mean``) is chosen by
#: the caller; the plotted column is ``f"{stat}_{product}"`` (e.g. ``median_rfl``).
#: ``subdir`` is the figure sub-folder; radiance units are EMIT L1B's native
#: microflick (adjust ``y_label`` if you convert upstream).
PRODUCTS = {
    "rfl": {"product": "rfl", "subdir": "reflectance",
            "name": "Reflectance", "y_label": "Reflectance (unitless)"},
    "rad": {"product": "rad", "subdir": "radiance",
            "name": "Radiance",
            "y_label": "Radiance (µW cm⁻² sr⁻¹ nm⁻¹)"},
}


# ============================================================
# LOADING
# ============================================================
def load_patch_spectra(scenes: pd.DataFrame, pics_id: str, patch_id: str,
                       rfl_dir: Path, rad_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load each surviving scene's per-channel spectrum, sorted by acquisition time.

    Parameters
    ----------
    scenes : pd.DataFrame
        The QC-filtered per-scene table; needs ``scene_id`` and ``datetime``.
    pics_id, patch_id : str
        Identify the per-channel CSV filenames.
    rfl_dir, rad_dir : Path
        Directories holding the reflectance / radiance per-channel CSVs.

    Returns
    -------
    (loaded, missing) : tuple of list of dict
        ``loaded`` — scenes whose radiance CSV exists (radiance anchors the join,
        so it is required to draw either product), each a dict with keys
        ``scene_id``, ``timestamp``, ``datetime``, ``label`` (``YYYY-MM-DD``) and
        ``spec`` (the spectrum frame from :func:`load_scene_spectrum`). ``missing``
        — scenes whose radiance CSV is absent (``spec`` is None), so the runner
        can report them. Both are ordered oldest -> newest.
    """
    recs = scenes.sort_values("datetime").reset_index(drop=True)
    loaded, missing = [], []
    for _, r in recs.iterrows():
        sid = r["scene_id"]
        spec = load_scene_spectrum(sid, pics_id, patch_id, rfl_dir, rad_dir)
        dt = pd.to_datetime(r["datetime"])
        entry = {
            "scene_id": sid,
            "timestamp": acq_timestamp(sid),
            "datetime": dt,
            "label": dt.strftime("%Y-%m-%d"),
            "spec": spec,
        }
        (loaded if spec is not None else missing).append(entry)
    return loaded, missing


def derive_bad_bands(specs: list[dict]) -> np.ndarray:
    """Water-vapor bad-band centre wavelengths (nm) from the first RFL-bearing scene.

    The bad bands are a fixed instrument property (identical in every L2A
    granule), so a single reflectance-bearing scene defines the set for the whole
    patch. Returns an empty array if no scene carries reflectance.
    """
    for e in specs:
        wls = absorption_band_wavelengths(e["spec"])
        if len(wls):
            return wls
    return np.array([], dtype=float)


def mask_absorption(wl: np.ndarray, values: np.ndarray,
                    bad_bands: np.ndarray | None, tol: float = ABS_TOL_NM) -> np.ndarray:
    """Return *values* with entries at absorption-band wavelengths set to NaN.

    A NaN produces a **gap** in the plotted line, so blanking the absorption
    windows breaks every curve there. Reflectance is already NaN in those bands;
    this makes radiance break identically.
    """
    v = np.asarray(values, dtype=float).copy()
    if bad_bands is not None and len(bad_bands):
        bad = np.asarray(bad_bands, dtype=float)
        hit = (np.abs(np.asarray(wl, dtype=float)[:, None] - bad[None, :]) <= tol).any(axis=1)
        v[hit] = np.nan
    return v


# ============================================================
# PLOTTING
# ============================================================
def _legend_cols(n: int) -> int:
    """Number of legend columns for *n* scenes (keeps the legend from running off)."""
    return 1 if n <= 22 else 2 if n <= 44 else 3


def plot_patch_spectra(specs: list[dict], product_cfg: dict, stat: str,
                       bad_bands: np.ndarray, pics_id: str, patch_id: str):
    """Overlay every scene's spectrum for one product; return ``(fig, n_plotted)``.

    Each scene is one line, coloured along a sequential ramp by acquisition date
    (oldest -> newest) with a cycling line style; the legend lists the dates.
    Absorption bands are blanked (plain gaps). Scenes with no finite data for
    this product (e.g. reflectance absent) are silently skipped and excluded from
    the returned count. ``matplotlib`` is imported lazily.

    Parameters
    ----------
    specs : list of dict
        The ``loaded`` entries from :func:`load_patch_spectra` (date-ordered).
    product_cfg : dict
        One value of :data:`PRODUCTS` (``product``/``name``/``y_label``).
    stat : str
        ``"median"`` or ``"mean"``; the plotted column is ``f"{stat}_{product}"``.
    bad_bands : np.ndarray
        Absorption-band centre wavelengths to blank (from :func:`derive_bad_bands`).
    pics_id, patch_id : str
        Labels for the title.

    Returns
    -------
    (matplotlib.figure.Figure, int)
        The figure and the number of scenes actually drawn.
    """
    import matplotlib.pyplot as plt

    stat_col = f"{stat}_{product_cfg['product']}"
    cmap = plt.get_cmap(DATE_CMAP)
    n = len(specs)

    fig, ax = plt.subplots(figsize=(14, 7))
    plotted = 0
    for i, e in enumerate(specs):
        spec = e["spec"]
        if stat_col not in spec:
            continue
        wl = spec["central_wl"].to_numpy(dtype=float)
        y = mask_absorption(wl, spec[stat_col].to_numpy(dtype=float), bad_bands)
        if not np.isfinite(y).any():
            continue  # no data for this product in this scene (e.g. RFL absent)
        color = cmap(i / max(n - 1, 1))
        ax.plot(wl, y, color=color, ls=LINE_STYLES[i % len(LINE_STYLES)],
                lw=0.9, label=e["label"])
        plotted += 1

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(product_cfg["y_label"])
    ax.set_title(
        f"{pics_id} {patch_id} — per-scene {product_cfg['name'].lower()} spectra "
        f"({stat}, {plotted} QC-passed scene{'s' if plotted != 1 else ''})"
    )
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(title="Acquisition date", fontsize=7, ncol=_legend_cols(plotted),
                  loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, plotted
