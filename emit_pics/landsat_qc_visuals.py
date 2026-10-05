"""Build per-scene QA/QC multipanel figures for the Landsat 9 control.

The Landsat analog of :mod:`emit_pics.display_qc_visuals`, but much simpler: a
Landsat QC cube is a small georeferenced GeoTIFF (7 OLI-2 TOA bands + per-pixel
SZA/VZA; see :func:`emit_pics.landsat_cube_extract.write_cube_geotiff`), not a
streamed xarray cube with MASK/OBS band sets. For one scene this renders:

* **RGB** true-color (red/green/blue TOA), fixed stretch so brightness is
  comparable page-to-page across a run;
* **NIR** (865 nm) TOA reflectance — the EMIT-significant band, good for spotting
  thin cloud/dust the visible misses;
* **SZA** and **VZA** per-pixel maps (deg);
* a **TOA spectrum** panel overlaying *every* scene's 7-band in-patch spectrum
  (current scene bold, the rest faded) so a scene with a normal-looking RGB but an
  unrealistic spectrum can be caught during review — the same idea as the EMIT
  spectra panels, at 7 bands.

Pure figure library: no I/O beyond reading the spectra CSVs it is handed a loader
for, no ``__main__``, no backend selection (the caller picks Agg for PDF or an
interactive backend for the reviewer). Reads only what the cube + TOA per-band
tables already hold — no network.
"""

import glob
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from emit_pics import paths
from emit_pics.landsat_cube_extract import OLI2_BANDS

# ============================================================
# CONSTANTS
# ============================================================
#: Band indices into the cube's first 7 layers (OLI2_BANDS order:
#: coastal, blue, green, red, nir08, swir16, swir22).
_IDX = {b["asset"]: i for i, b in enumerate(OLI2_BANDS)}
RGB_IDX = (_IDX["red"], _IDX["green"], _IDX["blue"])
NIR_IDX = _IDX["nir08"]
SZA_LAYER, VZA_LAYER = len(OLI2_BANDS), len(OLI2_BANDS) + 1   # last two layers

#: Fixed RGB stretch (TOA reflectance) so pages are comparable across a run; the
#: mild gamma lifts desert midtones without blowing out dune crests.
RGB_VMAX = 0.45
RGB_GAMMA = 1.4


# ============================================================
# RGB
# ============================================================
def rgb_composite(data, vmax=RGB_VMAX, gamma=RGB_GAMMA):
    """True-ish color from a cube's (band, y, x) array, ONE shared scale for R/G/B.

    A single ``vmax`` across the three bands preserves the warm desert band ratios
    (red > green > blue) that independent per-band stretching would flatten to gray.
    NaN (invalid) pixels render black.
    """
    r, g, b = (data[i].astype("float64") for i in RGB_IDX)
    stack = np.dstack([r, g, b])
    out = np.clip(stack / vmax, 0, 1) ** (1 / gamma)
    return np.nan_to_num(out, nan=0.0)


# ============================================================
# SPECTRA CONTEXT
# ============================================================
def load_toa_spectra(pics_id, patch_id):
    """Every scene's 7-band in-patch TOA spectrum for the overlay panel.

    Reads the per-scene ``perchannel_toa`` CSVs for the patch (the same tables the
    drift fit consumes). Returns ``{scene_ts: {"wl", "median"}}`` (both 1-D arrays,
    band-center order). Missing/absent tables simply yield fewer overlay curves.
    """
    out = {}
    for f in glob.glob(str(paths.LS_TOA_DIR / f"{pics_id}_{patch_id}_*_perchannel_toa.csv")):
        m = re.search(r"_(\d{8}T\d{6})_perchannel", f)
        if not m:
            continue
        df = pd.read_csv(f).sort_values("central_wl")
        out[m.group(1)] = {"wl": df["central_wl"].to_numpy(float),
                           "median": df["median_toa"].to_numpy(float)}
    return out


# ============================================================
# PER-SCENE FIGURE
# ============================================================
def render_scene_figure(cube, *, pics_id, patch_id, scene_ts, acq_dt=None,
                        spectra=None, fig=None, resize=True):
    """Render one Landsat scene's QC page: RGB, NIR, SZA, VZA + a TOA spectrum panel.

    Parameters
    ----------
    cube : dict
        A cube from :func:`emit_pics.landsat_cube_extract.read_cube_geotiff`
        (``data`` = (band, y, x); layers 0-6 TOA, 7 = SZA, 8 = VZA).
    pics_id, patch_id, scene_ts : str
        Labels for the title.
    acq_dt : datetime, optional
        Acquisition datetime for the title.
    spectra : dict, optional
        Output of :func:`load_toa_spectra` — overlays every scene's spectrum with
        *scene_ts* drawn bold. ``None`` skips the overlay (only this scene's line).
    fig : matplotlib Figure, optional
        Reuse an existing figure (the interactive reviewer passes one); ``None``
        creates a new one. *resize* controls whether the reused figure is resized.

    Returns the figure.
    """
    data = cube["data"]
    if fig is None:
        fig = plt.figure(figsize=(14, 7))
    else:
        fig.clf()
        if resize:
            fig.set_size_inches(14, 7)

    gs = fig.add_gridspec(2, 4, height_ratios=[2.4, 1.0], hspace=0.32, wspace=0.28)

    # --- RGB ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb_composite(data)); ax.set_title("RGB true-color"); ax.axis("off")

    # --- NIR ---
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(data[NIR_IDX], cmap="magma", vmin=0.3, vmax=0.75)
    ax.set_title("NIR 865 nm TOA"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- SZA / VZA ---
    for col, layer, name in ((2, SZA_LAYER, "SZA"), (3, VZA_LAYER, "VZA")):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data[layer], cmap="viridis")
        ax.set_title(f"{name} (deg)"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- TOA spectrum overlay (all scenes faded, this one bold) ---
    ax = fig.add_subplot(gs[1, :])
    if spectra:
        for ts, s in spectra.items():
            if ts == scene_ts:
                continue
            ax.plot(s["wl"], s["median"], color="0.75", lw=0.6, zorder=1)
        this = spectra.get(scene_ts)
        if this is not None:
            ax.plot(this["wl"], this["median"], color="#d62728", lw=2.0,
                    marker="o", ms=4, zorder=3, label="this scene")
            ax.legend(loc="lower center", fontsize=8, ncol=1)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Median TOA reflectance")
    ax.set_title("In-patch TOA spectrum (this scene vs. all)")
    ax.grid(alpha=0.3)

    dt_str = f"  |  {acq_dt}" if acq_dt is not None else ""
    fig.suptitle(f"Landsat 9 TOA QC — {pics_id.replace('_', ' ')}  {patch_id}  "
                 f"{scene_ts}{dt_str}", y=0.99, fontsize=12)
    return fig
