"""Build per-scene QA/QC multipanel figures over a patch.

For one EMIT acquisition this library streams the in-patch pixels and renders a
single figure array: RGB true-color, every L2A MASK band (binary flags plus the
continuous AOD550 / H2O bands), the OBS-derived SZA and VZA, and a context panel
overlaying the PICS, granule, and patch boundaries with the nadir track.  Below
the map panels two adjacent spectra panels (reflectance | radiance) overlay
*every* scene covering the patch — the current page's scene bold red, the rest
faded grey — so a scene with a normal-looking RGB/MASK but an unrealistic
spectrum can be caught during the manual QC pass.  It is *display only*: no
statistics are written to disk and no differencing is done.

Pure function library: no top-level execution, no hardcoded parameters, no
``__main__`` block, no import side effects.  ``earthaccess.login`` is not called
here; the caller authenticates before streaming.

Low-level helpers proven elsewhere in the package are reused rather than
duplicated:

* timestamp parsing / band-by-label lookup / KDTree patch mask / granule lookup
  from :mod:`emit_pics.id_matched_pairs`;
* asset-URL selection and cube windowing from :mod:`emit_pics.extract_patch_stats`;
* absorption-band masking and per-product labels from
  :mod:`emit_pics.display_spectra` (the spectra panels reuse the same per-channel
  RAD/RFL stats CSVs and masking as the standalone spectra step).

Map panels are windowed to the in-patch block and pixels outside the patch
polygon are blanked (NaN -> transparent), so every panel except the final
context panel is patch-specific.  The context panel is full-scene by design.
The spectra panels are not patch-windowed — they plot the pre-computed
per-channel in-patch statistics loaded by the caller.
"""

import warnings

import earthaccess
import numpy as np
import geopandas as gpd
import shapely
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from emit_pics.id_matched_pairs import band_index
from emit_pics.extract_patch_stats import (
    OBS_TOKEN,
    RFL_FILLS,
    RFL_TOKEN,
    MASK_TOKEN,
    find_asset_url,
    window_2d,
)
from emit_pics.display_spectra import PRODUCTS, mask_absorption

# CONSTANTS
#: MASK band labels that are binary flags (rendered discrete green/red).  Any
#: MASK band whose label is NOT in this set is treated as continuous (colorbar).
BINARY_FLAGS = {
    "Cloud flag", "Cirrus flag", "Water flag",
    "Spacecraft Flag", "Dilated Cloud Flag", "Aggregate Flag",
}

#: OBS band labels located by string (never a hardcoded index).
OBS_SZA_LABEL = "To-sun zenith (0 to 90 degrees from zenith)"
OBS_VZA_LABEL = "To-sensor zenith (0 to 90 degrees from zenith)"

#: Fill value common to MASK / OBS bands (NaN'd before display).
FILL = -9999.0

#: RGB display configuration.  Bands are chosen by nearest wavelength (nm) and
#: the stretch is FIXED so brightness is comparable page-to-page across a run.
RGB_WL_NM = (640.0, 550.0, 470.0)   # (R, G, B) target wavelengths
RGB_STRETCH = (0.0, 0.45)           # (vmin, vmax) reflectance, shared by all pages

#: Discrete flag colors: 0 = good (green), 1 = flagged (red).
FLAG_GOOD_COLOR = "#2ca02c"
FLAG_BAD_COLOR  = "#d62728"

#: Continuous colormaps.
MASK_CONT_CMAP = "viridis"   # AOD550, H2O
GEOM_CMAP      = "plasma"    # SZA, VZA

#: Grid width (panels per row); height adapts to the panel count.
NCOLS = 4

#: Spectra-panel styling.  The current page's scene is drawn bold red on top of
#: every other patch scene (thin grey), so an anomalous spectrum stands out even
#: when the RGB/MASK panels look clean.
SPECTRA_CURRENT_COLOR = "#d62728"   # this page's scene (bold, on top)
SPECTRA_BG_COLOR      = "0.6"        # every other scene covering the patch
SPECTRA_STAT_DEFAULT  = "median"     # in-patch statistic plotted per scene


# GRANULE OPENERS
def open_rfl_mask(granule) -> dict:
    """Open the L2A reflectance file and its sibling MASK file once.

    Reflectance / wavelengths / lon / lat come from the reflectance handle; the
    full mask cube and its band labels come from the MASK handle.  The whole
    mask band set is exposed (not just the Aggregate Flag), so the figure can
    render every band the file reports.

    Parameters
    ----------
    granule : earthaccess result object
        The L2A_RFL granule (also bundles MASK and RFLUNCERT).

    Returns
    -------
    dict
        ``refl`` (lazy cube, bands last), ``wl`` (1-D), ``lon``/``lat`` (2-D),
        ``mask_cube`` (lazy, bands last), ``mask_labels`` (list of str), and
        ``_handles`` (objects to close).
    """
    rfl_url = find_asset_url(granule, RFL_TOKEN)
    fh = earthaccess.open([rfl_url])[0]
    ds_rfl = xr.open_dataset(fh, engine="h5netcdf")
    ds_wl = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")
    ds_loc = xr.open_dataset(fh, engine="h5netcdf", group="location")

    mask_url = find_asset_url(granule, MASK_TOKEN)
    fh_m = earthaccess.open([mask_url])[0]
    ds_m = xr.open_dataset(fh_m, engine="h5netcdf")
    ds_mb = xr.open_dataset(fh_m, engine="h5netcdf", group="sensor_band_parameters")

    return {
        "refl": ds_rfl["reflectance"],
        "wl": np.asarray(ds_wl["wavelengths"].values, dtype=float),
        "lon": np.asarray(ds_loc["lon"].values, dtype=float),
        "lat": np.asarray(ds_loc["lat"].values, dtype=float),
        "mask_cube": ds_m["mask"],
        "mask_labels": [str(x) for x in ds_mb["mask_bands"].values],
        "_handles": [ds_rfl, ds_wl, ds_loc, fh, ds_m, ds_mb, fh_m],
    }


def open_obs(granule) -> dict:
    """Open the bundled OBS geometry file from an L1B_RAD granule once.

    OBS shares the (downtrack, crosstrack) grid with RAD/RFL, so lon/lat are not
    read here; the caller reuses the L2A lon/lat for the windowing and footprint
    panel.

    Parameters
    ----------
    granule : earthaccess result object
        The L1B_RAD granule (also bundles OBS).

    Returns
    -------
    dict
        ``obs_cube`` (lazy, bands last), ``obs_bands`` (raw label array for
        :func:`band_index`), and ``_handles`` (objects to close).
    """
    obs_url = find_asset_url(granule, OBS_TOKEN)
    fh = earthaccess.open([obs_url])[0]
    ds_obs = xr.open_dataset(fh, engine="h5netcdf")
    ds_ob = xr.open_dataset(fh, engine="h5netcdf", group="sensor_band_parameters")
    return {
        "obs_cube": ds_obs["obs"],
        "obs_bands": ds_ob["observation_bands"].values,
        "_handles": [ds_obs, ds_ob, fh],
    }


# PURE DATA HELPERS
def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    """Index of the channel whose central wavelength is closest to *target_nm*."""
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=float) - target_nm)))


def patch_only(arr: np.ndarray, in_sub: np.ndarray, fills=()) -> np.ndarray:
    """Blank a windowed 2-D array outside the patch (and at fill values).

    *fills* are set to NaN (``np.isclose``), then every pixel where *in_sub* is
    ``False`` is set to NaN, so display and per-panel stats use in-patch pixels
    only.
    """
    a = np.asarray(arr, dtype=float).copy()
    for f in fills:
        a[np.isclose(a, f)] = np.nan
    a[~in_sub] = np.nan
    return a


def build_rgba(refl_win: np.ndarray, wavelengths: np.ndarray, in_sub: np.ndarray,
               stretch=RGB_STRETCH, rgb_wl=RGB_WL_NM, fills=RFL_FILLS) -> np.ndarray:
    """Assemble a fixed-stretch RGBA image from a windowed reflectance cube.

    Out-of-patch and fill pixels are made fully transparent so the patch reads
    cleanly against the page.

    Parameters
    ----------
    refl_win : np.ndarray
        ``(h, w, bands)`` reflectance window.
    wavelengths : np.ndarray
        ``(bands,)`` central wavelengths (drives R/G/B band selection).
    in_sub : np.ndarray
        ``(h, w)`` boolean in-patch mask over the window.

    Returns
    -------
    np.ndarray
        ``(h, w, 4)`` RGBA in [0, 1].
    """
    ri, gi, bi = (nearest_band(wavelengths, t) for t in rgb_wl)
    rgb = np.stack(
        [refl_win[..., ri], refl_win[..., gi], refl_win[..., bi]], axis=-1
    ).astype(float)
    for f in fills:
        rgb[np.isclose(rgb, f)] = np.nan

    lo, hi = stretch
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)

    alpha = in_sub.astype(float)
    alpha[np.any(~np.isfinite(rgb), axis=-1)] = 0.0
    rgb = np.nan_to_num(rgb, nan=0.0)
    return np.dstack([rgb, alpha])


def band_stats(arr: np.ndarray, binary: bool = False) -> dict:
    """Per-panel summary over finite in-patch pixels.

    Binary panels return ``{"pct_flagged": ...}``; continuous panels return
    ``mean`` / ``std`` / ``min`` / ``max``.
    """
    a = np.asarray(arr, dtype=float)
    finite = a[np.isfinite(a)]
    if binary:
        pct = (100.0 * np.count_nonzero(finite > 0) / finite.size
               if finite.size else np.nan)
        return {"pct_flagged": pct}
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def band_units(label: str) -> str:
    """Colorbar unit string for a continuous MASK band, inferred from its label."""
    if "g cm-2" in label or "H2O" in label:
        return r"g cm$^{-2}$"
    if "AOD" in label:
        return "AOD (unitless)"
    if "zenith" in label or "azimuth" in label or "degrees" in label:
        return "degrees"
    return ""


def compute_nadir_track(vza_full: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                        fill: float = FILL):
    """Full-scene nadir track: per downtrack row, the min-VZA crosstrack pixel.

    Parameters
    ----------
    vza_full : np.ndarray
        ``(downtrack, crosstrack)`` to-sensor zenith over the WHOLE scene.
    lon, lat : np.ndarray
        ``(downtrack, crosstrack)`` geolocation over the whole scene.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(nadir_lon, nadir_lat)`` 1-D arrays (possibly empty).
    """
    v = np.asarray(vza_full, dtype=float).copy()
    v[np.isclose(v, fill)] = np.nan
    nlon, nlat = [], []
    for dr in range(v.shape[0]):
        row = v[dr, :]
        if np.all(np.isnan(row)):
            continue
        ct = int(np.nanargmin(row))
        nlon.append(lon[dr, ct])
        nlat.append(lat[dr, ct])
    return np.asarray(nlon), np.asarray(nlat)


def granule_footprint_4326(umm: dict):
    """Granule footprint polygon (EPSG:4326) from the UMM GPolygon boundary.

    All GPolygons are unioned, so a multi-polygon footprint is handled rather
    than silently taking only the first ring.
    """
    gp = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
    polys = [
        shapely.geometry.Polygon(
            [(p["Longitude"], p["Latitude"]) for p in g["Boundary"]["Points"]]
        )
        for g in gp
    ]
    return polys[0] if len(polys) == 1 else shapely.unary_union(polys)


def load_pics_polygon(pics_gpkg, pics_name: str):
    """Load one PICS boundary (EPSG:4326) by site name from the PICS GeoPackage.

    *pics_name* is the underscored id (e.g. ``"Libya_4"``); the site_name field
    in the gpkg uses spaces, so the underscore is reverted for the lookup.
    """
    g = gpd.read_file(pics_gpkg)
    spaced = pics_name.replace("_", " ")
    sub = g[g["site_name"] == spaced]
    if sub.empty:
        raise ValueError(
            f"PICS site '{spaced}' not found in {pics_gpkg}. "
            f"Available: {sorted(g['site_name'].tolist())}"
        )
    return gpd.GeoSeries([sub.geometry.iloc[0]], crs=g.crs).to_crs(4326).iloc[0]


# FIGURE ASSEMBLY
def _draw_flag(ax, fig, arr):
    """Discrete green(0)/red(1) flag panel, no colorbar."""
    cmap = ListedColormap([FLAG_GOOD_COLOR, FLAG_BAD_COLOR])
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm,
              interpolation="nearest")
    st = band_stats(arr, binary=True)
    txt = (f"{st['pct_flagged']:.1f}% flagged"
           if np.isfinite(st["pct_flagged"]) else "no valid px")
    _stat_box(ax, txt)


def _draw_continuous(ax, fig, arr, cmap_name, units):
    """Continuous colorbar panel with a mu/sigma/[min,max] stat box."""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(units, fontsize=7)
    st = band_stats(arr, binary=False)
    if np.isfinite(st["mean"]):
        txt = (f"$\\mu$={st['mean']:.3f}\n$\\sigma$={st['std']:.3f}\n"
               f"[{st['min']:.2f}, {st['max']:.2f}]")
    else:
        txt = "no valid px"
    _stat_box(ax, txt)


def _stat_box(ax, txt):
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))


def _draw_spectra_panel(ax, spectra, product, current_ts):
    """Overlay every patch scene's spectrum for one product; highlight the current.

    Every scene covering the patch is drawn thin grey; the current page's scene
    (matched by acquisition *current_ts*) is drawn bold red on top so an
    unrealistic spectrum is obvious even when the map panels look clean.  The
    water-vapor absorption bands are blanked to NaN gaps (same masking as
    :mod:`emit_pics.display_spectra`), so every curve breaks in the same windows.

    Parameters
    ----------
    ax : matplotlib axes
    spectra : dict or None
        ``{"entries": [...], "bad_bands": np.ndarray, "stat": "median"|"mean"}``
        as assembled by the runner (``entries`` are the date-ordered scene dicts
        from :func:`emit_pics.display_spectra.load_patch_spectra`).  ``None`` /
        empty draws a placeholder note (no per-channel CSVs available).
    product : str
        ``"rfl"`` or ``"rad"`` — selects the ``PRODUCTS`` config and the plotted
        column ``f"{stat}_{product}"``.
    current_ts : str or None
        Acquisition timestamp (``YYYYMMDDThhmmss``) of the current page's scene;
        the matching entry is the bold-red curve.  No match -> only grey curves
        plus a "no spectrum" note (soft-skip).
    """
    cfg = PRODUCTS[product]
    ax.set_title(f"{cfg['name']} spectra", fontsize=9)
    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel(cfg["y_label"], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    entries = (spectra or {}).get("entries") or []
    if not entries:
        ax.text(0.5, 0.5, "no per-channel spectra\n(run run_single_patch_cube_extract.py)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.4")
        return

    stat = spectra.get("stat", SPECTRA_STAT_DEFAULT)
    bad_bands = spectra.get("bad_bands")
    stat_col = f"{stat}_{product}"

    n_bg = 0
    current = None
    for e in entries:
        spec = e["spec"]
        if stat_col not in spec:
            continue
        wl = spec["central_wl"].to_numpy(dtype=float)
        y = mask_absorption(wl, spec[stat_col].to_numpy(dtype=float), bad_bands)
        if not np.isfinite(y).any():
            continue  # this scene has no data for this product (e.g. RFL absent)
        if current_ts is not None and e["timestamp"] == current_ts:
            current = (wl, y, e["label"])  # draw last so it sits on top
            continue
        ax.plot(wl, y, color=SPECTRA_BG_COLOR, lw=0.6, alpha=0.6, zorder=1)
        n_bg += 1

    if current is not None:
        wl, y, label = current
        ax.plot(wl, y, color=SPECTRA_CURRENT_COLOR, lw=2.0, zorder=3,
                label=f"this scene ({label})")
        ax.legend(fontsize=7, loc="upper right", frameon=True)
    else:
        ax.text(0.5, 0.98, "this scene: no spectrum", transform=ax.transAxes,
                ha="center", va="top", fontsize=8,
                color=SPECTRA_CURRENT_COLOR, fontweight="bold")

    ax.margins(x=0.01)
    ax.set_title(
        f"{cfg['name']} spectra "
        f"({n_bg} other scene{'s' if n_bg != 1 else ''} in grey)", fontsize=9)


def render_scene_figure(rfl: dict, obs: dict, *, mask: np.ndarray, bounds,
                        patch_geom_4326, pics_poly,
                        patch_id: str, granule_ur: str, acq_dt,
                        granule_poly=None, ncols: int = NCOLS,
                        draw_nadir: bool = False,
                        spectra=None, acq_ts=None, fig=None, resize=True):
    """Render one acquisition's multipanel QA/QC figure.

    Map panels (a grid ``ncols`` wide): RGB true-color, every MASK band (binary
    flags discrete green/red, continuous bands with a colorbar), SZA, VZA, and a
    full-scene context panel (PICS / granule / patch boundaries, optionally with
    the nadir track).  All map panels are windowed to the in-patch block; the
    context panel is full-scene.  Below the grid, a full-width row holds two
    spectra panels (reflectance | radiance) overlaying every patch scene with the
    current scene bold red (see :func:`_draw_spectra_panel`).

    Parameters
    ----------
    rfl, obs : dict
        Outputs of :func:`open_rfl_mask` and :func:`open_obs`.
    mask : np.ndarray
        Full-scene ``(downtrack, crosstrack)`` in-patch boolean mask.
    bounds : tuple
        ``(r0, r1, c0, c1)`` window from ``window_bounds(mask)`` (not ``None``).
    patch_geom_4326, pics_poly : shapely geometry
        Patch polygon and PICS boundary in EPSG:4326 (for the context panel).
    granule_poly : shapely geometry or None
        Granule footprint in EPSG:4326 (context overlay).  ``None`` omits the
        outline — e.g. when a cached cube carries no footprint.  Build it from a
        live granule with :func:`granule_footprint_4326`.
    patch_id, granule_ur : str
    acq_dt : datetime
        Acquisition datetime (page header).
    draw_nadir : bool
        Draw the nadir track on the context panel.  This requires reading the
        **whole-scene** VZA array, so it is off by default; the VZA panel itself
        is always windowed (cheap).  Enable only when the cosmetic track is worth
        the full-granule read.
    spectra : dict or None
        Pre-loaded per-scene spectra for the two bottom panels
        (``{"entries", "bad_bands", "stat"}``; see :func:`_draw_spectra_panel`).
        ``None`` / empty draws placeholder notes — the map panels still render.
    acq_ts : str or None
        Acquisition timestamp (``YYYYMMDDThhmmss``) of this page's scene, used to
        pick out the bold-red curve in the spectra panels.
    fig : matplotlib.figure.Figure or None
        Existing figure to draw into (cleared first).  ``None`` creates a new one.
        The interactive reviewer passes one figure and reuses it across scenes.
    resize : bool
        When reusing a passed ``fig``, whether to re-apply the computed figure
        size on this redraw.  ``True`` (default, the PDF path) keeps page sizes
        consistent; ``False`` (the interactive reviewer) preserves a window the
        user has resized or maximized instead of snapping it back each scene.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r0, r1, c0, c1 = bounds
    in_sub = mask[r0:r1, c0:c1]
    gran_poly = granule_poly   # already a polygon (or None); no UMM parsing here

    # --- OBS geometry indices (by label) ---
    sza_idx = band_index(obs["obs_bands"], OBS_SZA_LABEL)
    vza_idx = band_index(obs["obs_bands"], OBS_VZA_LABEL)

    # --- VZA panel is always windowed; the nadir track (full-scene read) is opt-in ---
    if draw_nadir:
        vza_full = np.asarray(obs["obs_cube"][:, :, vza_idx].values, dtype=float)
        nadir_lon, nadir_lat = compute_nadir_track(vza_full, rfl["lon"], rfl["lat"])
        vza_win = patch_only(vza_full[r0:r1, c0:c1], in_sub, fills=(FILL,))
    else:
        nadir_lon = nadir_lat = np.asarray([])
        vza_win = patch_only(
            window_2d(obs["obs_cube"][:, :, vza_idx], bounds), in_sub, fills=(FILL,)
        )
    sza_win = patch_only(
        window_2d(obs["obs_cube"][:, :, sza_idx], bounds), in_sub, fills=(FILL,)
    )

    # --- RGB from the windowed reflectance cube ---
    refl_win = np.asarray(rfl["refl"][r0:r1, c0:c1, :].values, dtype=float)
    rgba = build_rgba(refl_win, rfl["wl"], in_sub)

    # --- assemble panel descriptors ---
    panels = [{"title": "RGB true-color", "kind": "rgb", "data": rgba}]
    for i, lbl in enumerate(rfl["mask_labels"]):
        band = patch_only(
            window_2d(rfl["mask_cube"][:, :, i], bounds), in_sub, fills=(FILL,)
        )
        if lbl in BINARY_FLAGS:
            panels.append({"title": lbl, "kind": "flag", "data": band})
        else:
            panels.append({"title": lbl, "kind": "cont", "data": band,
                           "cmap": MASK_CONT_CMAP, "units": band_units(lbl)})
    panels.append({"title": "SZA (to-sun zenith)", "kind": "cont",
                   "data": sza_win, "cmap": GEOM_CMAP, "units": "degrees"})
    panels.append({"title": "VZA (to-sensor zenith)", "kind": "cont",
                   "data": vza_win, "cmap": GEOM_CMAP, "units": "degrees"})
    panels.append({"title": "footprints + nadir track", "kind": "context"})

    n = len(panels)
    nrows = int(np.ceil(n / ncols))

    # Layout: the map grid (nrows x ncols) sits on top; one extra full-width row
    # below holds the two spectra panels.  A GridSpec lets the spectra row span
    # the width while the map panels keep the square 4-per-row arrangement.
    # A caller may pass ``fig`` to redraw into an existing window (the interactive
    # reviewer reuses one figure across scenes); it is cleared first.
    figsize = (4 * ncols, 4 * nrows + 4.5)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()
        # Re-applying the fixed figsize on every redraw fights an interactive
        # caller that has sized/maximized its window (the QC reviewer): each new
        # scene would snap the window back to `figsize` and overflow the monitor.
        # `resize=False` keeps the reused window's current size; the PDF path
        # (default True) still gets consistent page dimensions.
        if resize:
            fig.set_size_inches(figsize)
    gs = fig.add_gridspec(nrows + 1, ncols,
                          height_ratios=[1] * nrows + [1.15])
    axes = np.array([fig.add_subplot(gs[r, c])
                     for r in range(nrows) for c in range(ncols)])

    for ax, panel in zip(axes, panels):
        kind = panel["kind"]
        if kind == "rgb":
            ax.imshow(panel["data"], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        elif kind == "flag":
            _draw_flag(ax, fig, panel["data"])
            ax.set_xticks([]); ax.set_yticks([])
        elif kind == "cont":
            _draw_continuous(ax, fig, panel["data"], panel["cmap"], panel["units"])
            ax.set_xticks([]); ax.set_yticks([])
        elif kind == "context":
            gpd.GeoSeries([pics_poly], crs=4326).boundary.plot(
                ax=ax, color="black", lw=1.6, label="PICS")
            if gran_poly is not None:
                gpd.GeoSeries([gran_poly], crs=4326).boundary.plot(
                    ax=ax, color="C1", lw=1.4, label="granule")
            gpd.GeoSeries([patch_geom_4326], crs=4326).boundary.plot(
                ax=ax, color="C0", lw=1.8, label="patch")
            if nadir_lon.size:
                ax.plot(nadir_lon, nadir_lat, color="red", lw=1.2, ls="--",
                        label="nadir track")
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            ax.legend(fontsize=6, loc="upper right")
            ax.set_aspect("equal")
        ax.set_title(panel["title"], fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    # --- bottom full-width row: reflectance | radiance spectra ---
    half = max(ncols // 2, 1)
    ax_rfl = fig.add_subplot(gs[nrows, 0:half])
    ax_rad = fig.add_subplot(gs[nrows, half:ncols])
    _draw_spectra_panel(ax_rfl, spectra, "rfl", acq_ts)
    _draw_spectra_panel(ax_rad, spectra, "rad", acq_ts)

    # shared flag legend (single per page)
    flag_handles = [
        Patch(facecolor=FLAG_GOOD_COLOR, edgecolor="none", label="good (0)"),
        Patch(facecolor=FLAG_BAD_COLOR, edgecolor="none", label="flagged (1)"),
    ]
    fig.legend(handles=flag_handles, loc="lower center", ncol=2, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01),
               title="MASK flags")

    fig.suptitle(
        f"{patch_id}  |  {granule_ur}\n"
        f"{acq_dt:%Y-%m-%d}   MASK bands + OBS geometry + footprints + spectra "
        f"(in-patch pixels for maps; current scene bold red in spectra)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
