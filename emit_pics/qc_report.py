"""Per-patch QA/QC report orchestration (PDF pages + worksheet refresh).

Assembles one patch's QC deliverables from the outputs of the cube-extract step:
a multipanel per-scene PDF (:func:`emit_pics.display_qc_visuals.render_scene_figure`)
and a refreshed keep/remove worksheet (:func:`emit_pics.qc_gate.build_qc_sheet`).

It is the orchestration layer shared by two thin runners:

* ``scripts/run_display_qc_visuals.py`` — one patch, and
* ``scripts/run_PICS_allpatches_qc.py`` — every patch of a size in a PICS.

Design notes
------------
* **Cube-first, lazy login.** Each page renders from a cached pixel cube when one
  exists (fully offline); only if a scene must be *streamed* is Earthdata
  authentication triggered — once, via :class:`EarthdataLogin`.  A run over
  already-cached cubes therefore touches no network at all.
* **``make_pdf`` toggle.** With ``make_pdf=False`` the render loop is skipped
  entirely and only the worksheet is refreshed — the lean path when verdicts are
  entered through the interactive reviewer instead of a printed PDF.
* This module *may* import ``earthaccess`` and matplotlib, unlike the pure figure
  library :mod:`emit_pics.display_qc_visuals`; the caller sets the matplotlib
  backend (``Agg`` for PDF output).
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from emit_pics.id_matched_pairs import (
    fetch_granule,
    in_patch_mask,
    load_patch_geom_4326,
    sniff_delimiter,
)
from emit_pics.extract_patch_stats import (
    close_handles,
    load_patch_dict,
    pair_acquisitions,
    split_products,
    window_bounds,
    with_retry,
)
from emit_pics.config import pics_polygon
from emit_pics.display_qc_visuals import (
    granule_footprint_4326,
    open_obs,
    open_rfl_mask,
    render_scene_figure,
)
from emit_pics.display_spectra import derive_bad_bands, load_patch_spectra
from emit_pics.single_patch_cube_extract import (
    cube_to_render_inputs,
    read_cube_netcdf,
)
from emit_pics.qc_gate import build_qc_sheet


# ============================================================
# CONFIG
# ============================================================
@dataclass
class QCConfig:
    """All inputs for one patch's QC report (paths + options)."""
    pics_id: str
    patch_id: str
    # inputs / outputs
    dict_path: Path
    bnds_path: Path
    geom_path: Path
    qc_path: Path
    out_pdf: Path
    cube_dir: Path
    mask_dir: Path
    rfl_dir: Path
    rad_dir: Path
    # options
    rad_short: str = "EMITL1BRAD"
    rfl_short: str = "EMITL2ARFL"
    prefer_cube: bool = True
    draw_nadir: bool = False
    spectra_stat: str = "median"
    make_pdf: bool = True


def default_qc_config(repo_root, pics_id: str, patch_id: str, **opts) -> QCConfig:
    """Build a :class:`QCConfig` with the repo-standard path layout.

    *opts* override any option field (e.g. ``prefer_cube=False``,
    ``make_pdf=False``, ``spectra_stat="mean"``).
    """
    data = Path(repo_root) / "data"
    patches = data / "ROIs" / "patches"
    qc = data / "qc"
    return QCConfig(
        pics_id=pics_id, patch_id=patch_id,
        dict_path=patches / f"patch_EMIT_dict_{pics_id}.csv",
        bnds_path=patches / f"patches_{pics_id}_bnds.gpkg",
        geom_path=data / "geometry" / f"geometry_{pics_id}_{patch_id}.csv",
        qc_path=qc / f"{pics_id}_{patch_id}_qc.csv",
        out_pdf=qc / f"{pics_id}_{patch_id}_QC_snapshots.pdf",
        cube_dir=data / "pixel_cubes",
        mask_dir=data / "mask_summary",
        rfl_dir=data / "reflectance",
        rad_dir=data / "radiance",
        **opts,
    )


class EarthdataLogin:
    """One-shot, lazy Earthdata authentication shared across many patches.

    :meth:`ensure` logs in the first time it is called and is a no-op after; a
    run that only reads cached cubes never calls it, so it never touches the
    network.
    """

    def __init__(self):
        self._done = False

    def ensure(self):
        if not self._done:
            import earthaccess
            earthaccess.login(persist=True)
            print("Authenticated to NASA Earthdata (stream needed).")
            self._done = True


# ============================================================
# SPECTRA CONTEXT
# ============================================================
def load_spectra_context(cfg: QCConfig):
    """Load every patch scene's per-channel spectrum once for the spectra panels.

    Reads the per-scene geometry table (all scenes — QC is *where* keep/drop is
    decided, so no filtering) and pulls each scene's RAD/RFL per-channel stats.
    Returns ``({"entries", "bad_bands", "stat"}, missing)``; an absent geometry
    table yields an empty context (panels render empty rather than failing).
    """
    if not cfg.geom_path.exists():
        print(f"  no geometry table ({cfg.geom_path.name}); spectra panels will "
              f"be empty. Run run_single_patch_cube_extract.py to populate them.")
        return {"entries": [], "bad_bands": None, "stat": cfg.spectra_stat}, []

    scenes = pd.read_csv(cfg.geom_path)
    scenes["datetime"] = pd.to_datetime(scenes["datetime"])
    loaded, missing = load_patch_spectra(scenes, cfg.pics_id, cfg.patch_id,
                                         cfg.rfl_dir, cfg.rad_dir)
    bad_bands = derive_bad_bands(loaded) if loaded else None
    return {"entries": loaded, "bad_bands": bad_bands, "stat": cfg.spectra_stat}, missing


# ============================================================
# PER-ACQUISITION RENDER
# ============================================================
def _render_from_cube(cfg, cube_path, ts, dt, patch_geom, pics_poly, spectra, pdf) -> str:
    """Render one acquisition's page from a cached pixel cube — no network."""
    ds = read_cube_netcdf(cube_path)
    try:
        inp = cube_to_render_inputs(ds)
        if inp is None:
            print(f"  skipped {cube_path.name}: cube missing reflectance/mask/geometry")
            return "missing"
        fig = render_scene_figure(
            inp["rfl"], inp["obs"], mask=inp["mask"], bounds=inp["bounds"],
            patch_geom_4326=patch_geom, pics_poly=pics_poly,
            granule_poly=inp["granule_poly"], patch_id=cfg.patch_id,
            granule_ur=str(ds.attrs.get("rfl_ur", "")), acq_dt=dt, draw_nadir=False,
            spectra=spectra, acq_ts=ts,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote page for {ds.attrs.get('timestamp', '?')} (from cube)")
        return "produced"
    finally:
        ds.close()


def _process_acquisition(cfg, acq, patch_geom, pics_poly, spectra, pdf, login) -> str:
    """Render one acquisition's page (cube-first, else stream) and append to *pdf*.

    Returns ``"produced"`` / ``"missing"``, or raises on transient stream errors
    that survive retries.  Streaming triggers ``login.ensure()`` first, so a
    cube-only page never authenticates.
    """
    ts, dt = acq["timestamp"], acq["datetime"]

    # Cube-first: render locally if a cached cube exists for this acquisition.
    if cfg.prefer_cube:
        cube_path = cfg.cube_dir / f"cube_{cfg.pics_id}_{cfg.patch_id}_{ts}.nc"
        if cube_path.exists():
            return _render_from_cube(cfg, cube_path, ts, dt, patch_geom,
                                     pics_poly, spectra, pdf)

    # Streaming fallback — both products are required for a complete page.
    if acq["rfl_ur"] is None or acq["rad_ur"] is None:
        absent = [name for name, ur in
                  (("L2A_RFL", acq["rfl_ur"]), ("L1B_RAD", acq["rad_ur"]))
                  if ur is None]
        print(f"  skipped {ts}: missing {', '.join(absent)}")
        return "missing"

    login.ensure()  # only now do we need the network
    gran_l2a = fetch_granule(cfg.rfl_short, acq["rfl_ur"])
    rfl = with_retry(lambda: open_rfl_mask(gran_l2a), label=f"{ts} L2A")
    try:
        gran_l1b = fetch_granule(cfg.rad_short, acq["rad_ur"])
        obs = with_retry(lambda: open_obs(gran_l1b), label=f"{ts} OBS")
        try:
            mask = in_patch_mask(rfl["lon"], rfl["lat"], patch_geom)
            bounds = window_bounds(mask)
            if bounds is None:
                print(f"  skipped {ts}: no in-patch pixels")
                return "missing"
            fig = render_scene_figure(
                rfl, obs, mask=mask, bounds=bounds,
                patch_geom_4326=patch_geom, pics_poly=pics_poly,
                granule_poly=granule_footprint_4326(gran_l2a["umm"]),
                patch_id=cfg.patch_id, granule_ur=acq["rfl_ur"], acq_dt=dt,
                draw_nadir=cfg.draw_nadir, spectra=spectra, acq_ts=ts,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote page for {ts}")
            return "produced"
        finally:
            close_handles(obs["_handles"])
    finally:
        close_handles(rfl["_handles"])


# ============================================================
# PER-PATCH DRIVER
# ============================================================
def run_patch_qc(cfg: QCConfig, login: EarthdataLogin, dict_df=None) -> dict:
    """Build one patch's QC report; return a summary dict.

    When ``cfg.make_pdf`` is true, renders one PDF page per acquisition
    (cube-first) into ``cfg.out_pdf``; either way it refreshes the keep/remove
    worksheet ``cfg.qc_path`` from the geometry table.  *dict_df* may be a
    pre-loaded patch-granule dict (the batch runner loads it once and shares it);
    ``None`` loads it here.

    Returns ``{"patch_id", "n_acquisitions", "produced", "errored", "missing",
    "pdf", "qc_path", "n_scenes", "n_new"}`` (``pdf`` is ``None`` when no PDF was
    written).
    """
    summary = {"patch_id": cfg.patch_id, "n_acquisitions": 0, "produced": 0,
               "errored": 0, "missing": 0, "pdf": None, "qc_path": cfg.qc_path,
               "n_scenes": 0, "n_new": 0}

    if cfg.make_pdf:
        if dict_df is None:
            dict_df = load_patch_dict(cfg.dict_path, sep=sniff_delimiter(cfg.dict_path))
        rad_urs, rfl_urs = split_products(dict_df, cfg.patch_id)
        acquisitions = pair_acquisitions(rad_urs, rfl_urs)
        summary["n_acquisitions"] = len(acquisitions)
        print(f"Patch {cfg.patch_id}: {len(rad_urs)} RAD, {len(rfl_urs)} RFL "
              f"granule(s) -> {len(acquisitions)} acquisition(s)")

        if acquisitions:
            patch_geom = load_patch_geom_4326(cfg.bnds_path, cfg.patch_id)
            pics_poly = pics_polygon(cfg.pics_id)
            spectra, missing_spec = load_spectra_context(cfg)
            print(f"  {len(spectra['entries'])} scene(s) with radiance stats "
                  f"-> spectra overlay" + (f"; {len(missing_spec)} without a "
                  f"radiance CSV get no bold-red curve." if missing_spec else "."))

            cfg.out_pdf.parent.mkdir(parents=True, exist_ok=True)
            produced, errored, missing = [], [], []
            with PdfPages(cfg.out_pdf) as pdf:
                for k, acq in enumerate(acquisitions, 1):
                    ts = acq["timestamp"]
                    print(f"  [{k}/{len(acquisitions)}] {ts}")
                    try:
                        status = _process_acquisition(cfg, acq, patch_geom,
                                                      pics_poly, spectra, pdf, login)
                    except Exception as ex:
                        print(f"    FAILED {ts}: {type(ex).__name__}: {ex}")
                        errored.append(ts)
                        continue
                    (produced if status == "produced" else missing).append(ts)

            # A PdfPages with no saved pages leaves an unusable file; drop it.
            if not produced and cfg.out_pdf.exists():
                cfg.out_pdf.unlink()
            summary.update(produced=len(produced), errored=len(errored),
                           missing=len(missing),
                           pdf=(cfg.out_pdf if produced else None))
            if produced:
                print(f"  PDF -> {cfg.out_pdf} ({len(produced)} page(s))")
        else:
            print("  no acquisitions cover this patch; no PDF.")

    # Refresh the keep/remove worksheet (non-destructive: prior verdicts kept).
    if cfg.geom_path.exists():
        sheet, new_ts = build_qc_sheet(
            cfg.geom_path, cfg.qc_path, mask_dir=cfg.mask_dir,
            pics_id=cfg.pics_id, patch_id=cfg.patch_id,
        )
        summary.update(n_scenes=len(sheet), n_new=len(new_ts))
        print(f"  worksheet -> {cfg.qc_path} ({len(sheet)} scene(s)"
              f"{f', {len(new_ts)} new' if new_ts else ''})")
    else:
        print(f"  no geometry table ({cfg.geom_path.name}); worksheet not "
              "generated. Run run_single_patch_cube_extract.py first.")

    return summary
