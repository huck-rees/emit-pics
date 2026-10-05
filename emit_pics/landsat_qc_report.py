"""Per-patch Landsat QC report orchestration (PDF pages + worksheet refresh).

The Landsat analog of :mod:`emit_pics.qc_report`, simpler because the Landsat QC
cubes are local GeoTIFFs (no streaming, no Earthdata login): every page renders
offline from a cached cube. Assembles one patch's QC deliverables:

* a multipanel per-scene PDF (:func:`emit_pics.landsat_qc_visuals.render_scene_figure`),
  one page per QC-cube scene, and
* a refreshed keep/remove worksheet via :func:`emit_pics.qc_gate.build_qc_sheet`
  (reused unchanged — the Landsat geometry table carries the columns it needs;
  ``mask_dir=None`` so ``pct_flagged`` is simply blank).

The worksheet is the same opt-out schema and gate the EMIT side uses, so the
existing :func:`emit_pics.qc_gate.apply_qc_filter` / ``is_reviewed`` consume it
directly downstream.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from emit_pics import paths
from emit_pics.config import canonical_token
from emit_pics.qc_gate import build_qc_sheet
from emit_pics.landsat_cube_extract import read_cube_geotiff
from emit_pics.landsat_qc_visuals import load_toa_spectra, render_scene_figure


@dataclass
class LandsatQCConfig:
    """Inputs for one Landsat patch's QC report (paths + options)."""
    pics_id: str
    patch_id: str
    geom_path: Path
    qc_path: Path
    cube_dir: Path
    out_pdf: Path
    make_pdf: bool = True


def default_qc_config(pics_id, patch_id, **opts) -> LandsatQCConfig:
    """Build a :class:`LandsatQCConfig` from the repo-standard Landsat path layout.

    ``cube_dir`` follows ``paths.LS_CUBE_DIR``, so a caller that redirected the
    cubes off-drive with :func:`emit_pics.paths.set_ls_cube_dir` before calling
    this picks up the external location automatically.
    """
    tok = canonical_token(pics_id)
    return LandsatQCConfig(
        pics_id=tok, patch_id=patch_id,
        geom_path=paths.ls_geometry(pics_id, patch_id),
        qc_path=paths.ls_qc_sheet(pics_id, patch_id),
        cube_dir=paths.LS_CUBE_DIR,
        out_pdf=paths.LS_QC_DIR / f"{tok}_{patch_id}_QC_snapshots.pdf",
        **opts,
    )


def run_patch_qc(cfg: LandsatQCConfig) -> dict:
    """Build one Landsat patch's QC report; return a summary dict.

    With ``cfg.make_pdf`` true, renders one offline PDF page per cached QC cube
    (in geometry-table/date order) into ``cfg.out_pdf``; either way it refreshes
    the keep/remove worksheet from the geometry table. Scenes lacking a cube are
    skipped in the PDF but still appear in the worksheet (so they can be reviewed
    once a cube exists).

    Returns ``{patch_id, n_scenes, n_new, produced, missing_cube, pdf}``.
    """
    summary = {"patch_id": cfg.patch_id, "n_scenes": 0, "n_new": 0,
               "produced": 0, "missing_cube": 0, "pdf": None}

    if not cfg.geom_path.exists():
        print(f"  no geometry table ({cfg.geom_path.name}); run the extractor first.")
        return summary

    geom = pd.read_csv(cfg.geom_path).sort_values("timestamp").reset_index(drop=True)
    geom["datetime"] = pd.to_datetime(geom["datetime"])

    if cfg.make_pdf:
        spectra = load_toa_spectra(cfg.pics_id, cfg.patch_id)
        produced, missing = [], []
        cfg.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(cfg.out_pdf) as pdf:
            for k, row in geom.iterrows():
                ts = str(row["timestamp"])
                cube_path = cfg.cube_dir / f"cube_{cfg.pics_id}_{cfg.patch_id}_{ts}.tif"
                if not cube_path.exists():
                    missing.append(ts)
                    print(f"  [{k+1}/{len(geom)}] {ts}: no cube -> skipped in PDF")
                    continue
                cube = read_cube_geotiff(cube_path)
                fig = render_scene_figure(cube, pics_id=cfg.pics_id,
                                          patch_id=cfg.patch_id, scene_ts=ts,
                                          acq_dt=row["datetime"], spectra=spectra)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                produced.append(ts)
                print(f"  [{k+1}/{len(geom)}] {ts}: page written")
        if not produced and cfg.out_pdf.exists():
            cfg.out_pdf.unlink()          # a PdfPages with no pages is unusable
        summary.update(produced=len(produced), missing_cube=len(missing),
                       pdf=(cfg.out_pdf if produced else None))
        if produced:
            print(f"  PDF -> {cfg.out_pdf} ({len(produced)} page(s))")

    # Refresh the keep/remove worksheet (reuses the EMIT gate; mask_dir=None).
    sheet, new_ts = build_qc_sheet(cfg.geom_path, cfg.qc_path, mask_dir=None,
                                   pics_id=cfg.pics_id, patch_id=cfg.patch_id)
    summary.update(n_scenes=len(sheet), n_new=len(new_ts))
    print(f"  worksheet -> {cfg.qc_path} ({len(sheet)} scene(s)"
          f"{f', {len(new_ts)} new' if new_ts else ''})")
    return summary
