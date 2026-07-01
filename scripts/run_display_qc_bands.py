#!/usr/bin/env python
"""Render per-scene QA/QC snapshots for one patch into a single PDF.

For every EMIT acquisition that fully covers the patch (read from the
patch-granule dictionary; no live search), this script:

1. Pairs the L1B_RAD and L2A_RFL granules by acquisition timestamp.
2. Opens each acquisition's files once (L2A reflectance + MASK, L1B OBS) and
   builds one multipanel QA/QC figure: RGB, every MASK band, SZA, VZA, and a
   context panel with the PICS / granule / patch boundaries and nadir track.
3. Writes one page per acquisition (chronological) to a single PDF.

A pair missing either product (L2A_RFL or L1B_RAD) is skipped and noted; no
pre-filtering on cloud / AOD is done at this stage.

Cube-first: if ``run_patch_extract.py`` already cached a pixel cube for an
acquisition, its page is rebuilt locally from that cube (no network, no
re-masking); otherwise the granule is streamed.  Set ``PREFER_CUBE = False`` to
always stream.  The nadir track needs whole-scene VZA and so is drawn only on the
streaming path.

Edit the user-configurable inputs below, then run::

    python scripts/run_display_qc_bands.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# USER-CONFIGURABLE INPUTS
# ============================================================
PICS_ID  = "Libya_4"       # underscored site id (input + output file paths)
PATCH_ID = "10_7_1"        # patch_id within the dictionary / bnds gpkg

RAD_SHORT_NAME = "EMITL1BRAD"   # L1B radiance (+ bundled OBS) collection
RFL_SHORT_NAME = "EMITL2ARFL"   # L2A reflectance (+ MASK) collection

PREFER_CUBE = True   # render from the cached pixel cube when present (no network)
DRAW_NADIR  = False  # draw the nadir track? requires a whole-scene VZA read per page
                     # (streaming path only; a cube has no whole-scene VZA)

# Paths (repo-relative — identical on every machine)
_REPO_ROOT = Path(__file__).resolve().parent.parent          # .../emit-pics
DATA_ROOT = _REPO_ROOT / "data"
PATCH_DIR = DATA_ROOT / "ROIs" / "patches"
CUBE_DIR  = DATA_ROOT / "pixel_cubes"
FIG_DIR   = _REPO_ROOT / "figures" / "QC"

DICT_PATH = PATCH_DIR / f"patch_EMIT_dict_{PICS_ID}.csv"
BNDS_PATH = PATCH_DIR / f"patches_{PICS_ID}_bnds.gpkg"
OUT_PDF   = FIG_DIR / f"{PICS_ID}_{PATCH_ID}_QC_snapshots.pdf"

# ============================================================
# Ensure the project root is on sys.path so 'emit_pics' is
# importable when running this script directly.
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (write straight to PDF)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import earthaccess

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
from emit_pics.display_qc_bands import (
    granule_footprint_4326,
    open_obs,
    open_rfl_mask,
    render_scene_figure,
)
from emit_pics.patch_extract import cube_to_render_inputs, read_cube_netcdf


def render_from_cube(cube_path, dt, patch_geom, pics_poly, pdf) -> str:
    """Render one acquisition's page from a cached pixel cube — no network.

    Returns ``"produced"`` (page written) or ``"missing"`` (the cube lacks the
    reflectance / MASK / geometry needed for a full page).  The nadir track is
    unavailable from a cube, so it is never drawn here.
    """
    ds = read_cube_netcdf(cube_path)
    try:
        inp = cube_to_render_inputs(ds)
        if inp is None:
            print(f"  skipped {cube_path.name}: cube missing reflectance/mask/geometry")
            return "missing"
        fig = render_scene_figure(
            inp["rfl"], inp["obs"], mask=inp["mask"], bounds=inp["bounds"],
            patch_geom_4326=patch_geom, pics_poly=pics_poly,
            granule_poly=inp["granule_poly"], patch_id=PATCH_ID,
            granule_ur=str(ds.attrs.get("rfl_ur", "")), acq_dt=dt, draw_nadir=False,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote page for {ds.attrs.get('timestamp', '?')} (from cube)")
        return "produced"
    finally:
        ds.close()


def process_acquisition(acq, patch_geom, pics_poly, pdf) -> str:
    """Render and append one acquisition's page to *pdf*.

    Cube-first: if a cached pixel cube exists for the acquisition it is rendered
    locally (no network); otherwise the granule is streamed.  Returns
    ``"produced"`` (page written), ``"missing"`` (a required product or in-patch
    pixels were absent, nothing written), or raises on transient errors that
    survive retries, for the caller to record as an error drop.
    """
    ts, dt = acq["timestamp"], acq["datetime"]

    # Cube-first: render locally if a cached cube exists for this acquisition.
    if PREFER_CUBE:
        cube_path = CUBE_DIR / f"cube_{PICS_ID}_{PATCH_ID}_{ts}.nc"
        if cube_path.exists():
            return render_from_cube(cube_path, dt, patch_geom, pics_poly, pdf)

    # Streaming fallback — both products are required for a complete page.
    if acq["rfl_ur"] is None or acq["rad_ur"] is None:
        absent = [name for name, ur in
                  (("L2A_RFL", acq["rfl_ur"]), ("L1B_RAD", acq["rad_ur"]))
                  if ur is None]
        print(f"  skipped {ts}: missing {', '.join(absent)}")
        return "missing"

    gran_l2a = fetch_granule(RFL_SHORT_NAME, acq["rfl_ur"])
    rfl = with_retry(lambda: open_rfl_mask(gran_l2a), label=f"{ts} L2A")
    try:
        gran_l1b = fetch_granule(RAD_SHORT_NAME, acq["rad_ur"])
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
                patch_id=PATCH_ID, granule_ur=acq["rfl_ur"], acq_dt=dt,
                draw_nadir=DRAW_NADIR,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote page for {ts}")
            return "produced"
        finally:
            close_handles(obs["_handles"])
    finally:
        close_handles(rfl["_handles"])


def main() -> None:
    # 0. Authenticate to NASA Earthdata (persisted for streaming reads).
    earthaccess.login(persist=True)
    print("Authenticated to NASA Earthdata.\n")

    # 1. Load the patch-granule dict and pair acquisitions.
    #    Auto-detect the delimiter (the dict on disk is comma-separated, though
    #    a tab-separated copy is also supported).
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    rad_urs, rfl_urs = split_products(dict_df, PATCH_ID)
    acquisitions = pair_acquisitions(rad_urs, rfl_urs)
    print(f"Patch {PATCH_ID}: {len(rad_urs)} RAD, {len(rfl_urs)} RFL granule(s) "
          f"-> {len(acquisitions)} acquisition(s)")
    if not acquisitions:
        print("No acquisitions cover this patch. Exiting.")
        sys.exit(1)

    # 2. Patch geometry + PICS boundary (both EPSG:4326 for the context panel).
    patch_geom = load_patch_geom_4326(BNDS_PATH, PATCH_ID)
    pics_poly = pics_polygon(PICS_ID)

    # 3. Render every acquisition into one PDF; bucket outcomes.
    os.makedirs(FIG_DIR, exist_ok=True)
    produced, errored, missing = [], [], []
    with PdfPages(OUT_PDF) as pdf:
        for k, acq in enumerate(acquisitions, 1):
            ts = acq["timestamp"]
            print(f"\n[{k}/{len(acquisitions)}] {ts}")
            try:
                status = process_acquisition(acq, patch_geom, pics_poly, pdf)
            except Exception as ex:
                print(f"  FAILED {ts}: {type(ex).__name__}: {ex}")
                errored.append(ts)
                continue
            {"produced": produced, "missing": missing}[status].append(ts)

    # A PdfPages with no saved pages leaves an unusable file; drop it.
    if not produced and OUT_PDF.exists():
        OUT_PDF.unlink()

    # 4. Completeness summary.
    print("\n--- Completeness ---")
    print(f"  acquisitions expected:    {len(acquisitions)}")
    print(f"  pages produced:           {len(produced)}")
    print(f"  dropped by errors:        {len(errored)}")
    print(f"  no page (missing prod):   {len(missing)}")
    if produced:
        print(f"\nWrote {len(produced)} page(s) -> {OUT_PDF}")
    else:
        print("\nNo pages produced; no PDF written.")
    if errored:
        print(f"  WARNING: {len(errored)} acquisition(s) dropped by errors "
              f"after retries -- PDF is INCOMPLETE; consider re-running: "
              f"{errored}")


if __name__ == "__main__":
    main()
