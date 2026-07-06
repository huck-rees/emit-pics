# EMIT PICS Temporal Stability

Assessing the radiometric temporal stability of NASA's EMIT imaging spectrometer over North African pseudo-invariant calibration sites (PICS).

## Repository layout

```
emit_pics/      Python package — pure function libraries (no side effects on import)
config/         sites.yaml — PICS site definitions (bounding boxes + metadata)
scripts/        Command-line entry points (edit the inputs at the top, then run)
data/           Pipeline inputs/outputs (contents gitignored; folder kept via .gitkeep)
figures/        Generated figures (contents gitignored; folder kept via .gitkeep)
notebooks/      Exploration and analysis notebooks
tests/          Unit tests
```

All paths are **repo-relative**: every script writes under `data/` and
`figures/` inside the repo, so it runs identically on any machine straight
after a `git clone` — no per-machine path edits. The *contents* of `data/` and
`figures/` (patches, footprints, dictionaries, per-channel stats, geometry
tables, pixel cubes, figures) are gitignored; only `.gitkeep` placeholders are
committed so the folders exist on a fresh clone.

`config/sites.yaml` is the **single source of truth** for PICS site geometry —
each site is an axis-aligned bounding box in EPSG:4326. Nothing external needs
to be copied to run the pipeline; the site rectangle is built from the YAML, and
the companion `northafrican_pics.gpkg` (for GIS use) can be regenerated from it.

## Package modules

| Module | Role |
| --- | --- |
| `patch_generator` | Tile a PICS boundary into nested, fully-contained square patches. |
| `id_granules` | Search NASA Earthdata; determine which granules fully cover each patch. |
| `patch_extract` | **Unified per-acquisition extraction** — open each granule once, build the in-patch mask once, dispatch to all extractors, and persist the raw pixel cube. |
| `extract_patch_stats` | Low-level openers / windowing / per-channel statistics (shared helpers). |
| `id_matched_pairs` | Matched-pair detection and SZA/VZA-over-time plotting (pure). |
| `display_qc_bands` | Per-scene QA/QC multipanel figure rendering. |
| `config` | PICS site geometry from `config/sites.yaml` — `pics_polygon` / `pics_geodataframe` build the ROI rectangle; `write_pics_gpkg` regenerates the GeoPackage. |

The libraries are side-effect-free; each is driven by a thin runner in
`scripts/` that handles authentication, file I/O, and figure/CSV output.

## Environment setup

```bash
conda env create -f environment.yml
conda activate emit-pics
```

You also need NASA Earthdata credentials. The streaming scripts authenticate via
`earthaccess.login()`, which looks for credentials in `~/.netrc` (or `_netrc` on
Windows) and falls back to an interactive prompt. See the
[earthaccess docs](https://earthaccess.readthedocs.io/) for details.

That is the *only* per-machine setup. Because paths are repo-relative and site
geometry lives in `config/sites.yaml`, a fresh clone is ready to run once the
conda env exists and Earthdata credentials are in place — no data files or path
edits to copy over.

## Pipeline

Each script exposes its inputs (site id, patch id, paths) as constants at the
top — edit those, then run. Steps 1–2 are global to a PICS site; steps 3–6 are
per-patch analyses that all branch off the patch–granule dictionary.

1. **`python scripts/run_patch_generator.py`**
   Tile the PICS site into nested square patches →
   `patches_{site}_bnds.gpkg` + `patches_{site}_atts.csv`. *(No Earthdata.)*

2. **`python scripts/run_id_granules.py`**
   Search Earthdata and find granule coverage per patch → granule footprints,
   per-patch coverage counts, and the **patch–granule dictionary**
   `patch_EMIT_dict_{site}.csv` that every downstream step reads.

Then, for a chosen patch:

3. **`python scripts/run_patch_extract.py`**
   The single streaming pass. Opens each granule **once**, builds the in-patch
   mask **once**, and writes per-channel L1B radiance / L2A reflectance stats,
   the per-scene SZA/VZA **geometry table**, MASK-band summaries, and (opt-in)
   the raw in-patch **pixel cube** (NetCDF). Optional Aggregate-Flag screening of
   the per-channel stats.

4. **`python scripts/run_display_qc_bands.py`**
   Per-scene QA/QC snapshot PDF (RGB, every MASK band, SZA/VZA, footprint
   context) for one patch, **and** it creates/refreshes the patch's **QC
   worksheet** `data/qc/{site}_{patch}_qc.csv` (one row per scene).

   *→ Manual step:* open the PDF, and in the worksheet set `keep=FALSE` (with a
   `reason`: cloud / coverage / artifact / other) for any bad scene, then set
   `reviewed=TRUE` in the top row. Verdicts are **opt-out** (default keep) and
   preserved across re-runs.

5. **`python scripts/run_id_matched_pairs.py`**
   Pure **local** post-processor (no Earthdata): reads the geometry table,
   **applies the QC gate** (drops `keep=FALSE` scenes; refuses to run until the
   worksheet is `reviewed=TRUE`, unless `REQUIRE_QC_REVIEW=False`), then writes
   the matched-pair CSV and the SZA/VZA-over-time figure.

6. **`python scripts/run_delta_calculations.py`**
   Pure **local** post-processor (no Earthdata): joins the matched-pairs CSV
   (step 5) with the per-channel RAD/RFL stats (step 3) and writes a tidy long
   CSV of the pairwise **t1→t2** percent change in reflectance and radiance
   (median & mean) at every wavelength — one row per `(matched pair × wavelength)`
   → `data/delta_r/{site}_{patch}_delta.csv`. No aggregation or figures; the
   cross-PICS analysis is a separate future step.

   The two scenes are aligned **by band order** (band *i* ↔ band *i*), not by
   exact wavelength — EMIT recalibrated its wavelengths mid-record (~0.088 nm
   shift between processing eras), so an exact match would silently drop
   cross-era pairs. `absorption_band_flag` marks the deep water-vapor bands,
   derived from where L2A reflectance is fill (so it flags radiance rows there
   too). Any pair that can't be formed is reported **loudly** and the run exits
   non-zero: *skipped* (a scene's radiance stats are missing — re-run step 3) or
   *degenerate* (the two scenes' band grids don't correspond).

Because the QC gate is enforced in step 5, run **step 4 before step 5** and do
the manual inspection in between; QC exclusions then propagate automatically to
step 6 (which only sees surviving matched pairs). Step 5 requires step 3 (the
geometry table); step 6 requires steps 3 and 5.

**Optional:** `python scripts/run_export_pics_gpkg.py` regenerates
`data/ROIs/northafrican_pics.gpkg` from `config/sites.yaml` for viewing the
site boxes in QGIS. It is a *derived* output (gitignored), never a pipeline
input — the scripts read the geometry straight from the YAML.

## Pixel-cube cache

`run_patch_extract.py` can persist the raw, **unscreened** in-patch pixel cube
per `(patch, acquisition)` to NetCDF (`CACHE_PIXELS = True`). It stores the
polygon-contained pixels flattened to `(pixel, band)` with fill values intact,
plus per-pixel SZA/VZA, lon/lat, every MASK band, and full-scene `row`/`col`
indices. Because nothing is screened on write, downstream screening (Aggregate
Flag, AOD) and BRDF correction become **read-time** transforms over local data —
no re-streaming.

Cache size scales with patch **area** (quadratic in edge length): ~32 MB per
product per acquisition at 10 km, ~16× that at 40 km. Enable caching only for the
patches your granule-set-cover selection actually uses; do not cache all nested
sizes en masse.
