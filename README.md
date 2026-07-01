# EMIT PICS Temporal Stability

Assessing the radiometric temporal stability of NASA's EMIT imaging spectrometer over North African pseudo-invariant calibration sites (PICS).

## Repository layout

```
emit_pics/      Python package — pure function libraries (no side effects on import)
config/         YAML site definitions (PICS coordinates, ROI parameters)
scripts/        Command-line entry points (edit the inputs at the top, then run)
notebooks/      Exploration and analysis notebooks
tests/          Unit tests
```

Pipeline outputs (patches, footprints, dictionaries, per-channel stats, geometry
tables, pixel cubes, figures) are written under a sibling `Data/` tree and are
gitignored.

## Package modules

| Module | Role |
| --- | --- |
| `patch_generator` | Tile a PICS boundary into nested, fully-contained square patches. |
| `id_granules` | Search NASA Earthdata; determine which granules fully cover each patch. |
| `patch_extract` | **Unified per-acquisition extraction** — open each granule once, build the in-patch mask once, dispatch to all extractors, and persist the raw pixel cube. |
| `extract_patch_stats` | Low-level openers / windowing / per-channel statistics (shared helpers). |
| `id_matched_pairs` | Matched-pair detection and SZA/VZA-over-time plotting (pure). |
| `display_qc_bands` | Per-scene QA/QC multipanel figure rendering. |
| `config` | Load PICS site definitions from `config/sites.yaml`. |

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

## Pipeline

Each script exposes its inputs (site id, patch id, paths) as constants at the
top — edit those, then run. Steps 1–2 are global to a PICS site; steps 3–5 are
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

4. **`python scripts/run_id_matched_pairs.py`**
   Pure **local** post-processor (no Earthdata): reads the geometry table from
   step 3 and writes the matched-pair CSV and the SZA/VZA-over-time figure.

5. **`python scripts/run_display_qc_bands.py`**
   Per-scene QA/QC snapshot PDF (RGB, every MASK band, SZA/VZA, footprint
   context) for one patch.

Step 4 requires step 3 to have run first for that patch (it reads the geometry
table). Steps 3–5 are otherwise independent.

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
