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
| `single_patch_cube_extract` | **Unified per-acquisition extraction for one patch** — open each granule once, build the in-patch mask once, dispatch to all extractors, and persist the raw pixel cube. |
| `PICS_allpatches_cube_extract` | **Batched extraction of every patch of one size** — loop-inverted (open each granule once, index its geolocation once, extract every patch it covers); reuses the single-patch openers, derivations and persistence. |
| `extract_patch_stats` | Low-level openers / windowing / per-channel statistics (shared helpers). |
| `id_matched_pairs` | Matched-pair detection and SZA/VZA-over-time plotting (pure). |
| `display_qc_visuals` | Per-scene QA/QC multipanel figure rendering (map panels + reflectance/radiance spectra overlay). |
| `qc_report` | Per-patch QC report orchestration — PDF pages (cube-first, lazy Earthdata login) + non-destructive worksheet refresh; shared by the single-patch and batched QC runners. |
| `qc_review` | Interactive keep/remove reviewer (pure matplotlib) that renders each scene from the cached cubes and writes verdicts straight to the per-patch `qc.csv`. |
| `display_spectra` | Overlay every QC-surviving scene's RAD/RFL spectrum for a patch (one series per scene, date-ramped; absorption bands masked). |
| `delta_calculations` | Per-wavelength **t1→t2** % change across matched pairs (band-order join; tidy long CSV). |
| `PICS_allpatches_aggregate_pairs` | **Option A** — collapse each acquisition pair's spatial-replicate patches to one median ΔR per channel (with `n_patches` + spread diagnostics), removing the pseudo-replication before the drift fit. |
| `matched_pairs_drift_analysis` | Cross-PICS drift: per-channel Theil–Sen slope of ΔR vs Δt, Kendall-τ + BH-FDR significance, and the drift-spectrum / per-channel figures. |
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

3. **`python scripts/run_single_patch_cube_extract.py`**
   The single streaming pass for **one** patch. Opens each granule **once**,
   builds the in-patch mask **once**, and writes per-channel L1B radiance / L2A
   reflectance stats, the per-scene SZA/VZA **geometry table**, MASK-band
   summaries, and (opt-in) the raw in-patch **pixel cube** (NetCDF). Optional
   Aggregate-Flag screening of the per-channel stats.

   *Batched alternative:* **`python scripts/run_PICS_allpatches_cube_extract.py`**
   extracts **every patch of a given size** (`PATCH_SIZE`, e.g. `10` km) across
   the whole PICS in one pass, writing the identical per-patch outputs. It
   **inverts the loop nesting** — iterating acquisitions on the outside and
   patches on the inside — so each granule is opened once and its full-scene
   lon/lat read and KDTree built once, instead of once *per patch it covers*
   (which a naive wrapper over the single-patch runner would do; within a PICS one
   granule covers most patches). Granule opens / geolocation reads / KDTree builds
   drop from `O(#granules × #patches)` to `O(#granules)`; only the windowed
   spectral reads stay per-patch. Prefer it when processing a whole size at once;
   use the single-patch runner for one-off patches or re-runs.

4. **`python scripts/run_single_patch_qc_products.py`**
   Per-scene QA/QC snapshot PDF for one patch — each page has the map panels
   (RGB, every MASK band, SZA/VZA, footprint context) **and**, below them, two
   adjacent spectra panels (reflectance | radiance) overlaying every scene that
   covers the patch, with the current page's scene bold red and the rest faded
   grey. The spectra let you reject scenes whose RGB/MASK look fine but whose
   spectrum is unrealistic. The PDF is written to
   `data/qc/{site}_{patch}_QC_snapshots.pdf` — the same folder as the QC
   worksheet. The step also creates/refreshes that **QC worksheet**
   `data/qc/{site}_{patch}_qc.csv` (one row per scene).

   The spectra panels reuse the per-channel RAD/RFL stats CSVs from step 3
   (`SPECTRA_STAT` selects `median`/`mean`; the deep water-vapor absorption bands
   are blanked as gaps in both products). No QC filtering is applied here — this
   is the step that *decides* what to keep — so every scene in the geometry table
   is overlaid; a scene missing its radiance CSV simply has no bold-red curve
   (soft-skip, noted in the console).

   Rendering is **cube-first with a lazy Earthdata login**: pages rebuild from the
   cached pixel cubes when present (fully offline), and authentication is
   triggered only if a scene must actually be streamed. `MAKE_PDF = False` skips
   the PDF and only refreshes the worksheet.

   *Batched alternative:* **`python scripts/run_PICS_allpatches_qc_products.py`** runs this
   QC step for **every patch of a size** across the PICS in one pass (refreshing
   all their worksheets, PDFs optional via `MAKE_PDF`). With cubes cached it
   touches no network. Pairs naturally with the batched extractor.

   *→ Enter verdicts one of two ways:*
   - **Interactive (recommended):** **`python scripts/run_qc_review.py`** (one
     patch) or **`python scripts/run_PICS_allpatches_qc_review.py`** (every patch
     of a size in one session) — a pure-matplotlib reviewer that renders each scene
     from the cached cubes and writes keep/remove verdicts straight to `qc.csv`.
     Mouse buttons *or* keyboard (`k` keep, `r` remove, `1`–`4` remove-with-reason,
     ←/→ navigate, `v` mark the patch reviewed, `q` quit); verdicts save on every
     action. In the all-patch reviewer, leaving a patch that isn't yet marked
     `reviewed` arms a one-key confirm gate (`v` to mark & continue, forward again
     to skip anyway, back to return), and quitting with any patch still unreviewed
     asks the same — so a finished patch never gets left unmarked by accident.
   - **By hand:** open the PDF and in the worksheet set `keep=FALSE` (with a
     `reason`: cloud / coverage / artifact / other) for any bad scene, then set
     `reviewed=TRUE` in the top row.

   Either way verdicts are **opt-out** (default keep), preserved across re-runs,
   and gated: downstream steps refuse to run until `reviewed=TRUE`.

5. **`python scripts/run_id_matched_pairs.py`**
   Pure **local** post-processor (no Earthdata): reads the geometry table,
   **applies the QC gate** (drops `keep=FALSE` scenes; refuses to run until the
   worksheet is `reviewed=TRUE`, unless `REQUIRE_QC_REVIEW=False`), then writes
   the matched-pair CSV and the SZA/VZA-over-time figure.
   **Batched alternative:** `python scripts/run_PICS_allpatches_id_matched_pairs.py`
   runs this for every patch of a size in one pass (skipping any patch that is
   unreviewed, missing its geometry table, or left with <2 scenes) and amends the
   attribute table once with each patch's matched-pair counts.

6. **`python scripts/run_delta_calculations.py`**
   Pure **local** post-processor (no Earthdata): joins the matched-pairs CSV
   (step 5) with the per-channel RAD/RFL stats (step 3) and writes a tidy long
   CSV of the pairwise **t1→t2** percent change in reflectance and radiance
   (median & mean) at every wavelength — one row per `(matched pair × wavelength)`
   → `data/delta_r/per_patch/{site}_{patch}_delta.csv`. No aggregation or figures;
   the per-PICS collapse is step 7 and the cross-PICS analysis is step 8.

   The two scenes are aligned **by band order** (band *i* ↔ band *i*), not by
   exact wavelength — EMIT recalibrated its wavelengths mid-record (~0.088 nm
   shift between processing eras), so an exact match would silently drop
   cross-era pairs. `absorption_band_flag` marks the deep water-vapor bands,
   derived from where L2A reflectance is fill (so it flags radiance rows there
   too). Any pair that can't be formed is reported **loudly** and the run exits
   non-zero: *skipped* (a scene's radiance stats are missing — re-run step 3) or
   *degenerate* (the two scenes' band grids don't correspond).
   **Batched alternative:** `python scripts/run_PICS_allpatches_delta_calculations.py`
   runs this for every patch of a size in one pass; each patch still writes
   whatever succeeded, the dropped-pair accounting is aggregated across the whole
   run (listed by patch, exit 2 if any pair dropped anywhere), and a patch whose
   matched-pairs CSV is absent is reported without tripping that exit code.

Because the QC gate is enforced in step 5, run **step 4 before step 5** and do
the manual inspection in between; QC exclusions then propagate automatically to
step 6 (which only sees surviving matched pairs). Step 5 requires step 3 (the
geometry table); step 6 requires steps 3 and 5.

*Parallel diagnostic (optional):* **`python scripts/run_display_spectra.py`**
runs alongside step 5 on the **same inputs** — it applies the identical QC gate,
then overlays every surviving scene's spectrum for the patch, one series per
scene (coloured along a sequential ramp by acquisition date, cycling line style,
date-labelled legend). Two high-resolution (1000 dpi) PNGs are written,
`figures/spectra/reflectance/{PICS}_{PATCH}_{stat}_rfl_spectra.png` and
`figures/spectra/radiance/{PICS}_{PATCH}_{stat}_rad_spectra.png`. Water-vapor
absorption bands are masked by data availability (where L2A reflectance is fill)
in **both** products, so every curve breaks in the same windows. Requires only
step 3 (per-channel CSVs + geometry table) and the reviewed QC worksheet.
**`python scripts/run_PICS_allpatches_display_spectra.py`** is the batched
counterpart: it renders the same survivor-spectra overlays for *every* patch of a
size in one run — the quickest way to re-check a whole PICS after a QC pass
(missed scenes stand out as stray curves), reporting per patch how many scenes
survived and skipping any that are unreviewed or fully pruned.

Then, once step 6 has run for the patches you care about (across sites):

7. **`python scripts/run_PICS_allpatches_aggregate_pairs.py`**
   Pure **local** per-PICS post-processor that removes the **pseudo-replication**
   in the drift fit (*option A*). The step-6 tables carry one row per
   `(patch × acquisition-pair × channel)`, so the *same* acquisition pair (same
   two timestamps, hence the same sensor state and atmosphere) seen by many
   spatial-replicate patches would otherwise enter the Theil–Sen fit as many
   near-identical points, artificially narrowing the uncertainty. For one PICS at
   one `PATCH_SIZE`, this pools every `data/delta_r/per_patch/{site}_{patch}_delta.csv`
   and, per channel, collapses each unique acquisition pair's spatial replicates
   to a single ΔR — the **median across patches** (robust to an anomalous patch,
   matching the downstream estimator) — so every acquisition pair contributes
   exactly one point. Writes `data/delta_r/aggregated/{site}_{size}_delta.csv`
   (one row per `(acquisition pair × channel)`, same delta-table columns so it is
   a drop-in for step 8), plus the diagnostics `n_patches` and the across-patch
   spread (`*_std`/`*_min`/`*_max` of the median deltas) — not used to weight the
   fit here, but exactly what a future inverse-variance (*option B*) scheme would
   need. Prints a collapse summary quantifying the effective-N inflation removed
   (e.g. *"142 patch-pair rows → 38 unique acquisition pairs; median 3 patches/pair"*).
   Aggregation is **within one PICS** — a `pair_id` (timestamp pair) is unique to
   a site since the PICS are far apart, so there is no cross-PICS collision.

8. **`python scripts/run_matched_pairs_drift_analysis.py`**
   Pure **local** cross-PICS analysis: pools the step-7 aggregated delta CSVs of
   one patch size (`PATCH_SIZE`, the first field of `patch_id`, e.g. `10` km) and,
   per EMIT channel, fits the **radiometric drift rate** as the robust
   **Theil–Sen** slope of ΔR vs Δt (median statistic, %/yr), with a **Kendall-τ**
   p-value and **Benjamini–Hochberg** FDR control across the 285 channels. Writes
   `data/drift/{size}_drift.csv` (per-channel stats, RAD + RFL), a
   `figures/matched_pairs/{size}_drift_spectrum.png` (drift rate vs wavelength,
   95% CI whiskers, BH-significance colouring), and — one per channel —
   `figures/matched_pairs/individual_channels/{size}_{wavelength}.png` (two-panel
   ΔR-vs-Δt scatter, RAD | RFL; colour/shape = PICS, **marker size ∝ `n_patches`**
   so more-replicated pairs stay visible even though option A does not weight the
   fit by it, each panel overlaid with its Theil–Sen fit line + a stats box).
   The estimator is unchanged — it just operates on the deduplicated rows. By
   default `USE_AGGREGATED = True` reads `data/delta_r/aggregated/`; set it to
   `False` to fit the raw per-patch deltas from `data/delta_r/per_patch/`
   (pseudo-replicated, dots sized by mean |SZA|) for comparison.
   **Water-vapor absorption bands are excluded from both products** (reflectance
   is fill there, radiance is atmospherically contaminated) — they appear as
   shaded gaps in the spectrum and are skipped in the per-channel figures.
   Requires the delta CSVs to carry the `SZA_t1`/`SZA_t2`/`band_index` columns
   (re-run step 6, then step 7, if they predate that schema).

**Optional:** `python scripts/run_export_pics_gpkg.py` regenerates
`data/ROIs/northafrican_pics.gpkg` from `config/sites.yaml` for viewing the
site boxes in QGIS. It is a *derived* output (gitignored), never a pipeline
input — the scripts read the geometry straight from the YAML.

## Pixel-cube cache

Both `run_single_patch_cube_extract.py` and `run_PICS_allpatches_cube_extract.py`
can persist the raw, **unscreened** in-patch pixel cube
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
