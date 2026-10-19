# EMIT PICS Temporal Stability

Assessing the radiometric temporal stability of NASA's EMIT imaging spectrometer over North African pseudo-invariant calibration sites (PICS).

The method: over a PICS the surface is assumed stable, so for pairs of acquisitions matched in viewing/illumination geometry, any per-wavelength change in reflectance/radiance is a proxy for sensor drift. Pooling those changes across many acquisition pairs and sites and regressing them against the time gap gives a per-channel **drift rate** (%/year).

## Repository layout

```
run_config.yaml   The ONE place to set the analysis target + knobs (edit this, then run)
emit_pics/        Python package — pure function libraries + config/paths/pipeline/provenance
config/           sites.yaml — PICS site definitions (bounding boxes + metadata)
scripts/          Thin command-line runners (read run_config.yaml) + run_pipeline.py orchestrator
data/             Pipeline inputs/outputs (contents gitignored; folder kept via .gitkeep)
figures/          Generated figures (contents gitignored; folder kept via .gitkeep)
docs/             Data dictionary
notebooks/        Exploration and analysis notebooks
tests/            Unit tests (pytest)
```

All paths are **repo-relative** — every script writes under `data/` and `figures/` inside the repo, so it runs identically on any machine after a `git clone`, with no per-machine path edits. The *contents* of `data/` and `figures/` are gitignored; only `.gitkeep` placeholders are committed so the folders exist on a fresh clone.

Two single sources of truth keep the pipeline portable:

- **`config/sites.yaml`** — PICS site geometry (each site an axis-aligned bounding box in EPSG:4326). The site rectangle is built from the YAML; the companion `northafrican_pics.gpkg` (for GIS use) can be regenerated from it.
- **`run_config.yaml`** — the analysis target (site / patch / size) and all tunable parameters. Every script reads it, so you set inputs once instead of editing each script.

## Quick start

```bash
conda env create -f environment.yml
conda activate emit-pics
pip install -e . --no-deps          # make `emit_pics` importable (scripts rely on it)
```

You also need NASA Earthdata credentials. The streaming steps authenticate via `earthaccess.login()`, which reads `~/.netrc` (or `_netrc` on Windows) and falls back to an interactive prompt — see the [earthaccess docs](https://earthaccess.readthedocs.io/).

The Landsat 9 control branch additionally reads Collection 2 Level-1 TOA assets from the **requester-pays** `usgs-landsat` S3 bucket (`us-west-2`), so it needs AWS credentials in the default boto3 chain — `~/.aws/credentials`, or the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables (you are billed egress for these reads). The requester-pays and GDAL `/vsis3/` flags are set automatically by `scripts/run_landsat_cube_extract.py`; the LandsatLook STAC catalog itself needs no credentials.

Then edit **`run_config.yaml`** (at minimum `site`, `patch`, `patch_size`) and run the pipeline stage by stage:

```bash
python scripts/run_pipeline.py setup      # tile patches + find granule coverage (Earthdata)
python scripts/run_pipeline.py extract    # stream pixels + build QC products (Earthdata)
#   → inspect the QC products and mark each worksheet reviewed (see "QC gate" below)
python scripts/run_pipeline.py analyze    # matched pairs → per-wavelength deltas → aggregate
python scripts/run_pipeline.py drift      # cross-PICS matched-pairs drift fit + figures
python scripts/run_pipeline.py brdf       # pooled quasi-BRDF drift: assemble → single-channel → spectrum
python scripts/run_pipeline.py landsat    # OLI-2 control: extract → QC → drift → robustness (AWS)
python scripts/run_pipeline.py compare    # EMIT vs OLI-2 at the seven OLI-2 bands
python scripts/run_pipeline.py figures    # publication figures + Table 1
python scripts/run_pipeline.py appendix   # cos(SZA)–time check + leave-one/two-out
```

`python scripts/run_pipeline.py all` runs every stage in order (it stops at the manual QC gate until the worksheets are marked reviewed). Add `--dry-run` to any stage to print the exact steps without executing them.

## Configuration — `run_config.yaml`

`site` accepts any spelling (`libya_4`, `"Libya 4"`, `Libya_4`); it is resolved to a single canonical filename token. Keys:

| Section | Key | Meaning |
| --- | --- | --- |
| *(top level)* | `site`, `patch`, `patch_size` | **Required.** Analysis target: PICS site, single-patch id (e.g. `10_7_1`), and patch-size token (e.g. `10`, km). |
| `products` | `rad_short_name`, `rfl_short_name` | EMIT collection short-names (`EMITL1BRAD`, `EMITL2ARFL`). |
| `matching` | `sza_tol`, `vza_tol`, `min_pair_days` | Matched-pair tolerances (degrees) on \|ΔSZA\| and \|ΔVZA\|, plus the minimum time separation (days) for a pair (`0` = off). |
| `qc` | `require_review` | If true, downstream steps refuse to run until the QC worksheet is reviewed. |
| `spectra` | `stat` | In-patch statistic to plot/compare (`median` or `mean`). |
| `radiometry` | `earth_sun_normalize`, `e0_method` | Normalize L1B radiance to 1 AU at read time (removes the seasonal 1/d² solar-flux leak; on by default), and the `E0` provider (`table` ephemeris or `spencer` closed-form). Setting `earth_sun_normalize: false` writes the uncorrected derived products under a `variant_label` subfolder instead of overwriting the corrected baseline. |
| `drift` | `alpha`, `n_bootstrap`, `seed` | Per-channel significance level (uncorrected bootstrap p); matched-pairs bootstrap iterations; RNG seed. |
| `brdf` | `site_patches`, `exclude_sites`, `target_wl`, `min_scenes`, `n_bootstrap`, `per_site_boot`, `spectrum_n_boot`, `gamma_cap` | Pooled multi-site drift: the roster (one patch per PICS; `null` skips a site), excluded sites, single-channel wavelength, per-site scene floor, the three bootstrap counts, and the unstable-fit cap. Reuses `drift.alpha` + `qc.require_review`. |
| `extract` | `cache_pixels`, `want_obs`, `filter_agg_flag`, `min_valid_frac` | Cache the raw pixel cube; stream OBS geometry; Aggregate-Flag screening + its threshold. |
| `granule_search` | `date_start`, `date_end`, `patch_size_km` | Earthdata search window and patch size (step 2). |
| `patch_gen` | `min_patch_km`, `increment_km` | Nested-patch tiling parameters (step 1). |
| `landsat` | `stac_url`, `collection`, `platform`, `max_cloud`, `min_coverage` | Landsat 9 OLI-2 STAC search for the control branch (LandsatLook catalog, collection, platform, and the cloud / in-patch-coverage thresholds). |

A few **execution-only toggles** stay in the relevant script's header (they change *how* a step runs, not *what* is analysed): `RESUME` (batched extractor), `PREFER_CUBE` / `DRAW_NADIR` / `MAKE_PDF` (QC products), and `ONLY_BANDS` (drift figures).

## The orchestrator — `run_pipeline.py`

`python scripts/run_pipeline.py <stage|all> [--dry-run]`

Every stage operates on all patches of `patch_size` (the batched runners) and on the `run_config.yaml` rosters (BRDF / Landsat).

| Stage | Runs | Notes |
| --- | --- | --- |
| `setup` | `run_patch_generator` → `run_id_granules` | Once per site; uses Earthdata. |
| `extract` | `cube_extract` → `qc_products` | Uses Earthdata; then **review QC**. |
| `analyze` | `id_matched_pairs` → `delta_calculations` → `aggregate_pairs` | Local; QC-gated. |
| `drift` | `matched_pairs_drift_analysis` | Local; cross-PICS matched-pairs drift. |
| `brdf` | `brdf_assemble` → `brdf_single_channel` → `brdf_drift_spectrum` | Local; pooled multi-site. |
| `landsat` | `cube_extract` → `qc_products` → `brdf_drift_spectrum` → `robustness` | OLI-2 control; AWS; then **review the Landsat QC**. |
| `compare` | `emit_oli2_band_comparison` | EMIT vs OLI-2 at the seven bands. |
| `figures` | `publication_figures` → `data_volume_table` | Main figures + Table 1. |
| `appendix` | `cos_sza_corr` → `brdf_loo` | cos(SZA)–time check; leave-one/two-out (Appendix B/C). |
| `all` | every stage above, in order | Stops at the manual QC gate until the worksheets are reviewed. |

Each step is an isolated subprocess that reads `run_config.yaml` itself, so a failure stops the pipeline with that step's own message. The patch-size sensitivity (Appendix A) needs a separate 20 km run — set 20 km patches + `brdf.label: 20km`, re-run `extract` → `brdf`, then `run_patch_size_sensitivity.py` — so it is not part of `appendix`.

The per-patch `run_single_patch_*.py` scripts remain for one-off single-patch inspection but are not driven by the orchestrator. You can also run any underlying script directly for granular control.

## Package modules

| Module | Role |
| --- | --- |
| `config` | PICS site geometry from `config/sites.yaml`; `canonical_token` resolves any site spelling to the filename token; `pics_geodataframe` / `write_pics_gpkg`. |
| `run_config` | Load `run_config.yaml` into a typed `RunConfig` (defaults + loud fail on missing keys). |
| `paths` | Single source for every `data/`/`figures/` path — one function per product. |
| `pipeline` | Ordered stage → script resolution used by `run_pipeline.py`. |
| `provenance` | Write a `<output>.json` sidecar (git commit, timestamp, versions, run config) beside each data product. |
| `patch_generator` | Tile a PICS boundary into nested, fully-contained square patches. |
| `id_granules` | Search NASA Earthdata; determine which granules fully cover each patch. |
| `single_patch_cube_extract` | Unified per-acquisition extraction for one patch — open each granule once, mask once, derive all outputs, persist the raw pixel cube. |
| `PICS_allpatches_cube_extract` | Batched extraction of every patch of one size — loop-inverted (open each granule once); reuses the single-patch derivations. |
| `extract_patch_stats` | Low-level openers / windowing / per-channel statistics (shared helpers). |
| `earth_sun` | Earth–Sun distance + eccentricity factor `E0` (ephemeris table or Spencer closed-form) for the read-time 1-AU radiance normalization. |
| `id_matched_pairs` | Matched-pair detection and SZA/VZA-over-time plotting (pure). |
| `qc_gate` | The manual QC gate: filter `keep=FALSE` scenes; `is_reviewed` / `apply_qc_filter`. |
| `display_qc_visuals` / `qc_report` / `qc_review` | QC figure rendering, PDF orchestration, and the interactive keep/remove reviewer. |
| `display_spectra` | Overlay every QC-surviving scene's RAD/RFL spectrum for a patch. |
| `delta_calculations` | Per-wavelength **t1→t2** % change across matched pairs (band-order join; tidy long CSV). |
| `PICS_allpatches_aggregate_pairs` | **Option A** — collapse each acquisition pair's spatial-replicate patches to one median ΔR per channel, removing pseudo-replication before the fit. |
| `matched_pairs_drift_analysis` / `matched_pairs_bootstrap` | Cross-PICS drift: per-channel Theil–Sen slope of ΔR vs Δt with a PICS-stratified matched-pairs bootstrap CI (shared plumbing + figure in the former, the estimator in the latter). |
| `brdf_drift` | Pooled multi-site quasi-BRDF drift: per-site + joint `(α+β·cosθ)(1+γt)` fits, bootstraps, the shared-drift consistency check, and the design-table assembly. |
| `brdf_loo` | Leave-one-out and leave-two-out refits of the pooled BRDF drift (Appendix B). |
| `patch_size_sensitivity` | Compare the BRDF drift spectrum across patch sizes (Appendix A). |
| `cos_sza_corr` | cos(SZA)-vs-time collinearity check (table + figure) for the BRDF design. |
| `landsat_cube_extract` | Landsat 9 OLI-2 TOA extraction over a patch (STAC + windowed `rasterio` COG reads); the Landsat analog of the EMIT cube extract. |
| `landsat_brdf` | Landsat design-table assembly + per-site / leave-one-out drift, reusing `brdf_drift`'s fit. |
| `landsat_qc_report` / `landsat_qc_visuals` / `landsat_qc_review` | Landsat QC snapshot PDF, figure rendering, and the interactive keep/remove reviewer. |

The function libraries are side-effect-free; each is driven by a thin runner in `scripts/`.

## Pipeline detail

Steps 1–2 are global to a PICS site; steps 3–6 are per-patch analyses branching off the patch–granule dictionary; steps 7–8 pool them. The orchestrator runs these in order, but each is a standalone script.

1. **`run_patch_generator.py`** — Tile the PICS into nested square patches → `patches_{site}_bnds.gpkg` + `patches_{site}_atts.csv`. *(No Earthdata.)*

2. **`run_id_granules.py`** — Search Earthdata for granule coverage per patch → granule footprints, per-patch coverage counts, and the **patch–granule dictionary** `patch_EMIT_dict_{site}.csv` that every downstream step reads.

3. **`run_single_patch_cube_extract.py`** (batched: **`run_PICS_allpatches_cube_extract.py`**) — Open each covering granule **once**, build the in-patch mask **once**, and write per-channel L1B radiance / L2A reflectance stats, the per-scene SZA/VZA **geometry table**, MASK-band summaries, and (opt-in, `extract.cache_pixels`) the raw in-patch **pixel cube** (NetCDF). Optional Aggregate-Flag screening (`extract.filter_agg_flag` + `min_valid_frac`). The batched runner **inverts the loop nesting** (acquisitions outer, patches inner) so granule opens / geolocation reads / KDTree builds drop from `O(#granules × #patches)` to `O(#granules)`; only windowed spectral reads stay per-patch.

4. **`run_single_patch_qc_products.py`** (batched: **`run_PICS_allpatches_qc_products.py`**) — Per-scene QA/QC snapshot PDF (`data/qc/{site}_{patch}_QC_snapshots.pdf`): map panels (RGB, every MASK band, SZA/VZA, footprint context) plus adjacent reflectance | radiance spectra panels overlaying every scene, so you can reject scenes whose spectrum is unrealistic even when the imagery looks fine. Also creates/refreshes the **QC worksheet** `data/qc/{site}_{patch}_qc.csv` (one row per scene). Rendering is **cube-first with a lazy Earthdata login** — pages rebuild from cached cubes when present (fully offline).

   **QC gate.** Enter keep/remove verdicts either interactively (**`run_qc_review.py`** / **`run_PICS_allpatches_qc_review.py`** — a matplotlib reviewer with mouse/keyboard verdicts written straight to `qc.csv`) or by hand (set `keep=FALSE` with a `reason`, then `reviewed=TRUE` in the top row). Verdicts are **opt-out** (default keep), preserved across re-runs, and gated: steps 5–6 refuse to run until `reviewed=TRUE` (unless `qc.require_review` is false).

5. **`run_id_matched_pairs.py`** (batched: **`run_PICS_allpatches_id_matched_pairs.py`**) — Pure **local** post-processor: read the geometry table, **apply the QC gate**, then write the matched-pair CSV and the SZA/VZA-over-time figure. A matched pair is two acquisitions with \|ΔSZA\| ≤ `matching.sza_tol`, \|ΔVZA\| ≤ `matching.vza_tol`, and at least `matching.min_pair_days` apart.

6. **`run_delta_calculations.py`** (batched: **`run_PICS_allpatches_delta_calculations.py`**) — Join the matched-pairs CSV with the per-channel stats and write a tidy long CSV of the pairwise **t1→t2** percent change (median & mean, RFL + RAD) at every wavelength → `data/delta_r/per_patch/{site}_{patch}_delta.csv`. Scenes are aligned **by band order** (band *i* ↔ band *i*), not by exact wavelength — EMIT recalibrated its wavelengths mid-record (~0.09 nm shift between processing eras), so an exact match would silently drop cross-era pairs. `absorption_band_flag` marks deep water-vapor bands (where L2A reflectance is fill). Any pair that can't be formed is reported **loudly** and the run exits non-zero: *skipped* (missing radiance stats — re-run step 3) or *degenerate* (band grids don't correspond).

   *Parallel diagnostic (optional):* **`run_display_spectra.py`** (batched counterpart exists) overlays every QC-surviving scene's spectrum for a patch (date-ramped, absorption bands masked) → `figures/spectra/{reflectance,radiance}/`.

7. **`run_PICS_allpatches_aggregate_pairs.py`** — Pure **local** per-PICS step that removes the **pseudo-replication** in the drift fit (*option A*). The step-6 tables carry one row per `(patch × acquisition-pair × channel)`, so the *same* acquisition pair seen by many spatial-replicate patches would otherwise enter the fit as many near-identical points, artificially narrowing the uncertainty. Per channel, this collapses each unique acquisition pair's spatial replicates to a single ΔR — the **median across patches** — so every pair contributes exactly one point → `data/delta_r/aggregated/{site}_{size}_delta.csv`, plus diagnostics `n_patches` and across-patch spread (`*_std`/`*_min`/`*_max`).

8. **`run_matched_pairs_drift_analysis.py`** — Pure **local** cross-PICS analysis: pool the aggregated delta CSVs of one patch size across all PICS and, per EMIT channel, fit the **drift rate** with a **PICS-stratified matched-pairs bootstrap**. The point estimate is the robust **Theil–Sen** slope of ΔR vs Δt (%/yr) on one representative ΔR per pair; the 95% CI and two-sided p-value come from resampling the matched pairs with replacement, stratified by PICS (`drift.n_bootstrap` resamples, `drift.seed`). Writes `data/drift/{size}_mp_drift.csv`, `figures/matched_pairs/{size}_mp_drift_spectrum.png` (drift vs wavelength, 95% CI whiskers, raw-significance colouring), and one two-panel ΔR-vs-Δt scatter per channel. Water-vapor absorption bands are excluded from the fit (shaded gaps).

**Optional:** `run_export_pics_gpkg.py` regenerates `data/ROIs/northafrican_pics.gpkg` from `config/sites.yaml` for QGIS. It is a *derived* output (gitignored), never a pipeline input.

### BRDF — pooled multi-site drift (parallel branch)

An **independent second drift estimate** that consumes the same extracted data (per-channel stats + geometry + QC gate) but answers the drift question with a regression instead of matched pairs. Every PICS contributes **one representative patch = one "site"**; all sites share a single drift rate while each keeps its own brightness/BRDF terms, fitting the `2S+1` model `ρ̂ₛ = (αₛ + βₛ·cosθ)(1 + γt)` (`t` = years since the earliest scene across all sites, a common clock; `γ` = the one shared drift). Geometry is *modelled out* via the `cosθ` term rather than matched away. Configure the roster in the `run_config.yaml` `brdf:` block, then run the three scripts (or `run_pipeline.py brdf`):

9. **`run_brdf_assemble.py`** — Resolve the roster; build the pooled per-scene **design table** (per site: QC-surviving, reflectance-bearing scenes with median ρ at the channel nearest `brdf.target_wl`; common clock `t`, `g = cos SZA`); apply the `brdf.min_scenes` floor (**warns** on drops). Writes `data/brdf/brdf_design.csv`, `data/brdf/brdf_leverage.csv`, and the leverage/confounding scatter.

10. **`run_brdf_single_channel.py`** — At the single channel nearest `brdf.target_wl`: the **shared-drift consistency check** (each site's standalone γ + bootstrap CI; a `pass`/`marginal`/`fail` verdict on whether all CIs overlap — computed, **warned loudly if not clean, but never blocking**, and recorded in the sidecar); the joint pooled fit (per-site α/β + shared γ); the stratified within-site bootstrap of γ; and the quasi-BRDF-correction collapse figure. Writes `brdf_persite_gamma_{wl}nm.csv`, `brdf_joint_fit_{wl}nm.csv`, `brdf_bootstrap_gamma_{wl}nm.csv` + three figures.

11. **`run_brdf_drift_spectrum.py`** — Repeat the pooled joint fit at **every** EMIT channel (water-vapor bands excluded), with a two-sided (uncorrected) bootstrap p per channel, plus an unstable-channel flag (`brdf.gamma_cap`). Writes `data/brdf/brdf_drift_spectrum.csv` (the multi-site analog of the matched-pairs drift spectrum) + the drift-spectrum figure.

> **Trust the pooled γ only if the consistency check is clean.** A `fail` (per-site CIs disjoint) means the pooled value blends differing true drifts — a red flag, not a result; a `marginal` warrants caution. Mauritania 2 is excluded by default (too few QC-surviving scenes to constrain γ); the `min_scenes` floor enforces this generally.

### Landsat 9 OLI-2 control (parallel branch)

An independent **control** on the EMIT drift: the same sensor-agnostic quasi-BRDF fit applied to Landsat 9 OLI-2 top-of-atmosphere reflectance over the same PICS, used to (a) screen each candidate PICS for temporal invariance and (b) cross-check the EMIT drift at the seven OLI-2 reflective bands. OLI-2's onboard calibration makes a fitted drift attributable to the surface rather than the sensor. TOA reflectance only (no Earth–Sun normalization), the seven reflective bands, read as windowed COGs from the LandsatLook STAC (requester-pays AWS; see Quick start).

Set the site/patch and date window in `run_config.yaml`, then:

```bash
python scripts/run_landsat_cube_extract.py          # STAC + windowed COG reads (AWS)
python scripts/run_landsat_qc_products.py           # per-scene QC worksheet + PDF
python scripts/run_landsat_qc_review.py             # manual keep/remove (opt-out gate)
python scripts/run_landsat_brdf_drift_spectrum.py   # per-band OLI-2 drift
python scripts/run_landsat_brdf_robustness.py       # per-site + leave-one-out invariance
python scripts/run_emit_oli2_band_comparison.py     # EMIT vs OLI-2 at the seven bands
```

The OLI-2 relative spectral response used for the band comparison lives in `data/landsat/oli2_rsr/` (source workbook + README); `scripts/build_oli2_rsr_tidy.py` regenerates the tidy CSV from the USGS workbook (needs `openpyxl`).

## Provenance

Every data-product write also emits a sidecar `<output>.json` recording the UTC timestamp, git commit / branch / dirty flag, script name, Python + key package versions, and the exact `run_config` used. A CSV on disk can't tell you how it was made; its sidecar can — so every result is traceable to an exact code state. Sidecars live beside the (gitignored) data and travel with it when results are archived (e.g. to Zenodo).

## Data products

The columns and units of every output CSV/table are documented in **[`docs/data_dictionary.md`](docs/data_dictionary.md)**.

## Pixel-cube cache

Both cube-extract runners can persist the raw, **unscreened** in-patch pixel cube per `(patch, acquisition)` to NetCDF (`extract.cache_pixels`). It stores the polygon-contained pixels flattened to `(pixel, band)` with fill values intact, plus per-pixel SZA/VZA, lon/lat, every MASK band, and full-scene `row`/`col` indices. Because nothing is screened on write, downstream screening (Aggregate Flag, AOD) and BRDF correction become **read-time** transforms over local data — no re-streaming. Cache size scales with patch **area**: ~32 MB per product per acquisition at 10 km, ~16× that at 40 km. Cache only the patches your analysis actually uses.

## Tests

```bash
pytest
```

Unit tests cover the core science (band-order delta join, the PICS-stratified matched-pairs bootstrap drift statistics — verified by recovering a known injected slope with a bracketing CI — matched-pair detection, the QC gate, and the BRDF fits, verified by recovering a known injected `(αₛ, βₛ, γ)`, plus the shared-drift consistency verdicts) and the config/paths/pipeline plumbing. Run them before and after any refactor to confirm the numbers are unchanged.
