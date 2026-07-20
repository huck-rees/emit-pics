# Data dictionary

Columns and units for every output the EMIT-PICS pipeline writes. All outputs
land under `data/` (and `figures/`), which is gitignored; each **data product**
is accompanied by a `<filename>.json` [provenance sidecar](../README.md#provenance)
recording the git commit, timestamp, package versions, and `run_config` used.

**Conventions.** `{site}` is the canonical site token (e.g. `Libya_4`); `{patch}`
is a patch id (e.g. `10_7_1`); `{size}` is the patch-size token in km (e.g. `10`);
`{ts}` is an acquisition timestamp `YYYYMMDDThhmmss`.

**Units.** Wavelength — nm · Radiance — µW/cm²/sr/nm (EMIT L1B) · Reflectance —
unitless (EMIT L2A) · Angles (SZA/VZA) — degrees · Time separation — days ·
Drift rate — %/year · Deltas — percent.

---

## Directory map

| Path | Product |
| --- | --- |
| `data/ROIs/patches/patches_{site}_bnds.gpkg`, `_atts.csv` | Patch boundaries + attributes (step 1) |
| `data/ROIs/patches/patch_EMIT_dict_{site}.csv` | Patch–granule dictionary (step 2) |
| `data/ROIs/emit_granule_footprints_{site}.gpkg` | Granule footprints (step 2) |
| `data/radiance/{site}_{patch}_{ts}_perchannel_rad.csv` | Per-channel radiance stats (step 3) |
| `data/reflectance/{site}_{patch}_{ts}_perchannel_rfl.csv` | Per-channel reflectance stats (step 3) |
| `data/geometry/geometry_{site}_{patch}.csv` | Per-scene SZA/VZA geometry table (step 3) |
| `data/mask_summary/{site}_{patch}_{ts}_mask_summary.csv` | MASK-band summary (step 3) |
| `data/pixel_cubes/cube_{site}_{patch}_{ts}.nc` | Raw in-patch pixel cube, NetCDF (step 3, opt-in) |
| `data/qc/{site}_{patch}_qc.csv`, `_QC_snapshots.pdf` | QC worksheet + snapshot PDF (step 4) |
| `data/matched_pairs/{site}_{patch}_matched_pairs.csv` | Matched pairs (step 5) |
| `data/delta_r/per_patch/{site}_{patch}_delta.csv` | Per-patch per-wavelength deltas (step 6) |
| `data/delta_r/aggregated/{site}_{size}_delta.csv` | Per-PICS aggregated deltas (step 7) |
| `data/drift/{size}_drift.csv` | Cross-PICS per-channel drift stats (step 8) |

---

## 1. Patch attribute table — `patches_{site}_atts.csv`

One row per patch.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `patch_id` | text | Patch id, `{size}_{row}_{col}` (e.g. `10_7_1`). |
| `size_km` | int (km) | Patch edge length. |
| `n_matched_pairs_rad` | int (nullable) | Matched-pair count for radiance (appended by step 5). |
| `n_matched_pairs_rfl` | int (nullable) | Matched-pair count for reflectance (appended by step 5). |

## 2. Patch–granule dictionary — `patch_EMIT_dict_{site}.csv`

One row per `(patch, product, granule)`. Tab-separated. Geometry columns are populated for RAD rows (RFL inherits from the coincident RAD by timestamp).

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `patch_id` | text | Patch id. |
| `size_km` | int (km) | Patch edge length. |
| `product` | text | `L1B_RAD` or `L2A_RFL`. |
| `granule_ur` | text | EMIT GranuleUR. |
| `sza_min`, `sza_std`, `sza_mean`, `sza_max` | degrees | In-patch solar-zenith distribution (appended by step 3). |
| `vza_min`, `vza_std`, `vza_mean`, `vza_max` | degrees | In-patch view-zenith distribution (appended by step 3). |

## 3a. Per-channel stats — `{site}_{patch}_{ts}_perchannel_{rad,rfl}.csv`

One row per EMIT channel. Statistic keys: `min, 1qt, median, 3qt, max, mean, std`
(`1qt`/`3qt` = 25th/75th percentile), suffixed `_rad` or `_rfl`.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `central_wl` | nm | Channel centre wavelength. |
| `min_rad … std_rad` | µW/cm²/sr/nm | Radiance in-patch stats (rad file). |
| `min_rfl … std_rfl` | unitless | Reflectance in-patch stats (rfl file). |

## 3b. Per-scene geometry table — `geometry_{site}_{patch}.csv`

One row per scene covering the patch.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `scene_id`, `granule_ur` | text | RAD GranuleUR (scene identity). |
| `timestamp` | `YYYYMMDDThhmmss` | Acquisition timestamp. |
| `datetime` | ISO datetime | Acquisition time. |
| `has_rfl` | bool | Whether an L2A reflectance product exists for the scene. |
| `sza_mean, sza_std, sza_min, sza_max, sza_p25, sza_p50, sza_p75` | degrees | In-patch solar-zenith distribution. |
| `vza_mean, vza_std, vza_min, vza_max, vza_p25, vza_p50, vza_p75` | degrees | In-patch view-zenith distribution. |
| `n_pixels` | int | In-patch pixel count. |

## 3c. MASK summary — `{site}_{patch}_{ts}_mask_summary.csv`

One row per L2A MASK band.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `mask_band` | text | MASK band label (e.g. `Aggregate Flag`). |
| `kind` | text | `binary` or `continuous`. |
| `pct_flagged` | percent | Flagged fraction for binary bands (NaN for continuous). |
| `mean, std, min, max` | band units | Stats for continuous bands (NaN for binary). |
| `n_valid` | int | Non-fill in-patch pixel count. |

## 4. QC worksheet — `{site}_{patch}_qc.csv`

One row per scene. `reviewed` is patch-level (only the top row is read). Opt-out:
a blank or `TRUE` `keep` means the scene is kept.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `reviewed` | TRUE/FALSE | Patch-level: has the manual review been completed (top row only). |
| `scene_ts` | `YYYYMMDDThhmmss` | Scene timestamp. |
| `datetime` | ISO datetime | Acquisition time. |
| `has_rfl` | TRUE/FALSE | Reflectance product present. |
| `n_pixels` | int | In-patch pixel count. |
| `pct_flagged` | percent | Aggregate-Flag flagged-% (glance metric). |
| `keep` | TRUE/FALSE | Keep verdict (opt-out; blank = keep). |
| `reason` | text | `cloud` / `coverage` / `artifact` / `other`. |
| `note` | text | Free-text detail. |

## 5. Matched pairs — `{site}_{patch}_matched_pairs.csv`

One row per unordered scene pair.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `t1_scene_id`, `t2_scene_id` | text | The two scenes' GranuleURs (t1 earlier). |
| `t1_scene_dt`, `t2_scene_dt` | ISO datetime | Acquisition times. |
| `t1_scene_SZA`, `t2_scene_SZA`, `t1_scene_VZA`, `t2_scene_VZA` | degrees | In-patch mean geometry. |
| `delta_SZA`, `delta_VZA` | degrees | \|Δ\| of the in-patch means. |
| `delta_time` | days | \|Δ\| acquisition time. |
| `matched_pair` | TRUE/FALSE | Both tolerances met. |
| `t1_has_rfl`, `t2_has_rfl` | bool | Reflectance available per scene. |

## 6. Per-patch delta — `{site}_{patch}_delta.csv`

One row per `(matched pair × band)`.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `PICS_ID`, `patch_id`, `pair_id` | text | Site token, patch id, `{t1_ts}_{t2_ts}` pair key. |
| `t1_scene_id`, `t2_scene_id` | text | The two scenes. |
| `t1_dt`, `t2_dt` | ISO datetime | Acquisition times. |
| `dt_days` | days | Time separation. |
| `SZA_t1`, `SZA_t2` | degrees | Per-scene mean \|SZA\|. |
| `delta_SZA`, `delta_VZA` | degrees | Geometry differences. |
| `band_index` | int 0–284 | EMIT channel index (stable cross-scene identity). |
| `wavelength`, `wavelength_t2` | nm | t1's canonical grid; t2's label for the same band. |
| `rfl_t1_median … rfl_t2_mean` | unitless | Per-scene reflectance median/mean. |
| `delta_rfl_pct_median`, `delta_rfl_pct_mean` | percent | 100·(t2−t1)/t1 reflectance. |
| `rad_t1_median … rad_t2_mean` | µW/cm²/sr/nm | Per-scene radiance median/mean. |
| `delta_rad_pct_median`, `delta_rad_pct_mean` | percent | 100·(t2−t1)/t1 radiance. |
| `absorption_band_flag` | bool | Deep water-vapor band (L2A reflectance is fill). |

## 7. Aggregated delta — `{site}_{size}_delta.csv`

One row per `(unique acquisition pair × band)` — spatial-replicate patches
collapsed by median (option A). It keeps the pair identity/geometry/band columns
and the four delta values (`PICS_ID`, `pair_id`, `t1_scene_id`, `t2_scene_id`,
`t1_dt`, `t2_dt`, `dt_days`, `SZA_t1`, `SZA_t2`, `delta_SZA`, `delta_VZA`,
`band_index`, `wavelength`, `absorption_band_flag`, `delta_rfl_pct_median`,
`delta_rfl_pct_mean`, `delta_rad_pct_median`, `delta_rad_pct_mean`) but drops the
per-patch-only columns (`patch_id`, `wavelength_t2`, and the raw per-scene
`rfl_t*`/`rad_t*` values). Added columns:

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `patch_size` | int (km) | Patch size the aggregation was run at. |
| `n_patches` | int | Number of spatial-replicate patches collapsed into this row. |
| `delta_rfl_pct_median_{std,min,max}` | percent | Across-patch spread of the median reflectance delta. |
| `delta_rad_pct_median_{std,min,max}` | percent | Across-patch spread of the median radiance delta. |

## 8. Drift stats — `{size}_drift.csv`

One row per `(band × product)`; absorption-band rows carry NaN stats.

| Column | Type/unit | Meaning |
| --- | --- | --- |
| `product` | text | `rad` or `rfl`. |
| `band_index` | int 0–284 | EMIT channel index. |
| `wavelength` | nm | Channel wavelength (median across pairs). |
| `absorption_band` | bool | Water-vapor band (excluded from the fit). |
| `n` | int | Number of `(pair)` points fitted for the band. |
| `slope_pct_yr` | %/year | Theil–Sen drift rate (NaN if unfitted). |
| `ci_low_pct_yr`, `ci_high_pct_yr` | %/year | 95% confidence interval on the slope. |
| `intercept_pct` | percent | Theil–Sen intercept (bias diagnostic; ≈0 expected). |
| `kendall_tau` | −1…+1 | Kendall rank correlation. |
| `p_value` | float | Unadjusted Kendall-τ p-value. |
| `raw_significant` | bool | `p_value ≤ alpha`. |
| `bh_qvalue` | float | Benjamini–Hochberg adjusted p-value. |
| `bh_significant` | bool | BH-significant at the FDR target (`drift.alpha`). |
