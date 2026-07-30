# BRDF / quasi-BRDF drift — pipeline integration plan

Port `notebooks/BRDF_drift_workflow_allPICS.ipynb` into the main pipeline as a
library module + three executable scripts that parallel the notebook exactly and
export the same data and figures. **No method changes** — same model, same
bootstrap seeds, same formulas, so outputs reproduce the notebook numerically.

## Decisions (locked)

1. **Consistency check (§4b)** — compute the verdict, **warn loudly if violated,
   but proceed**. Record the verdict in the provenance sidecar. Never hard-block.
2. **Roster config** — `site_patches` lives in `run_config.yaml` (new `brdf:`
   block), the single source of truth.
3. **Orchestrator** — build the three scripts *and* add a `brdf` stage to
   `run_pipeline.py`.
4. **Scope** — pooled multi-site version only (this notebook). Single-site is
   handled as a **roster of one** — no separate fork.

Extra: no synthetic-data *walkthrough* deliverable. A minimal synthetic
fit-recovery **unit test** is kept only where it is the only way to assert the
fit is correct.

## Where it fits

BRDF drift is a **parallel drift branch**, a sibling to the existing matched-pairs
Theil–Sen drift. Both read the *same upstream products* (per-channel RAD/RFL cube
CSVs + geometry + the QC gate); they differ only in the statistic. BRDF is
inherently **multi-site pooled**, so it does **not** obey the `single` /
`allpatches` mode fork — it always runs on a roster of one-patch-per-site.

---

## New library module: `emit_pics/brdf_drift.py`

One module (repo convention: one analysis = one module, cf. `delta_calculations.py`,
`matched_pairs_drift_analysis.py`), split internally into a pure-math core and an
IO/assembly shell.

### Pure model & fitting (deterministic; fixed `default_rng` seeds)
- `quasi_brdf_resid(x, S, sidx, g, t, rho)` — the `(a+b·g)(1+γt)−ρ` residual.
- `fit_site(g, t, rho) -> {alpha, beta, gamma, n}` — standalone 3-param fit (§4b).
- `boot_site_gamma(g, t, rho, n_boot, seed=1) -> (gamma, (lo, hi))` — per-site CI (§4b).
- `fit_joint(frame, product) -> {sites, alpha[], beta[], gamma, success, sidx, ...}` — pooled `2S+1` fit (§5).
- `bootstrap_joint_gamma(frame, product, n_boot, seed=42) -> {gamma, se, ci, draws}` — stratified within-site γ bootstrap (§7).
- `channel_gamma(rho, g_all, t_all, site_all, n_boot, rng) -> (gamma, (lo, hi), pval, n)` — pooled γ + two-sided bootstrap p at one channel (§8).
- `consistency_check(persite_df) -> {verdict, overlap_lo, overlap_hi, per_product}` — **new**: the machine version of the §4b eyeball test. `verdict ∈ {pass, marginal, fail}` from whether all per-site γ CIs share a common overlap (per product). Used to warn, not block.

### IO / assembly (reuses existing loaders)
- `resolve_roster(cfg) -> (roster, unmapped, excluded)` — sites.yaml → drop `exclude_sites` → map to `site_patches` (§1).
- `load_site_obs(pics_id, patch, target_wl, require_qc_review) -> (df|None, wl_used, note)` — per-scene QC-surviving, `has_rfl`, median ρ at the nearest band (§2). Reuses `apply_qc_filter`, `load_scene_spectrum`.
- `assemble_design(roster, cfg) -> (obs, wl_by_site, T0)` — stack sites; set common clock `t` (yr since global earliest scene) and `g = cos(SZA)` (§2).
- `leverage_summary(obs) -> df` — per-site n, first/last, SZA/cosSZA span, t-span, `corr(cosSZA,t)` (§3).
- `apply_min_scenes_gate(obs, summary, min_scenes) -> (obs, summary, dropped)` (§3).
- `channel_matrix(obs, product, nband) -> np.ndarray` — scenes × bands ρ matrix (§8).
- `water_vapor_mask(ref_perchannel) -> (wlgrid, excl_mask, fit_bands)` — fixed-instrument band exclusion from one reflectance scene (§8).

### Reused as-is (not rebuilt)
`load_scene_spectrum` +
`_perchannel_path` (`delta_calculations`), `apply_qc_filter` / `is_reviewed`
(`qc_gate`), `load_sites` / `canonical_token` (`config`), plus `paths`,
`provenance`.

---

## Config additions

### `run_config.yaml` — new `brdf:` block
```yaml
brdf:
  site_patches:              # one representative patch per PICS (one patch = one site)
    Libya_4:   "10_7_1"
    Algeria_5: null          # null = skip until you fill a patch id
  exclude_sites: [Mauritania_2]
  target_wl:       858.0     # nm; nearest channel used
  min_scenes:      4         # per-site post-QC floor
  n_bootstrap:     1000      # pooled γ (§7)
  per_site_boot:   500       # per-site γ CI (§4b)
  spectrum_n_boot: 300       # per-channel (§8)
  gamma_cap:       500.0     # unstable-channel flag (§8)
```
Reuse the existing global `drift.alpha` (0.05) and `qc.require_review`.

### `emit_pics/run_config.py` — new `RunConfig` fields
`brdf_site_patches: dict`, `brdf_exclude_sites: list`, `brdf_target_wl: float`,
`brdf_min_scenes: int`, `brdf_n_bootstrap: int`, `brdf_per_site_boot: int`,
`brdf_spectrum_n_boot: int`, `brdf_gamma_cap: float`. Populated from `sec("brdf")`.
`as_dict()` already serializes nested dicts → provenance captures the full roster.

---

## Path additions (`emit_pics/paths.py`)

New dirs (added to `_ALL_OUTPUT_DIRS`): `BRDF_DIR = DATA/"brdf"`,
`BRDF_FIG_DIR = FIGURES/"brdf"`. Filenames are **not** keyed by a single site
(pooled); single-channel products carry the target wavelength.

| Function | Path |
|---|---|
| `brdf_design()` | `data/brdf/brdf_design.csv` |
| `brdf_leverage()` | `data/brdf/brdf_leverage.csv` |
| `brdf_persite_gamma(wl)` | `data/brdf/brdf_persite_gamma_{wl:.0f}nm.csv` |
| `brdf_joint_fit(wl)` | `data/brdf/brdf_joint_fit_{wl:.0f}nm.csv` |
| `brdf_bootstrap_gamma(wl)` | `data/brdf/brdf_bootstrap_gamma_{wl:.0f}nm.csv` |
| `brdf_drift_spectrum()` | `data/brdf/brdf_drift_spectrum.csv` |
| `brdf_leverage_fig(wl)` | `figures/brdf/brdf_leverage_scatter_{wl:.0f}nm.png` |
| `brdf_persite_gamma_fig(wl)` | `figures/brdf/brdf_persite_gamma_{wl:.0f}nm.png` |
| `brdf_corrected_fig(wl)` | `figures/brdf/brdf_corrected_{wl:.0f}nm.png` |
| `brdf_bootstrap_fig(wl)` | `figures/brdf/brdf_bootstrap_{wl:.0f}nm.png` |
| `brdf_drift_spectrum_fig(kind)` | `figures/brdf/brdf_drift_spectrum_{kind}.png` (`kind ∈ {bh, raw}`) |

---

## Scripts — three thin executables (`scripts/run_brdf_*.py`)

Each follows the house header: `cfg = load_config()`, paths via `emit_pics.paths`,
`provenance.write_sidecar(...)` after every data write, loud failure on missing
prerequisites.

### 1. `run_brdf_assemble.py` — roster + design table + leverage gate (§1–§3, §4a)
- Resolve roster; report unmapped/excluded sites.
- Assemble pooled design table; apply `MIN_SCENES` gate (warn on drops).
- **Outputs:** `brdf_design.csv` (site, patch, timestamp, datetime, sza_mean, t, g,
  rho_rad, rho_rfl), `brdf_leverage.csv`; figure `brdf_leverage_scatter_{wl}nm.png`
  (§4a, ρ vs t and ρ vs cosSZA colored by site) + pooled `corr(cosSZA,t)`.

### 2. `run_brdf_single_channel.py` — single-channel fit + diagnostics (§4b, 5, 6, 7)
- Reads `brdf_design.csv`.
- §4b per-site standalone γ + bootstrap CI → `consistency_check` → **warn loudly if
  verdict ≠ pass**, then proceed.
- §5 joint pooled fit (per-site α/β + shared γ). §7 stratified bootstrap of the
  pooled γ.
- **Outputs:** `brdf_persite_gamma_{wl}nm.csv` (product, site, n, gamma_pct_yr,
  ci_lo, ci_hi), `brdf_joint_fit_{wl}nm.csv` (product, site, alpha, beta, +
  shared gamma_pct_yr, success, n), `brdf_bootstrap_gamma_{wl}nm.csv` (product,
  gamma_pct_yr, boot_se, ci_lo, ci_hi, n, n_sites); figures
  `brdf_persite_gamma_{wl}nm.png` (§4b forest), `brdf_corrected_{wl}nm.png`
  (§6 collapse), `brdf_bootstrap_{wl}nm.png` (§7 histogram). Sidecar carries the
  §4b verdict.

### 3. `run_brdf_drift_spectrum.py` — γ(λ) spectrum (§8) — the expensive one
- Reads `brdf_design.csv`; builds per-channel ρ matrices; water-vapor exclusion.
- Pooled joint fit at every fit channel × 2 products; uncorrected bootstrap p;
  unstable-channel flag (`gamma_cap`).
- **Outputs:** `brdf_drift_spectrum.csv` (product, band_index, wavelength,
  gamma_pct_yr, ci_lo_pct_yr, ci_hi_pct_yr, p_value, sig_raw, n,
  unstable) — columns mirror the matched-pairs drift spectrum; figure
  `brdf_drift_spectrum.png` (rad top / rfl bottom, water-vapor bands shaded,
  red = significant).

---

## Orchestrator (`emit_pics/pipeline.py` + `scripts/run_pipeline.py`)

Add `"brdf"` to `STAGE_ORDER` and a `STEPS["brdf"]` entry. Because BRDF ignores the
mode fork, the same three scripts fill **both** slots of each `(single, allpatches)`
pair, so `scripts_for("brdf", mode)` returns them for either mode:
```python
"brdf": [
    ("run_brdf_assemble.py",       "run_brdf_assemble.py"),
    ("run_brdf_single_channel.py", "run_brdf_single_channel.py"),
    ("run_brdf_drift_spectrum.py", "run_brdf_drift_spectrum.py"),
],
```
`run_pipeline.py` help text notes the `brdf` stage is pooled (mode-agnostic).

---

## Tests (`tests/`)

- `test_brdf_drift.py` — (a) **fit-recovery**: build tiny arrays from known
  α/β/γ, assert `fit_site` / `fit_joint` recover them and CIs bracket truth (the
  only rigorous way to check the fit — justified synthetic use); (b)
  `consistency_check` verdicts on hand-built overlapping vs disjoint CIs;
  (c) `resolve_roster` exclusion/mapping; (d) `apply_min_scenes_gate` drops.
- Extend `test_run_config.py` (new `brdf_*` fields + defaults) and `test_paths.py`
  (new path functions) and `test_pipeline.py` (`brdf` stage resolves in both modes).
- No golden spectrum test (needs real per-channel CSVs + long bootstrap); covered
  by unit tests + a manual smoke run against real data.

---

## Docs

- `README.md` — add the `brdf` stage to the orchestrator table, `brdf_drift.py`
  to the module table, and a short "BRDF drift (pooled multi-site)" subsection.
- `docs/data_dictionary.md` — columns + units for the 6 new products and the
  `data/brdf/` + `figures/brdf/` layout.

---

## Build order (incremental, test-as-we-go)

1. Config + paths + `RunConfig` fields (+ their tests). Foundation.
2. `brdf_drift.py` pure-math core (+ recovery / consistency unit tests).
3. `brdf_drift.py` IO/assembly (roster, load_site_obs, design, leverage, matrices).
4. Script 1 (assemble) → run on real data; eyeball design + leverage + §4a scatter.
5. Script 2 (single_channel) → reproduce §4b/5/6/7; compare numbers to the notebook.
6. Script 3 (drift_spectrum) → reproduce §8; compare figure to the notebook.
7. `pipeline.py` wiring + `run_pipeline.py` + tests.
8. README + data_dictionary + confirm provenance sidecars on every product.

## Fidelity guarantee

Model, formulas, and seeds (`default_rng(1)` per-site, `(42)` pooled, `(2024)`
spectrum) are copied verbatim so the scripts reproduce the notebook's numbers and
figures. This port only relocates code into the library/script/config structure —
it does not alter the analysis.
