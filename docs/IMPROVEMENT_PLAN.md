# EMIT-PICS Improvement Plan

**Decisions:** single `run_config.yaml` · plain-Python `run_pipeline.py` · convert all scripts in one pass · public/publication-grade polish.

**Guiding rule:** *the science does not change.* Every step below either moves boilerplate into one place, adds files, or adds a safety net — none alters a formula, tolerance, or output-file location. We prove that with change #5, which is why it goes **first**.

---

## Change #5 (done FIRST) — Test suite as the safety net

**Goal:** Lock down current numerical behavior *before* touching anything, so the refactors in #1/#4 can't silently change results.

**New files:**
- `pyproject.toml` — adds `pytest` config (also used by #4 packaging).
- `tests/conftest.py` — tiny synthetic fixtures (hand-built DataFrames/arrays with known answers).
- `tests/test_delta_calculations.py`, `test_drift_analysis.py`, `test_matched_pairs.py`, `test_aggregate_pairs.py`, `test_qc_gate.py`, `test_patch_generator.py`, `test_config_paths.py`.
- `tests/golden/` — one small end-to-end regression fixture (see below).
- `.github/workflows/tests.yml` — run `pytest` on every push (public repo → CI expected).

**What gets tested (all already pure — verified in the code inventory):**

| Module | Function | Known-answer test |
|---|---|---|
| `delta_calculations` | `_pct_change`, `absorption_band_flag`, `compute_pair_delta` | % formula incl. near-zero denominator; **band-order join + ±3 nm wavelength-drift tolerance**; grid-mismatch → empty frame |
| `matched_pairs_bootstrap` | `point_slope`, `resample_slope`, `bootstrap_channel`, `compute_bootstrap_drift_spectrum` | Theil–Sen point estimate recovers a known slope; PICS-stratified with-replacement resample; CI brackets truth and is wider than a patch-draw-only scheme; <3-point guard; seed-reproducible |
| `id_matched_pairs` | `find_matched_pairs`, `in_patch_mask`, `acq_timestamp` | 3 scenes with set SZA/VZA → known matched flags; point in/out of polygon |
| `PICS_allpatches_aggregate_pairs` | `aggregate_by_pair`, `collapse_summary` | 2 patches, same acquisition pair → single median row, `n_patches=2` |
| `qc_gate` | `apply_qc_filter`, `_keep` | `keep=FALSE` row dropped; unreviewed sheet raises `QCReviewRequired` |
| `patch_generator` | `_tile_one_size` | fully-contained-only tiling on a unit box |

**The golden test (the real insurance):** build a miniature synthetic dataset (a handful of fake per-channel CSVs + a matched-pairs CSV), run the *pure* delta→aggregate→drift chain, and snapshot the output CSVs. After each later change we re-run it; identical output = the science is provably untouched.

**Effort:** ~1–1.5 days. **Risk:** none (additive). **Verify:** `pytest` green on the current, unmodified code.

---

## Change #1 — One `run_config.yaml` + one `paths.py`

**Goal:** Kill the biggest redundancy — the repeated `USER-CONFIGURABLE INPUTS` blocks and the ~13 duplicated `*_DIR` path constants and filename templates copied into every script.

**New files:**
- `run_config.yaml` (repo root) — the single place you edit. Sketch:
  ```yaml
  site: Libya_4          # any spelling; resolved via config.get_site
  patch: "10_7_1"        # single-patch steps
  patch_size: "10"       # batched steps
  products: {rad_short_name: EMITL1BRAD, rfl_short_name: EMITL2ARFL}
  matching: {sza_tol: 2.0, vza_tol: 5.0}
  qc: {require_review: true}
  spectra: {stat: median}
  drift: {alpha: 0.05, n_bootstrap: 1000, seed: 0}
  extract: {cache_pixels: true, want_obs: true, filter_agg_flag: false, min_valid_frac: 0.60}
  granule_search: {date_start: "2022-08-01", date_end: "2026-06-30", patch_size_km: 10}
  patch_gen: {min_patch_km: 10, increment_km: 10}
  ```
- `emit_pics/run_config.py` — `load_config(path=None)` returns a typed object (dataclass) with sensible defaults, so a missing key fails loudly with a clear message.
- `emit_pics/paths.py` — the **one** module that owns the `data/`+`figures/` layout and every filename template. One function per product, e.g.:
  ```python
  def geometry(site, patch)          -> Path   # data/geometry/geometry_{tok}_{patch}.csv
  def perchannel(site, patch, ts, product) -> Path
  def matched_pairs_csv(site, patch) -> Path
  def delta_per_patch(site, patch)   -> Path
  def delta_aggregated(site, size)   -> Path
  def drift_csv(size)                -> Path
  def ensure_dirs()                  # mkdir -p all output dirs
  ```
  It uses `canonical_token()` (change #4) so the token matches today's filenames byte-for-byte.

**Changed:** every `run_*.py` header shrinks from ~25 lines of constants to ~3:
```python
from emit_pics.run_config import load_config
from emit_pics import paths
cfg = load_config()
```
All `to_csv`/`savefig` targets become `paths.something(cfg.site, cfg.patch)`.

**Non-breaking safeguards:** `paths.*` resolves to the identical locations; the golden test + a one-time diff of real outputs (below) confirm byte-identical files.

**Effort:** ~2–3 days for all 17 scripts. **Risk:** low, caught by tests. **Verify:** re-run one real patch end-to-end, `diff` new outputs vs. a pre-change backup — expect zero differences.

---

## Change #2 — `run_pipeline.py` orchestrator

**Goal:** Encode the step order + the single/batched fork + the QC-gate pause, so you stop hand-remembering "which script next."

**New file:** `scripts/run_pipeline.py` (plain Python). Design:
- Reads `run_config.yaml`; takes `--mode single|allpatches` and `--stage prep|analyze|drift`.
- Calls the existing scripts via `subprocess` (each step stays isolated — a crash in one doesn't corrupt the orchestrator; and because the scripts now read `run_config.yaml`, no arguments need threading through).
- **`prep`** runs steps 1→4 (generate → granules → extract → QC products), then **stops** with a clear message: *"Review QC now (`run_qc_review.py`), mark `reviewed=TRUE`, then run `--stage analyze`."*
- **`analyze`** runs 5→7 (matched pairs → deltas → aggregate) for the current site; it checks the QC gate first and refuses if unreviewed (reusing `qc_gate`), so the pause is enforced, not just suggested.
- **`drift`** runs step 8 (cross-PICS) once you've processed the sites you want — kept separate because step 8 pools *all* sites of a size.
- Re-running is safe: expensive extraction already has `RESUME=True`; the local post-processors are cheap.

**Effort:** ~1 day. **Risk:** low (a thin wrapper; changes no step's internals). **Verify:** a full `prep`→review→`analyze`→`drift` pass on one site reproduces today's manual sequence.

---

## Change #3 — Provenance sidecars

**Goal:** Make every output self-documenting and reproducible — publication-grade traceability.

**New file:** `emit_pics/provenance.py` with `write_sidecar(output_path, params)` that writes `output_path + ".json"` containing: ISO timestamp, `git` commit + dirty flag, script name, Python + key package versions (numpy/pandas/scipy/earthaccess), and the exact `run_config` values used.

**Changed:** since we're already editing every script for #1, add **one line** after each primary write:
```python
provenance.write_sidecar(out_path, cfg.as_dict())
```
The write-point inventory already pinpoints each script's primary output, so this is mechanical.

**Note:** sidecars live in `data/` (gitignored) and travel with the data — when you archive results to Zenodo for the paper, data + sidecars go together, and each number is traceable to an exact commit.

**Effort:** ~0.5 day. **Risk:** none (additive). **Verify:** every output has a matching `.json`; commit hash matches `git rev-parse HEAD`.

---

## Change #4 — Naming consistency, dedup, docs & public-release polish

**A. Consistent site token + library dedup (the redundancy your ask targets):**
- Add `canonical_token(site)` to `config.py` (= `site_name.replace(" ", "_")`, matching today's filenames). Route `paths.py` and the two upstream scripts (`id_granules`, `patch_generator`) through it, so the token is derived one way everywhere.
- Consolidate the genuine duplicates the inventory found: `load_patch_dict` + `split_products` (duplicated in `id_matched_pairs.py` **and** `extract_patch_stats.py`) → keep one, re-import in the other. `open_obs` / `open_rfl_mask` (3–4 near-copies) → one shared `emit_pics/_streaming.py`. Signatures unchanged; guarded by the #5 tests.

**B. Data dictionary & docs:**
- `docs/data_dictionary.md` — every CSV product, its columns + units, and which step writes it (built from the existing `DELTA_CSV_COLS` / `AGG_CSV_COLS` / `PAIR_CSV_COLS` / `STAT_COLS` constants) + the `data/` and `figures/` directory map.
- Update `README.md` to the new `run_config.yaml` + `run_pipeline.py` workflow.

**C. Packaging & reproducibility (public release):**
- `pyproject.toml` — makes `emit_pics` pip-installable (`pip install -e .`); **deletes the `sys.path.insert` block from all 17 scripts**.
- Pin the environment: add a locked `environment.lock.yml` (`conda env export`) alongside the human-readable `environment.yml`, so the paper is reproducible years later.
- `LICENSE` + `CITATION.cff`. ⚠️ **One thing to check, not me:** the repo path contains "JPL," and NASA/JPL work often requires institutional sign-off on license choice and public release. **Confirm the license and clearance with your advisor/JPL before we add `LICENSE` or make anything public** — leave a placeholder and do not push anything public until cleared.

**Effort:** ~2 days (dedup + docs + packaging). **Risk:** dedup is the only code-touching part → covered by tests. **Verify:** `pytest` green; `pip install -e .` then `import emit_pics` works with no `sys.path` hack.

---

## Suggested order & the non-breaking proof

1. **Branch** (`git checkout -b refactor-workflow`).
2. **#5 tests + golden snapshot** on the *current* code — the safety net.
3. **#4A** `canonical_token` + `pyproject.toml` (editable install, drop `sys.path`) + library dedup — run tests.
4. **#1** `run_config.yaml` + `paths.py`, convert all scripts — run tests.
5. **#3** provenance (same edit pass as #1).
6. **#2** `run_pipeline.py`.
7. **#4B/C** docs, data dictionary, pinned env, CITATION (license pending clearance).
8. **Final proof:** re-run one real patch end-to-end and `diff` outputs against a pre-refactor backup → **zero differences** = science provably intact.

**Total rough effort:** ~7–9 focused days. Biggest payoff per hour: **#5 then #1** (they remove the "edit 10 files" foot-gun and make everything after them safe).
