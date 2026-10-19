#!/usr/bin/env python
"""Compute per-wavelength % change across matched pairs for EVERY patch of a size.

The batched counterpart to ``run_delta_calculations.py``: instead of one patch, it
enumerates every patch of ``PATCH_SIZE`` in the PICS (the ``{size}_*`` rows of the
patch-granule dict) and computes the pairwise t1->t2 deltas for each one, writing
a tidy long CSV per patch to ``data/delta_r/per_patch/{PICS}_{PATCH}_delta.csv``
(one row per ``matched pair x wavelength``). Pure **local** post-processor (no
Earthdata): per patch it reads that patch's matched-pairs CSV (step 5) and the
per-channel RAD/RFL stats (from the cube extraction), joins the two scenes of each
matched pair **by band order**, and records the median & mean percent change in
reflectance and radiance at every wavelength.

Loud-fail accounting is **aggregated across the whole run**. A matched pair can
drop out two ways, both data-integrity problems that must never pass silently:
*skipped* (a scene's radiance per-channel CSV is missing — re-run the extraction)
or *degenerate* (the two scenes' band grids don't correspond). Each patch still
writes whatever succeeded; at the end every dropped pair is listed with its cause
and its patch, and the run exits non-zero (2) if any pair was dropped anywhere.
A patch whose matched-pairs CSV is absent (step 5 not run / blocked at QC for it)
is reported and skipped, but does **not** trip the drop exit code.

Prerequisite: run ``run_PICS_allpatches_id_matched_pairs.py`` (step 5) and the
cube extraction (per-channel CSVs) for the patches first. Set the target in
run_config.yaml, then run::

    python scripts/run_PICS_allpatches_delta_calculations.py
"""

import sys

import numpy as np
import pandas as pd

from emit_pics import paths, provenance
from emit_pics.config import canonical_token
from emit_pics.run_config import load_config
from emit_pics.delta_calculations import (
    DELTA_CSV_COLS,
    WL_MATCH_TOL_NM,
    absorption_band_wavelengths,
    compute_pair_delta,
    load_matched_pairs,
    load_scene_spectrum,
    pair_metadata,
    scene_perchannel_path,
)
from emit_pics.extract_patch_stats import load_patch_dict
from emit_pics.id_matched_pairs import sniff_delimiter
from emit_pics.PICS_allpatches_cube_extract import patch_ids_for_size

# INPUTS — set the analysis target in run_config.yaml, not here.
cfg = load_config()
paths.set_variant(cfg.output_variant)   # nest derived outputs under an active radiometry variant
PICS_ID    = canonical_token(cfg.site)   # canonical filename/label token, e.g. "Libya_4"
PATCH_SIZE = cfg.patch_size

# Repo-relative paths — every name is built in emit_pics.paths.
RFL_DIR   = paths.RFL_DIR
RAD_DIR   = paths.RAD_DIR
OUT_DIR   = paths.DELTA_PER_PATCH_DIR
DICT_PATH = paths.patch_dict(PICS_ID)


def process_patch(pid: str) -> dict:
    """Compute one patch's delta table; return a per-patch report (never exits).

    Mirrors ``run_delta_calculations.main`` for a single patch. The report dict
    carries ``status`` (``"produced"`` | ``"empty"`` | ``"no_matched"`` |
    ``"missing_csv"``), the matched/with-delta pair counts, and the ``skipped`` /
    ``degenerate`` drop lists + ``missing_rad`` paths for the aggregated loud-fail.
    """
    matched_path = paths.matched_pairs(PICS_ID, pid)
    out_path = paths.delta_per_patch(PICS_ID, pid)
    rep = {"pid": pid, "status": None, "n_matched": 0, "n_with_deltas": 0,
           "skipped": [], "degenerate": [], "missing_rad": {}}

    if not matched_path.exists():
        print(f"  {pid}: no matched-pairs CSV ({matched_path.name}); skipped.")
        rep["status"] = "missing_csv"
        return rep

    # 1. Matched (TRUE-only) pairs for this patch.
    pairs = load_matched_pairs(matched_path)
    rep["n_matched"] = len(pairs)
    if pairs.empty:
        print(f"  {pid}: 0 matched pair(s); nothing to compute.")
        rep["status"] = "no_matched"
        return rep

    # 2. Load each referenced scene's spectrum ONCE (radiance anchors the join;
    #    a scene with no radiance CSV is unusable — record the exact missing file).
    scene_ids = pd.unique(pairs[["t1_scene_id", "t2_scene_id"]].values.ravel())
    specs = {}
    for sid in scene_ids:
        spec = load_scene_spectrum(sid, PICS_ID, pid, RFL_DIR, RAD_DIR,
                                   earth_sun_normalize=cfg.earth_sun_normalize,
                                   e0_method=cfg.e0_method)
        specs[sid] = spec
        if spec is None:
            rep["missing_rad"][sid] = scene_perchannel_path(sid, PICS_ID, pid,
                                                            RAD_DIR, "rad")

    # 3. Derive the water-vapor bad-band wavelengths ONCE from the first
    #    reflectance-bearing scene (fixed instrument property).
    bad_bands = np.array([], dtype=float)
    for spec in specs.values():
        if spec is None:
            continue
        wls = absorption_band_wavelengths(spec)
        if len(wls):
            bad_bands = wls
            break

    # 4. Per pair: compute per-wavelength deltas; track the two drop modes.
    frames = []
    for _, row in pairs.iterrows():
        meta = pair_metadata(row, PICS_ID, pid)
        t1, t2 = specs[row["t1_scene_id"]], specs[row["t2_scene_id"]]
        if t1 is None or t2 is None:
            offenders = [(side, sid) for side, sid, spec in
                         (("t1", row["t1_scene_id"], t1),
                          ("t2", row["t2_scene_id"], t2)) if spec is None]
            rep["skipped"].append({"pair_id": meta["pair_id"], "offenders": offenders})
            continue
        df = compute_pair_delta(t1, t2, meta, bad_bands)
        if df.empty or int(df["rad_t2_median"].notna().sum()) == 0:
            rep["degenerate"].append({"pair_id": meta["pair_id"],
                                      "t1": row["t1_scene_id"], "t2": row["t2_scene_id"]})
            continue
        frames.append(df)

    # 5. Write whatever succeeded.
    if frames:
        table = pd.concat(frames, ignore_index=True)[DELTA_CSV_COLS]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        provenance.write_sidecar(out_path, cfg.as_dict())
        rep["n_with_deltas"] = int(table["pair_id"].nunique())
        rep["status"] = "produced"
    else:
        rep["status"] = "empty"

    n_drop = len(rep["skipped"]) + len(rep["degenerate"])
    drop_note = (f"; {n_drop} dropped ({len(rep['skipped'])} no-rad, "
                 f"{len(rep['degenerate'])} degenerate)" if n_drop else "")
    tail = "" if frames else "; no CSV written"
    print(f"  {pid}: {rep['n_with_deltas']}/{rep['n_matched']} pair(s) -> deltas"
          f"{drop_note}{tail}")
    return rep


def report_drops(reports: list) -> int:
    """Print the aggregated loud-fail section; return the total dropped-pair count.

    Lists every skipped / degenerate pair grouped by patch (with the exact missing
    radiance files and the fix), mirroring the single-patch runner but spanning the
    whole run. Returns 0 when nothing was dropped.
    """
    with_drops = [r for r in reports if r["skipped"] or r["degenerate"]]
    n_drop = sum(len(r["skipped"]) + len(r["degenerate"]) for r in with_drops)
    if not n_drop:
        return 0

    bar = "!" * 70
    print("\n" + bar)
    print(f"!! INCOMPLETE: {n_drop} matched pair(s) DROPPED across "
          f"{len(with_drops)} patch(es).")
    print(bar)
    all_missing = {}
    for r in with_drops:
        print(f"\n  patch {r['pid']}:")
        if r["skipped"]:
            print(f"    {len(r['skipped'])} skipped — missing radiance stats:")
            for s in r["skipped"]:
                print(f"      pair {s['pair_id']}:")
                for side, sid in s["offenders"]:
                    exp = r["missing_rad"].get(sid, "?")
                    print(f"        {side} scene missing radiance CSV: {sid}")
                    print(f"           expected -> {exp}")
                    all_missing[str(exp)] = True
        if r["degenerate"]:
            print(f"    {len(r['degenerate'])} degenerate — band grids do not "
                  f"correspond:")
            for d in r["degenerate"]:
                print(f"      pair {d['pair_id']}:  {d['t1']}  x  {d['t2']}")
    if all_missing:
        print(f"\n  {len(all_missing)} unique missing radiance file(s):")
        for p in sorted(all_missing):
            print(f"    - {p}")
        print("    Fix: re-run the cube extraction for the affected patch(es), "
              "then re-run this script.")
    print(f"\n  Degenerate cause: unequal band count, or a per-band wavelength gap "
          f"> {WL_MATCH_TOL_NM} nm (inspect central_wl before changing it).")
    return n_drop


def main() -> None:
    # 1. Enumerate this size's patches from the patch-granule dict.
    dict_df = load_patch_dict(DICT_PATH, sep=sniff_delimiter(DICT_PATH))
    patch_ids = patch_ids_for_size(dict_df, PATCH_SIZE)
    if not patch_ids:
        print(f"No patches of size {PATCH_SIZE!r} in {DICT_PATH.name}. Exiting.")
        sys.exit(1)

    print(f"{PICS_ID}, size {PATCH_SIZE} km: computing per-wavelength deltas for "
          f"{len(patch_ids)} patch(es).")

    # 2. Compute deltas per patch; collect reports.
    reports = [process_patch(pid) for pid in patch_ids]

    # 3. Summary tally.
    tally = {"produced": 0, "empty": 0, "no_matched": 0, "missing_csv": 0}
    for r in reports:
        tally[r["status"]] += 1
    n_deltas = sum(r["n_with_deltas"] for r in reports)
    print("\n--- Summary ---")
    print(f"  patches with delta CSV:      {tally['produced']}")
    print(f"  patches all-dropped (no CSV):{tally['empty']}")
    print(f"  patches with 0 matched pairs:{tally['no_matched']}")
    print(f"  patches missing matched CSV: {tally['missing_csv']}")
    print(f"  total pairs with deltas:     {n_deltas}")
    print(f"\nDelta CSVs -> {OUT_DIR}")
    if tally["missing_csv"]:
        print("  Some patches have no matched-pairs CSV — run "
              "run_PICS_allpatches_id_matched_pairs.py (step 5) for them first.")

    # 4. Aggregated loud fail — never let a dropped pair pass silently.
    n_drop = report_drops(reports)
    if n_drop:
        sys.exit(2)


if __name__ == "__main__":
    main()
