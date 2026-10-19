"""Generate Table 1 (data volume per PICS) from the analysis data products.

Per PICS, in manuscript order, counts:
  OLI-2 scenes   - QC-surviving Landsat-9 scenes at the site's BRDF patch
  EMIT total     - distinct EMIT acquisitions extracted over the site
  Matched pairs  - unique geometry-matched acquisition pairs (aggregated deltas)
  BRDF scenes    - QC-surviving EMIT scenes at the single BRDF patch

Writes data/table1_data_volume.csv and prints a paste-ready LaTeX table body.
Run after the analyze + brdf stages and the Landsat QC review.
"""
import glob
import os
import re

import pandas as pd

from emit_pics.paths import REPO_ROOT

DATA = REPO_ROOT / "data"
SITE_ORDER = ["Libya_1", "Libya_4", "Algeria_3", "Algeria_5", "Mauritania_1"]
LABEL = {s: s.replace("_", "~") for s in SITE_ORDER}

# manuscript Table 1 values, for a regression check (oli2, total, matched, brdf)
REFERENCE = {
    "Libya_1": (59, 57, 18, 16), "Libya_4": (69, 47, 8, 13),
    "Algeria_3": (67, 98, 15, 14), "Algeria_5": (63, 78, 26, 15),
    "Mauritania_1": (51, 55, 6, 7),
}


def _kept_scenes(qc_csv) -> int:
    """Count QC-surviving scenes in a worksheet (opt-out gate: keep unless FALSE)."""
    df = pd.read_csv(qc_csv)
    return int((df["keep"].astype(str).str.upper() != "FALSE").sum())


def _distinct_acquisitions(site: str, size: str) -> int:
    """Distinct EMIT acquisition timestamps extracted over a site, any patch."""
    rx = re.compile(rf"^{re.escape(site)}_{size}_\d+_\d+_(\d{{8}}T\d{{6}})_perchannel_rad\.csv$")
    pattern = str(DATA / "radiance" / f"{site}_{size}_*_perchannel_rad.csv")
    ts = {m.group(1) for p in glob.glob(pattern) if (m := rx.match(os.path.basename(p)))}
    return len(ts)


def build_table() -> pd.DataFrame:
    design = pd.read_csv(DATA / "brdf" / "brdf_design.csv")
    brdf_n = design.groupby("site").size()
    roster = design.groupby("site")["patch"].first()   # one BRDF patch per site
    rows = []
    for site in SITE_ORDER:
        patch = roster[site]
        size = patch.split("_")[0]
        rows.append({
            "site": site,
            "oli2_scenes": _kept_scenes(DATA / "landsat" / "qc" / f"{site}_{patch}_qc.csv"),
            "emit_total_scenes": _distinct_acquisitions(site, size),
            "matched_pairs": pd.read_csv(
                DATA / "delta_r" / "aggregated" / f"{site}_{size}_delta.csv")["pair_id"].nunique(),
            "brdf_scenes": int(brdf_n[site]),
        })
    return pd.DataFrame(rows)


def latex_body(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.iterrows():
        lines.append(f"    {LABEL[r.site]:<12} & {r.oli2_scenes} & {r.emit_total_scenes} "
                     f"& {r.matched_pairs} & {r.brdf_scenes} \\\\")
    return "\n".join(lines)


def main():
    df = build_table()
    out = DATA / "table1_data_volume.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}\n")
    print("LaTeX table body:\n" + latex_body(df))

    mism = [r.site for _, r in df.iterrows()
            if (r.oli2_scenes, r.emit_total_scenes, r.matched_pairs, r.brdf_scenes)
            != REFERENCE[r.site]]
    print("\nregression check vs manuscript Table 1: "
          + ("PASS (all cells match)" if not mism else f"DIFF at {mism}"))


if __name__ == "__main__":
    main()
