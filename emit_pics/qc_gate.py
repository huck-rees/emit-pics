"""Manual scene-level QC gate: a per-patch worksheet + a downstream filter.

After rendering the QC snapshot PDF you inspect each scene by eye. Bad scenes
(cloud, missing coverage, artifacts) are marked in an editable per-patch
worksheet; downstream steps then drop them. The design is **opt-out**: every
scene defaults to ``keep=TRUE`` and you only mark the rejects.

Two halves live here, both pure (no ``sys.exit``; the gate raises
:class:`QCReviewRequired` for the runner to handle):

* :func:`build_qc_sheet` — create/refresh ``data/qc/{PICS}_{PATCH}_qc.csv`` from
  the per-scene geometry table (co-located with inspection, called by
  ``run_display_qc_visuals``). Non-destructive: existing verdicts are preserved on
  refresh; newly appearing scenes are appended as ``keep=TRUE``.
* :func:`apply_qc_filter` — drop ``keep=FALSE`` scenes wherever the geometry
  table is consumed (``run_id_matched_pairs``, and so, transitively, the delta
  calculations and any future scene-keyed step).

Review safety: the worksheet carries a single patch-level ``reviewed`` flag (top
row only). Because opt-out means an untouched sheet (all TRUE) is
indistinguishable from a reviewed-all-good one, the gate refuses to proceed
until you set ``reviewed=TRUE`` (unless the caller passes
``require_review=False``). Adding new scenes on refresh resets it to FALSE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from emit_pics.extract_patch_stats import AGG_FLAG_LABEL

#: Controlled vocabulary for the ``reason`` column (free-text ``note`` for detail).
QC_REASONS = ["cloud", "coverage", "artifact", "other"]

#: Worksheet column order. ``reviewed`` is patch-level — only the top row's
#: value is read; the rest are left blank.
QC_SHEET_COLS = [
    "reviewed", "scene_ts", "datetime", "has_rfl", "n_pixels", "pct_flagged",
    "keep", "reason", "note",
]


class QCReviewRequired(Exception):
    """Raised by :func:`apply_qc_filter` when the QC gate blocks a run."""


# ============================================================
# PARSING HELPERS
# ============================================================
def _truthy(v) -> bool:
    """Interpret a reviewed/boolean cell as True (TRUE/1/YES/Y)."""
    return str(v).strip().upper() in {"TRUE", "1", "YES", "Y"}


def _keep(v) -> bool:
    """Interpret a ``keep`` cell: kept unless explicitly rejected.

    Rejection tokens: FALSE / 0 / NO / N / X / EXCLUDE. Blank stays kept
    (opt-out), so clearing a cell never silently drops a scene.
    """
    return str(v).strip().upper() not in {"FALSE", "0", "NO", "N", "X", "EXCLUDE"}


def _blank(v) -> str:
    """Normalize NaN/None to an empty string for text columns."""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# ============================================================
# GLANCE METRIC
# ============================================================
def _agg_pct(mask_dir, pics_id: str, patch_id: str, ts: str) -> float:
    """Aggregate-Flag flagged-% for one scene from its MASK summary, or NaN."""
    if mask_dir is None:
        return np.nan
    path = Path(mask_dir) / f"{pics_id}_{patch_id}_{ts}_mask_summary.csv"
    if not path.exists():
        return np.nan
    try:
        m = pd.read_csv(path)
        hit = m.loc[m["mask_band"] == AGG_FLAG_LABEL, "pct_flagged"]
        return float(hit.iloc[0]) if len(hit) else np.nan
    except (KeyError, ValueError, IndexError):
        return np.nan


# ============================================================
# WORKSHEET BUILD / REFRESH
# ============================================================
def build_qc_sheet(geom_path, qc_path, *, mask_dir=None,
                   pics_id: str, patch_id: str):
    """Create or non-destructively refresh a patch's QC worksheet.

    Rows come from the per-scene geometry table (one per scene, PDF-page order).
    On refresh, ``keep`` / ``reason`` / ``note`` are preserved per ``scene_ts``
    and any new scenes are appended as ``keep=TRUE``. The patch-level
    ``reviewed`` flag is preserved, but reset to FALSE if new scenes appeared.

    Returns
    -------
    (pd.DataFrame, list[str])
        The written worksheet and the list of newly added ``scene_ts`` values.
    """
    geom = pd.read_csv(geom_path).sort_values("datetime").reset_index(drop=True)

    base = pd.DataFrame({
        "scene_ts": geom["timestamp"].astype(str),
        "datetime": geom["datetime"].astype(str),
        "has_rfl": geom["has_rfl"],
        "n_pixels": geom["n_pixels"] if "n_pixels" in geom.columns else np.nan,
        "pct_flagged": [_agg_pct(mask_dir, pics_id, patch_id, ts)
                        for ts in geom["timestamp"].astype(str)],
    })
    base["keep"] = "TRUE"
    base["reason"] = ""
    base["note"] = ""

    reviewed_prev, prev_ts = False, set()
    qc_path = Path(qc_path)
    if qc_path.exists():
        prev = pd.read_csv(qc_path)
        if "reviewed" in prev.columns and len(prev):
            reviewed_prev = _truthy(prev["reviewed"].iloc[0])
        prev_by_ts = {str(r["scene_ts"]): r for _, r in prev.iterrows()}
        prev_ts = set(prev_by_ts)
        for i, ts in base["scene_ts"].items():
            if ts in prev_by_ts:
                p = prev_by_ts[ts]
                base.at[i, "keep"] = "TRUE" if _keep(p.get("keep", "TRUE")) else "FALSE"
                base.at[i, "reason"] = _blank(p.get("reason"))
                base.at[i, "note"] = _blank(p.get("note"))

    new_ts = [ts for ts in base["scene_ts"] if ts not in prev_ts]
    reviewed_val = reviewed_prev and not new_ts   # new scenes invalidate review

    base.insert(0, "reviewed", "")
    if len(base):
        base.loc[base.index[0], "reviewed"] = "TRUE" if reviewed_val else "FALSE"
    base = base[QC_SHEET_COLS]

    qc_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(qc_path, index=False)
    return base, new_ts


# ============================================================
# DOWNSTREAM FILTER
# ============================================================
def apply_qc_filter(scenes: pd.DataFrame, qc_path, *,
                    require_review: bool = True, timestamp_col: str = "timestamp"):
    """Drop QC-excluded scenes from a per-scene table.

    Parameters
    ----------
    scenes : pd.DataFrame
        Per-scene rows (e.g. from the geometry table), keyed by *timestamp_col*.
    qc_path : path-like
        The patch's QC worksheet.
    require_review : bool
        If True, raise :class:`QCReviewRequired` unless the worksheet exists and
        its ``reviewed`` flag is TRUE. If False, apply whatever exclusions exist
        (or none, if the sheet is absent) without gating.
    timestamp_col : str
        Column in *scenes* holding the scene timestamp to match ``scene_ts``.

    Returns
    -------
    (pd.DataFrame, dict)
        The filtered scenes and a report:
        ``{n_before, n_after, n_excluded, reasons: {reason: count}}``.

    Raises
    ------
    QCReviewRequired
        When *require_review* is True and the sheet is missing or unreviewed.
    """
    qc_path = Path(qc_path)
    if not qc_path.exists():
        if require_review:
            raise QCReviewRequired(
                f"QC worksheet not found: {qc_path}\n"
                "Run scripts/run_display_qc_visuals.py to create it, inspect the "
                "PDF, mark bad scenes (keep=FALSE), and set reviewed=TRUE in the "
                "top row. Set REQUIRE_QC_REVIEW=False to bypass the gate."
            )
        report = {"n_before": len(scenes), "n_after": len(scenes),
                  "n_excluded": 0, "reasons": {}}
        return scenes.reset_index(drop=True), report

    qc = pd.read_csv(qc_path)
    reviewed = _truthy(qc["reviewed"].iloc[0]) if ("reviewed" in qc.columns and len(qc)) else False
    if require_review and not reviewed:
        raise QCReviewRequired(
            f"QC worksheet is not marked reviewed: {qc_path}\n"
            "Inspect the QC PDF, mark any bad scenes (keep=FALSE) with a reason, "
            "then set reviewed=TRUE in the top row and re-run. "
            "Set REQUIRE_QC_REVIEW=False to bypass the gate."
        )

    excluded = {str(r["scene_ts"]): _blank(r.get("reason")) or "unspecified"
                for _, r in qc.iterrows() if not _keep(r.get("keep", "TRUE"))}

    ts = scenes[timestamp_col].astype(str)
    filtered = scenes[~ts.isin(excluded)].reset_index(drop=True)

    reasons: dict[str, int] = {}
    for scene_ts in ts[ts.isin(excluded)]:
        r = excluded[scene_ts]
        reasons[r] = reasons.get(r, 0) + 1

    report = {
        "n_before": len(scenes), "n_after": len(filtered),
        "n_excluded": len(scenes) - len(filtered), "reasons": reasons,
    }
    return filtered, report
