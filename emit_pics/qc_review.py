"""Interactive keep/remove reviewer that writes QC verdicts straight to the CSV.

A single-user desktop tool (pure matplotlib — no server, no notebook, no extra
dependency) for entering the keep/remove verdicts the QC gate consumes.  It
renders the **same** multipanel QC figure the PDF uses
(:func:`emit_pics.display_qc_visuals.render_scene_figure`), but **from the cached
pixel cubes** (offline), and attaches Keep/Remove controls that update the
per-patch worksheet ``data/qc/{PICS}_{PATCH}_qc.csv`` live via the same
non-destructive schema :mod:`emit_pics.qc_gate` defines.

Two windows:

* **viewer** — the QC multipanel, redrawn in place per scene (one reused figure);
* **controls** — persistent Keep / Remove / Prev / Next / Reviewed / Quit buttons,
  a reason picker, and a status readout.

Every action is available by mouse *and* keyboard (keyboard is far faster for bulk
review):

    k          keep this scene            r          remove (uses reason picker)
    1-4        remove with reason 1-4     left/right  previous / next scene
    space      next                       v          mark this patch reviewed
    q / esc    save & quit

Verdicts are written to the worksheet immediately on each action (crash-safe), so
there is no explicit "save".  A scene with no cached cube shows a placeholder (you
can still mark it, but re-run extraction with ``CACHE_PIXELS=True`` to see it).
"""

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from emit_pics.config import pics_polygon
from emit_pics.id_matched_pairs import load_patch_geom_4326
from emit_pics.single_patch_cube_extract import (
    cube_to_render_inputs,
    read_cube_netcdf,
)
from emit_pics.display_qc_visuals import render_scene_figure
from emit_pics.qc_gate import QC_REASONS, QC_SHEET_COLS, build_qc_sheet
from emit_pics.qc_report import default_qc_config, load_spectra_context


def _write_sheet(df: pd.DataFrame, qc_path) -> None:
    """Persist the worksheet in the canonical column order (crash-safe write)."""
    df[QC_SHEET_COLS].to_csv(qc_path, index=False)


class QCReviewer:
    """Stateful reviewer over one or more patches of a PICS."""

    def __init__(self, repo_root, pics_id: str, patch_ids, *,
                 spectra_stat: str = "median"):
        self.pics_id = pics_id
        self.pics_poly = pics_polygon(pics_id)
        self.spectra_stat = spectra_stat
        self.patches = {}      # pid -> dict(cfg, df, qc_path, geom, spectra, dt_by_ts)
        self.items = []        # flat [(pid, scene_ts), ...] in worksheet (date) order
        self.i = 0
        self.reason = QC_REASONS[0]
        self._n_no_cube = 0

        for pid in patch_ids:
            self._load_patch(repo_root, pid)

        if not self.items:
            raise SystemExit(
                "No reviewable scenes found. Run run_single_patch_cube_extract.py / "
                "run_PICS_allpatches_cube_extract.py (with CACHE_PIXELS=True) and the "
                "QC step to create cubes + worksheets first.")

    # ---- loading -------------------------------------------------------
    def _load_patch(self, repo_root, pid: str) -> None:
        cfg = default_qc_config(repo_root, self.pics_id, pid,
                                prefer_cube=True, spectra_stat=self.spectra_stat,
                                make_pdf=False)
        if not cfg.qc_path.exists():
            if cfg.geom_path.exists():
                build_qc_sheet(cfg.geom_path, cfg.qc_path, mask_dir=cfg.mask_dir,
                               pics_id=self.pics_id, patch_id=pid)
            else:
                print(f"  skip {pid}: no worksheet and no geometry table.")
                return
        try:
            geom = load_patch_geom_4326(cfg.bnds_path, pid)
        except Exception as ex:
            print(f"  skip {pid}: no patch geometry ({ex}).")
            return

        # Read every column as text (fill blanks) so writing a string verdict into
        # an all-empty reason/note column never hits pandas' dtype-mismatch path.
        df = pd.read_csv(cfg.qc_path, dtype=str).fillna("")
        spectra, _ = load_spectra_context(cfg)
        dt_by_ts = dict(zip(df["scene_ts"], pd.to_datetime(df["datetime"])))
        self.patches[pid] = {"cfg": cfg, "df": df, "qc_path": cfg.qc_path,
                             "geom": geom, "spectra": spectra, "dt_by_ts": dt_by_ts}
        self.items += [(pid, ts) for ts in df["scene_ts"].tolist()]
        self._n_no_cube += sum(
            not (cfg.cube_dir / f"cube_{self.pics_id}_{pid}_{ts}.nc").exists()
            for ts in df["scene_ts"])

    # ---- current-item accessors ---------------------------------------
    def _cur(self):
        pid, ts = self.items[self.i]
        return pid, ts, self.patches[pid]

    # ---- UI ------------------------------------------------------------
    def _build_ui(self) -> None:
        self.viewer_fig = plt.figure("QC review — viewer", figsize=(16, 12))
        self.ctrl_fig = plt.figure("QC review — controls", figsize=(7.5, 2.6))
        self._widgets = []

        def add_btn(rect, label, cb):
            ax = self.ctrl_fig.add_axes(rect)
            b = Button(ax, label)
            b.on_clicked(lambda _evt: cb())
            self._widgets.append(b)

        add_btn([0.02, 0.68, 0.17, 0.26], "Keep (k)", self._keep)
        add_btn([0.21, 0.68, 0.17, 0.26], "Remove (r)", self._remove)
        add_btn([0.02, 0.38, 0.11, 0.24], "< Prev", self._prev)
        add_btn([0.14, 0.38, 0.11, 0.24], "Next >", self._next)
        add_btn([0.26, 0.38, 0.12, 0.24], "Reviewed (v)", self._mark_reviewed)
        add_btn([0.26, 0.08, 0.12, 0.24], "Quit (q)", self._quit)

        ax_r = self.ctrl_fig.add_axes([0.42, 0.10, 0.22, 0.84])
        ax_r.set_title("remove reason", fontsize=8)
        self.radio = RadioButtons(ax_r, QC_REASONS)
        self.radio.on_clicked(self._set_reason)
        self._widgets.append(self.radio)

        self.status = self.ctrl_fig.text(
            0.67, 0.5, "", va="center", ha="left", fontsize=9, family="monospace")

        for f in (self.viewer_fig, self.ctrl_fig):
            f.canvas.mpl_connect("key_press_event", self._on_key)

    # ---- rendering -----------------------------------------------------
    def _render(self) -> None:
        pid, ts, P = self._cur()
        cube_path = P["cfg"].cube_dir / f"cube_{self.pics_id}_{pid}_{ts}.nc"
        rendered = False
        if cube_path.exists():
            ds = read_cube_netcdf(cube_path)
            try:
                inp = cube_to_render_inputs(ds)
                if inp is not None:
                    render_scene_figure(
                        inp["rfl"], inp["obs"], mask=inp["mask"], bounds=inp["bounds"],
                        patch_geom_4326=P["geom"], pics_poly=self.pics_poly,
                        granule_poly=inp["granule_poly"], patch_id=pid,
                        granule_ur=str(ds.attrs.get("rfl_ur", "")),
                        acq_dt=P["dt_by_ts"].get(ts), draw_nadir=False,
                        spectra=P["spectra"], acq_ts=ts, fig=self.viewer_fig)
                    rendered = True
            finally:
                ds.close()
        if not rendered:
            self.viewer_fig.clf()
            self.viewer_fig.text(
                0.5, 0.5,
                f"no cached cube for\n{pid}   {ts}\n\n"
                f"(re-run extraction with CACHE_PIXELS=True to view this scene;\n"
                f"you can still mark a verdict)",
                ha="center", va="center", fontsize=14, color="0.35")
        self.viewer_fig.canvas.draw_idle()
        self._sync_radio()
        self._update_status()

    def _sync_radio(self) -> None:
        """Reflect the current scene's stored reason in the reason picker."""
        _, ts, P = self._cur()
        row = P["df"]["scene_ts"] == ts
        keep = str(P["df"].loc[row, "keep"].iloc[0]).upper() != "FALSE"
        reason = str(P["df"].loc[row, "reason"].iloc[0])
        if not keep and reason in QC_REASONS and self.radio.value_selected != reason:
            self.radio.set_active(QC_REASONS.index(reason))  # updates self.reason too

    def _update_status(self) -> None:
        pid, ts, P = self._cur()
        df = P["df"]
        row = df["scene_ts"] == ts
        keep = str(df.loc[row, "keep"].iloc[0]).upper() != "FALSE"
        reason = str(df.loc[row, "reason"].iloc[0]) if not keep else ""
        reviewed = str(df.loc[df.index[0], "reviewed"]).upper() == "TRUE"
        pts = df["scene_ts"].tolist()
        j, n = pts.index(ts), len(pts)
        n_removed = int((df["keep"].astype(str).str.upper() == "FALSE").sum())
        verdict = "KEEP" if keep else f"REMOVE({reason})"
        self.status.set_text(
            f"patch {pid}\n"
            f"scene {j + 1}/{n}   [{self.i + 1}/{len(self.items)} overall]\n"
            f"{ts}\n"
            f"verdict : {verdict}\n"
            f"removed in patch : {n_removed}\n"
            f"reviewed : {'YES' if reviewed else 'no'}\n"
            f"reason picker : {self.reason}")
        self.ctrl_fig.canvas.draw_idle()
        try:
            self.viewer_fig.canvas.manager.set_window_title(
                f"QC — {pid} {ts}  [{self.i + 1}/{len(self.items)}]  {verdict}")
        except Exception:
            pass

    # ---- verdict actions ----------------------------------------------
    def _apply(self, keep: bool, reason: str = "") -> None:
        pid, ts, P = self._cur()
        df = P["df"]
        row = df["scene_ts"] == ts
        df.loc[row, "keep"] = "TRUE" if keep else "FALSE"
        df.loc[row, "reason"] = "" if keep else reason
        _write_sheet(df, P["qc_path"])
        self._advance()

    def _keep(self):
        self._apply(True)

    def _remove(self, reason=None):
        self._apply(False, reason or self.reason)

    def _reason_key(self, idx: int):
        self.reason = QC_REASONS[idx]
        self.radio.set_active(idx)
        self._remove(self.reason)

    def _set_reason(self, label):
        self.reason = label  # radio callback

    def _mark_reviewed(self):
        pid, _, P = self._cur()
        P["df"].loc[P["df"].index[0], "reviewed"] = "TRUE"
        _write_sheet(P["df"], P["qc_path"])
        self._update_status()
        print(f"  patch {pid} marked reviewed.")

    # ---- navigation ----------------------------------------------------
    def _advance(self):
        self.i = min(self.i + 1, len(self.items) - 1)
        self._render()

    def _next(self):
        self.i = min(self.i + 1, len(self.items) - 1)
        self._render()

    def _prev(self):
        self.i = max(self.i - 1, 0)
        self._render()

    def _quit(self):
        plt.close("all")

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "k":
            self._keep()
        elif k == "r":
            self._remove()
        elif k in ("1", "2", "3", "4") and int(k) - 1 < len(QC_REASONS):
            self._reason_key(int(k) - 1)
        elif k in ("right", " "):
            self._next()
        elif k == "left":
            self._prev()
        elif k == "v":
            self._mark_reviewed()
        elif k in ("q", "escape"):
            self._quit()

    # ---- run -----------------------------------------------------------
    def run(self):
        self._build_ui()
        self._render()
        print("Interactive QC review — keys:")
        print("  k=keep  r=remove  1-4=remove w/ reason  <-/->=prev/next  "
              "v=mark patch reviewed  q=quit")
        if self._n_no_cube:
            print(f"  ({self._n_no_cube} scene(s) have no cached cube; shown as "
                  f"placeholders.)")
        plt.show(block=True)

        reviewed = [pid for pid, P in self.patches.items()
                    if str(P["df"].loc[P["df"].index[0], "reviewed"]).upper() == "TRUE"]
        print(f"\nDone. {len(reviewed)}/{len(self.patches)} patch(es) marked "
              f"reviewed: {reviewed if reviewed else '(none)'}")
        print("Verdicts saved to the per-patch worksheets under data/qc/.")
