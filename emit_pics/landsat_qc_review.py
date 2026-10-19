"""Interactive keep/remove reviewer for the Landsat 9 QC cubes.

The Landsat analog of :mod:`emit_pics.qc_review`: a single-user desktop tool
(pure matplotlib) that renders each scene's QC multipanel
(:func:`emit_pics.landsat_qc_visuals.render_scene_figure`) from the cached GeoTIFF
cubes and attaches Keep/Remove controls that write verdicts straight into the
per-patch worksheet ``data/landsat/qc/{PICS}_{PATCH}_qc.csv`` via the same
non-destructive schema :mod:`emit_pics.qc_gate` defines (reused unchanged).

It reuses the EMIT reviewer's cross-backend window-placement helpers and mirrors
its keys, but is deliberately single-patch-simpler (no multi-patch boundary gate):

    k          keep this scene          r          remove (uses reason picker)
    1-4        remove with reason 1-4   left/right  previous / next scene
    space      next                     v          mark this patch reviewed
    q / esc    save & quit

Verdicts are written immediately on each action (crash-safe). A scene with no
cached cube shows a placeholder (still markable). Point the cube dir at wherever
the cubes live (e.g. an external drive) via
:func:`emit_pics.paths.set_ls_cube_dir` before constructing the reviewer.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from emit_pics import paths
from emit_pics.config import canonical_token
from emit_pics.qc_gate import QC_REASONS, build_qc_sheet
from emit_pics.landsat_cube_extract import read_cube_geotiff
from emit_pics.landsat_qc_visuals import load_toa_spectra, render_scene_figure
from emit_pics.qc_review_ui import (
    write_sheet as _write_sheet,
    screen_workarea as _screen_workarea,
    set_window_geometry as _set_window_geometry,
    move_window as _move_window,
)


class LandsatQCReviewer:
    """Stateful keep/remove reviewer over one Landsat patch."""

    def __init__(self, pics_id: str, patch_id: str):
        self.pics_id = canonical_token(pics_id)
        self.patch_id = patch_id
        self.reason = QC_REASONS[0]
        self.i = 0
        self._fitted = False

        geom_path = paths.ls_geometry(pics_id, patch_id)
        self.qc_path = paths.ls_qc_sheet(pics_id, patch_id)
        if not geom_path.exists():
            raise SystemExit(f"No geometry table for {self.pics_id} {patch_id}; "
                             "run the Landsat extractor first.")
        if not self.qc_path.exists():
            build_qc_sheet(geom_path, self.qc_path, mask_dir=None,
                           pics_id=self.pics_id, patch_id=patch_id)

        # All columns as text so writing a verdict never hits a dtype-mismatch path.
        self.df = pd.read_csv(self.qc_path, dtype=str).fillna("")
        self.dt_by_ts = dict(zip(self.df["scene_ts"], self.df["datetime"]))
        self.spectra = load_toa_spectra(self.pics_id, patch_id)
        self.items = self.df["scene_ts"].tolist()
        self._n_no_cube = sum(not self._cube_path(ts).exists() for ts in self.items)
        if not self.items:
            raise SystemExit("Worksheet has no scenes.")

    # ---- paths / accessors --------------------------------------------
    def _cube_path(self, ts):
        return paths.ls_pixel_cube(self.pics_id, self.patch_id, ts)

    def _cur_ts(self):
        return self.items[self.i]

    # ---- UI ------------------------------------------------------------
    def _build_ui(self):
        self.viewer_fig = plt.figure("Landsat QC — viewer", figsize=(14, 7))
        self.ctrl_fig = plt.figure("Landsat QC — controls", figsize=(7.5, 2.6))
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

        self.status = self.ctrl_fig.text(0.67, 0.5, "", va="center", ha="left",
                                         fontsize=9, family="monospace")
        for f in (self.viewer_fig, self.ctrl_fig):
            f.canvas.mpl_connect("key_press_event", self._on_key)

    # ---- window placement (dock controls under the viewer) ------------
    def _place_windows(self) -> bool:
        vmgr = self.viewer_fig.canvas.manager
        area = _screen_workarea(vmgr)
        if area is None:
            return False
        sw, sh = area
        ctrl_h = min(300, max(200, int(sh * 0.22)))
        avail_h = max(200, sh - ctrl_h)
        w_in, h_in = self.viewer_fig.get_size_inches()
        dpi = self.viewer_fig.get_dpi()
        scale = min(sw / (w_in * dpi), avail_h / (h_in * dpi), 1.0)
        if scale < 1.0:
            self.viewer_fig.set_dpi(max(30.0, dpi * scale))
            self.viewer_fig.canvas.draw_idle()
        vh = int(round(self.viewer_fig.get_size_inches()[1] * self.viewer_fig.get_dpi()))
        _move_window(vmgr, 0, 0)
        _set_window_geometry(self.ctrl_fig.canvas.manager,
                             0, min(vh, avail_h), min(sw, 1000), ctrl_h)
        self._fitted = True
        return True

    def _install_one_shot_placement(self):
        state = {"done": False}

        def _cb(_evt):
            if not state["done"] and self._place_windows():
                state["done"] = True

        for f in (self.viewer_fig, self.ctrl_fig):
            f.canvas.mpl_connect("draw_event", _cb)
        self._place_windows()

    # ---- rendering -----------------------------------------------------
    def _render(self):
        ts = self._cur_ts()
        cube_path = self._cube_path(ts)
        if cube_path.exists():
            cube = read_cube_geotiff(cube_path)
            render_scene_figure(cube, pics_id=self.pics_id, patch_id=self.patch_id,
                                scene_ts=ts, acq_dt=self.dt_by_ts.get(ts),
                                spectra=self.spectra, fig=self.viewer_fig,
                                resize=not self._fitted)
        else:
            self.viewer_fig.clf()
            self.viewer_fig.text(0.5, 0.5,
                                 f"no cached cube for\n{self.patch_id}   {ts}\n\n"
                                 f"(re-run extraction with cube caching to view;\n"
                                 f"you can still mark a verdict)",
                                 ha="center", va="center", fontsize=14, color="0.35")
        self.viewer_fig.canvas.draw_idle()
        self._sync_radio()
        self._update_status()

    def _sync_radio(self):
        ts = self._cur_ts()
        row = self.df["scene_ts"] == ts
        keep = str(self.df.loc[row, "keep"].iloc[0]).upper() != "FALSE"
        reason = str(self.df.loc[row, "reason"].iloc[0])
        if not keep and reason in QC_REASONS and self.radio.value_selected != reason:
            self.radio.set_active(QC_REASONS.index(reason))

    def _update_status(self):
        ts = self._cur_ts()
        row = self.df["scene_ts"] == ts
        keep = str(self.df.loc[row, "keep"].iloc[0]).upper() != "FALSE"
        reason = str(self.df.loc[row, "reason"].iloc[0]) if not keep else ""
        reviewed = str(self.df.loc[self.df.index[0], "reviewed"]).upper() == "TRUE"
        n_removed = int((self.df["keep"].astype(str).str.upper() == "FALSE").sum())
        verdict = "KEEP" if keep else f"REMOVE({reason})"
        self.status.set_text(
            f"patch {self.patch_id}\n"
            f"scene {self.i + 1}/{len(self.items)}\n"
            f"{ts}\n"
            f"verdict : {verdict}\n"
            f"removed in patch : {n_removed}\n"
            f"reviewed : {'YES' if reviewed else 'no'}\n"
            f"reason picker : {self.reason}")
        self.ctrl_fig.canvas.draw_idle()
        try:
            self.viewer_fig.canvas.manager.set_window_title(
                f"Landsat QC — {self.patch_id} {ts} [{self.i+1}/{len(self.items)}] {verdict}")
        except Exception:
            pass

    # ---- verdict actions ----------------------------------------------
    def _apply(self, keep: bool, reason: str = ""):
        ts = self._cur_ts()
        row = self.df["scene_ts"] == ts
        self.df.loc[row, "keep"] = "TRUE" if keep else "FALSE"
        self.df.loc[row, "reason"] = "" if keep else reason
        _write_sheet(self.df, self.qc_path)
        self._next()

    def _keep(self):
        self._apply(True)

    def _remove(self, reason=None):
        self._apply(False, reason or self.reason)

    def _reason_key(self, idx: int):
        self.reason = QC_REASONS[idx]
        self.radio.set_active(idx)
        self._remove(self.reason)

    def _set_reason(self, label):
        self.reason = label

    def _mark_reviewed(self):
        self.df.loc[self.df.index[0], "reviewed"] = "TRUE"
        _write_sheet(self.df, self.qc_path)
        print(f"  patch {self.patch_id} marked reviewed.")
        self._update_status()

    # ---- navigation ----------------------------------------------------
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
        self._install_one_shot_placement()
        print("Landsat interactive QC review — keys:")
        print("  k=keep  r=remove  1-4=remove w/ reason  <-/->=prev/next  "
              "v=mark reviewed  q=quit")
        if self._n_no_cube:
            print(f"  ({self._n_no_cube} scene(s) have no cached cube; shown as "
                  f"placeholders.)")
        plt.show(block=True)

        reviewed = str(self.df.loc[self.df.index[0], "reviewed"]).upper() == "TRUE"
        n_removed = int((self.df["keep"].astype(str).str.upper() == "FALSE").sum())
        print(f"\nDone. patch {self.patch_id}: {n_removed} removed, "
              f"reviewed={'YES' if reviewed else 'NO'}.")
        print(f"Verdicts saved to {self.qc_path}")
