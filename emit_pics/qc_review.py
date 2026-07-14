"""Interactive keep/remove reviewer that writes QC verdicts straight to the CSV.

A single-user desktop tool (pure matplotlib — no server, no notebook, no extra
dependency) for entering the keep/remove verdicts the QC gate consumes.  It
renders the **same** multipanel QC figure the PDF uses
(:func:`emit_pics.display_qc_visuals.render_scene_figure`), but **from the cached
pixel cubes** (offline), and attaches Keep/Remove controls that update the
per-patch worksheet ``data/qc/{PICS}_{PATCH}_qc.csv`` live via the same
non-destructive schema :mod:`emit_pics.qc_gate` defines.

Two windows, auto-tiled to the screen on launch (viewer filling the top, controls
docked in a fixed strip along the bottom) so they open fitted and non-overlapping
and stay put — the viewer no longer snaps back to a fixed size on each new scene:

* **viewer** — the QC multipanel, redrawn in place per scene (one reused figure);
* **controls** — persistent Keep / Remove / Prev / Next / Reviewed / Quit buttons,
  a reason picker, and a status readout.

Placement is best-effort and backend-detected (TkAgg / Qt); on a backend whose
window geometry cannot be set it simply opens at the toolkit's default position.

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
from emit_pics.display_qc_visuals import NCOLS, render_scene_figure
from emit_pics.qc_gate import QC_REASONS, QC_SHEET_COLS, build_qc_sheet
from emit_pics.qc_report import default_qc_config, load_spectra_context


def _write_sheet(df: pd.DataFrame, qc_path) -> None:
    """Persist the worksheet in the canonical column order (crash-safe write)."""
    df[QC_SHEET_COLS].to_csv(qc_path, index=False)


# ---------------------------------------------------------------------------
# Cross-backend window placement (TkAgg / Qt).  Best-effort: any backend whose
# window API is unavailable silently no-ops, so the reviewer still runs — the
# windows just open at the toolkit's default positions.
# ---------------------------------------------------------------------------
def _screen_workarea(manager):
    """Screen ``(width, height)`` in pixels for a figure's window, or ``None``."""
    win = getattr(manager, "window", None)
    if win is None:
        return None
    if hasattr(win, "winfo_screenwidth"):                    # TkAgg
        try:
            return int(win.winfo_screenwidth()), int(win.winfo_screenheight())
        except Exception:
            return None
    try:                                                     # Qt (5/6)
        screen = win.screen() if hasattr(win, "screen") else None
        if screen is not None:
            ag = screen.availableGeometry()
            return int(ag.width()), int(ag.height())
    except Exception:
        pass
    return None


def _set_window_geometry(manager, x, y, w, h):
    """Position+size a figure's top-level window (TkAgg / Qt); no-op otherwise."""
    win = getattr(manager, "window", None)
    if win is None:
        return
    x, y, w, h = int(x), int(y), int(w), int(h)
    try:
        if hasattr(win, "wm_geometry"):                      # TkAgg
            win.wm_geometry(f"{w}x{h}+{x}+{y}")
        elif hasattr(win, "setGeometry"):                    # Qt
            win.setGeometry(x, y, w, h)
        elif hasattr(win, "move") and hasattr(win, "resize"):
            win.move(x, y)
            win.resize(w, h)
    except Exception:
        pass


def _move_window(manager, x, y):
    """Move a figure's window without changing its size (TkAgg / Qt); no-op else."""
    win = getattr(manager, "window", None)
    if win is None:
        return
    x, y = int(x), int(y)
    try:
        if hasattr(win, "wm_geometry"):                      # TkAgg: "+x+y" moves only
            win.wm_geometry(f"+{x}+{y}")
        elif hasattr(win, "move"):                           # Qt
            win.move(x, y)
    except Exception:
        pass


def _best_ncols(n_panels, avail_w, avail_h, base):
    """Column count that lets the panel grid fill the screen with the biggest panels.

    The map grid is ``ncols`` wide over ``ceil(n_panels/ncols)`` rows, plus one
    full-width spectra row; :func:`render_scene_figure` sizes it ``(4*ncols,
    4*rows + 4.5)`` inches.  Fitting that (fixed-aspect) block into the available
    screen area, the achievable scale is ``min(avail_w/w_in, avail_h/h_in)``.
    Because the layout is tall, height usually binds, so *more* columns (fewer
    rows) yields a larger scale — i.e. bigger panels.  Return the ``ncols >= base``
    that maximizes the scale.
    """
    best, best_scale = base, -1.0
    for c in range(base, max(base, n_panels) + 1):
        rows = -(-n_panels // c)                # ceil division
        w_in, h_in = 4 * c, 4 * rows + 4.5
        scale = min(avail_w / w_in, avail_h / h_in)
        if scale > best_scale:
            best, best_scale = c, scale
    return best


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
        self._ncols = None      # chosen at fit time to fill the screen (None = default)
        self._fitted = False    # once True, redraws stop re-snapping the viewer size

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

    # ---- window placement ---------------------------------------------
    def _infer_n_panels(self):
        """Total QC panel count (RGB + every MASK band + SZA + VZA + context).

        Read from the first available cube's MASK bands; ``None`` if no cube can
        be read (then column selection falls back to the render default).
        """
        for pid, ts in self.items:
            cp = self.patches[pid]["cfg"].cube_dir / f"cube_{self.pics_id}_{pid}_{ts}.nc"
            if not cp.exists():
                continue
            try:
                with read_cube_netcdf(cp) as ds:
                    labels = ds.attrs.get("mask_labels")
                    n_mask = (len(labels) if labels is not None
                              else int(ds.sizes.get("mask_band", 0)))
                return 1 + n_mask + 3      # RGB + masks + SZA + VZA + context
            except Exception:
                continue
        return None

    def _place_windows(self) -> bool:
        """Fit the viewer to the screen (proportionally) and dock the controls.

        Rather than stretch the viewer window to the screen's aspect ratio — which
        distorts the fixed-aspect panel grid and shrinks the map/mask panels — this
        (a) widens the column count so the tall grid uses the screen's width, then
        (b) scales the *whole* figure down by lowering its DPI so it fits without
        changing panel proportions.  The viewer is pinned top-left; the controls
        dock in a fixed strip just below it, so the two never overlap.  Returns
        ``True`` once placed (window realized), ``False`` to retry later.
        """
        vmgr = self.viewer_fig.canvas.manager
        area = _screen_workarea(vmgr)
        if area is None:
            return False
        sw, sh = area
        ctrl_h = min(300, max(200, int(sh * 0.22)))     # bottom strip for controls
        avail_h = max(200, sh - ctrl_h)

        # (a) Pick the column count that best fills the screen, and re-render at it.
        n_panels = self._infer_n_panels()
        if n_panels:
            best = _best_ncols(n_panels, sw, avail_h, NCOLS)
            if best != (self._ncols or NCOLS):
                self._ncols = best
                self._render()      # re-lay out at the new width (resize=True still)

        # (b) Shrink to fit by DPI (aspect preserved), never enlarge.
        w_in, h_in = self.viewer_fig.get_size_inches()
        dpi = self.viewer_fig.get_dpi()
        scale = min(sw / (w_in * dpi), avail_h / (h_in * dpi), 1.0)
        if scale < 1.0:
            self.viewer_fig.set_dpi(max(30.0, dpi * scale))
            self.viewer_fig.canvas.draw_idle()

        # Pin viewer top-left (keep its fitted size); dock controls just below it.
        vh = int(round(self.viewer_fig.get_size_inches()[1] * self.viewer_fig.get_dpi()))
        _move_window(vmgr, 0, 0)
        _set_window_geometry(self.ctrl_fig.canvas.manager,
                             0, min(vh, avail_h), min(sw, 1000), ctrl_h)
        self._fitted = True         # freeze size for subsequent scene redraws
        return True

    def _install_one_shot_placement(self) -> None:
        """Place the windows once, as soon as a backend window exists.

        Managers usually exist before ``plt.show`` (so we try immediately), but
        some backends realize the window only on first draw — hence the
        ``draw_event`` fallback, which retries until placement succeeds and then
        stops, so later per-scene redraws never re-snap a window the user nudged.
        """
        state = {"done": False}

        def _cb(_evt):
            if state["done"]:
                return
            if self._place_windows():
                state["done"] = True

        for f in (self.viewer_fig, self.ctrl_fig):
            f.canvas.mpl_connect("draw_event", _cb)
        self._place_windows()   # eager attempt (usually enough)

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
                        spectra=P["spectra"], acq_ts=ts, fig=self.viewer_fig,
                        ncols=(self._ncols or NCOLS), resize=not self._fitted)
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
        self._install_one_shot_placement()
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
