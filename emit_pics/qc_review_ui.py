"""Shared helpers for the interactive QC reviewers (EMIT + Landsat).

Dependency-light utilities both desktop reviewers use: the crash-safe worksheet
write and the best-effort cross-backend (TkAgg / Qt) window placement. Imports
only pandas and the QC schema, so neither reviewer pulls the other's heavier
imports (the EMIT reviewer's streaming stack / the Landsat reviewer's rasterio).

Window placement is best-effort and backend-detected: any backend whose window
API is unavailable silently no-ops, so the reviewers still run — the windows
just open at the toolkit's default positions.
"""

import pandas as pd

from emit_pics.qc_gate import QC_SHEET_COLS


def write_sheet(df: pd.DataFrame, qc_path) -> None:
    """Persist the worksheet in the canonical column order (crash-safe write)."""
    df[QC_SHEET_COLS].to_csv(qc_path, index=False)


def screen_workarea(manager):
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


def set_window_geometry(manager, x, y, w, h):
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


def move_window(manager, x, y):
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
