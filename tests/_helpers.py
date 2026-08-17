"""Shared, deterministic fixtures for the test suite.

Pure builders only — no file/network I/O — so every test is fast and
reproducible.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def make_spec(wls, med_rfl, mean_rfl, med_rad, mean_rad):
    """Build a per-scene spectrum frame in the shape ``load_scene_spectrum`` returns.

    Columns: ``central_wl, median_rfl, mean_rfl, median_rad, mean_rad``.
    """
    return pd.DataFrame({
        "central_wl": np.asarray(wls, dtype=float),
        "median_rfl": np.asarray(med_rfl, dtype=float),
        "mean_rfl": np.asarray(mean_rfl, dtype=float),
        "median_rad": np.asarray(med_rad, dtype=float),
        "mean_rad": np.asarray(mean_rad, dtype=float),
    })


def make_meta(**overrides):
    """Scalar per-pair metadata dict accepted by ``compute_pair_delta``."""
    meta = {
        "PICS_ID": "Libya_4", "patch_id": "10_7_1", "pair_id": "P",
        "t1_scene_id": "S1", "t2_scene_id": "S2",
        "t1_dt": datetime(2023, 1, 1), "t2_dt": datetime(2023, 2, 1),
        "dt_days": 31.0,
        "SZA_t1": 30.0, "SZA_t2": 31.0, "delta_SZA": 1.0, "delta_VZA": 0.5,
    }
    meta.update(overrides)
    return meta
