"""Shared, deterministic fixtures for the test suite.

Pure builders only — no file/network I/O — so every test is fast and
reproducible. Used by the unit tests and the golden regression test.
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


def build_golden_delta_df():
    """A fixed synthetic long delta table with a KNOWN injected drift.

    Two PICS, five acquisition pairs each (Δt = 100..300 days), three channels:
      * band 0 @ 500 nm  — radiance drift +2.0 %/yr (reflectance +1.0 %/yr)
      * band 1 @ 650 nm  — radiance drift -1.0 %/yr (reflectance -0.5 %/yr)
      * band 2 @ 1400 nm — water-vapor absorption band (excluded from the fit)

    Because ΔR is exactly ``slope · Δt / 365.25``, a correct Theil–Sen fit must
    recover ``slope_pct_yr`` = the injected slope to machine precision. This is
    both a functional check and the input to the golden-file regression test.
    """
    dts = [100.0, 150.0, 200.0, 250.0, 300.0]
    bands = [(0, 500.0, 2.0, False), (1, 650.0, -1.0, False),
             (2, 1400.0, 0.0, True)]
    rows = []
    for pics in ["Libya_4", "Algeria_3"]:
        for k, d in enumerate(dts):
            for bidx, wl, slope, is_abs in bands:
                rows.append({
                    "PICS_ID": pics,
                    "pair_id": f"{pics}_{k}",
                    "band_index": bidx,
                    "wavelength": wl,
                    "dt_days": d,
                    "delta_rad_pct_median": slope * d / 365.25,
                    "delta_rfl_pct_median": 0.5 * slope * d / 365.25,
                    "absorption_band_flag": is_abs,
                })
    return pd.DataFrame(rows)
