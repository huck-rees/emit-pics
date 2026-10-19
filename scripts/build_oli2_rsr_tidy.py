"""Rebuild data/landsat/oli2_rsr/L9_OLI2_RSR_tidy.csv from the USGS workbook.

Reads the seven reflective-band sheets of ``L9_OLI2_RSR.xlsx`` (band-average
RSR, normalized to peak = 1) and writes the long-format table that the EMIT/OLI-2
band-synthesis analysis consumes. Cirrus and Pan are omitted. Requires openpyxl.
Source and provenance: see the README in the same directory.
"""
import warnings

import pandas as pd

from emit_pics.paths import REPO_ROOT

RSR_DIR = REPO_ROOT / "data" / "landsat" / "oli2_rsr"
XLSX = RSR_DIR / "L9_OLI2_RSR.xlsx"
TIDY = RSR_DIR / "L9_OLI2_RSR_tidy.csv"

# workbook sheet -> (OLI-2 asset name, band number, nominal center nm)
BANDS = [
    ("CoastalAerosol", "coastal", 1, 443),
    ("Blue", "blue", 2, 482),
    ("Green", "green", 3, 561),
    ("Red", "red", 4, 654),
    ("NIR", "nir08", 5, 865),
    ("SWIR1", "swir16", 6, 1609),
    ("SWIR2", "swir22", 7, 2201),
]


def build() -> pd.DataFrame:
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl "unknown extension" notice
        xl = pd.ExcelFile(XLSX)
        for sheet, band, band_n, center in BANDS:
            d = xl.parse(sheet)
            frames.append(pd.DataFrame({
                "band": band,
                "band_n": band_n,
                "center_nm": center,
                "wavelength_nm": d["Wavelength"].astype(float),
                "rsr": d["BA RSR [watts]"],
                "rsr_std": d["stdev(RSR)"],
            }))
    return pd.concat(frames, ignore_index=True)


def main():
    df = build()
    df.to_csv(TIDY, index=False)
    print(f"wrote {TIDY} ({len(df)} rows, {df['band'].nunique()} bands)")


if __name__ == "__main__":
    main()
