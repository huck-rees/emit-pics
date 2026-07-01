#!/usr/bin/env python
"""Regenerate the PICS GeoPackage from ``config/sites.yaml`` (optional, for GIS).

``config/sites.yaml`` is the single source of truth for PICS geometry; the
pipeline never needs the GeoPackage. This helper materializes one anyway —
handy for eyeballing the site boxes in QGIS. The output is a *derived* file
(gitignored under ``data/``), not an input.

    python scripts/run_export_pics_gpkg.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

OUT_GPKG = _REPO_ROOT / "data" / "ROIs" / "northafrican_pics.gpkg"

from emit_pics.config import write_pics_gpkg


def main() -> None:
    path = write_pics_gpkg(OUT_GPKG)
    print(f"Wrote PICS GeoPackage from sites.yaml -> {path}")


if __name__ == "__main__":
    main()
