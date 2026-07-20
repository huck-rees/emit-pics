#!/usr/bin/env python
"""Regenerate the PICS GeoPackage from ``config/sites.yaml`` (optional, for GIS).

``config/sites.yaml`` is the single source of truth for PICS geometry; the
pipeline never needs the GeoPackage. This helper materializes one anyway —
handy for eyeballing the site boxes in QGIS. The output is a *derived* file
(gitignored under ``data/``), not an input.

    python scripts/run_export_pics_gpkg.py
"""

from emit_pics import paths
from emit_pics.config import write_pics_gpkg

OUT_GPKG = paths.pics_gpkg()


def main() -> None:
    path = write_pics_gpkg(OUT_GPKG)
    print(f"Wrote PICS GeoPackage from sites.yaml -> {path}")


if __name__ == "__main__":
    main()
