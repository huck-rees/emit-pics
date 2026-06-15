"""Load and validate site configuration from YAML."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sites.yaml"


def load_sites(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load all PICS site definitions from a YAML config file.

    Returns a dict keyed by site id (e.g. 'libya_4') with fields:
        name, center_lat, center_lon, roi_size_km,
        and optional: elevation_m, utm_zone, cnes_bounds.

    Sites with null coordinates are included — validation happens in
    get_site() when you actually request one.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sites", {})


def get_site(site_id: str, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate a single site by id.

    Returns the full site dict including any optional fields
    (elevation_m, utm_zone, cnes_bounds).

    Raises KeyError if the site doesn't exist, ValueError if its
    coordinates are still null.
    """
    sites = load_sites(config_path)
    if site_id not in sites:
        raise KeyError(
            f"Unknown site '{site_id}'. Available: {', '.join(sites.keys())}"
        )
    info = sites[site_id]
    if info.get("center_lat") is None or info.get("center_lon") is None:
        raise ValueError(
            f"Site '{site_id}' has null coordinates. "
            "Fill in center_lat/center_lon in config/sites.yaml first."
        )
    return info
