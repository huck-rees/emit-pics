"""Load and validate site configuration from YAML."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sites.yaml"


def load_sites(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load PICS site definitions from a YAML config file.

    Returns a dict keyed by site id (e.g. 'libya_4') with fields:
        name, center_lat, center_lon, roi_size_km.

    Raises ValueError if any site has null coordinates (reminds user to
    fill in the TODOs).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sites = cfg.get("sites", {})
    for site_id, info in sites.items():
        if info.get("center_lat") is None or info.get("center_lon") is None:
            raise ValueError(
                f"Site '{site_id}' has null coordinates. "
                "Fill in center_lat/center_lon in config/sites.yaml first."
            )
    return sites
