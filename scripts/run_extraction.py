#!/usr/bin/env python
"""Search and download EMIT granules for a PICS site.

Usage:
    python scripts/run_extraction.py --site libya_4 \
        --start 2023-01-01 --end 2023-12-31

Requires:
    - NASA Earthdata credentials in .netrc/_netrc or interactive login.
    - Site coordinates filled in config/sites.yaml.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'emit_pics' is importable
# when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from emit_pics.config import get_site
from emit_pics.extract import (
    authenticate,
    download_granules,
    search_all_products,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search and download EMIT granules for a PICS site."
    )
    parser.add_argument(
        "--site",
        required=True,
        help="Site key from config/sites.yaml, e.g. 'libya_4'.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (ISO-8601), e.g. '2023-01-01'.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (ISO-8601), e.g. '2023-12-31'.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directory to store downloaded granules (default: data/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search only; don't download. Writes the manifest of search results.",
    )
    args = parser.parse_args()

    # Load the requested site (only validates that one site's coordinates).
    try:
        site = get_site(args.site)
    except (KeyError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    lat = site["center_lat"]
    lon = site["center_lon"]
    data_dir = Path(args.data_dir)

    # Authenticate to Earthdata.
    authenticate()

    # Search all three EMIT products.
    logger.info(
        "Searching for EMIT granules at %s (%.4f, %.4f) from %s to %s",
        site["name"], lat, lon, args.start, args.end,
    )
    all_results = search_all_products(lat, lon, args.start, args.end)

    total = sum(len(v) for v in all_results.values())
    logger.info("Total granules found across all products: %d", total)

    if total == 0:
        logger.warning("No granules found. Check coordinates and date range.")
        sys.exit(0)

    # Write manifest (even for dry runs — useful for inspecting search results).
    manifest_path = write_manifest(all_results, args.site, data_dir)
    logger.info("Manifest written to %s", manifest_path)

    if args.dry_run:
        logger.info("Dry run — skipping download.")
        return

    # Download each product.
    for label, granules in all_results.items():
        if granules:
            product_dir = data_dir / label.lower()
            logger.info("Downloading %d %s granules...", len(granules), label)
            download_granules(granules, product_dir)

    logger.info("Done. Granules saved to %s", data_dir)


if __name__ == "__main__":
    main()
