#!/usr/bin/env python3
"""Download 3DOptix optical filter catalog metadata.

Crawls the public 3DOptix catalog pages for filters, extracts physical
properties (Brand, Subtype, Material, Cut-On/Cut-Off wavelengths, shape, thickness, etc.),
and stores them in the MFDB reference database (spectra.db).

Usage:
    python -m chisurf.plugins.spectra_downloader.download.threed_optix --db <path> [--max-pages <N>] [--brand <brand_name>]
"""

import argparse
import logging
import re
import time
import urllib.parse
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.3doptix.com"
CATALOG_FILTER_URL = f"{BASE_URL}/catalog/optics/filter/"

# 3DOptix catalog subtype label → canonical component kind (resolved via
# COMPONENT_KINDS in mfdb_adapter to its category + default spectrum type).
SUBTYPE_KIND_MAP: dict[str, str] = {
    "Bandpass Filter": "bandpass",
    "Longpass Filter": "longpass",
    "Shortpass Filter": "shortpass",
    "Notch Filter": "notch",
    "Dichroic Filter": "dichroic",
    "Neutral Density Filter": "nd",
    "Laser-line Filter": "laserline",
    "Color Glass": "colorglass",
    "Hot/Cold Mirror": "hot_cold_mirror",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def parse_wl(val: str | None) -> float | None:
    """Extract wavelength float value from text (e.g. '550 nm' -> 550.0)."""
    if not val:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", val)
    return float(m.group(1)) if m else None

def scrape_product_detail(url: str) -> Dict[str, Any]:
    """Fetch and parse detailed page for a specific catalog item."""
    res = {}
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return res
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. External product webpage URL
        info_card = soup.find(class_=re.compile("Product-module__info"))
        if info_card:
            for div in info_card.find_all("div"):
                dt = div.find("dt")
                dd = div.find("dd")
                if dt and dd and dt.get_text(strip=True) == "Product webpage":
                    a = dd.find("a", href=True)
                    if a:
                        res["external_url"] = a["href"]

        # 2. Detailed properties & material parameters
        properties = {}
        material = {}
        for section in soup.find_all(class_=re.compile("Product-module__parameters")):
            header = section.find("h2")
            if header:
                header_text = header.get_text(strip=True).lower()
                target_dict = None
                if "properties" in header_text:
                    target_dict = properties
                elif "material" in header_text:
                    target_dict = material
                
                if target_dict is not None:
                    for div in section.find_all("div"):
                        dt = div.find("dt")
                        dd = div.find("dd")
                        if dt and dd:
                            target_dict[dt.get_text(strip=True)] = dd.get_text(strip=True)
        res["properties"] = properties
        res["material"] = material

        # 3. Description text & coating
        desc_el = soup.find(class_=re.compile("Product-module__description"))
        description = ""
        if desc_el:
            description = desc_el.get_text(" ", strip=True)
        res["description"] = description

        # Extract coating using regex
        coating = None
        coating_match = re.search(r"coated with (?:'|\")?([a-zA-Z0-9\-\_\.\s]+?)(?:'|\"|,|\.|\sand|\sfor|\sa\s|\saoi)", description)
        if not coating_match:
            coating_match = re.search(r"coating[^.]+?is (?:'|\")?([a-zA-Z0-9\-\_\.\s]+?)(?:'|\"|,|\.|\sand|\sfor|\sa\s)", description)
        if not coating_match:
            coating_match = re.search(r"identified as (?:'|\")?([a-zA-Z0-9\-\_\.\s]+?)(?:'|\"|,|\.|\sand|\sfor|\sa\s)", description)
        
        if coating_match:
            coating = coating_match.group(1).strip().strip("'").strip('"')
        res["coating"] = coating

    except Exception as e:
        logger.error(f"Error scraping detail page {url}: {e}")
    
    return res

def download_threed_optix_to_db(
    db: FluorophoreDatabase,
    max_pages: int = 0,
    brand_filter: str | None = None
) -> Dict[str, int]:
    """Scrape 3DOptix catalog and save to database.

    Parameters
    ----------
    db : FluorophoreDatabase
        Database adapter handle.
    max_pages : int, default=0
        Maximum number of index pages to scrape. 0 means scrape all.
    brand_filter : str, optional
        If specified, only scrape items matching this brand (case insensitive).
    """
    counts: Dict[str, int] = {}
    page = 0
    consecutive_empty_pages = 0

    print(f"Starting 3DOptix filter scraper (brand_filter={brand_filter or 'All'}, max_pages={max_pages or 'All'})")

    while True:
        if max_pages > 0 and page >= max_pages:
            break

        print(f"Fetching catalog index page {page + 1} ...")
        url = f"{CATALOG_FILTER_URL}?p={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"Error: Non-200 status code {resp.status_code} for page {page}")
                break
            
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                print(f"No table found on page {page}. Index page might be empty.")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 2:
                    break
                page += 1
                continue
            
            consecutive_empty_pages = 0
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]
            
            if not rows:
                print(f"No rows found on page {page}.")
                break
            
            print(f"Found {len(rows)} items on page {page + 1}.")
            
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                
                # Extract index page columns
                a_tag = cols[0].find("a", href=True)
                if not a_tag:
                    continue
                
                full_name = a_tag.get_text(strip=True)
                detail_path = a_tag["href"]
                detail_url = urllib.parse.urljoin(BASE_URL, detail_path)
                
                item_brand = cols[4].get_text(strip=True)
                
                # Apply brand filter if specified
                if brand_filter and item_brand.lower() != brand_filter.lower():
                    continue

                item_subtype = cols[3].get_text(strip=True)
                item_material = cols[5].get_text(strip=True) if len(cols) > 5 else None
                item_shape = cols[6].get_text(strip=True) if len(cols) > 6 else None

                part_number = full_name.split()[0].strip()

                print(f"Scraping {part_number} ({item_brand} - {item_subtype}) ...")
                
                # Fetch details from product detail page
                details = scrape_product_detail(detail_url)
                time.sleep(0.5)  # Respectful delay
                
                # Skip if detail fetch failed
                if not details:
                    continue

                # Merge properties
                item_properties = details.get("properties", {})
                if item_shape and "Shape" not in item_properties:
                    item_properties["Shape"] = item_shape
                
                # Clean brand name for the probe-type id (lowercase, alnum + _).
                brand_clean = re.sub(r'[^a-z0-9_]', '_', item_brand.lower().strip())
                kind = SUBTYPE_KIND_MAP.get(item_subtype, "other")

                # Collect every scraped property into one dict; register_component
                # canonicalizes keys (Cut-On → cut_on, Cut-Off → cut_off, …).
                props: Dict[str, Any] = {"Brand": item_brand, "Subtype": item_subtype}
                props.update(item_properties)
                for mat_k, mat_v in details.get("material", {}).items():
                    props[f"Material {mat_k}"] = mat_v
                coating = details.get("coating")
                if coating:
                    props["Coating"] = coating
                ext_url = details.get("external_url")
                if ext_url:
                    props["External Link"] = ext_url

                with db:
                    db.register_component(
                        name=part_number,
                        source="3doptix",
                        kind=kind,
                        source_ref=ext_url or part_number,
                        type_name=f"{brand_clean}_{kind}",
                        type_display=f"{item_brand} {item_subtype}",
                        description=f"{item_brand} {item_subtype} – {part_number}",
                        properties=props,
                    )

                counts[kind] = counts.get(kind, 0) + 1

        except Exception as e:
            logger.error(f"Error processing page {page}: {e}")
            break
        
        page += 1

    print("\nImport summary:")
    for tname, cnt in sorted(counts.items()):
        print(f"  {tname:<35s} {cnt:3d} items")
    print(f"\nTotal: {sum(counts.values())} items imported.")
    return counts

def main() -> None:
    """CLI entry point for threed_optix.py script."""
    parser = argparse.ArgumentParser(
        description="Download 3DOptix optical filter catalog metadata into MFDB",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DATABASE_PATH),
        help="MFDB SQLite database path (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum catalog pages to scrape (default: 10, set to 0 for unlimited)",
    )
    parser.add_argument(
        "--brand",
        default=None,
        help="Filter items by brand name (e.g. optosigma, thorlabs)",
    )
    args = parser.parse_args()

    db = FluorophoreDatabase(args.db)
    with db:
        download_threed_optix_to_db(db, max_pages=args.max_pages, brand_filter=args.brand)
    print("3DOptix catalog scrape complete.")

if __name__ == "__main__":
    main()
