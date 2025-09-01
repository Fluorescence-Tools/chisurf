import os
import json
import requests
import argparse
import re
import numpy as np
import sys
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

# Add the parent directory to the path to import the database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SpectraDatabase

BASE_URL = "https://ahf.de/produkte/spektralanalytik-photonik/optische-filter/"
DOWNLOAD_DIR = "ahf_filters"

visited_pages = set()
downloaded_files = {}
products = {}

SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".zip", ".svg", ".mp4", ".mp3")

def is_valid_spectrum_url(url):
    return url.lower().endswith(".txt") and "/spectrum/raw/" in url.lower()

def should_skip_url(url):
    return urlparse(url).path.lower().endswith(SKIP_EXTENSIONS)

def extract_product_name(url, product_info):
    # Try to extract product name from URL or product info
    # Example: Extract "F39-xxx" or similar pattern
    pattern = r'[A-Z]\d{2,3}[-_][A-Za-z0-9]+'

    # First try to find in URL
    url_match = re.search(pattern, url)
    if url_match:
        return url_match.group(0)

    # Then try to find in product info
    info_match = re.search(pattern, product_info)
    if info_match:
        return info_match.group(0)

    # If no match found, use a part of the filename as fallback
    filename = os.path.basename(url)
    file_basename, _ = os.path.splitext(filename)
    return file_basename

def get_spectrum_type(filename):
    # Determine if the spectrum is absorption, emission, or other
    lower_filename = filename.lower()
    if "abs" in lower_filename or "transmission" in lower_filename:
        return "absorption_spectrum"
    elif "ems" in lower_filename or "emission" in lower_filename:
        return "emission_spectrum"
    else:
        return "other_spectrum"

def parse_spectrum_data(content):
    """Parse spectrum data from raw content."""
    try:
        # Convert bytes to string
        text = content.decode('utf-8')

        # Split into lines
        lines = text.strip().split('\n')

        # Skip header lines (usually 2)
        data_lines = lines[2:] if len(lines) > 2 else lines

        wavelengths = []
        values = []

        for line in data_lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    wavelength = float(parts[0])
                    value = float(parts[1])
                    wavelengths.append(wavelength)
                    values.append(value)
                except ValueError:
                    continue

        return np.array(wavelengths), np.array(values)
    except Exception as e:
        print(f"Error parsing spectrum data: {e}")
        return None, None

def download_spectrum_file(file_url, source_page, product_info, db):
    filename = os.path.basename(file_url)
    file_basename, _ = os.path.splitext(filename)

    # Extract product name
    product_name = extract_product_name(file_url, product_info)

    if filename in downloaded_files:
        return

    print(f"Downloading: {file_url} for product {product_name}")
    try:
        r = requests.get(file_url, timeout=10)
        r.raise_for_status()

        # Parse the spectrum data
        wavelengths, values = parse_spectrum_data(r.content)
        if wavelengths is None or values is None:
            print(f"Failed to parse spectrum data from {file_url}")
            return

        downloaded_files[filename] = file_url

        # Get the item type ID (create if it doesn't exist)
        type_id = db.add_item_type(DOWNLOAD_DIR, "AHF Filters")

        # Get or create the item
        item_data = db.get_item_by_name_and_type(product_name, type_id)
        if item_data:
            item_id = item_data[0]
        else:
            # Add the item to the database
            item_id = db.add_item(product_name, type_id, product_info)
            print(f"Added item: {product_name}")

            # Add source page as an optical property
            db.add_optical_property(item_id, "source_page", source_page)

        # Determine spectrum type and add to database
        spectrum_type_str = get_spectrum_type(filename)
        if spectrum_type_str == "absorption_spectrum":
            db_spectrum_type = "absorption"
        elif spectrum_type_str == "emission_spectrum":
            db_spectrum_type = "emission"
        else:
            db_spectrum_type = "transmission"  # Default for filters

        # Add the spectrum to the database
        db.add_spectrum(item_id, db_spectrum_type, wavelengths, values)
        print(f"Added {db_spectrum_type} spectrum for {product_name}")

        # Initialize product metadata if not exists (for backward compatibility)
        if product_name not in products:
            products[product_name] = {
                "name": product_name,
                "description": product_info,
                "source_page": source_page,
                "optical_properties": {},
                "structure_images": [],
                "absorption_spectrum": "",
                "emission_spectrum": "",
                "other_spectra": []
            }

    except Exception as e:
        print(f"Failed to download {file_url}: {e}")

def extract_info(soup, page_url, db):
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(page_url, href)

        if should_skip_url(full_url):
            continue

        if is_valid_spectrum_url(full_url):
            container = link.find_parent()
            product_info = container.get_text(strip=True) if container else ""
            download_spectrum_file(full_url, page_url, product_info, db)

def crawl(url, db):
    if url in visited_pages:
        return
    visited_pages.add(url)

    print(f"Crawling: {url}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    extract_info(soup, url, db)

    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(url, href)
        if BASE_URL in full_url and not should_skip_url(full_url):
            crawl(full_url, db)

def save_all_metadata():
    """Save metadata.json files for all products (for backward compatibility)"""
    if not products:
        return

    print(f"Saving metadata for {len(products)} products")
    for product_name, metadata in products.items():
        product_dir = os.path.join(DOWNLOAD_DIR, product_name)
        os.makedirs(product_dir, exist_ok=True)

        metadata_path = os.path.join(product_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata for {product_name}")

def main():
    """Main function with command-line argument handling"""
    parser = argparse.ArgumentParser(description="Download AHF filter spectra")
    parser.add_argument("-o", "--output", help="Output directory for downloaded files", default="ahf_filters")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata files (for backward compatibility)")
    args = parser.parse_args()

    global DOWNLOAD_DIR
    DOWNLOAD_DIR = args.output

    print(f"Downloading AHF filter spectra")

    # Initialize the database
    db = SpectraDatabase()
    with db:
        # Create tables if they don't exist
        db.create_tables()

        # Crawl the website and store data in the database
        crawl(BASE_URL, db)

    # For backward compatibility, save metadata files if requested
    if args.save_metadata:
        save_all_metadata()

    print(f"Downloaded {len(downloaded_files)} spectrum files for {len(products)} products")
    print("Data has been stored in the database.")

if __name__ == "__main__":
    main()
