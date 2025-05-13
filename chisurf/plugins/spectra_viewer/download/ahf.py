import os
import json
import requests
import argparse
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

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

def download_spectrum_file(file_url, source_page, product_info):
    filename = os.path.basename(file_url)
    file_basename, _ = os.path.splitext(filename)

    # Extract product name
    product_name = extract_product_name(file_url, product_info)

    # Create product directory structure
    product_dir = os.path.join(DOWNLOAD_DIR, product_name)
    spectra_dir = os.path.join(product_dir, "spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    # Set file path in the spectra directory
    filepath = os.path.join(spectra_dir, filename)

    if filename in downloaded_files:
        return

    print(f"Downloading: {file_url} for product {product_name}")
    try:
        r = requests.get(file_url, timeout=10)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        downloaded_files[filename] = file_url

        # Initialize product metadata if not exists
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

        # Update product metadata with spectrum info
        spectrum_type = get_spectrum_type(filename)
        if spectrum_type == "absorption_spectrum":
            products[product_name]["absorption_spectrum"] = filename
        elif spectrum_type == "emission_spectrum":
            products[product_name]["emission_spectrum"] = filename
        else:
            products[product_name]["other_spectra"].append(filename)

    except Exception as e:
        print(f"Failed to download {file_url}: {e}")

def extract_info(soup, page_url):
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(page_url, href)

        if should_skip_url(full_url):
            continue

        if is_valid_spectrum_url(full_url):
            container = link.find_parent()
            product_info = container.get_text(strip=True) if container else ""
            download_spectrum_file(full_url, page_url, product_info)

def crawl(url):
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
    extract_info(soup, url)

    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(url, href)
        if BASE_URL in full_url and not should_skip_url(full_url):
            crawl(full_url)

def save_all_metadata():
    """Save metadata.json files for all products"""
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
    args = parser.parse_args()

    global DOWNLOAD_DIR
    DOWNLOAD_DIR = args.output

    print(f"Downloading AHF filter spectra to {DOWNLOAD_DIR}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    crawl(BASE_URL)
    save_all_metadata()

    print(f"Downloaded {len(downloaded_files)} spectrum files for {len(products)} products")

if __name__ == "__main__":
    main()
