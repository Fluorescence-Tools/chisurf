import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import json
import time
import argparse
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to import the database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SpectraDatabase

BASE_URL = "https://www.atto-tec.com"
START_URL = f"{BASE_URL}/produkte/Fluorescent-Labels/"
BASE_DIR = "atto_dyes"  # Default output directory
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_all_links(start_url):
    visited = set()
    to_visit = [start_url]
    product_pages = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                href = urllib.parse.urljoin(BASE_URL, link['href'])
                if href.startswith(START_URL) and href not in visited:
                    to_visit.append(href)
                if "/ATTO-" in href and href.endswith(".html"):
                    product_pages.append(href)
        except Exception as e:
            print(f"Error visiting {url}: {e}")

    return list(set(product_pages))

def download_file(url, path):
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {path}")
    else:
        print(f"Failed to download: {url}")

def parse_optical_properties(soup):
    properties = {}
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                properties[key] = value
    return properties

def get_description(soup):
    desc = ""
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text(strip=True)
        if "ATTO" in text:
            desc += text + " "
    return desc.strip()

def get_image_links(soup):
    image_urls = []
    for meta in soup.find_all('meta', property="og:image"):
        img_url = meta.get("content")
        if img_url:
            image_urls.append(img_url)
    return image_urls

def download_spectrum(url):
    """Download a spectrum file and return its data as numpy arrays."""
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to download spectrum: {url}")
        return None, None

    # Parse the text file
    lines = response.text.strip().split('\n')
    if len(lines) < 3:
        print(f"Invalid spectrum file format: {url}")
        return None, None

    # Skip the first two lines (header)
    data_lines = lines[2:]
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

def process_product_page(url, db):
    """Process a product page and store data in the database."""
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')

    dye_name = url.split('/')[-1].replace('.html', '')

    # Get the item type ID (create if it doesn't exist)
    type_id = db.add_item_type(BASE_DIR, "Atto Dyes")

    # Get the description and optical properties
    description = get_description(soup)
    optical_properties = parse_optical_properties(soup)

    # Add the item to the database
    item_id = db.add_item(dye_name, type_id, description)
    print(f"Added item: {dye_name}")

    # Add optical properties
    for prop_name, prop_value in optical_properties.items():
        db.add_optical_property(item_id, prop_name, prop_value)

    # Process spectra
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith(".txt") and ("abs" in href.lower() or "ems" in href.lower()):
            full_url = urllib.parse.urljoin(url, href)
            spectrum_type = "absorption" if "abs" in href.lower() else "emission"

            # Download and parse the spectrum
            wavelengths, values = download_spectrum(full_url)
            if wavelengths is not None and values is not None:
                # Add the spectrum to the database
                db.add_spectrum(item_id, spectrum_type, wavelengths, values)
                print(f"Added {spectrum_type} spectrum for {dye_name}")

    # Download and add structure images
    image_urls = get_image_links(soup)

    for image_url in image_urls:
        # Download the image directly to memory
        response = requests.get(image_url, headers=HEADERS)
        if response.status_code == 200:
            image_data = response.content
            image_filename = os.path.basename(image_url)

            # Add the image directly to the database
            db.add_image(item_id, image_filename, image_data)
            print(f"Downloaded and added image: {image_filename}")
        else:
            print(f"Failed to download image: {image_url}")

    print(f"Processed {dye_name}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download ATTO dye spectra and information")
    parser.add_argument("-o", "--output", help="Output directory for downloaded files", default="atto_dyes")
    args = parser.parse_args()

    # Set the global BASE_DIR
    global BASE_DIR
    BASE_DIR = args.output

    print(f"Downloading ATTO dye information")

    # Initialize the database
    db = SpectraDatabase()
    with db:
        db.create_tables()

        # Get product links and process them
        product_links = get_all_links(START_URL)
        print(f"Found {len(product_links)} product pages.")
        for url in product_links:
            print(f"Processing {url}")
            process_product_page(url, db)
            time.sleep(1)

    print("Download complete. Data saved to database.")

if __name__ == "__main__":
    main()
