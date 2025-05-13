import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import json
import time
import argparse

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

def process_product_page(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')

    dye_name = url.split('/')[-1].replace('.html', '')
    base_dir = os.path.join(BASE_DIR, dye_name)
    spectra_dir = os.path.join(base_dir, "spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    metadata = {
        "name": dye_name,
        "description": "",
        "optical_properties": {},
        "structure_images": [],
        "absorption_spectrum": "",
        "emission_spectrum": ""
    }

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith(".txt") and ("abs" in href.lower() or "ems" in href.lower()):
            full_url = urllib.parse.urljoin(url, href)
            filename = full_url.split('/')[-1]
            filepath = os.path.join(spectra_dir, filename)
            download_file(full_url, filepath)
            if "abs" in filename.lower():
                metadata["absorption_spectrum"] = filename
            elif "ems" in filename.lower():
                metadata["emission_spectrum"] = filename

    metadata["optical_properties"] = parse_optical_properties(soup)
    metadata["description"] = get_description(soup)

    image_urls = get_image_links(soup)
    for i, image_url in enumerate(image_urls):
        image_filename = os.path.basename(image_url)
        image_path = os.path.join(base_dir, image_filename)
        download_file(image_url, image_path)
        metadata["structure_images"].append(image_filename)

    with open(os.path.join(base_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata for {dye_name}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download ATTO dye spectra and information")
    parser.add_argument("-o", "--output", help="Output directory for downloaded files", default="atto_dyes")
    args = parser.parse_args()

    # Set the global BASE_DIR
    global BASE_DIR
    BASE_DIR = args.output

    print(f"Downloading ATTO dye information to {BASE_DIR}")
    os.makedirs(BASE_DIR, exist_ok=True)

    # Get product links and process them
    product_links = get_all_links(START_URL)
    print(f"Found {len(product_links)} product pages.")
    for url in product_links:
        print(f"Processing {url}")
        process_product_page(url)
        time.sleep(1)

    print(f"Download complete. Data saved to {BASE_DIR}")

if __name__ == "__main__":
    main()
