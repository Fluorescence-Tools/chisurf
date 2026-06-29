#!/usr/bin/env python3
"""Download ATTO-TEC dye spectra from the Internet Archive (Wayback Machine).

ATTO-TEC was acquired by Leica and its live site no longer publishes the per-dye
spectra. The last good ``atto-tec.com`` product pages — optical properties +
absorption/emission ``.txt`` spectra — are archived on the Wayback Machine, so
this scraper sources them from there: the CDX API lists the archived
``/ATTO-*.html`` pages and each page (and its spectra files) is fetched raw
(``…id_/…``). Records are written through the canonical ``register_component``
contract (source ``atto``, kind ``organic_dye``).
"""

import json
import logging
import re
import urllib.parse

import numpy as np
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (ChiSurf spectra downloader)"}
CDX_URL = "http://web.archive.org/cdx/search/cdx"
_WAYBACK_RE = re.compile(r"^https?://web\.archive\.org/web/(\d+)(?:id_)?/(.*)$")


def archived_atto_pages() -> dict[str, str]:
    """Return ``{original_url: timestamp}`` for archived ATTO product pages."""
    resp = requests.get(
        CDX_URL,
        params={
            "url": "atto-tec.com/ATTO-*",
            "output": "json",
            "filter": "statuscode:200",
            "collapse": "urlkey",
            "fl": "original,timestamp",
        },
        headers=HEADERS,
        timeout=60,
    )
    rows = resp.json()[1:]  # drop header row
    pages: dict[str, str] = {}
    for original, timestamp in rows:
        if re.search(r"/ATTO-\w+\.html$", original, re.IGNORECASE):
            pages[original] = timestamp
    return pages


def _raw_wayback(ts: str, url: str) -> str:
    """Return the raw (toolbar-free) Wayback URL for ``url`` at ``ts``."""
    return f"https://web.archive.org/web/{ts}id_/{url}"


def _to_raw(url: str) -> str:
    """Rewrite a toolbar Wayback URL to its raw ``…id_/…`` form."""
    m = _WAYBACK_RE.match(url)
    return _raw_wayback(m.group(1), m.group(2)) if m else url


def _clean_property(key: str, raw: str) -> str:
    """Normalise an ATTO property value (German decimals, εmax exponent, QY %)."""
    value = (raw or "").strip()
    if not value:
        return value
    # εmax like "9,0×104" → 9.0 × 10^4
    if "ε" in key or "epsilon" in key.lower():
        m = re.search(r"([\d.,]+)\s*[×x]\s*10\^?(\d+)", value)
        if m:
            base = float(m.group(1).replace(",", "."))
            return str(base * (10 ** int(m.group(2))))
    value = value.replace(",", ".")
    # ηfl is the quantum yield in percent → fraction
    if key.strip().startswith("η") or "ηfl" in key:
        try:
            return str(float(value) / 100.0)
        except ValueError:
            return value
    return value


def parse_optical_properties(soup: BeautifulSoup) -> dict[str, str]:
    """Parse the ATTO optical-property table.

    Keys (λabs / λfl / ηfl / εmax / τfl …) are returned verbatim so
    ``register_component`` can canonicalise them (λabs→abs_max, λfl→em_max,
    ηfl→qy, εmax→ext_coeff, τfl→lifetime).
    """
    props: dict[str, str] = {}
    table = soup.find("table")
    if not table:
        return props
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 2:
            key = cols[0].get_text(strip=True)
            if key:
                props[key] = _clean_property(key, cols[1].get_text(strip=True))
    return props


def _download_spectrum(url: str):
    """Fetch an archived ATTO spectrum ``.txt`` → (wavelengths, intensities)."""
    try:
        resp = requests.get(_to_raw(url), headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
    except Exception:
        return None
    wl, vals = [], []
    for line in resp.text.strip().splitlines()[2:]:  # skip 2-line header
        parts = line.strip().replace(",", ".").split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) >= 2:
            try:
                wl.append(float(parts[0]))
                vals.append(float(parts[1]))
            except ValueError:
                continue
    if len(wl) < 3:
        return None
    return np.array(wl), np.array(vals)


def _description(soup: BeautifulSoup) -> str:
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if "ATTO" in text:
            return text
    return ""


def process_archived_page(db, page_url: str, ts: str) -> bool:
    """Fetch one archived ATTO page and register the dye. Returns success."""
    try:
        resp = requests.get(_raw_wayback(ts, page_url), headers=HEADERS, timeout=30)
    except Exception as e:
        logger.error("ATTO page %s: %s", page_url, e)
        return False
    soup = BeautifulSoup(resp.text, "html.parser")

    name = page_url.rstrip("/").split("/")[-1].replace(".html", "")
    name = name.replace("-", " ").upper().replace("ATTO ", "ATTO ")

    spectra: dict[str, tuple] = {}
    for link in soup.find_all("a", href=True):
        href = link["href"]
        low = href.lower()
        if low.endswith(".txt") and ("abs" in low or "em" in low):
            stype = "absorption" if "abs" in low else "emission"
            # On an id_ page the links are the ORIGINAL (now-dead) atto-tec.com
            # URLs — fetch each one's archived copy from the same snapshot.
            if not href.startswith("http"):
                href = urllib.parse.urljoin(page_url, href)
            spec = _download_spectrum(_raw_wayback(ts, href))
            if spec is not None:
                spectra[stype] = spec

    db.register_component(
        name=name,
        source="atto",
        kind="organic_dye",
        source_ref=page_url,
        description=_description(soup),
        properties=parse_optical_properties(soup),
        spectra=spectra,
    )
    return True


def download_atto_from_wayback(db, limit: int = 0) -> int:
    """Download all archived ATTO dyes from the Wayback Machine."""
    pages = archived_atto_pages()
    print(f"Found {len(pages)} archived ATTO product pages.")
    count = 0
    for url, ts in sorted(pages.items()):
        if limit and count >= limit:
            break
        print(f"  {url}")
        if process_archived_page(db, url, ts):
            count += 1
    db.conn.commit()
    print(f"ATTO (Wayback) sync complete: {count} dyes.")
    return count


def main():
    """Download ATTO dye records (from the Wayback Machine) into the staging DB."""
    from chisurf.plugins.spectra_downloader.download._base import scraper_main

    def _add(parser):
        parser.add_argument("--limit", type=int, default=0,
                            help="Max number of dyes to import (0 = all).")

    scraper_main(
        "Download ATTO-TEC dye spectra from the Wayback Machine into the staging DB",
        lambda db, args: download_atto_from_wayback(db, limit=args.limit),
        _add,
    )


if __name__ == "__main__":
    main()
