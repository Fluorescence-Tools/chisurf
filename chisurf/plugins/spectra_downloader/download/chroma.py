#!/usr/bin/env python3
"""Download Chroma Technology optical filter and fluorochrome spectra into MFDB.

Sources data from Chroma's public Spectra Viewer API
(https://www.chroma.com/spectra-viewer). Spectral curves are
provided as tab-separated ASCII files (wavelength in nm, value
as fraction) for 650+ filters and 200+ fluorochromes.

Component types included:
  - Bandpass, longpass, shortpass, and notch filters
  - Dichroic beamsplitters and mirrors
  - Excitation and emission filters (fluorescence sets)
  - Light source emission spectra
  - Fluorochrome excitation / emission spectra

Usage:
    python -m chisurf.plugins._dev.spectra_downloader.download.chroma --db <path>
"""

import concurrent.futures
import json
import logging
import urllib.request
from typing import Any

import numpy as np

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)

logger = logging.getLogger(__name__)

DATA_PROVIDER_URL = "https://www.chroma.com/api/sv/data-providers"
ASCII_BASE_URL = "https://www.chroma.com/files/part_spectra"


# Map Chroma filter types → MFDB probe info
FILTER_TYPE_MAP: dict[str, tuple[str, str, str]] = {
    "BP":  ("chroma_bandpass",    "Chroma Bandpass Filter",       "transmission"),
    "EX":  ("chroma_excitation",  "Chroma Excitation Filter",     "transmission"),
    "EM":  ("chroma_emission",    "Chroma Emission Filter",       "transmission"),
    "BS":  ("chroma_dichroic",    "Chroma Dichroic Beamsplitter", "transmission"),
    "AS":  ("chroma_astronomy",   "Chroma Astronomy Filter",      "transmission"),
    "MV":  ("chroma_machine_vision", "Chroma Machine Vision Filter", "transmission"),
    "ND":  ("chroma_nd",          "Chroma ND Filter",             "transmission"),
    "BEAMSPLITTER": ("chroma_beamsplitter", "Chroma Beamsplitter", "transmission"),
    "MIRROR":       ("chroma_mirror",       "Chroma Mirror",       "reflectance"),
    "POLARIZER":    ("chroma_polarizer",    "Chroma Polarizer",    "transmission"),
    "TRI":          ("chroma_tristimulus",  "Chroma Tristimulus Filter", "transmission"),
}

# Light sources → MFDB probe type
LIGHTSOURCE_TYPE = ("chroma_lightsource", "Chroma Light Source", "emission")

# Fluorochrome → MFDB probe type
FLUOROCHROME_TYPE = ("chroma_fluorochrome", "Chroma Fluorochrome", "excitation")


def _fetch_json(url: str, data: dict | None = None) -> Any:
    """Fetch JSON from a URL, optionally with POST data."""
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ChromaDownloader/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _fetch_ascii_spectrum(sp_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Download and parse a tab-separated ASCII spectrum file.

    Returns (wavelength_nm, value) arrays, or None on failure.
    """
    url = f"{ASCII_BASE_URL}/{sp_id}-ascii.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    wl_list: list[float] = []
    val_list: list[float] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) >= 2:
            try:
                wl = float(parts[0])
                val = float(parts[1])
                if np.isfinite(wl) and np.isfinite(val):
                    wl_list.append(wl)
                    val_list.append(val)
            except (ValueError, TypeError):
                continue
    if len(wl_list) < 3:
        return None
    arr = np.column_stack([np.array(wl_list), np.array(val_list)])
    arr = arr[arr[:, 0].argsort()]
    return arr[:, 0], arr[:, 1]


def _sanitize_name(raw: str) -> str:
    """Clean up a part-number string for use as a chromophore_name."""
    return raw.strip().split(" - ")[0].strip()


def _import_filter_item(
    db: FluorophoreDatabase,
    item: dict,
    counts: dict[str, int],
) -> None:
    """Import a single Chroma filter item into the database."""
    sp_id = item.get("spID")
    if not sp_id:
        return
    chroma_type = item.get("Type")
    if chroma_type not in FILTER_TYPE_MAP:
        return
    type_name, type_display, spectrum_type = FILTER_TYPE_MAP[chroma_type]
    raw_name = item.get("Number", "")
    name = _sanitize_name(raw_name)
    if not name:
        return
    spectrum = _fetch_ascii_spectrum(sp_id)
    if spectrum is None:
        return
    wl, val = spectrum
    with db:
        type_id = db.add_probe_type(type_name, type_display)
        item_id = db.add_probe(
            chromophore_name=name,
            type_id=type_id,
            description=f"Chroma {type_display} – {raw_name}",
        )
        db.add_spectrum(item_id, spectrum_type, wl, val)
        db.add_optical_property(item_id, "Origin", f"Chroma ({type_display})")
    counts[type_name] = counts.get(type_name, 0) + 1


def _import_light_source_item(
    db: FluorophoreDatabase,
    item: dict,
    counts: dict[str, int],
) -> None:
    """Import a single Chroma light source item into the database."""
    sp_id = item.get("spID")
    if not sp_id:
        return
    type_name, type_display, spectrum_type = LIGHTSOURCE_TYPE
    name = item.get("Title", "").strip()
    if not name:
        return
    spectrum = _fetch_ascii_spectrum(sp_id)
    if spectrum is None:
        return
    wl, val = spectrum
    with db:
        type_id = db.add_probe_type(type_name, type_display)
        probe_id = db.add_probe(
            chromophore_name=name,
            type_id=type_id,
            description=f"Chroma {type_display} – {item.get('source', '')}",
        )
        db.add_spectrum(probe_id, spectrum_type, wl, val)
        db.add_optical_property(probe_id, "Origin", "Chroma (Light Source)")
    counts[type_name] = counts.get(type_name, 0) + 1


def _import_fluorochrome_item(
    db: FluorophoreDatabase,
    item: dict,
    counts: dict[str, int],
) -> None:
    """Import a single Chroma fluorochrome item into the database.

    Stores both excitation and emission spectra.
    """
    ex_id = item.get("spExID")
    em_id = item.get("spEmID")
    if not ex_id or not em_id:
        return
    name = item.get("Title", "").strip()
    if not name:
        return
    ex_spec = _fetch_ascii_spectrum(ex_id)
    em_spec = _fetch_ascii_spectrum(em_id)
    if ex_spec is None or em_spec is None:
        return
    ex_wl, ex_val = ex_spec
    em_wl, em_val = em_spec
    type_name, type_display, _ = FLUOROCHROME_TYPE
    with db:
        type_id = db.add_probe_type(type_name, type_display)
        probe_id = db.add_probe(
            chromophore_name=name,
            type_id=type_id,
            description=f"Chroma Fluorochrome – {name}",
        )
        db.add_spectrum(probe_id, "excitation", ex_wl, ex_val)
        db.add_spectrum(probe_id, "emission", em_wl, em_val)
        db.add_optical_property(probe_id, "Origin", "Chroma (Fluorochrome)")
    counts[type_name] = counts.get(type_name, 0) + 1


def _import_item_batch(
    db: FluorophoreDatabase,
    items: list[dict],
    import_fn,
    label: str,
    counts: dict[str, int],
) -> None:
    """Download spectra in parallel, then import sequentially."""
    sp_id_key = "spID"

    # Pre-fetch all spectra in parallel
    spectra_map: dict[int, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {}
        for item in items:
            sp_id = item.get(sp_id_key) or item.get("spExID")
            if sp_id:
                futures[pool.submit(_fetch_ascii_spectrum, sp_id)] = (item, sp_id)
        for future in concurrent.futures.as_completed(futures):
            item, sp_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    spectra_map[id(item)] = result
            except Exception:
                pass

    # Import sequentially
    imported = 0
    for item in items:
        if id(item) in spectra_map:
            import_fn(db, item, counts)
            imported += 1
        elif sp_id_key in item and item[sp_id_key]:
            pass  # no data for this item

    print(f"  {label}: {imported} items imported.")


def download_chroma_to_db(db: FluorophoreDatabase) -> dict[str, int]:
    """Download and import all Chroma spectral data into MFDB.

    Parameters
    ----------
    db : FluorophoreDatabase
        Open database handle.

    Returns
    -------
    dict[str, int]
        Mapping from probe-type name → count of imported items.
    """
    counts: dict[str, int] = {}

    # --- Filters (650 items) ---
    print("Fetching Chroma filter list …")
    data = _fetch_json(f"{DATA_PROVIDER_URL}/filters", data={})
    filter_items = data.get("data", [])
    print(f"  {len(filter_items)} filters found.")

    for item in filter_items:
        _import_filter_item(db, item, counts)

    # --- Light sources (13 with spectral data) ---
    print("\nFetching Chroma light source list …")
    data = _fetch_json(f"{DATA_PROVIDER_URL}/lightsources", data={})
    source_items = data.get("data", [])
    print(f"  {len(source_items)} light sources found.")

    for item in source_items:
        _import_light_source_item(db, item, counts)

    # --- Fluorochromes (203 items) ---
    print("\nFetching Chroma fluorochrome list …")
    data = _fetch_json(f"{DATA_PROVIDER_URL}/fluorochromes", data={})
    fluoro_items = data.get("data", [])
    print(f"  {len(fluoro_items)} fluorochromes found.")

    for item in fluoro_items:
        _import_fluorochrome_item(db, item, counts)

    print("\nImport summary:")
    for tname, cnt in sorted(counts.items()):
        known = dict(
            [v[:2] for v in FILTER_TYPE_MAP.values()]
            + [LIGHTSOURCE_TYPE[:2], FLUOROCHROME_TYPE[:2]]
        )
        display = known.get(tname, tname)
        print(f"  {display:<40s} {cnt:4d} items")
    print(f"\nTotal: {sum(counts.values())} items imported.")
    return counts


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Chroma optical component spectra into MFDB",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DATABASE_PATH),
        help="MFDB SQLite database path (default: %(default)s)",
    )
    args = parser.parse_args()

    db = FluorophoreDatabase(args.db)
    with db:
        download_chroma_to_db(db)
    print("Chroma data download complete.")


if __name__ == "__main__":
    main()
