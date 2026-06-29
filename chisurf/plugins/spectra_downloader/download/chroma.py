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


# Map each Chroma filter "Type" code → a canonical component kind. The kind
# resolves (via COMPONENT_KINDS in mfdb_adapter) to the canonical category and
# default spectrum type, so filters/dichroics/mirrors land in the right tab.
FILTER_KIND_MAP: dict[str, str] = {
    "BP": "bandpass",
    "EX": "excitation_filter",
    "EM": "emission_filter",
    "BS": "dichroic",
    "AS": "astronomy",
    "MV": "machine_vision",
    "ND": "nd",
    "BEAMSPLITTER": "beamsplitter",
    "MIRROR": "mirror",
    "POLARIZER": "polarizer",
    "TRI": "tristimulus",
}


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
    """Import a single Chroma filter / dichroic / mirror item."""
    sp_id = item.get("spID")
    if not sp_id:
        return
    kind = FILTER_KIND_MAP.get(item.get("Type"))
    if kind is None:
        return
    raw_name = item.get("Number", "")
    name = _sanitize_name(raw_name)
    if not name:
        return
    spectrum = _fetch_ascii_spectrum(sp_id)
    if spectrum is None:
        return
    with db:
        db.register_component(
            name=name,
            source="chroma",
            kind=kind,
            source_ref=str(sp_id),
            description=f"Chroma {raw_name}".strip(),
            spectra=spectrum,
        )
    counts[kind] = counts.get(kind, 0) + 1


def _import_light_source_item(
    db: FluorophoreDatabase,
    item: dict,
    counts: dict[str, int],
) -> None:
    """Import a single Chroma light source item."""
    sp_id = item.get("spID")
    if not sp_id:
        return
    name = item.get("Title", "").strip()
    if not name:
        return
    spectrum = _fetch_ascii_spectrum(sp_id)
    if spectrum is None:
        return
    with db:
        db.register_component(
            name=name,
            source="chroma",
            kind="light_source",
            source_ref=str(sp_id),
            description=f"Chroma Light Source – {item.get('source', '')}".strip(" –"),
            spectra=spectrum,
        )
    counts["light_source"] = counts.get("light_source", 0) + 1


def _import_fluorochrome_item(
    db: FluorophoreDatabase,
    item: dict,
    counts: dict[str, int],
) -> None:
    """Import a single Chroma fluorochrome (excitation + emission spectra)."""
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
    with db:
        db.register_component(
            name=name,
            source="chroma",
            kind="fluorochrome",
            source_ref=f"{ex_id}/{em_id}",
            description=f"Chroma Fluorochrome – {name}",
            spectra={"excitation": ex_spec, "emission": em_spec},
        )
    counts["fluorochrome"] = counts.get("fluorochrome", 0) + 1


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

    print("\nImport summary (by component kind):")
    for kind, cnt in sorted(counts.items()):
        print(f"  {kind:<20s} {cnt:4d} items")
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
