import json
import logging
import urllib.request

import numpy as np

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)

logger = logging.getLogger(__name__)

FPBASE_API_URL = "https://www.fpbase.org/api/proteins/"

def fetch_fpbase_proteins(url=FPBASE_API_URL):
    """Fetch a page of proteins/summary from FPbase and any subsequent pages.

    Returns:
        tuple: (list_of_proteins, next_page_url)
    """
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            results = data.get('results', data) if isinstance(data, dict) else data
            next_url = data.get('next') if isinstance(data, dict) else None
            return results, next_url
    except Exception as e:
        logger.error(f"Failed to fetch FPbase summary from {url}: {e}")
        return [], None

def fetch_fpbase_spectra(slug):
    """Fetch the full spectra (ex, em, ec) for a given protein slug.

    Returns:
        dict: A dictionary mapping spectrum types (e.g., 'absorption', 'emission') to (x, y) numpy arrays.
    """
    try:
        # We query the spectra endpoint. Some items might be under proteins, others under generic spectra.
        url = f"https://www.fpbase.org/api/proteins/spectra/?format=json&protein__slug={slug}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())

            spectra_dict = {}
            for item in data:
                # The item list usually contains states.
                for spec in item.get('spectra', []):
                    # the item represents one spectrum record
                    state = spec.get('state', '')
                    # Map FPbase internal names to our standard types
                    q = state.lower()

                    if 'ex' in q or 'abs' in q:
                        stype = 'absorption'
                    elif 'em' in q:
                        stype = 'emission'
                    elif 'ec' in q or 'ext' in q:
                        stype = 'extinction'
                    else:
                        stype = q

                    d = spec.get('data', [])
                    if d:
                        arr = np.array(d)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            x, y = arr[:, 0], arr[:, 1]
                            spectra_dict[stype] = (x, y)

            return spectra_dict

    except Exception as e:
        logger.error(f"Failed to fetch FPbase spectra for {slug}: {e}")
        return {}

def download_fpbase_to_db(db):
    """Download FPbase summaries and spectra for records with spectra.

    Ensures every item in the database has at least one spectrum.
    """
    print("Fetching list of items with spectra from FPbase...")
    try:
        # This endpoint returns a list of objects with 'slug', 'name', and 'spectra'
        spectra_url = "https://www.fpbase.org/api/proteins/spectra/?format=json"
        req = urllib.request.Request(spectra_url, headers={'User-Agent': 'Mozilla/5.0 (ChiSurf)'})
        with urllib.request.urlopen(req) as response:
            spectra_list = json.loads(response.read())
    except Exception as e:
        logger.error(f"Failed to fetch spectra list: {e}")
        return 0

    if not spectra_list:
        print("No spectra found in FPbase spectra API.")
        return 0

    slugs_with_spectra = {item['slug']: item for item in spectra_list}
    print(f"Found {len(slugs_with_spectra)} items with digital spectra data.")

    print("Fetching protein metadata...")
    all_metadata = {}
    url = f"{FPBASE_API_URL}?format=json&limit=100"
    while url:
        proteins, url = fetch_fpbase_proteins(url)
        if proteins:
            for p in proteins:
                all_metadata[p.get('slug')] = p
        else:
            break

    print(f"Retrieved metadata for {len(all_metadata)} proteins.")

    with db:
        type_id = db.add_probe_type("fpbase", "FPbase")
        count = 0

        for slug, spec_item in slugs_with_spectra.items():
            meta = all_metadata.get(slug, {})
            name = spec_item.get('name') or meta.get('name')
            if not name:
                continue

            # Metadata extraction
            is_fp = meta.get('seq') is not None
            origin = "FPbase (Protein)" if is_fp else "FPbase (Organic Dye)"
            desc = meta.get('description', '') or ''

            # Create / update probe
            item_id = db.add_probe(chromophore_name=name, type_id=type_id, description=desc)
            db.add_optical_property(item_id, "fpbase_slug", slug)
            db.add_optical_property(item_id, "Origin", origin)

            # Add optical properties from metadata
            state = meta.get('default_state')
            if not state and meta.get('states'):
                state = meta.get('states')[0]
            if state:
                for k, db_k in [('qy', 'Quantum Yield'), ('ext_coeff', 'Extinction Coefficient'),
                                ('ex_max', 'Excitation Max'), ('em_max', 'Emission Max')]:
                    val = state.get(k)
                    if val is not None:
                        db.add_optical_property(item_id, db_k, str(val))

            # Add the spectra data immediately
            for spec in spec_item.get('spectra', []):
                state_name = spec.get('state', '').lower()
                stype = 'absorption' if ('ex' in state_name or 'abs' in state_name) else \
                        'emission' if 'em' in state_name else \
                        'transmission' if 'trans' in state_name else state_name

                data = spec.get('data', [])
                if data:
                    arr = np.array(data)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        db.add_spectrum(item_id, stype, arr[:, 0], arr[:, 1])

            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/{len(slugs_with_spectra)} items...")

        db.conn.commit()

    print(f"FPbase sync complete. {count} items with spectra registered.")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FPbase spectra into MFDB")
    parser.add_argument("--db", help="MFDB SQLite database path", default=str(DEFAULT_DATABASE_PATH))
    args = parser.parse_args()

    print("Starting FPbase data fetch...")
    db = FluorophoreDatabase(args.db)
    with db:
        download_fpbase_to_db(db)
    print("FPbase database update complete.")
