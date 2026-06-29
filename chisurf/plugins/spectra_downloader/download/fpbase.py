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
FPBASE_GRAPHQL_URL = "https://www.fpbase.org/graphql/"

# FPbase spectra are categorised (P protein, D dye, F filter, L light, C camera)
# with a subtype. Map the optical-component ones to a canonical kind + spectrum
# type. Proteins/dyes come from the richer proteins REST API above, so only the
# instrument categories (cameras/detectors, light sources, filters) are pulled
# from GraphQL here.
FPBASE_OPTICS_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("C", "QE"): ("detector", "quantum_efficiency"),   # cameras, SPADs, hybrid PMTs
    ("L", "PD"): ("light_source", "emission"),         # light-source power distribution
    ("F", "BP"): ("filter", "transmission"),
    ("F", "LP"): ("filter", "transmission"),
    ("F", "SP"): ("filter", "transmission"),
    ("F", "BX"): ("filter", "transmission"),
    ("F", "BM"): ("filter", "transmission"),
    ("F", "BS"): ("dichroic", "transmission"),
}


def _fpbase_graphql(query: str):
    """POST a GraphQL query to FPbase and return the ``data`` payload."""
    req = urllib.request.Request(
        FPBASE_GRAPHQL_URL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (ChiSurf)"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read()).get("data") or {}


def download_fpbase_optics_to_db(db, categories=("C",)) -> int:
    """Download FPbase instrument spectra (cameras/detectors, lights, filters).

    FPbase hosts many detectors the proteins API does not — cameras, hybrid
    PMTs and single-photon detectors (e.g. the Thorlabs SPCMxxA SPAD). They are
    fetched from the GraphQL ``spectra`` catalogue and registered with the
    canonical kind/category (detector → quantum_efficiency, etc.).

    Parameters
    ----------
    db : FluorophoreDatabase
        Open database handle.
    categories : tuple of str
        FPbase categories to import: ``C`` (cameras/detectors), ``L`` (light
        sources), ``F`` (filters/dichroics). Defaults to detectors only.
    """
    print(f"Fetching FPbase instrument spectra {categories} via GraphQL …")
    listing = _fpbase_graphql("{ spectra { id category subtype owner { name } } }")
    specs = [s for s in listing.get("spectra", []) if s.get("category") in categories]
    print(f"  {len(specs)} candidate spectra.")

    count = 0
    with db:
        for s in specs:
            kind_stype = FPBASE_OPTICS_MAP.get((s.get("category"), s.get("subtype")))
            if not kind_stype:
                continue
            kind, spectrum_type = kind_stype
            name = (s.get("owner") or {}).get("name")
            if not name:
                continue
            try:
                one = _fpbase_graphql(f'{{ spectrum(id: {int(s["id"])}) {{ data }} }}')
                data = (one.get("spectrum") or {}).get("data") or []
                arr = np.array(data, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 3:
                    continue
                db.register_component(
                    name=name,
                    source="fpbase",
                    kind=kind,
                    source_ref=str(s["id"]),
                    description=f"FPbase {kind} – {name}",
                    spectra={spectrum_type: (arr[:, 0], arr[:, 1])},
                )
                count += 1
            except Exception as e:  # pragma: no cover - network/parse resilience
                logger.error(f"Failed FPbase spectrum {s.get('id')} ({name}): {e}")
        db.conn.commit()

    print(f"  FPbase instruments imported: {count}")
    return count

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
        count = 0

        for slug, spec_item in slugs_with_spectra.items():
            meta = all_metadata.get(slug, {})
            name = spec_item.get('name') or meta.get('name')
            if not name:
                continue

            is_fp = meta.get('seq') is not None
            desc = meta.get('description', '') or ''

            # Collect spectra, keyed by canonical spectrum type.
            spectra: dict[str, tuple] = {}
            for spec in spec_item.get('spectra', []):
                state_name = spec.get('state', '').lower()
                stype = 'absorption' if ('ex' in state_name or 'abs' in state_name) else \
                        'emission' if 'em' in state_name else \
                        'transmission' if 'trans' in state_name else state_name
                data = spec.get('data', [])
                if data:
                    arr = np.array(data)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        spectra[stype] = (arr[:, 0], arr[:, 1])

            # A transmission spectrum marks an optical filter; otherwise it is a
            # fluorescent protein (has a sequence) or an organic dye.
            if 'transmission' in spectra:
                kind = "filter"
            elif is_fp:
                kind = "fluorescent_protein"
            else:
                kind = "organic_dye"

            # Optical properties (register_component canonicalizes the keys).
            properties: dict[str, str] = {"fpbase_slug": slug}
            state = meta.get('default_state')
            if not state and meta.get('states'):
                state = meta.get('states')[0]
            if state:
                for k, db_k in [('qy', 'Quantum Yield'), ('ext_coeff', 'Extinction Coefficient'),
                                ('ex_max', 'Excitation Max'), ('em_max', 'Emission Max')]:
                    val = state.get(k)
                    if val is not None:
                        properties[db_k] = str(val)

            db.register_component(
                name=name,
                source="fpbase",
                kind=kind,
                source_ref=slug,
                description=desc,
                properties=properties,
                spectra=spectra,
            )

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
    parser.add_argument("--optics", default="C",
                        help="FPbase instrument categories to also import: C=detectors, "
                             "L=light sources, F=filters (comma-separated; empty to skip).")
    args = parser.parse_args()

    print("Starting FPbase data fetch...")
    db = FluorophoreDatabase(args.db)
    with db:
        download_fpbase_to_db(db)
        cats = tuple(c.strip() for c in args.optics.split(",") if c.strip())
        if cats:
            download_fpbase_optics_to_db(db, categories=cats)
    print("FPbase database update complete.")
