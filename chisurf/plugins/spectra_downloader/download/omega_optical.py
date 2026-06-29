import csv
import requests
from typing import Any, List, Tuple

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import FluorophoreDatabase


class OmegaOpticalDownloader:
    def __init__(self, db: FluorophoreDatabase):
        self.db = db

    def download_and_store(self, product_id: str, name: str, filter_type: str) -> None:
        """
        Download CSV for a specific product and store in MFDB.
        
        Parameters
        ----------
        product_id : str
            Omega product identifier (e.g., 'W2806')
        name : str
            Human readable name (e.g., '650LP')
        filter_type : str
            Internal filter type for probe_type (e.g., 'longpass')
        """
        # Try transmission CSV first, fall back to optical density
        for suffix in ["_transmission.csv", "_od.csv"]:
            csv_url = f"https://www.omegafilters.com/files/spectral-curve/{product_id}{suffix}"
            try:
                response = requests.get(csv_url, timeout=60)
                response.raise_for_status()
                break  # Success
            except requests.exceptions.RequestException:
                continue  # Try next suffix
        else:
            raise RuntimeError(f"Could not download any CSV for product for product {product_id}")

        # Parse CSV - expect wavelength,transmission or wavelength,OD
        wavelengths: List[float] = []
        intensities: List[float] = []
        
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):  # Skip comments/empty
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0].strip())
                val = float(parts[1].strip())
                wavelengths.append(wl)
                intensities.append(val)
            except ValueError:
                continue  # Skip malformed lines

        if not wavelengths:
            raise RuntimeError(f"No valid data parsed from CSV for {product_id}")

        # Store in MFDB through the canonical ingestion contract. ``filter_type``
        # already matches a COMPONENT_KINDS key (longpass/shortpass/bandpass/
        # dichroic), so the category and spectrum group are derived consistently.
        with self.db:
            self.db.register_component(
                name=f"Omega {name}",
                source="omega",
                kind=filter_type,
                source_ref=product_id,
                description=f"Omega Optical {filter_type} filter - {name}",
                spectra={"transmission": (wavelengths, intensities)},
            )


# Example SKUs (Omega has no public catalogue API; these are demonstration parts).
EXAMPLE_PRODUCTS = [
    ("W2806", "650LP", "longpass"),
    ("W3272", "630SP", "shortpass"),
    ("W253", "330BP10", "bandpass"),
    ("W3450", "420SP", "shortpass"),
    ("W201", "490SP", "shortpass"),
    ("XF2017/25.7*36", "560DRLP", "dichroic"),
]


def download_omega_to_db(db: FluorophoreDatabase) -> None:
    """Download the example Omega Optical spectra into an open staging DB."""
    downloader = OmegaOpticalDownloader(db)
    for product_id, name, ftype in EXAMPLE_PRODUCTS:
        try:
            print(f"Processing Omega {name} ({product_id})...")
            downloader.download_and_store(product_id, name, ftype)
            print("  Success")
        except Exception as e:
            print(f"  Failed: {e}")


def main():
    """CLI entry point."""
    from chisurf.plugins.spectra_downloader.download._base import scraper_main

    def _add(parser):
        parser.add_argument("--product", default=None,
                            help="Download a specific Omega product ID (for testing).")

    def _run(db, args):
        if args.product:
            OmegaOpticalDownloader(db).download_and_store(
                args.product, f"Omega-{args.product}", "unknown")
        else:
            download_omega_to_db(db)

    scraper_main("Download Omega Optical spectra into the staging DB", _run, _add)


if __name__ == "__main__":
    main()