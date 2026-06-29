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


def download_omega_to_db(db_path: str = None) -> None:
    """
    Download all available Omega Optical spectra into MFDB.
    
    This is a placeholder - full implementation would require scraping
    Omega's website to get product list. For now, demonstrates with examples.
    """
    db = FluorophoreDatabase(db_path or ":memory:")
    
    # Example products - in reality this would come from scraping or API
    # These are example SKUs we saw in our web search
    example_products = [
        ("W2806", "650LP", "longpass"),
        ("W3272", "630SP", "shortpass"),
        ("W253", "330BP10", "bandpass"),
        ("W3450", "420SP", "shortpass"),
        ("W201", "490SP", "shortpass"),
        ("XF2017/25.7*36", "560DRLP", "dichroic"),
    ]
    
    for product_id, name, ftype in example_products:
        try:
            print(f"Processing Omega {name} ({product_id})...")
            downloader = OmegaOpticalDownloader(db)
            downloader.download_and_store(product_id, name, ftype)
            print(f"  Success")
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Omega Optical spectra into MFDB"
    )
    parser.add_argument(
        "--db", 
        default=None, 
        help="MFDB SQLite database path (default: in-memory)"
    )
    parser.add_argument(
        "--product",
        help="Download specific Omega product ID (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.product:
        # Single product mode
        db = FluorophoreDatabase(args.db or ":memory:")
        downloader = OmegaOpticalDownloader(db)
        # Would need name and type - simplified for CLI
        downloader.download_and_store(
            args.product, 
            f"Omega-{args.product}", 
            "unknown"
        )
    else:
        # Full catalog mode (placeholder)
        download_omega_to_db(args.db)