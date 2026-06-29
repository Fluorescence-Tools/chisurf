#!/usr/bin/env python3
"""Import PhotochemCAD 3.1 Common Compounds into the Spectra Viewer database.

This script reads the `Common Compounds DB.db` text database and the associated
`.abs.txt` / `.ems.txt` spectra and `.tif` structure images from PhotochemCAD
and populates the ChiSurf MFDB-backed `spectra.db`.

All PhotochemCAD metadata is stored as optical properties so it is visible in
Spectra Viewer (item info table) and usable by the Förster-radius calculator
(via `Quantum Yield` and `Extinction Coefficient`).

Usage (from repo root):
    python -m chisurf.plugins.spectra_viewer.download.photochemcad_common_compounds
or simply run it via the Spectra Viewer GUI (Tools → Download Data →
"Download Photochemcad Common Compounds").
"""

import csv
from pathlib import Path

import numpy as np

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)


def _parse_float(value: str):
    """Best-effort conversion of numeric strings like "2,860" → 2860.0.

    Returns None if parsing fails.
    """
    if value is None:
        return None
    txt = str(value).strip().strip('"')
    if not txt:
        return None
    txt = txt.replace(",", "")
    try:
        return float(txt)
    except ValueError:
        return None


def _load_pcad_spectrum(path: Path):
    """Load a PhotochemCAD spectrum (.abs.txt / .ems.txt).

    Files are tab-separated with a one-line header. This function is robust
    against stray blank / non-numeric lines and returns (wavelengths, values)
    as numpy arrays, or (None, None) on failure.
    """
    xs = []
    ys = []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Skip header line (e.g. "Wavelength (nm)\tA01 abs1 N")
                if line_no == 0 and not line[0].isdigit():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    wl = float(parts[0])
                    val = float(parts[1])
                except ValueError:
                    continue
                xs.append(wl)
                ys.append(val)
    except OSError as e:
        print(f"  ERROR: could not read spectrum file {path}: {e}")
        return None, None

    if not xs:
        print(f"  WARNING: no numeric data found in spectrum file {path}")
        return None, None

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def import_photochemcad_common_compounds(db: FluorophoreDatabase, common_dir: Path):
    """Import all entries from PhotochemCAD Common Compounds into spectra.db.

    Parameters
    ----------
    db : FluorophoreDatabase
        Open database handle.
    common_dir : Path
        Directory containing `Common Compounds DB.db` and all *.abs/*.ems/*.tif
        files (e.g. `playground/PhotochemCAD 3.1/Common Compounds`).
    """
    db_path = common_dir / "Common Compounds DB.db"
    if not db_path.exists():
        print(f"ERROR: PhotochemCAD DB file not found: {db_path}")
        return

    print(f"Using PhotochemCAD DB: {db_path}")

    # Ensure tables exist
    db.create_tables()

    imported = 0

    with db.conn:  # type: ignore[attr-defined]
        with db_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                print("ERROR: PhotochemCAD DB file is empty.")
                return

            # Header layout (indices) from the file:
            # 0: '#' (index)
            # 1: 'Absorption'
            # 2: 'Name'
            # 3: 'Structure'
            # 4: 'Class'
            # 5: 'File' (absorption spectrum file)
            # 6: 'Wavelength'
            # 7: 'Epsilon'
            # 8: 'Solvent' (absorption)
            # 9: 'Instrument' (absorption)
            # 10: 'Date' (absorption)
            # 11: 'Reference' (absorption)
            # 12: 'Inv' (absorption investigator)
            # 13: 'Emission'
            # 14: 'File' (emission spectrum file)
            # 15: 'Solvent' (emission)
            # 16: 'Quantum Yield'
            # 17: 'Instrument' (emission)
            # 18: 'Date' (emission)
            # 19: 'Reference' (emission)
            # 20: 'Inv' (emission investigator)

            for row in reader:
                if not row or all(not c.strip() for c in row):
                    continue

                # Pad to at least 21 columns to avoid IndexError
                if len(row) < 21:
                    row += [""] * (21 - len(row))

                idx = row[0].strip().strip("#")
                name = row[2].strip().strip('"')
                if not name:
                    # Skip malformed entries
                    continue

                structure_file = row[3].strip().strip('"')
                cls = row[4].strip()

                abs_file = row[5].strip().strip('"')
                abs_wavelength = row[6].strip()
                epsilon = row[7].strip()
                abs_solvent = row[8].strip()
                abs_instrument = row[9].strip()
                abs_date = row[10].strip()
                abs_reference = row[11].strip()
                abs_investigator = row[12].strip()

                em_file = row[14].strip().strip('"')
                em_solvent = row[15].strip()
                quantum_yield = row[16].strip()
                em_instrument = row[17].strip()
                em_date = row[18].strip()
                em_reference = row[19].strip()
                em_investigator = row[20].strip()

                # Build a human-readable description field
                desc_lines = []
                if cls:
                    desc_lines.append(f"Class: {cls}")
                if abs_reference:
                    desc_lines.append(f"Absorption ref: {abs_reference}")
                if em_reference:
                    desc_lines.append(f"Emission ref: {em_reference}")
                description = "\n".join(desc_lines)

                # Collect optical properties into a dict (register_component
                # canonicalizes keys like "Quantum Yield" → qy).
                properties: dict[str, str] = {}

                def add_prop(label: str, value: str):
                    v = (value or "").strip()
                    if v:
                        properties[label] = v

                # Core metadata
                add_prop("PhotochemCAD Index", idx)
                add_prop("Class", cls)

                # Absorption info
                add_prop("Absorption max wavelength (nm)", abs_wavelength)
                add_prop("Epsilon", epsilon)

                eps_float = _parse_float(epsilon)
                if eps_float is not None:
                    # Name chosen to integrate with Förster-radius UI
                    properties["Extinction Coefficient"] = str(eps_float)

                add_prop("Absorption solvent", abs_solvent)
                add_prop("Absorption instrument", abs_instrument)
                add_prop("Absorption date", abs_date)
                add_prop("Absorption reference", abs_reference)
                add_prop("Absorption investigator", abs_investigator)

                # Emission info
                add_prop("Emission solvent", em_solvent)
                add_prop("Quantum Yield", quantum_yield)
                add_prop("Emission instrument", em_instrument)
                add_prop("Emission date", em_date)
                add_prop("Emission reference", em_reference)
                add_prop("Emission investigator", em_investigator)

                # File names for traceability
                add_prop("Absorption file", abs_file)
                add_prop("Emission file", em_file)

                # Collect spectra.
                spectra: dict[str, tuple] = {}
                if abs_file:
                    abs_path = common_dir / abs_file
                    if abs_path.exists():
                        wl, val = _load_pcad_spectrum(abs_path)
                        if wl is not None and val is not None:
                            spectra["absorption"] = (wl, val)
                    else:
                        print(f"  WARNING: absorption file not found: {abs_path}")
                if em_file:
                    em_path = common_dir / em_file
                    if em_path.exists():
                        wl, val = _load_pcad_spectrum(em_path)
                        if wl is not None and val is not None:
                            spectra["emission"] = (wl, val)
                    else:
                        print(f"  WARNING: emission file not found: {em_path}")

                # Register through the canonical ingestion contract.
                item_id = db.register_component(
                    name=name,
                    source="photochemcad",
                    kind="organic_dye",
                    source_ref=idx,
                    description=description,
                    properties=properties,
                    spectra=spectra,
                )

                # Import structure image
                if structure_file:
                    img_path = common_dir / structure_file
                    if img_path.exists():
                        try:
                            with open(img_path, "rb") as img_f:
                                img_data = img_f.read()
                            img_ext = img_path.suffix.lower().replace(".", "")
                            db.add_probe_image(item_id, img_data, fmt=img_ext, image_name=img_path.name)
                        except Exception as e:  # pragma: no cover - defensive
                            print(f"  WARNING: failed to add image {img_path}: {e}")
                    else:
                        print(f"  WARNING: structure image not found: {img_path}")

                imported += 1
                if imported % 50 == 0:
                    print(f"  Imported {imported} compounds so far...")

    print(f"Finished PhotochemCAD import: {imported} compounds imported.")


def main():
    """Entry point for CLI / Spectra Viewer integration."""
    import argparse

    # Default path: repo_root / playground / "PhotochemCAD 3.1" / "Common Compounds"
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]  # .../chisurf repo root (top-level)
    default_common_dir = repo_root / "playground" / "PhotochemCAD 3.1" / "Common Compounds"

    parser = argparse.ArgumentParser(
        description=(
            "Import PhotochemCAD 3.1 Common Compounds into the Spectra Viewer "
            "SQLite database (spectra.db)."
        )
    )
    parser.add_argument(
        "--pcad-dir",
        default=str(default_common_dir),
        help=(
            "Directory containing 'Common Compounds DB.db' and the *.abs/*.ems/*.tif "
            "files (default: %(default)s)."
        ),
    )
    parser.add_argument("--db", help="MFDB SQLite database path", default=str(DEFAULT_DATABASE_PATH))

    args = parser.parse_args()
    common_dir = Path(args.pcad_dir)

    if not common_dir.exists():
        print(f"ERROR: PhotochemCAD Common Compounds directory not found: {common_dir}")
        return

    print(f"PhotochemCAD Common Compounds directory: {common_dir}")

    db = FluorophoreDatabase(args.db)
    with db:
        import_photochemcad_common_compounds(db, common_dir)


if __name__ == "__main__":
    main()
