#!/usr/bin/env python3
"""
Import spectra from qfe_spectraviewer (QuickFit) into ChiSurf spectra_viewer database.

This script imports fluorophores, filters, light sources, and detectors from the
QuickFit qfe_spectraviewer plugin assets into the ChiSurf spectra_viewer database.

Usage:
    python import_qfe_spectra.py /path/to/qfe_spectraviewer/assets
"""

import argparse
import os
from pathlib import Path

import numpy as np

from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
    DEFAULT_DATABASE_PATH,
    FluorophoreDatabase,
)


def parse_ini_file(ini_path):
    """Parse an .ini file and return a dictionary of sections."""
    # Allow duplicate keys by reading manually
    with open(ini_path, encoding='utf-8') as f:
        content = f.read()

    # Split into sections
    sections = {}
    current_section = None
    current_data = {}

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            # Save previous section
            if current_section:
                sections[current_section] = current_data
            current_section = line[1:-1]
            current_data = {}
        elif '=' in line and current_section:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            current_data[key] = value

    # Save last section
    if current_section:
        sections[current_section] = current_data

    return sections

def load_spectrum(spec_path):
    """Load a .spec file and return wavelength and intensity arrays."""
    try:
        data = np.loadtxt(spec_path, delimiter=',', comments='#')
        if data.shape[1] >= 2:
            wavelengths = data[:, 0]
            if data.shape[1] >= 3:  # Fluorophore format: wavelength, absorption, emission
                abs_values = data[:, 1]
                em_values = data[:, 2]
                return wavelengths, abs_values, em_values
            else:  # Filter/lightsource format: wavelength, intensity
                values = data[:, 1]
                return wavelengths, values
    except Exception as e:
        print(f"Error loading spectrum {spec_path}: {e}")
        return None, None

def _qfe_properties(props, exclude):
    """Collect non-spectrum .ini keys as a property dict for register_component."""
    return {k: v for k, v in props.items() if k not in exclude}


def import_fluorophores(db, assets_dir, ini_data):
    """Import fluorophore data from .ini and .spec files."""
    print("Importing fluorophores...")

    for name, props in ini_data.items():
        try:
            spectra = {}
            abs_spec = props.get('spectrum_abs', '')
            if abs_spec and os.path.exists(assets_dir / abs_spec):
                result = load_spectrum(assets_dir / abs_spec)
                if result and len(result) >= 2:
                    spectra['absorption'] = (result[0], result[1])
            em_spec = props.get('spectrum_fl', '')
            if em_spec and os.path.exists(assets_dir / em_spec):
                result = load_spectrum(assets_dir / em_spec)
                if result and len(result) >= 2:
                    spectra['emission'] = (result[0], result[1])

            db.register_component(
                name=name,
                source="qfe",
                kind="organic_dye",
                source_ref=name,
                description=props.get('reference', ''),
                properties=_qfe_properties(props, {'spectrum_fl', 'spectrum_abs', 'folder', 'reference'}),
                spectra=spectra,
            )
            print(f"Imported fluorophore: {name}")
        except Exception as e:
            print(f"Error importing fluorophore {name}: {e}")


def import_filters(db, assets_dir, ini_data):
    """Import filter data from .ini and .spec files."""
    print("Importing filters...")

    for name, props in ini_data.items():
        try:
            spectra = None
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    spectra = {'transmission': (wavelengths, values)}

            db.register_component(
                name=name,
                source="qfe",
                kind="filter",
                source_ref=name,
                description=props.get('description', ''),
                properties=_qfe_properties(props, {'spectrum', 'folder', 'description'}),
                spectra=spectra,
            )
            print(f"Imported filter: {name}")
        except Exception as e:
            print(f"Error importing filter {name}: {e}")


def import_lightsources(db, assets_dir, ini_data):
    """Import light source data from .ini and .spec files."""
    print("Importing light sources...")

    for name, props in ini_data.items():
        try:
            spectra = None
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    spectra = {'emission': (wavelengths, values)}

            db.register_component(
                name=name,
                source="qfe",
                kind="light_source",
                source_ref=name,
                description=props.get('description', ''),
                properties=_qfe_properties(props, {'spectrum', 'folder', 'description'}),
                spectra=spectra,
            )
            print(f"Imported light source: {name}")
        except Exception as e:
            print(f"Error importing light source {name}: {e}")


def import_detectors(db, assets_dir, ini_data):
    """Import detector data from .ini and .spec files."""
    print("Importing detectors...")

    for name, props in ini_data.items():
        try:
            spectra = None
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    spectra = {'quantum_efficiency': (wavelengths, values)}

            db.register_component(
                name=name,
                source="qfe",
                kind="detector",
                source_ref=name,
                description=props.get('description', ''),
                properties=_qfe_properties(props, {'spectrum', 'folder', 'description'}),
                spectra=spectra,
            )
            print(f"Imported detector: {name}")
        except Exception as e:
            print(f"Error importing detector {name}: {e}")

def main():
    """Import QuickFit spectra assets into the configured MFDB database."""
    parser = argparse.ArgumentParser(description="Import qfe_spectraviewer spectra into ChiSurf database")
    parser.add_argument("assets_dir", help="Path to qfe_spectraviewer assets directory")
    parser.add_argument("--db", help="MFDB SQLite database path", default=str(DEFAULT_DATABASE_PATH))
    args = parser.parse_args()

    assets_path = Path(args.assets_dir)
    if not assets_path.exists():
        print(f"Assets directory does not exist: {assets_path}")
        return

    print(f"Importing spectra from: {assets_path}")

    # Initialize database
    db = FluorophoreDatabase(args.db)
    with db:

        # Import fluorophores
        fluorophore_ini = assets_path / "fluorophors.ini"
        if fluorophore_ini.exists():
            fluorophore_data = parse_ini_file(fluorophore_ini)
            import_fluorophores(db, assets_path, fluorophore_data)

        # Import filters
        filters_ini = assets_path / "filters.ini"
        if filters_ini.exists():
            filters_data = parse_ini_file(filters_ini)
            import_filters(db, assets_path, filters_data)

        # Import light sources
        lightsources_ini = assets_path / "ligtsources.ini"
        if lightsources_ini.exists():
            lightsources_data = parse_ini_file(lightsources_ini)
            import_lightsources(db, assets_path, lightsources_data)

        # Import detectors
        detectors_ini = assets_path / "detectors.ini"
        if detectors_ini.exists():
            detectors_data = parse_ini_file(detectors_ini)
            import_detectors(db, assets_path, detectors_data)

    print("Import complete!")

if __name__ == "__main__":
    main()
