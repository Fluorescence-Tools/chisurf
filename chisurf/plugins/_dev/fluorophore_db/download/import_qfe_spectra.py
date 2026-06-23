#!/usr/bin/env python3
"""
Import spectra from qfe_spectraviewer (QuickFit) into ChiSurf spectra_viewer database.

This script imports fluorophores, filters, light sources, and detectors from the
QuickFit qfe_spectraviewer plugin assets into the ChiSurf spectra_viewer database.

Usage:
    python import_qfe_spectra.py /path/to/qfe_spectraviewer/assets
"""

import os
import sys
import configparser
import numpy as np
from pathlib import Path
import argparse

# Add the parent directory to the path to import the database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chisurf.fio.mmcif.db import FluorophoreDatabase

def parse_ini_file(ini_path):
    """Parse an .ini file and return a dictionary of sections."""
    config = configparser.ConfigParser()
    # Allow duplicate keys by reading manually
    with open(ini_path, 'r', encoding='utf-8') as f:
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

def import_fluorophores(db, assets_dir, ini_data):
    """Import fluorophore data from .ini and .spec files."""
    print("Importing fluorophores...")

    # Add probe type
    fluorophore_type_id = db.add_probe_type("fluorophore", "Fluorophores")

    for name, props in ini_data.items():
        try:
            # Get basic properties
            description = props.get('reference', '')
            folder = props.get('folder', 'other')

            # Add probe
            item_id = db.add_probe(chromophore_name=name, type_id=fluorophore_type_id, description=description)

            # Add optical properties
            for prop_key, prop_value in props.items():
                if prop_key not in ['spectrum_fl', 'spectrum_abs', 'folder', 'reference']:
                    db.add_optical_property(item_id, prop_key, prop_value)

            # Load and add absorption spectrum
            abs_spec = props.get('spectrum_abs', '')
            if abs_spec and os.path.exists(assets_dir / abs_spec):
                result = load_spectrum(assets_dir / abs_spec)
                if result and len(result) >= 2:
                    wavelengths, abs_values = result[0], result[1]
                    db.add_spectrum(item_id, 'absorption', wavelengths, abs_values)

            # Load and add emission spectrum
            em_spec = props.get('spectrum_fl', '')
            if em_spec and os.path.exists(assets_dir / em_spec):
                result = load_spectrum(assets_dir / em_spec)
                if result and len(result) >= 2:
                    wavelengths, em_values = result[0], result[1]
                    db.add_spectrum(item_id, 'emission', wavelengths, em_values)

            print(f"Imported fluorophore: {name}")

        except Exception as e:
            print(f"Error importing fluorophore {name}: {e}")

def import_filters(db, assets_dir, ini_data):
    """Import filter data from .ini and .spec files."""
    print("Importing filters...")

    # Add probe type
    filter_type_id = db.add_probe_type("filter", "Filters")

    for name, props in ini_data.items():
        try:
            # Get basic properties
            description = props.get('description', '')
            folder = props.get('folder', 'other')

            # Add probe
            item_id = db.add_probe(chromophore_name=name, type_id=filter_type_id, description=description)

            # Add optical properties
            for prop_key, prop_value in props.items():
                if prop_key not in ['spectrum', 'folder', 'description']:
                    db.add_optical_property(item_id, prop_key, prop_value)

            # Load and add transmission spectrum
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    db.add_spectrum(item_id, 'transmission', wavelengths, values)

            print(f"Imported filter: {name}")

        except Exception as e:
            print(f"Error importing filter {name}: {e}")

def import_lightsources(db, assets_dir, ini_data):
    """Import light source data from .ini and .spec files."""
    print("Importing light sources...")

    # Add probe type
    lightsource_type_id = db.add_probe_type("lightsource", "Light Sources")

    for name, props in ini_data.items():
        try:
            # Get basic properties
            description = props.get('description', '')
            folder = props.get('folder', 'other')

            # Add probe
            item_id = db.add_probe(chromophore_name=name, type_id=lightsource_type_id, description=description)

            # Add optical properties
            for prop_key, prop_value in props.items():
                if prop_key not in ['spectrum', 'folder', 'description']:
                    db.add_optical_property(item_id, prop_key, prop_value)

            # Load and add emission spectrum
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    db.add_spectrum(item_id, 'emission', wavelengths, values)

            print(f"Imported light source: {name}")

        except Exception as e:
            print(f"Error importing light source {name}: {e}")

def import_detectors(db, assets_dir, ini_data):
    """Import tttr_channeldefinition data from .ini and .spec files."""
    print("Importing detectors...")

    # Add probe type
    detector_type_id = db.add_probe_type("tttr_channeldefinition", "Detectors")

    for name, props in ini_data.items():
        try:
            # Get basic properties
            description = props.get('description', '')
            folder = props.get('folder', 'other')

            # Add probe
            item_id = db.add_probe(chromophore_name=name, type_id=detector_type_id, description=description)

            # Add optical properties
            for prop_key, prop_value in props.items():
                if prop_key not in ['spectrum', 'folder', 'description']:
                    db.add_optical_property(item_id, prop_key, prop_value)

            # Load and add quantum efficiency spectrum
            spec_file = props.get('spectrum', '')
            if spec_file and os.path.exists(assets_dir / spec_file):
                wavelengths, values = load_spectrum(assets_dir / spec_file)
                if wavelengths is not None:
                    db.add_spectrum(item_id, 'quantum_efficiency', wavelengths, values)

            print(f"Imported tttr_channeldefinition: {name}")

        except Exception as e:
            print(f"Error importing tttr_channeldefinition {name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Import qfe_spectraviewer spectra into ChiSurf database")
    parser.add_argument("assets_dir", help="Path to qfe_spectraviewer assets directory")
    args = parser.parse_args()

    assets_path = Path(args.assets_dir)
    if not assets_path.exists():
        print(f"Assets directory does not exist: {assets_path}")
        return

    print(f"Importing spectra from: {assets_path}")

    # Initialize database
    db = FluorophoreDatabase()
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
