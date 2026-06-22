#!/usr/bin/env python3
"""
Debug helper for SM Acquisition simulation parameters.

This script helps debug and inspect simulation parameters without launching the full GUI.
"""

import sys
import json
from pathlib import Path

# Add chisurf path
chisurf_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chisurf_path))

def inspect_simulation_params():
    """Inspect and display current simulation parameters."""
    print("SM Acquisition Simulation Parameters Inspector")
    print("=" * 60)

    try:
        from chisurf.plugins._dev.photon_acquisition.tcspc_devices.simulation.wrapper import SimulationDevice

        device = SimulationDevice()

        print("\nDefault Simulation Parameters:")
        print("-" * 40)

        # Show all parameters
        for key, value in sorted(device.simulation_params.items()):
            if isinstance(value, list) and len(value) > 5:
                print(f"{key}: [{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] ({len(value)} items)")
            else:
                print(f"{key}: {value}")

        print("\nParameter Descriptions:")
        print("-" * 40)
        descriptions = {
            'N_species': 'Number of molecular species',
            'N_channels': 'Number of detection channels (2 = parallel/perpendicular)',
            'M': 'Initial number of molecules per species',
            'D': 'Diffusion coefficients (μm²/s)',
            'q': 'Brightness per species per channel (photons per molecule per μs)',
            'q_bg': 'Background count rate per channel (counts per μs)',
            'k_rad': 'Radiative decay rates (μs⁻¹)',
            'k_nrad': 'Non-radiative decay rates (μs⁻¹)',
            'box_xy': 'Lateral simulation box size (μm)',
            'box_z': 'Axial simulation box size (μm)',
            'focus_type': 'Focus geometry type (0=3D Gaussian)',
            'focus_param': 'Focus parameters [waist_xy, waist_z] (μm)',
            'dt': 'Diffusion time step (μs)',
            'N_ph_max': 'Maximum photons to generate',
            'N_ph_per_file': 'Photons per output SPC file',
            'pulsed_exc': 'Excitation mode (0=CW, 1=pulsed)',
            'ch_conversion': 'Channel routing mapping',
            'N_tac_channels': 'Number of TAC channels',
            'tac_dt': 'TAC channel width (ns)',
            'laser_period': 'Laser repetition period (ns)',
            'r0': 'Fundamental anisotropy',
            'g_factor': 'G-factor for anisotropy correction',
            'l1': 'Instrumental depolarization factor 1',
            'l2': 'Instrumental depolarization factor 2',
            'parallel_scatter': 'Parallel scattering background',
            'perp_scatter': 'Perpendicular scattering background',
            'parallel_dark': 'Parallel dark counts',
            'perp_dark': 'Perpendicular dark counts',
            'spc_output_path': 'Output directory for SPC files'
        }

        for key, desc in descriptions.items():
            if key in device.simulation_params:
                print(f"{key:15}: {desc}")

        print("\nSimulation will generate:")
        n_ph_max = device.simulation_params.get('N_ph_max', 1000)
        n_per_file = device.simulation_params.get('N_ph_per_file', 1000)
        n_files = (n_ph_max + n_per_file - 1) // n_per_file  # Ceiling division
        print(f"- {n_ph_max:,} total photons")
        print(f"- {n_files} output files")
        print(f"- {n_per_file:,} photons per file")

        # Test parameter validation
        print("\nParameter Validation:")
        print("-" * 40)

        errors = []
        warnings = []

        if device.simulation_params.get('N_species', 1) < 1:
            errors.append("N_species must be >= 1")

        n_channels = device.simulation_params.get('N_channels', 2)
        if n_channels not in [1, 2]:
            warnings.append(f"N_channels={n_channels}, typically 1 or 2 for single/dual channel detection")

        q = device.simulation_params.get('q', [])
        expected_q_len = device.simulation_params.get('N_species', 1) * n_channels
        if len(q) != expected_q_len:
            errors.append(f"q array length {len(q)} doesn't match N_species({device.simulation_params.get('N_species', 1)}) * N_channels({n_channels}) = {expected_q_len}")

        if device.simulation_params.get('N_ph_max', 0) <= 0:
            errors.append("N_ph_max must be > 0")

        if device.simulation_params.get('N_ph_per_file', 0) <= 0:
            errors.append("N_ph_per_file must be > 0")

        if errors:
            print("❌ ERRORS:")
            for error in errors:
                print(f"   - {error}")

        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")

        if not errors and not warnings:
            print("✅ All parameters appear valid")

    except Exception as e:
        print(f"❌ Error inspecting parameters: {e}")
        import traceback
        traceback.print_exc()

def save_default_params(filename="default_simulation_params.json"):
    """Save current default parameters to JSON file."""
    try:
        from chisurf.plugins._dev.photon_acquisition.tcspc_devices.simulation.wrapper import SimulationDevice

        device = SimulationDevice()
        params = device.simulation_params.copy()

        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"✅ Default parameters saved to {filename}")

    except Exception as e:
        print(f"❌ Error saving parameters: {e}")

def load_and_validate_params(filename):
    """Load parameters from JSON and validate them."""
    try:
        with open(filename, 'r') as f:
            params = json.load(f)

        print(f"Loaded parameters from {filename}")
        print(f"Parameters: {len(params)} total")

        # Validate key parameters
        required_params = ['N_species', 'N_channels', 'N_ph_max']
        for param in required_params:
            if param in params:
                print(f"  {param}: {params[param]}")
            else:
                print(f"  ❌ Missing required parameter: {param}")

        return params

    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return None

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SM Acquisition Simulation Parameter Debugger")
    parser.add_argument('--save', metavar='FILENAME', help='Save default parameters to JSON file')
    parser.add_argument('--load', metavar='FILENAME', help='Load and validate parameters from JSON file')

    args = parser.parse_args()

    if args.save:
        save_default_params(args.save)
    elif args.load:
        load_and_validate_params(args.load)
    else:
        inspect_simulation_params()
