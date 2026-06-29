#!/usr/bin/env python3
"""
Simple test script for debugging Burbulator DLL issues.
Run this script directly to test DLL loading and function calls.
"""

import os
import sys
import ctypes as ct
import traceback

# Add the Chisurf path to import simulation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Try to import tttrlib for validation
try:
    import tttrlib
    TTTRLIB_AVAILABLE = True
except ImportError:
    TTTRLIB_AVAILABLE = False
    print("[WARNING] tttrlib not available for validation")

try:
    import numpy as np
except ImportError:
    np = None
    print("[WARNING] numpy not available")

try:
    # High-level wrapper used by the SM acquisition simulation device
    from chisurf.plugins.core.acq.tcspc_devices.simulation.burbulator_dll_wrapper import (
        BurbulatorDLL,
        BurbulatorError,
    )
    WRAPPER_AVAILABLE = True
except Exception:
    BurbulatorDLL = None  # type: ignore
    BurbulatorError = RuntimeError  # type: ignore
    WRAPPER_AVAILABLE = False
    print("[WARNING] BurbulatorDLL wrapper not available for testing")

def test_wrapper_basic():
    """Basic test of the high-level BurbulatorDLL wrapper."""
    print("\n=== Testing BurbulatorDLL Wrapper: basic simulate_ov3 ===")

    if not WRAPPER_AVAILABLE:
        print("[SKIP] BurbulatorDLL wrapper not available")
        return True

    try:
        burb = BurbulatorDLL()
        print(f"[OK] BurbulatorDLL loaded library from {burb.path}")

        sim = burb.simulate_ov3(
            Nspecies=1,
            M=[10.0],
            D=[1.0],
            Nchannels=2,
            q=[0.1, 0.1],
            q_bg=[0.001, 0.001],
            k_rad=[0.0],
            k_nrad=[0.0],
            box_xy=2.0,
            box_z=4.0,
            focus_type=0,
            focus_param=[0.3, 2.0],
            dt=0.01,
            N_ph_max=1000,
        )
        N_ph = sim.get("N_ph", 0)
        print(f"[OK] simulate_ov3 returned {N_ph} photons")
        if N_ph <= 0:
            print("[WARNING] simulate_ov3 returned no photons")
        return True
    except BurbulatorError as e:
        print(f"[FAIL] BurbulatorDLL wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_wrapper_spc_file():
    """Test wrapper: simulate_ov3 + convert_to_spc132 + write_spc132_file + tttrlib validation."""
    print("\n=== Testing BurbulatorDLL Wrapper: SPC file generation and validation ===")

    if not WRAPPER_AVAILABLE:
        print("[SKIP] BurbulatorDLL wrapper not available")
        return True

    try:
        burb = BurbulatorDLL()
        print(f"[OK] BurbulatorDLL loaded library from {burb.path}")

        # Moderate-sized test to keep runtime reasonable
        Nspecies = 1
        Nchannels = 2
        N_ph_max = 50000

        sim = burb.simulate_ov3(
            Nspecies=Nspecies,
            M=[50.0],
            D=[3.0],
            Nchannels=Nchannels,
            q=[50.0, 50.0],
            q_bg=[0.001, 0.001],
            k_rad=[1.0],
            k_nrad=[0.0],
            box_xy=2.0,
            box_z=4.0,
            focus_type=0,
            focus_param=[0.3, 2.0],
            dt=0.01,
            N_ph_max=N_ph_max,
        )
        N_ph = sim.get("N_ph", 0)
        print(f"[OK] simulate_ov3 returned {N_ph} photons")
        if N_ph <= 0:
            print("[FAIL] simulate_ov3 returned no photons, skipping SPC test")
            return False

        # Convert to raw SPC-130 records (no header)
        ch_conversion = [8, 0, 9, 1, 10, 2]
        N_tac_channels = 4096
        tac_dt = 0.004069
        laser_period = 13.596

        spc_bytes, MT_ov, spc_i = burb.convert_to_spc132(
            pulsed_exc=0,
            Nchannels=Nchannels,
            data_T=sim["data_T"],
            data_t=sim["data_t"],
            data_N=sim["data_N"],
            data_species=sim["data_species"],
            data_molecule=sim["data_molecule"],
            tw=0.01,
            ch_conversion=ch_conversion,
            N_tac_channels=N_tac_channels,
            tac_dt=tac_dt,
            laser_period=laser_period,
            N_photons=N_ph,
        )
        print(f"[OK] convert_to_spc132 returned {spc_i} bytes, MT_ov={MT_ov}")
        if not spc_bytes:
            print("[FAIL] convert_to_spc132 produced no data")
            return False

        # Write a single SPC-132 file with header and validate with tttrlib
        spc_filename = "wrapper_test.spc"
        burb.write_spc132_file(spc_filename, spc_bytes, macro_time_clock=100)
        print(f"[OK] Wrote SPC-132 file {spc_filename}")

        if not TTTRLIB_AVAILABLE or np is None:
            print("[SKIP] tttrlib or numpy not available, skipping validation of wrapper SPC file")
            return True

        try:
            tttr_data = tttrlib.TTTR(spc_filename, 'SPC-130')
            macro_times = tttr_data.macro_times
            n_photons = len(macro_times)
            if n_photons == 0:
                print(f"[FAIL] Wrapper SPC file {spc_filename} contains no photons")
                return False
            print(f"[OK] Wrapper SPC file {spc_filename}: {n_photons} photons, macro_times range {macro_times.min()} - {macro_times.max()}")
        except Exception as e:
            print(f"[FAIL] Could not validate wrapper SPC file with tttrlib: {e}")
            traceback.print_exc()
            return False

        return True
    except BurbulatorError as e:
        print(f"[FAIL] Wrapper SPC pipeline failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Burbulator DLL Debug Test Script")
    print("=" * 40)

    if not test_wrapper_basic():
        print("Wrapper basic test failed")
        return

    if not test_wrapper_spc_file():
        print("Wrapper SPC file test failed")
        return

    print("\n" + "=" * 40)
    print("Test script completed successfully")

if __name__ == "__main__":
    main()
