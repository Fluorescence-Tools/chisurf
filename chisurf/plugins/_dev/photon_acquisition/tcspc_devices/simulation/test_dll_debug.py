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
    from chisurf.plugins._dev.photon_acquisition.tcspc_devices.simulation.burbulator_dll_wrapper import (
        BurbulatorDLL,
        BurbulatorError,
    )
    WRAPPER_AVAILABLE = True
except Exception:
    BurbulatorDLL = None  # type: ignore
    BurbulatorError = RuntimeError  # type: ignore
    WRAPPER_AVAILABLE = False
    print("[WARNING] BurbulatorDLL wrapper not available for testing")

def test_dll_loading():
    """Test basic DLL loading."""
    print("=== Testing DLL Loading ===")

    dll_path = r"e:/dev/tttrlib/playground/Burbulator/burbulator_x64.dll"
    print(f"DLL path: {dll_path}")
    print(f"DLL exists: {os.path.exists(dll_path)}")

    try:
        dll = ct.windll.LoadLibrary(dll_path)
        print("[OK] DLL loaded successfully")

        # Test function existence
        try:
            smdif_ov3 = dll.smdif_ov3
            print("[OK] smdif_ov3 function found")
        except AttributeError:
            print("[FAIL] smdif_ov3 function not found")
            return False

        return dll
    except Exception as e:
        print(f"[FAIL] DLL loading failed: {e}")
        traceback.print_exc()
        return None

def test_function_prototype(dll):
    """Test setting up function prototype."""
    print("\n=== Testing Function Prototype ===")

    try:
        # smdif_ov3 function signature (31 parameters)
        dll.smdif_ov3.argtypes = [
            ct.c_int,        # N_species
            ct.POINTER(ct.c_double),  # M
            ct.POINTER(ct.c_double),  # D
            ct.c_int,        # N_channels
            ct.POINTER(ct.c_double),  # q
            ct.POINTER(ct.c_double),  # q_bg
            ct.POINTER(ct.c_double),  # k_rad
            ct.POINTER(ct.c_double),  # k_nrad
            ct.c_double,     # box_xy
            ct.c_double,     # box_z
            ct.c_int,        # focus_type
            ct.POINTER(ct.c_double),  # focus_param
            ct.c_double,     # dt
            ct.c_int,        # N_ph_max
            ct.POINTER(ct.c_uint),    # data_T
            ct.POINTER(ct.c_double),  # data_t
            ct.POINTER(ct.c_short),   # data_N
            ct.POINTER(ct.c_short),   # data_species
            ct.POINTER(ct.c_int),     # data_molecule
            ct.POINTER(ct.c_uint),    # T0
            ct.POINTER(ct.c_int),     # Nmolecules
            ct.POINTER(ct.c_double),  # x
            ct.POINTER(ct.c_double),  # y
            ct.POINTER(ct.c_double),  # z
            ct.POINTER(ct.c_short),   # species
            ct.c_int,        # rmt1seed
            ct.POINTER(ct.c_uint),    # rmt1state
            ct.POINTER(ct.c_int),     # rmt1left
            ct.c_int,        # rmt2seed
            ct.POINTER(ct.c_uint),    # rmt2state
            ct.POINTER(ct.c_int),     # rmt2left
        ]
        dll.smdif_ov3.restype = ct.c_int
        print("[OK] Function prototype set successfully")

        # Add data2spc132_tac function prototype
        dll.data2spc132_tac.argtypes = [
            ct.c_int,        # pulsed_exc
            ct.c_int,        # N_channels
            ct.POINTER(ct.c_uint),    # data_T
            ct.POINTER(ct.c_double),  # data_t
            ct.POINTER(ct.c_short),   # data_N
            ct.POINTER(ct.c_short),   # data_species
            ct.POINTER(ct.c_int),     # data_molecule
            ct.c_uint,      # N_photons
            ct.c_double,    # tw (dt)
            ct.POINTER(ct.c_ushort),  # ch_conversion
            ct.c_int,       # N_tac_channels
            ct.c_double,    # tac_dt
            ct.c_double,    # laser_period
            ct.POINTER(ct.c_double),  # F (can be None)
            ct.POINTER(ct.c_int),     # lookup (can be None)
            ct.POINTER(ct.c_byte),    # spc_data (output)
            ct.POINTER(ct.c_ulong),   # MT_ov
            ct.POINTER(ct.c_ulong),   # spc_i
            ct.POINTER(ct.c_uint),    # rmt2state
            ct.POINTER(ct.c_int),     # rmt2left
        ]
        dll.data2spc132_tac.restype = ct.c_int
        print("[OK] data2spc132_tac prototype set successfully")

        return True

    except Exception as e:
        print(f"[FAIL] Function prototype setup failed: {e}")
        traceback.print_exc()
        return False

def test_simple_call(dll):
    """Test a simple DLL call with minimal parameters."""
    print("\n=== Testing Simple DLL Call ===")

    try:
        # Minimal parameters for testing
        Nspecies = 1
        Nchannels = 2
        N_ph_max = 5  # Very small number for testing

        # Arrays
        M = (ct.c_double * 1)(100.0)
        D = (ct.c_double * 1)(1.0)
        q = (ct.c_double * 2)(0.1, 0.1)
        q_bg = (ct.c_double * 2)(0.001, 0.001)
        k_rad = (ct.c_double * 1)(0.0)
        k_nrad = (ct.c_double * 1)(0.0)
        focus_param = (ct.c_double * 2)(0.3, 2.0)

        # Calculate array sizes exactly like C# code
        sum_M = sum(int(float(m)) for m in M)
        array_size = sum_M * 2 + 50
        print(f"sum_M = {sum_M}, array_size = {array_size}")
        print(f"Output arrays sized for {N_ph_max * 2} photons each")

        # Output arrays - size them like C# does: N_ph_max * 2
        data_T = (ct.c_uint * (N_ph_max * 2))()
        data_t = (ct.c_double * (N_ph_max * 2))()
        data_N = (ct.c_short * (N_ph_max * 2))()
        data_species = (ct.c_short * (N_ph_max * 2))()
        data_molecule = (ct.c_int * (N_ph_max * 2))()

        # Molecule state arrays
        x = (ct.c_double * array_size)()
        y = (ct.c_double * array_size)()
        z = (ct.c_double * array_size)()
        species = (ct.c_short * array_size)()

        # RNG state (initialize with zeros)
        T0 = ct.c_uint(0)
        Nmolecules = ct.c_int(0)

        rmt1seed = 12345
        rmt1state = (ct.c_uint * 624)(0)  # Mersenne Twister state array - correct size
        rmt1left = ct.c_int(0)

        rmt2seed = 54321
        rmt2state = (ct.c_uint * 624)(0)  # Mersenne Twister state array - correct size
        rmt2left = ct.c_int(0)

        print(f"Calling DLL with N_ph_max = {N_ph_max}")

        # Call the function
        result = dll.smdif_ov3(
            Nspecies, M, D, Nchannels, q, q_bg, k_rad, k_nrad,
            2.0, 4.0, 0, focus_param, 0.01, N_ph_max,
            data_T, data_t, data_N, data_species, data_molecule,
            ct.byref(T0), ct.byref(Nmolecules),
            x, y, z, species,
            rmt1seed, rmt1state, ct.byref(rmt1left),
            rmt1seed, rmt2state, ct.byref(rmt2left)  # Both use rmt1seed like C#
        )

        print(f"[OK] DLL call succeeded, returned: {result}")
        print(f"T0 = {T0.value}, Nmolecules = {Nmolecules.value}")

        # Print first few photons (safely bounded)
        max_photons_to_show = min(10, result, N_ph_max * 2)
        photon_count = 0
        for i in range(max_photons_to_show):
            if i < len(data_T) and data_T[i] > 0:
                print(f"Photon {photon_count}: T={data_T[i]}, t={data_t[i]}, N={data_N[i]}, species={data_species[i]}, molecule={data_molecule[i]}")
                photon_count += 1

        return True

    except Exception as e:
        print(f"[FAIL] DLL call failed: {e}")
        traceback.print_exc()
        return False

def test_minimal_call(dll):
    """Test a minimal DLL call with the smallest possible parameters."""
    print("\n=== Testing Minimal DLL Call ===")

    try:
        # Minimal parameters that should work
        Nspecies = 1
        Nchannels = 1  # Single channel
        N_ph_max = 1   # Just 1 photon

        # Arrays
        M = (ct.c_double * 1)(1.0)  # 1 molecule
        D = (ct.c_double * 1)(0.0)  # No diffusion
        q = (ct.c_double * 1)(1.0)  # High brightness
        q_bg = (ct.c_double * 1)(0.0)  # No background
        k_rad = (ct.c_double * 1)(0.0)
        k_nrad = (ct.c_double * 1)(0.0)
        focus_param = (ct.c_double * 2)(1.0, 1.0)

        # Calculate array sizes
        sum_M = sum(int(float(m)) for m in M)
        array_size = sum_M * 2 + 50
        print(f"Minimal test: sum_M = {sum_M}, array_size = {array_size}")

        # Output arrays - minimal size
        data_T = (ct.c_uint * 2)()
        data_t = (ct.c_double * 2)()
        data_N = (ct.c_short * 2)()
        data_species = (ct.c_short * 2)()
        data_molecule = (ct.c_int * 2)()

        # Molecule state arrays
        x = (ct.c_double * array_size)()
        y = (ct.c_double * array_size)()
        z = (ct.c_double * array_size)()
        species = (ct.c_short * array_size)()

        # RNG state
        T0 = ct.c_uint(0)
        Nmolecules = ct.c_int(0)
        rmt1seed = 12345
        rmt1state = (ct.c_uint * 624)(0)
        rmt1left = ct.c_int(0)
        rmt2seed = 54321
        rmt2state = (ct.c_uint * 624)(0)
        rmt2left = ct.c_int(0)

        print("Calling DLL with minimal parameters...")

        # Call the function
        result = dll.smdif_ov3(
            Nspecies, M, D, Nchannels, q, q_bg, k_rad, k_nrad,
            1.0, 1.0, 0, focus_param, 1.0, N_ph_max,  # Simple geometry and timing
            data_T, data_t, data_N, data_species, data_molecule,
            ct.byref(T0), ct.byref(Nmolecules),
            x, y, z, species,
            rmt1seed, rmt1state, ct.byref(rmt1left),
            rmt1seed, rmt2state, ct.byref(rmt2left)  # Both use rmt1seed like C#
        )

        print(f"[OK] Minimal call succeeded, returned: {result}")
        print(f"T0 = {T0.value}, Nmolecules = {Nmolecules.value}")

        return True

    except Exception as e:
        print(f"[FAIL] Minimal call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def test_parameter_variations(dll):
    """Test different parameter combinations."""
    print("\n=== Testing Parameter Variations ===")

    # Test one case at a time to isolate issues
    test_cases = [
        {"N_ph_max": 1, "M": [1.0], "name": "Single molecule, 1 photon"},
        {"N_ph_max": 10, "M": [10.0], "name": "10 molecules, 10 photons"},
        {"N_ph_max": 100, "M": [50.0], "name": "50 molecules, 100 photons"},
    ]

    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")

        try:
            Nspecies = 1
            Nchannels = 2
            N_ph_max = test_case['N_ph_max']

            # Arrays
            M_val = test_case['M'][0]
            M = (ct.c_double * 1)(M_val)
            D = (ct.c_double * 1)(1.0)
            q = (ct.c_double * 2)(0.1, 0.1)
            q_bg = (ct.c_double * 2)(0.001, 0.001)
            k_rad = (ct.c_double * 1)(0.0)
            k_nrad = (ct.c_double * 1)(0.0)
            focus_param = (ct.c_double * 2)(0.3, 2.0)

            # Calculate array sizes
            sum_M = sum(int(float(m)) for m in M)
            array_size = sum_M * 2 + 50

            # Output arrays
            data_T = (ct.c_uint * (N_ph_max * 2))()
            data_t = (ct.c_double * (N_ph_max * 2))()
            data_N = (ct.c_short * (N_ph_max * 2))()
            data_species = (ct.c_short * (N_ph_max * 2))()
            data_molecule = (ct.c_int * (N_ph_max * 2))()

            # Molecule state arrays
            x = (ct.c_double * array_size)()
            y = (ct.c_double * array_size)()
            z = (ct.c_double * array_size)()
            species = (ct.c_short * array_size)()

            # RNG state
            T0 = ct.c_uint(0)
            Nmolecules = ct.c_int(0)
            rmt1seed = 12345
            rmt1state = (ct.c_uint * 624)(0)  # Mersenne Twister state array
            rmt1left = ct.c_int(0)
            rmt2seed = 54321
            rmt2state = (ct.c_uint * 624)(0)  # Mersenne Twister state array
            rmt2left = ct.c_int(0)

            result = dll.smdif_ov3(
                Nspecies, M, D, Nchannels, q, q_bg, k_rad, k_nrad,
                2.0, 4.0, 0, focus_param, 0.01, N_ph_max,
                data_T, data_t, data_N, data_species, data_molecule,
                ct.byref(T0), ct.byref(Nmolecules),
                x, y, z, species,
                rmt1seed, rmt1state, ct.byref(rmt1left),
                rmt1seed, rmt2state, ct.byref(rmt2left)  # Both use rmt1seed like C#
            )

            print(f"[OK] Success: returned {result}, Nmolecules={Nmolecules.value}")

        except Exception as e:
            print(f"[FAIL] Failed: {e}")

def test_format_verification(dll):
    """Test that simulation output matches BH_SPC format."""
    print("\n=== Testing BH_SPC Format Verification ===")

    try:
        # Generate some photons
        Nspecies = 1
        Nchannels = 2
        N_ph_max = 5

        # Arrays
        M = (ct.c_double * 1)(10.0)
        D = (ct.c_double * 1)(1.0)
        q = (ct.c_double * 2)(0.1, 0.1)
        q_bg = (ct.c_double * 2)(0.001, 0.001)
        k_rad = (ct.c_double * 1)(0.0)
        k_nrad = (ct.c_double * 1)(0.0)
        focus_param = (ct.c_double * 2)(0.3, 2.0)

        sum_M = sum(int(float(m)) for m in M)
        array_size = sum_M * 2 + 50

        # Output arrays
        data_T = (ct.c_uint * (N_ph_max * 2))()
        data_t = (ct.c_double * (N_ph_max * 2))()
        data_N = (ct.c_short * (N_ph_max * 2))()
        data_species = (ct.c_short * (N_ph_max * 2))()
        data_molecule = (ct.c_int * (N_ph_max * 2))()

        # Molecule state arrays
        x = (ct.c_double * array_size)()
        y = (ct.c_double * array_size)()
        z = (ct.c_double * array_size)()
        species = (ct.c_short * array_size)()

        # RNG state
        T0 = ct.c_uint(0)
        Nmolecules = ct.c_int(0)
        rmt1seed = 12345
        rmt1state = (ct.c_uint * 624)(0)
        rmt1left = ct.c_int(0)
        rmt2seed = 54321
        rmt2state = (ct.c_uint * 624)(0)
        rmt2left = ct.c_int(0)

        # Debug: print all parameters passed to smdif_ov3
        print("\n[DEBUG TEST] smdif_ov3 parameters:")
        print(f"  Nspecies    = {Nspecies}")
        print(f"  M           = {[float(m) for m in M]}")
        print(f"  D           = {[float(d) for d in D]}")
        print(f"  Nchannels   = {Nchannels}")
        print(f"  q           = {[float(v) for v in q]}")
        print(f"  q_bg        = {[float(v) for v in q_bg]}")
        print(f"  k_rad       = {[float(v) for v in k_rad]}")
        print(f"  k_nrad      = {[float(v) for v in k_nrad]}")
        print(f"  box_xy      = {2.0}")
        print(f"  box_z       = {4.0}")
        print(f"  focus_type  = {0}")
        print(f"  focus_param = {[float(v) for v in focus_param]}")
        print(f"  dt          = {0.01}")
        print(f"  N_ph_max    = {N_ph_max}")
        print(f"  rmt1seed    = {rmt1seed}")
        print(f"  rmt2seed    = {rmt2seed}")

        # Get photons from simulation
        N_ph = dll.smdif_ov3(
            Nspecies, M, D, Nchannels, q, q_bg, k_rad, k_nrad,
            2.0, 4.0, 0, focus_param, 0.01, N_ph_max,
            data_T, data_t, data_N, data_species, data_molecule,
            ct.byref(T0), ct.byref(Nmolecules), x, y, z, species,
            rmt1seed, rmt1state, ct.byref(rmt1left),
            rmt1seed, rmt2state, ct.byref(rmt2left)  # Both use rmt1seed like C#
        )

        print(f"Generated {N_ph} photons from simulation")

        # Debug: show first few raw simulation data
        print(f"DEBUG TEST: First few data_T: {data_T[:5] if N_ph > 0 else 'None'}")
        print(f"DEBUG TEST: First few data_t: {data_t[:5] if N_ph > 0 else 'None'}")
        print(f"DEBUG TEST: First few data_N: {data_N[:5] if N_ph > 0 else 'None'}")
        if N_ph > 0:
            data_N_arr = np.array(data_N[:N_ph], dtype=int)
            dn_min = int(data_N_arr.min())
            dn_max = int(data_N_arr.max())
            dn_unique = np.unique(data_N_arr)
        else:
            dn_min = dn_max = "N/A"
            dn_unique = "N/A"

        print(
            f"DEBUG TEST: data_N range: min={dn_min}, max={dn_max}, unique={dn_unique}"
        )

        # Convert to BH_SPC format using data2spc132_tac (proper way)
        pulsed_exc = 0  # CW excitation
        ch_conversion = (ct.c_ushort * 6)(8, 0, 9, 1, 10, 2)  # Match C# code exactly
        N_tac_channels = 4096
        tac_dt = 0.004069  # ns per TAC channel
        laser_period = 13.596  # ns

        # IRF parameters (not used for CW, but provide properly sized dummy arrays)
        irf_size = Nspecies * Nchannels * N_tac_channels  # Required size for pulsed excitation
        F = (ct.c_double * irf_size)(0.0)  # Dummy array for CW excitation
        lookup = (ct.c_int * irf_size)(0)  # Dummy array for CW excitation

        # Output buffer - allocate extra space for potential overflow records
        spc_data_size = N_ph_max * 8  # 8 bytes per photon to account for overflows
        spc_data = (ct.c_byte * spc_data_size)()

        # Conversion parameters
        MT_ov = ct.c_ulong(0)
        spc_i = ct.c_ulong(0)

        print(f"[DEBUG] About to call data2spc132_tac with N_ph={N_ph}")
        print(f"[DEBUG] Array sizes: data_T={len(data_T)}, data_t={len(data_t)}, data_N={len(data_N)}")
        print(f"[DEBUG] spc_data_size={spc_data_size} (8 bytes/photon for overflows), irf_size={irf_size}")
        print(f"[DEBUG] First few data values: data_T[0]={data_T[0]}, data_t[0]={data_t[0]}, data_N[0]={data_N[0]}")

        # Call conversion function
        dll_conversion_success = False
        try:
            result = dll.data2spc132_tac(
                pulsed_exc, Nchannels, data_T, data_t, data_N, data_species, data_molecule,
                N_ph, 0.01, ch_conversion, N_tac_channels, tac_dt, laser_period,
                F, lookup, spc_data, ct.byref(MT_ov), ct.byref(spc_i),
                rmt2state, ct.byref(rmt2left)
            )
            if result == 1:
                dll_conversion_success = True
                print(f"[OK] DLL conversion succeeded, {spc_i.value} bytes written")
            else:
                print(f"[FAIL] DLL conversion failed with code {result}, falling back to manual conversion")
        except Exception as e:
            print(f"[CRASH] DLL conversion crashed ({e}), falling back to manual conversion")

        if dll_conversion_success:
            # Convert byte array to uint32 array
            n_records = min(spc_i.value // 4, spc_data_size // 4)  # Safety check
            records = []
            for i in range(n_records):
                # Reconstruct 32-bit photon record from 4 bytes (little-endian)
                byte0 = spc_data[i*4] & 0xFF
                byte1 = spc_data[i*4 + 1] & 0xFF  
                byte2 = spc_data[i*4 + 2] & 0xFF
                byte3 = spc_data[i*4 + 3] & 0xFF
                photon_record = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
                records.append(photon_record)

            print(f"DEBUG TEST: Converted to {len(records)} records")
            print(f"DEBUG TEST: First few records: {records[:5] if len(records) > 0 else 'None'}")

            # Debug: manual decode verification
            if len(records) > 0:
                print(f"DEBUG TEST: Manual decode verification:")
                for i in range(min(3, len(records))):
                    photon_record = records[i]
                    byte0 = spc_data[i*4] & 0xFF
                    byte1 = spc_data[i*4 + 1] & 0xFF
                    byte2 = spc_data[i*4 + 2] & 0xFF
                    byte3 = spc_data[i*4 + 3] & 0xFF
                    reconstructed = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
                    match = "✓" if photon_record == reconstructed else "✗"
                    print(f"  Record {i}: 0x{photon_record:08X} == 0x{reconstructed:08X} {match}")
        else:
            # Fallback to manual BH_SPC conversion
            print(f"Using manual BH_SPC conversion for {N_ph} photons")
            records = []
            for i in range(N_ph):
                tac_time = int(data_t[i] * 4096) % 4096  # Convert to 12-bit microtime
                channel = int(data_N[i]) & 0xFF  # Channel number
                # Map simulation channels to BH SPC channels
                if channel < len(ch_conversion):
                    channel = ch_conversion[channel]
                macrotime_diff = i * 10  # Simple macrotime difference

                # Create BH_SPC format photon record
                photon_record = (tac_time << 16) | (channel << 8) | (macrotime_diff & 0xFF)
                records.append(photon_record)

            print(f"DEBUG TEST: Fallback converted to {len(records)} records")
            print(f"DEBUG TEST: First few records: {records[:5] if len(records) > 0 else 'None'}")

        print(f"[OK] Converted {N_ph} photons to {len(records)} BH_SPC records")

        # Verify BH_SPC format from the properly converted records
        for i, record in enumerate(records[:5]):  # Show first 5
            # Extract components using BH_SPC bit layout
            byte0 = record & 0xFF
            byte1 = (record >> 8) & 0xFF
            byte2 = (record >> 16) & 0xFF
            byte3 = (record >> 24) & 0xFF

            if byte3 >= 0xC0:
                # Overflow record
                overflow_count = byte3 - 0xC0
                mt_overflow = (overflow_count << 24) | (byte2 << 16) | (byte1 << 8) | byte0
                print(f"Photon {i}: record=0x{record:08X} -> OVERFLOW: {mt_overflow}")
            else:
                # Regular photon record
                microtime = (record >> 16) & 0xFFF  # Bits 27-16
                channel_extracted = (record >> 8) & 0xFF  # Bits 15-8
                macrotime_low = record & 0xFF  # Bits 7-0

                print(f"Photon {i}: record=0x{record:08X}")
                print(f"  Extracted: microtime={microtime}, channel={channel_extracted}, macrotime_low={macrotime_low}")
                if dll_conversion_success:
                    print(f"  Raw data: t={data_t[i]:.6f}, N={data_N[i]}, T={data_T[i]}")
                else:
                    print(f"  Manual conversion: tac_time={int(data_t[i] * 4096) % 4096}, channel={int(data_N[i])}")

        conversion_type = "DLL" if dll_conversion_success else "manual"
        print(f"[OK] BH_SPC format verification completed (using {conversion_type} conversion)")
        return True

    except Exception as e:
        print(f"[FAIL] Format verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversion_and_spc_file(dll):
    """Test full conversion pipeline and write SPC file with 100000 photons."""
    print("\n=== Testing Conversion and SPC File Writing ===")

    try:
        # Use same parameter structure as C# RunOpenVDll
        Nspecies = 1  # Number of fluorescent species (1 = single fluorophore)
        Nchannels = 2  # Number of detection channels (typically 2 for anisotropy)
        N_ph_max_perfile = 300000  # Photons per batch (like C# N_ph_max_perfile, scaled up for efficiency)
        N_ph_max = 1000  # Total photons target - reduced for testing

        # Arrays (same as C# defaults)
        M = (ct.c_double * 1)(50.0)  # 50 molecules - controls photon generation rate
        D = (ct.c_double * 1)(3.0)   # Diffusion coefficient (μm²/μs)
        q = (ct.c_double * 2)(50.0, 50.0)      # Quantum yield per channel [parallel, perpendicular]
        q_bg = (ct.c_double * 2)(0.001, 0.001)  # Background quantum yield per channel
        k_rad = (ct.c_double * 1)(1.0)       # Radiative decay rate (0 = use natural lifetime)
        k_nrad = (ct.c_double * 1)(0.0)      # Non-radiative decay rate (0 = none)
        focus_param = (ct.c_double * 2)(0.3, 2.0)  # Focus parameters [waist_xy, waist_z] in μm

        # Calculate array sizes like C#
        sum_M = sum(int(float(m)) for m in M)
        array_size = sum_M * 2 + 50

        # Output arrays sized for per-file batch
        data_T = (ct.c_uint * (N_ph_max_perfile * 2))()
        data_t = (ct.c_double * (N_ph_max_perfile * 2))()
        data_N = (ct.c_short * (N_ph_max_perfile * 2))()
        data_species = (ct.c_short * (N_ph_max_perfile * 2))()
        data_molecule = (ct.c_int * (N_ph_max_perfile * 2))()

        # Molecule state arrays
        x = (ct.c_double * array_size)()
        y = (ct.c_double * array_size)()
        z = (ct.c_double * array_size)()
        species = (ct.c_short * array_size)()

        # RNG state
        T0 = ct.c_uint(0)
        Nmolecules = ct.c_int(0)
        rmt1seed = 12345
        rmt1state = (ct.c_uint * 624)(0)
        rmt1left = ct.c_int(0)
        rmt2state = (ct.c_uint * 624)(0)
        rmt2left = ct.c_int(0)

        # Conversion setup (same as C#)
        pulsed_exc = 0  # CW excitation
        ch_conversion = (ct.c_ushort * 6)(8, 0, 9, 1, 10, 2)  # Match C# code exactly
        N_tac_channels = 4096
        tac_dt = 0.004069  # ns per TAC channel
        laser_period = 13.596  # ns

        # IRF parameters (not used for CW, but provide properly sized dummy arrays)
        irf_size = Nspecies * Nchannels * N_tac_channels
        F = (ct.c_double * irf_size)(0.0)
        lookup = (ct.c_int * irf_size)(0)

        # Output buffer sized for batch like C#
        Bytes_perfile = 4 * N_ph_max_perfile
        spc_data = (ct.c_byte * (Bytes_perfile * 4))()

        # Tracking variables like C#
        N_ph_generated = 0
        filenumber = 0

        # Accumulated macrotime overflows and SPC write index (persist across batches)
        MT_ov = ct.c_ulong(0)
        spc_i = ct.c_ulong(0)

        # BH SPC-132 header (bh_spc132_header_t) written at start of each file
        # macro_time_clock in units of 1/10 ns; choose a plausible value (<500) so
        # tttrlib recognizes the file as BH132 when auto-detecting.
        macro_time_clock = 100
        bh_spc132_header_bytes = bytes((
            macro_time_clock & 0xFF,
            (macro_time_clock >> 8) & 0xFF,
            (macro_time_clock >> 16) & 0xFF,
            0x80  # invalid=1, unused=0
        ))

        print(f"Starting batch generation: target {N_ph_max} photons, batch size {N_ph_max_perfile}")

        # Main generation loop (like C# Run method)
        while N_ph_generated < N_ph_max:
            batch_size = min(N_ph_max_perfile, N_ph_max - N_ph_generated)
            print(f"Generating batch {filenumber}: {batch_size} photons (total so far: {N_ph_generated})")

            # Generate photon batch
            N_ph_last = dll.smdif_ov3(
                Nspecies, M, D, Nchannels, q, q_bg, k_rad, k_nrad,
                2.0, 4.0, 0, focus_param, 0.01, batch_size,  # focus_type = 0 (3D Gaussian, uniform CEF)
                data_T, data_t, data_N, data_species, data_molecule,
                ct.byref(T0), ct.byref(Nmolecules), x, y, z, species,
                rmt1seed, rmt1state, ct.byref(rmt1left),
                rmt1seed, rmt2state, ct.byref(rmt2left)  # Both RNGs use rmt1seed like C#
            )

            N_ph_generated += N_ph_last
            rmt1seed = -1  # Like C# code - reset seeds between batches
            rmt2seed = -1  # Also reset rmt2seed

            print(f"Generated {N_ph_last} photons in this batch")

            if N_ph_last == 0:
                print("[WARNING] No photons generated in batch, stopping")
                break

            # Convert to BH_SPC format
            result = dll.data2spc132_tac(
                pulsed_exc, Nchannels, data_T, data_t, data_N, data_species, data_molecule,
                N_ph_last, 0.01, ch_conversion, N_tac_channels, tac_dt, laser_period,
                F, lookup, spc_data, ct.byref(MT_ov), ct.byref(spc_i),
                rmt2state, ct.byref(rmt2left)
            )
            if result == 1:
                dll_conversion_success = True
                print(f"[OK] DLL conversion succeeded, {spc_i.value} bytes written")
            else:
                print(f"[FAIL] DLL conversion failed with code {result}")
                dll_conversion_success = False

            if dll_conversion_success:
                # Write SPC files in chunks of Bytes_perfile, shifting remaining bytes
                # This mirrors the C# RunOpenVDll.Run implementation, but we also
                # prepend a 4-byte BH SPC-132 header for tttrlib compatibility.
                while spc_i.value >= Bytes_perfile:
                    spc_filename = f"m{filenumber:03d}.spc"  # Match C# naming
                    with open(spc_filename, 'wb') as f:
                        f.write(bh_spc132_header_bytes)
                        file_bytes = bytes((int(b) & 0xFF) for b in spc_data[:Bytes_perfile])
                        f.write(file_bytes)
                    print(f"[OK] Wrote header + {Bytes_perfile} bytes to {spc_filename}")

                    # Shift remaining bytes down in the buffer
                    remaining = spc_i.value - Bytes_perfile
                    if remaining > 0:
                        for i in range(remaining):
                            spc_data[i] = spc_data[Bytes_perfile + i]
                    spc_i.value -= Bytes_perfile
                    filenumber += 1
            else:
                print(f"[FAIL] Skipping file write for batch {filenumber}")

            # Safety check to prevent infinite loop
            if filenumber > 20:  # Should not need more than 10 files for 100k photons
                print("[ERROR] Too many batches generated, stopping")
                break

        # After loop, write final partial SPC file if there are remaining bytes
        if dll_conversion_success and spc_i.value > 0:
            spc_filename = f"m{filenumber:03d}.spc"
            with open(spc_filename, 'wb') as f:
                f.write(bh_spc132_header_bytes)
                remaining_bytes = bytes((int(b) & 0xFF) for b in spc_data[:spc_i.value])
                f.write(remaining_bytes)
            print(f"[OK] Wrote header + final {spc_i.value} bytes to {spc_filename}")
            filenumber += 1

        print(f"\n[FINAL] Generated {N_ph_generated} photons total across {filenumber} files")

        # Verify results
        if N_ph_generated > 0:
            # Check the last file
            last_filename = f"m{filenumber-1:03d}.spc"
            if os.path.exists(last_filename):
                file_size = os.path.getsize(last_filename)
                print(f"[OK] Last file {last_filename} exists with {file_size} bytes")
            else:
                print(f"[WARNING] Last file {last_filename} not found")

        print("[OK] Batch conversion and SPC file writing test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Conversion and SPC file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spc_file_validation():
    """Test that all generated SPC files can be loaded by tttrlib."""
    print("\n=== Testing SPC File Validation with tttrlib ===")

    if not TTTRLIB_AVAILABLE:
        print("[SKIP] tttrlib not available, skipping validation")
        return True

    if np is None:
        print("[SKIP] numpy not available, skipping validation")
        return True

    # Find all generated .spc files
    spc_files = sorted([f for f in os.listdir('.') if f.startswith('m') and f.endswith('.spc')])
    if not spc_files:
        print("[FAIL] No SPC files found to validate")
        return False

    print(f"Found {len(spc_files)} SPC files to validate")

    failed_files = []
    total_photons = 0

    for spc_file in spc_files:
        try:
            print(f"Testing {spc_file}...")
            # Load the file with tttrlib
            # Explicitly specify BH SPC-130 container type to avoid unreliable
            # auto-detection on header-less simulation files.
            tttr_data = tttrlib.TTTR(spc_file, 'SPC-130')

            # Check that we can access macro_times
            macro_times = tttr_data.macro_times
            n_photons = len(macro_times)
            total_photons += n_photons

            # Require that each SPC file actually contains photons
            if n_photons == 0:
                print(f"  [FAIL] {spc_file}: no photons found in macro_times")
                failed_files.append((spc_file, "empty macro_times"))
                continue

            print(f"  [OK] {spc_file}: {n_photons} photons, macro_times range: {macro_times.min()} - {macro_times.max()}")

            # Basic validation - macro_times should be monotonically increasing
            if n_photons > 1 and not np.all(macro_times[1:] >= macro_times[:-1]):
                print(f"  [WARNING] {spc_file}: macro_times not monotonically increasing")
                failed_files.append((spc_file, "non-monotonic macro_times"))

        except Exception as e:
            print(f"  [FAIL] {spc_file}: {e}")
            failed_files.append((spc_file, str(e)))

    if failed_files:
        print(f"\n[FAIL] {len(failed_files)} out of {len(spc_files)} files failed validation:")
        for file, error in failed_files:
            print(f"  - {file}: {error}")
        return False
    else:
        print(f"\n[OK] All {len(spc_files)} files validated successfully")
        print(f"    Total photons across all files: {total_photons}")
        return True


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

    # Test DLL loading
    dll = test_dll_loading()
    if not dll:
        print("Cannot continue without DLL")
        return

    # Test function prototype
    if not test_function_prototype(dll):
        print("Cannot continue without proper prototype")
        return

    # Test simple call
    if not test_minimal_call(dll):
        print("Basic call failed, trying minimal test...")
        if not test_minimal_call(dll):
            print("Minimal call also failed, testing parameter variations...")

    # Test parameter variations
    test_parameter_variations(dll)

    # Test format verification
    test_format_verification(dll)

    # Test conversion and SPC file writing (raw DLL)
    if not test_conversion_and_spc_file(dll):
        print("File generation failed")
        return

    # Validate generated files with tttrlib (raw DLL)
    if not test_spc_file_validation():
        print("File validation failed")
        return

    # Test the high-level BurbulatorDLL wrapper as used by the SM acquisition plugin
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
