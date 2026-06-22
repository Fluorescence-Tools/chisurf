#!/usr/bin/env python3
"""
Test Streaming Simulation CLI

Tests the new queue-based streaming architecture WITHOUT GUI.
This is a 1:1 correspondence to what the GUI does.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add chisurf to path
chisurf_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(chisurf_root))

# Import the simulation device directly
from chisurf.plugins._dev.photon_acquisition.tcspc_devices.simulation.wrapper import SimulationDevice


def test_streaming_simulation(n_ph_max=1000000, n_ph_per_file=100000, output_dir=None):
    """Test streaming simulation architecture (same as GUI).
    
    Args:
        n_ph_max: Total photons to generate
        n_ph_per_file: Photons per SPC file
        output_dir: Output directory for SPC files
    """
    
    print("=" * 70)
    print("STREAMING SIMULATION TEST (CLI)")
    print("=" * 70)
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"simulation_output_cli_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize simulation device
    print("\n1. Initializing simulation device...")
    device = SimulationDevice()
    
    if not device.initialized:
        print("ERROR: Failed to initialize simulation device")
        return False
    
    print("   ✓ Device initialized")
    print(f"   ✓ DLL available: {device.simulator.dll_available}")
    
    # Configure simulation parameters
    print("\n2. Configuring simulation parameters...")
    device.simulation_params['N_ph_max'] = n_ph_max
    device.simulation_params['N_ph_per_file'] = n_ph_per_file
    device.simulation_params['spc_output_path'] = output_dir
    
    print(f"   ✓ Total photons: {n_ph_max:,}")
    print(f"   ✓ Photons per file: {n_ph_per_file:,}")
    print(f"   ✓ Expected files: {n_ph_max // n_ph_per_file}")
    
    # Start measurement (starts background generation)
    print("\n3. Starting measurement (background generation)...")
    start_time = time.time()
    
    success = device.start_measurement()
    if not success:
        print("ERROR: Failed to start measurement")
        return False
    
    print("   ✓ Background generation thread started")
    
    # Read data from queue (simulates GUI read_fifo loop)
    print("\n4. Reading data from queue (streaming)...")
    print("   (This simulates what the GUI does in its acquisition loop)")
    
    photon_count = 0
    batch_count = 0
    last_log_time = time.time()
    
    while device.measurement_running:
        # Read from FIFO (pulls from queue)
        data = device.read_fifo(max_words=32768)
        
        if len(data) > 0:
            photon_count += len(data)
            batch_count += 1
            
            # Log progress every second
            current_time = time.time()
            if current_time - last_log_time >= 1.0:
                elapsed = current_time - start_time
                rate = photon_count / elapsed if elapsed > 0 else 0
                progress = (photon_count / n_ph_max) * 100 if n_ph_max > 0 else 0
                print(f"   Batch {batch_count:3d}: {photon_count:8,} photons "
                      f"({progress:5.1f}%) @ {rate:,.0f} photons/s")
                last_log_time = current_time
        else:
            # No data yet, wait briefly
            time.sleep(0.01)
        
        # Check if generation is done
        if not device.measurement_running:
            break
    
    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total photons read: {photon_count:,}")
    print(f"Total batches: {batch_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average rate: {photon_count / elapsed:,.0f} photons/second")
    
    # Check output files
    print(f"\n5. Checking output files in {output_dir}...")
    spc_files = sorted(Path(output_dir).glob("m*.spc"))
    print(f"   ✓ Found {len(spc_files)} SPC files:")
    
    total_size = 0
    for spc_file in spc_files:
        size = spc_file.stat().st_size
        total_size += size
        print(f"      {spc_file.name}: {size:,} bytes")
    
    print(f"   ✓ Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    # Cleanup
    device.close()
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test streaming simulation (CLI equivalent of GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10k photons)
  python test_streaming_sim.py --n-ph-max 10000 --n-ph-per-file 5000
  
  # Full simulation (1M photons, default)
  python test_streaming_sim.py
  
  # Custom output directory
  python test_streaming_sim.py --output-dir my_simulation
        """
    )
    
    parser.add_argument(
        '--n-ph-max',
        type=int,
        default=1000000,
        help='Total number of photons to generate (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--n-ph-per-file',
        type=int,
        default=100000,
        help='Number of photons per SPC file (default: 100,000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for SPC files (default: auto-generated)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (10k photons, 5k per file)'
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        print("QUICK TEST MODE: 10,000 photons\n")
        args.n_ph_max = 10000
        args.n_ph_per_file = 5000
    
    # Run test
    try:
        success = test_streaming_simulation(
            n_ph_max=args.n_ph_max,
            n_ph_per_file=args.n_ph_per_file,
            output_dir=args.output_dir
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
