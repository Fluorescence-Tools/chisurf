#!/usr/bin/env python3
"""
Simple Standalone Simulation Runner (No GUI Dependencies)

This is a simplified version that directly calls the Burbulator DLL
without requiring PyQt5 or the full sm_acquisition plugin.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add paths
chisurf_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(chisurf_root))

# Import matplotlib (non-GUI backend)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available, plots will be skipped")
    plt = None

# Import the Burbulator DLL wrapper directly
sys.path.insert(0, str(Path(__file__).parent.parent / 'tcspc_devices' / 'simulation'))
from burbulator_dll_wrapper import BurbulatorDLL


def run_simulation(params, output_dir, logger=None):
    """Run simulation using Burbulator DLL directly."""
    
    try:
        dll = BurbulatorDLL()
    except Exception as e:
        print(f"ERROR: Failed to load Burbulator DLL: {e}")
        return None
    
    print(f"Simulation parameters:")
    for k, v in params.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{v[0]}, ..., {v[-1]}] ({len(v)} items)")
        else:
            print(f"  {k}: {v}")
    
    # Generate photons
    print(f"\nGenerating {params['N_ph_max']:,} photons...")
    
    # Convert parameter names to match DLL API
    dll_params = {
        'Nspecies': params.get('N_species', 1),
        'M': params.get('M', [50.0]),
        'D': params.get('D', [3.0]),
        'Nchannels': params.get('N_channels', 2),
        'q': params.get('q', [50.0, 50.0]),
        'q_bg': params.get('q_bg', [0.001, 0.001]),
        'k_rad': params.get('k_rad', [1.0]),
        'k_nrad': params.get('k_nrad', [0.0]),
        'box_xy': params.get('box_xy', 2.0),
        'box_z': params.get('box_z', 4.0),
        'focus_type': params.get('focus_type', 0),
        'focus_param': params.get('focus_param', [0.3, 2.0]),
        'dt': params.get('dt', 0.01),
        'N_ph_max': params.get('N_ph_max', 1000),
        'rmt1seed': params.get('rmt1seed', 12345),
        'rmt2seed': params.get('rmt2seed', 54321),
    }
    
    result = dll.simulate_ov3(**dll_params)
    
    if result is None:
        print("ERROR: Simulation failed")
        return None
    
    # Extract from dict
    n_photons = result['N_ph']
    data_T = result['data_T']
    data_t = result['data_t']
    data_N = result['data_N']
    data_species = result['data_species']
    data_molecule = result['data_molecule']
    T0 = result['T0']
    Nmolecules = result['Nmolecules']
    
    print(f"Generated {n_photons:,} photons from {Nmolecules} molecules (T0={T0})")
    
    # Convert to SPC format and write files
    spc_output_path = str(output_dir)
    params['spc_output_path'] = spc_output_path
    
    n_per_file = params.get('N_ph_per_file', 100000)
    n_files = (n_photons + n_per_file - 1) // n_per_file
    
    print(f"\nWriting {n_files} SPC files...")
    
    for file_idx in range(n_files):
        start_idx = file_idx * n_per_file
        end_idx = min(start_idx + n_per_file, n_photons)
        
        batch_size = end_idx - start_idx
        
        # Get batch data
        batch_T = data_T[start_idx:end_idx]
        batch_t = data_t[start_idx:end_idx]
        batch_N = data_N[start_idx:end_idx]
        batch_species = data_species[start_idx:end_idx]
        batch_molecule = data_molecule[start_idx:end_idx]
        
        # Convert to SPC
        spc_data, MT_ov, spc_i = dll.convert_to_spc132(
            pulsed_exc=params.get('pulsed_exc', 0),
            Nchannels=params.get('N_channels', 2),
            tw=params.get('dt', 0.01),
            N_tac_channels=params.get('N_tac_channels', 4096),
            tac_dt=params.get('tac_dt', 0.004069),
            laser_period=params.get('laser_period', 13.596),
            spc_data_bytes_per_photon=8,
            data_T=batch_T,
            data_t=batch_t,
            data_N=batch_N,
            data_species=batch_species,
            data_molecule=batch_molecule,
            ch_conversion=params.get('ch_conversion', [8, 0, 9, 1, 10, 2]),
            F=None,
            lookup=None,
            N_photons=batch_size
        )
        
        # Write SPC file
        filename = output_dir / f"m{file_idx:03d}.spc"
        with open(filename, 'wb') as f:
            f.write(spc_data)
        
        print(f"  Wrote {filename.name}: {len(spc_data):,} bytes ({batch_size:,} photons)")
    
    print(f"\nAll files written to: {output_dir}")
    
    return {
        'n_photons': n_photons,
        'n_molecules': Nmolecules,
        'n_files': n_files,
        'data_T': data_T,
        'data_t': data_t,
        'data_N': data_N
    }


def plot_results(data, params, output_dir):
    """Generate plots from simulation data."""
    
    if plt is None:
        print("Skipping plots (matplotlib not available)")
        return
    
    print("\nGenerating plots...")
    
    # Extract data
    microtimes = data['data_t']
    channels = data['data_N']
    n_tac = params.get('N_tac_channels', 4096)
    tac_dt = params.get('tac_dt', 0.004069)
    
    # Create decay histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TCSPC Decay Histograms', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for ch in range(4):
        ax = axes[ch]
        ch_mask = (channels == ch)
        ch_microtimes = microtimes[ch_mask]
        
        if len(ch_microtimes) > 0:
            # Convert to ns
            microtime_bins = np.arange(0, n_tac)
            time_ns = microtime_bins * tac_dt
            
            # Histogram
            counts, _ = np.histogram(ch_microtimes, bins=n_tac, range=(0, n_tac))
            
            ax.semilogy(time_ns, counts + 1, '-', linewidth=0.5)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Channel {ch} (N={len(ch_microtimes):,})')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.5)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Channel {ch}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decay_histograms.png', dpi=150)
    plt.close()
    print("  Saved decay_histograms.png")
    
    # Channel distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_ch, counts_ch = np.unique(channels, return_counts=True)
    ax.bar(unique_ch, counts_ch)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Photon Count')
    ax.set_title('Photon Distribution by Channel')
    ax.grid(True, alpha=0.3, axis='y')
    for ch, count in zip(unique_ch, counts_ch):
        ax.text(ch, count, f'{count:,}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_distribution.png', dpi=150)
    plt.close()
    print("  Saved channel_distribution.png")


def save_summary(data, params, output_dir):
    """Save simulation summary."""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in params.items()},
        'results': {
            'total_photons': int(data['n_photons']),
            'n_molecules': int(data['n_molecules']),
            'n_files': int(data['n_files'])
        },
        'channel_statistics': {}
    }
    
    channels = data['data_N']
    for ch in range(4):
        ch_count = int((channels == ch).sum())
        summary['channel_statistics'][ch] = {
            'photon_count': ch_count,
            'fraction': ch_count / data['n_photons'] if data['n_photons'] > 0 else 0
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("  Saved summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Simple SM Acquisition Simulation (No GUI dependencies)"
    )
    
    parser.add_argument('--n-photons', '-n', type=int, default=1000000,
                       help='Total photons to generate (default: 1,000,000)')
    parser.add_argument('--photons-per-file', '-p', type=int, default=100000,
                       help='Photons per SPC file (default: 100,000)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--diffusion', '-D', type=float, default=3.0,
                       help='Diffusion coefficient (μm²/s)')
    parser.add_argument('--brightness', '-q', type=float, default=50.0,
                       help='Molecular brightness (photons/molecule/μs)')
    parser.add_argument('--n-molecules', '-M', type=float, default=50.0,
                       help='Initial number of molecules')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"simulation_output_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SIMPLE SM ACQUISITION SIMULATION")
    print("=" * 70)
    print(f"Output: {output_dir.absolute()}")
    print(f"Photons: {args.n_photons:,}")
    print(f"Files: {(args.n_photons + args.photons_per_file - 1) // args.photons_per_file}")
    print("=" * 70)
    
    # Setup simulation parameters
    params = {
        'N_species': 1,
        'M': [args.n_molecules],
        'D': [args.diffusion],
        'N_channels': 2,
        'q': [args.brightness, args.brightness],
        'q_bg': [0.001, 0.001],
        'k_rad': [1.0],
        'k_nrad': [0.0],
        'box_xy': 2.0,
        'box_z': 4.0,
        'focus_type': 0,
        'focus_param': [0.3, 2.0],
        'dt': 0.01,
        'N_ph_max': args.n_photons,
        'N_ph_per_file': args.photons_per_file,
        'pulsed_exc': 0,
        'ch_conversion': [8, 0, 9, 1, 10, 2],
        'N_tac_channels': 4096,
        'tac_dt': 0.004069,
        'laser_period': 13.596,
        'rmt1seed': 12345,
        'rmt2seed': 54321
    }
    
    try:
        # Run simulation
        data = run_simulation(params, output_dir)
        
        if data is None:
            return 1
        
        # Generate plots
        if not args.no_plots:
            plot_results(data, params, output_dir)
        
        # Save summary
        save_summary(data, params, output_dir)
        
        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print(f"Output files in: {output_dir.absolute()}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
