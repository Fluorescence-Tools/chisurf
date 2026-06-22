#!/usr/bin/env python3
"""
Headless SM Acquisition Simulation Runner

This script runs photon simulations without the GUI and generates comprehensive output:
- Multiple SPC files based on N_ph_per_file batching
- Decay histograms (TCSPC) for each channel
- FCS correlation curves (auto and cross)
- Count rate traces
- Summary statistics

Usage:
    python run_simulation.py --n-photons 1000000 --output-dir ./sim_output
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Add chisurf path
chisurf_root = Path(__file__).parent.parent.parent.parent.parent
if str(chisurf_root) not in sys.path:
    sys.path.insert(0, str(chisurf_root))

# Import after path setup
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)


def setup_logging(output_dir, debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    log_file = Path(output_dir) / 'simulation.log'
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)


def create_simulation_device(params):
    """Create and configure simulation device."""
    # Import simulation wrapper bypassing the sm_acquisition __init__.py
    # which has PyQt/pyqtgraph dependencies
    import importlib
    
    # Temporarily mock PyQt5 modules to allow import
    import sys
    from types import ModuleType
    
    # Create mock modules
    mock_pyqt = ModuleType('PyQt5')
    mock_pyqt.QtCore = ModuleType('PyQt5.QtCore')
    mock_pyqt.QtWidgets = ModuleType('PyQt5.QtWidgets')
    mock_pyqt.QtCore.pyqtSignal = type('pyqtSignal', (), {})
    mock_pyqt.QtCore.QObject = type('QObject', (), {})
    
    sys.modules['PyQt5'] = mock_pyqt
    sys.modules['PyQt5.QtCore'] = mock_pyqt.QtCore
    sys.modules['PyQt5.QtWidgets'] = mock_pyqt.QtWidgets
    sys.modules['pyqtgraph'] = ModuleType('pyqtgraph')
    
    # Now import wrapper
    from chisurf.plugins._dev.photon_acquisition.tcspc_devices.simulation.wrapper import SimulationDevice
    
    device = SimulationDevice()
    
    # Update parameters
    device.simulation_params.update(params)
    
    return device


def run_simulation(device, logger):
    """Run the simulation and collect photon data."""
    logger.info("Initializing simulation device...")
    if not device.initialize(simulation=True):
        logger.error("Failed to initialize simulation device")
        return None
    
    logger.info("Starting measurement...")
    if not device.start_measurement():
        logger.error("Failed to start measurement")
        return None
    
    # Collect photon data
    logger.info("Collecting photon data...")
    all_photons = []
    all_microtimes = []
    all_channels = []
    
    batch_count = 0
    total_photons = 0
    
    while True:
        # Read FIFO data
        data = device.read_fifo(max_records=10000)
        
        if data is None or len(data) == 0:
            logger.info("End of data stream")
            break
        
        # Process photons using device's photon processor
        if hasattr(device, '_process_photons'):
            photons, microtimes, channels, overflow_count = device._process_photons(data)
            
            if len(photons) > 0:
                all_photons.extend(photons)
                all_microtimes.extend(microtimes)
                all_channels.extend(channels)
                total_photons += len(photons)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Collected {total_photons:,} photons in {batch_count} batches")
        else:
            # Fallback: just count records
            total_photons += len(data)
            batch_count += 1
            
        # Small delay to prevent tight loop
        time.sleep(0.01)
    
    logger.info(f"Simulation complete: {total_photons:,} total photons collected")
    
    device.stop_measurement()
    
    if total_photons == 0:
        logger.warning("No photons collected!")
        return None
    
    return {
        'photons': np.array(all_photons, dtype=np.int64),
        'microtimes': np.array(all_microtimes, dtype=np.uint16),
        'channels': np.array(all_channels, dtype=np.uint8),
        'total_count': total_photons
    }


def compute_decay_histogram(microtimes, channels, n_channels=4, n_bins=4096):
    """Compute decay histograms for each channel."""
    histograms = {}
    
    for ch in range(n_channels):
        ch_mask = (channels == ch)
        ch_microtimes = microtimes[ch_mask]
        
        if len(ch_microtimes) > 0:
            hist, bins = np.histogram(ch_microtimes, bins=n_bins, range=(0, n_bins))
            histograms[ch] = {
                'counts': hist,
                'bins': bins[:-1],
                'total': len(ch_microtimes)
            }
        else:
            histograms[ch] = {
                'counts': np.zeros(n_bins),
                'bins': np.arange(n_bins),
                'total': 0
            }
    
    return histograms


def compute_correlation(photons, channels, channel_pairs=None, n_casc=25, n_bins=3):
    """
    Compute FCS correlation curves using tttrlib if available, or skip.
    
    Args:
        photons: Array of macrotimes
        channels: Array of channel indices
        channel_pairs: List of (ch1, ch2) tuples for correlation. If None, compute all auto/cross.
        n_casc: Number of cascades
        n_bins: Bins per cascade
    """
    try:
        # Try to import tttrlib Correlator
        import sys
        from pathlib import Path
        tttrlib_path = Path(__file__).parent.parent.parent.parent / 'modules' / 'tttrlib' / 'ext' / 'python'
        if str(tttrlib_path) not in sys.path:
            sys.path.insert(0, str(tttrlib_path))
        
        import _tttrlib
        from Correlator import Correlator
        
        if channel_pairs is None:
            # Auto-correlations for channels 0 and 1
            channel_pairs = [(0, 0), (1, 1), (0, 1)]
        
        results = {}
        
        for ch1, ch2 in channel_pairs:
            # Extract photons for each channel
            mask1 = (channels == ch1)
            mask2 = (channels == ch2)
            
            stream1 = photons[mask1].astype(np.int64)
            stream2 = photons[mask2].astype(np.int64)
            
            if len(stream1) < 10 or len(stream2) < 10:
                continue
            
            # Compute correlation using tttrlib
            try:
                correlator = Correlator(
                    n_casc=n_casc,
                    n_bins=n_bins,
                    macro_times=(stream1, stream2)
                )
                
                taus = correlator.x_axis
                correlation = correlator.correlation
                
                pair_name = f"G{ch1}{ch2}" if ch1 == ch2 else f"G{ch1}{ch2}_cross"
                results[pair_name] = {
                    'taus': taus,
                    'correlation': correlation,
                    'n1': len(stream1),
                    'n2': len(stream2)
                }
            except Exception as e:
                print(f"Warning: Correlation failed for channels {ch1},{ch2}: {e}")
        
        return results
        
    except ImportError as e:
        print(f"Warning: tttrlib not available, skipping correlation calculation: {e}")
        return {}


def compute_count_rate(photons, channels, bin_width_ms=10, n_channels=4):
    """
    Compute count rate traces.
    
    Args:
        photons: Array of macrotimes (in units from device, typically 0.1 ms)
        channels: Array of channel indices
        bin_width_ms: Binning width in milliseconds
    """
    if len(photons) == 0:
        return {}
    
    # Convert macrotimes to milliseconds (assuming macrotime unit is ~0.1 ms)
    macrotime_unit_ms = 0.1  # Typical for BH SPC
    photons_ms = photons * macrotime_unit_ms
    
    # Compute time bins
    t_min = photons_ms.min()
    t_max = photons_ms.max()
    duration_ms = t_max - t_min
    
    if duration_ms <= 0:
        return {}
    
    n_bins = int(np.ceil(duration_ms / bin_width_ms))
    bins = np.linspace(t_min, t_max, n_bins + 1)
    
    results = {}
    
    for ch in range(n_channels):
        ch_mask = (channels == ch)
        ch_photons = photons_ms[ch_mask]
        
        if len(ch_photons) > 0:
            counts, _ = np.histogram(ch_photons, bins=bins)
            # Convert to kHz
            count_rate_khz = counts / (bin_width_ms / 1000.0) / 1000.0
            time_s = (bins[:-1] - t_min) / 1000.0  # Convert to seconds
            
            results[ch] = {
                'time': time_s,
                'rate': count_rate_khz,
                'mean_rate': np.mean(count_rate_khz),
                'total_counts': len(ch_photons)
            }
    
    return results


def plot_decay_histograms(histograms, output_file, params):
    """Plot and save decay histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TCSPC Decay Histograms', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    tac_dt = params.get('tac_dt', 0.004069)  # ns per bin
    
    for ch in range(4):
        ax = axes[ch]
        
        if ch in histograms and histograms[ch]['total'] > 0:
            bins = histograms[ch]['bins']
            counts = histograms[ch]['counts']
            time_ns = bins * tac_dt
            
            ax.semilogy(time_ns, counts + 1, 'o-', markersize=2, linewidth=0.5)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Channel {ch} (Total: {histograms[ch]["total"]:,})')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0.5)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Channel {ch} (No photons)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved decay histograms to {output_file}")


def plot_correlations(correlations, output_file, params):
    """Plot and save correlation curves."""
    if not correlations:
        print("No correlations to plot")
        return
    
    n_corr = len(correlations)
    fig, axes = plt.subplots(1, n_corr, figsize=(5 * n_corr, 4))
    fig.suptitle('FCS Correlation Curves', fontsize=14, fontweight='bold')
    
    if n_corr == 1:
        axes = [axes]
    
    dt = params.get('dt', 0.01)  # Diffusion timestep in μs
    
    for ax, (name, data) in zip(axes, correlations.items()):
        taus = data['taus']
        corr = data['correlation']
        
        # Convert tau to microseconds (taus are in units of dt)
        taus_us = taus * dt
        
        ax.semilogx(taus_us, corr, 'o-', markersize=3, linewidth=1)
        ax.set_xlabel('Lag time τ (μs)')
        ax.set_ylabel('G(τ)')
        ax.set_title(f'{name} (N1={data["n1"]}, N2={data["n2"]})')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlations to {output_file}")


def plot_count_rates(count_rates, output_file):
    """Plot and save count rate traces."""
    if not count_rates:
        print("No count rate data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Count Rate Traces', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for ch in range(4):
        ax = axes[ch]
        
        if ch in count_rates:
            data = count_rates[ch]
            ax.plot(data['time'], data['rate'], '-', linewidth=0.8)
            ax.axhline(y=data['mean_rate'], color='r', linestyle='--', 
                      label=f'Mean: {data["mean_rate"]:.2f} kHz')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Count Rate (kHz)')
            ax.set_title(f'Channel {ch} (Total: {data["total_counts"]:,} photons)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Count Rate (kHz)')
            ax.set_title(f'Channel {ch}')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved count rates to {output_file}")


def save_summary(data, histograms, correlations, count_rates, params, output_file):
    """Save simulation summary as JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'statistics': {
            'total_photons': int(data['total_count']) if data else 0,
            'channels': {}
        }
    }
    
    # Add channel statistics
    if data:
        for ch in range(4):
            ch_mask = (data['channels'] == ch)
            n_photons = int(ch_mask.sum())
            
            summary['statistics']['channels'][ch] = {
                'photon_count': n_photons,
                'fraction': n_photons / data['total_count'] if data['total_count'] > 0 else 0
            }
            
            if ch in count_rates:
                summary['statistics']['channels'][ch]['mean_count_rate_kHz'] = float(count_rates[ch]['mean_rate'])
    
    # Add correlation info
    if correlations:
        summary['correlations'] = {}
        for name, corr_data in correlations.items():
            summary['correlations'][name] = {
                'n_photons_1': int(corr_data['n1']),
                'n_photons_2': int(corr_data['n2']),
                'correlation_points': len(corr_data['taus'])
            }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Headless SM Acquisition Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 1 million photons, 100k per file
  python run_simulation.py --n-photons 1000000 --photons-per-file 100000
  
  # Quick test with 10k photons
  python run_simulation.py --n-photons 10000 --output-dir ./test_sim
  
  # Custom diffusion and brightness
  python run_simulation.py --n-photons 500000 --diffusion 5.0 --brightness 100
        """
    )
    
    parser.add_argument(
        '--n-photons', '-n',
        type=int,
        default=1000000,
        help='Total number of photons to generate (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--photons-per-file', '-p',
        type=int,
        default=100000,
        help='Photons per SPC output file (default: 100,000)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--diffusion', '-D',
        type=float,
        default=3.0,
        help='Diffusion coefficient in μm²/s (default: 3.0)'
    )
    
    parser.add_argument(
        '--brightness', '-q',
        type=float,
        default=50.0,
        help='Molecular brightness in photons/molecule/μs (default: 50.0)'
    )
    
    parser.add_argument(
        '--n-molecules', '-M',
        type=float,
        default=50.0,
        help='Initial number of molecules (default: 50)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (faster)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"simulation_output_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.debug)
    
    logger.info("=" * 70)
    logger.info("SM ACQUISITION HEADLESS SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Total photons: {args.n_photons:,}")
    logger.info(f"Photons per file: {args.photons_per_file:,}")
    logger.info(f"Expected files: {(args.n_photons + args.photons_per_file - 1) // args.photons_per_file}")
    logger.info("-" * 70)
    
    # Configure simulation parameters
    params = {
        'N_ph_max': args.n_photons,
        'N_ph_per_file': args.photons_per_file,
        'spc_output_path': str(output_dir.absolute()),
        'D': [args.diffusion],
        'q': [args.brightness, args.brightness],
        'M': [args.n_molecules],
    }
    
    try:
        # Create simulation device
        logger.info("Creating simulation device...")
        device = create_simulation_device(params)
        
        # Run simulation
        start_time = time.time()
        data = run_simulation(device, logger)
        elapsed_time = time.time() - start_time
        
        if data is None:
            logger.error("Simulation failed to produce data")
            return 1
        
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Throughput: {data['total_count'] / elapsed_time:.0f} photons/second")
        
        # Generate plots and analysis
        if not args.no_plots:
            logger.info("Generating analysis and plots...")
            
            # Decay histograms
            logger.info("Computing decay histograms...")
            histograms = compute_decay_histogram(
                data['microtimes'], 
                data['channels'],
                n_bins=device.simulation_params.get('N_tac_channels', 4096)
            )
            plot_decay_histograms(histograms, output_dir / 'decay_histograms.png', device.simulation_params)
            
            # Correlations
            logger.info("Computing FCS correlations...")
            correlations = compute_correlation(data['photons'], data['channels'])
            if correlations:
                plot_correlations(correlations, output_dir / 'correlations.png', device.simulation_params)
            
            # Count rates
            logger.info("Computing count rate traces...")
            count_rates = compute_count_rate(data['photons'], data['channels'])
            if count_rates:
                plot_count_rates(count_rates, output_dir / 'count_rates.png')
        else:
            histograms = {}
            correlations = {}
            count_rates = {}
        
        # Save summary
        save_summary(data, histograms, correlations, count_rates, 
                    device.simulation_params, output_dir / 'summary.json')
        
        logger.info("=" * 70)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"Output files in: {output_dir.absolute()}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
