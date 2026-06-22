#!/usr/bin/env python3
"""
CLI for SM Acquisition GUI with debugging support.

This script provides a command-line interface to launch the SM Acquisition GUI
with enhanced debugging and parameter configuration options.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add chisurf path
chisurf_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chisurf_path))

def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sm_acquisition_debug.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def create_standalone_app():
    """Create a standalone PyQt application for SM Acquisition."""
    from qtpy.QtWidgets import QApplication, QMainWindow, QMdiArea
    from qtpy.QtCore import Qt

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("SM Acquisition")
    app.setApplicationVersion("2.0")

    # Create main window
    main_window = QMainWindow()
    main_window.setWindowTitle("SM Acquisition - Standalone")
    main_window.resize(1400, 900)

    # Create MDI area
    mdi_area = QMdiArea()
    mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    main_window.setCentralWidget(mdi_area)
    main_window.mdiarea = mdi_area

    return app, main_window

def launch_sm_acquisition(args):
    """Launch SM Acquisition with given arguments."""
    logger = setup_logging(args.debug)

    try:
        # Import required modules
        logger.info("Loading SM Acquisition plugin...")

        if args.standalone:
            logger.info("Running in standalone mode")
            app, main_window = create_standalone_app()

            # Import and initialize SM Acquisition
            from chisurf.plugins._dev.photon_acquisition.main import SMAcquisitionManager
            manager = SMAcquisitionManager(standalone_main_window=main_window)

        else:
            logger.info("Running in chisurf mode")
            # Try to import chisurf
            try:
                import chisurf
                chisurf.run()
                return
            except ImportError as e:
                logger.error(f"chisurf not available: {e}")
                logger.info("Falling back to standalone mode")
                app, main_window = create_standalone_app()

                # Import and initialize SM Acquisition
                from chisurf.plugins._dev.photon_acquisition.main import SMAcquisitionManager
                manager = SMAcquisitionManager(standalone_main_window=main_window)

        # Configure simulation parameters if provided
        if hasattr(manager, 'device') and hasattr(manager.device, 'simulation_params'):
            if args.n_photons:
                manager.device.simulation_params['N_ph_max'] = args.n_photons
                logger.info(f"Set N_ph_max to {args.n_photons}")

            if args.photons_per_file:
                manager.device.simulation_params['N_ph_per_file'] = args.photons_per_file
                logger.info(f"Set N_ph_per_file to {args.photons_per_file}")

            if args.output_dir:
                manager.device.simulation_params['spc_output_path'] = args.output_dir
                logger.info(f"Set output path to {args.output_dir}")

            if args.species:
                manager.device.simulation_params['N_species'] = args.species
                logger.info(f"Set N_species to {args.species}")

        # Show debug info
        logger.info("SM Acquisition initialized successfully")
        logger.info("Available device types: BH SPC 830, PicoQuant, Simulation")
        logger.info("Select 'Simulation' from device dropdown, then 'Init Device' to start")

        if args.standalone:
            # Start the application
            logger.info("Starting GUI application...")
            sys.exit(app.exec())
        else:
            # chisurf handles the main loop
            pass

    except Exception as e:
        logger.error(f"Failed to launch SM Acquisition: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SM Acquisition GUI with debugging support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with debug logging
  python cli_sm_acquisition.py --debug

  # Launch in standalone mode
  python cli_sm_acquisition.py --standalone

  # Configure simulation parameters
  python cli_sm_acquisition.py --n-photons 1000000 --photons-per-file 100000 --output-dir ./simulation_data

  # Launch with chisurf integration
  python cli_sm_acquisition.py
        """
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--standalone', '-s',
        action='store_true',
        help='Run in standalone mode without chisurf'
    )

    parser.add_argument(
        '--n-photons', '-n',
        type=int,
        help='Number of photons to generate (N_ph_max)'
    )

    parser.add_argument(
        '--photons-per-file', '-p',
        type=int,
        help='Photons per output file (N_ph_per_file)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for SPC files'
    )

    parser.add_argument(
        '--species', '-S',
        type=int,
        default=1,
        help='Number of molecular species (default: 1)'
    )

    args = parser.parse_args()

    print("SM Acquisition CLI")
    print("=" * 50)
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Standalone mode: {'ON' if args.standalone else 'OFF'}")
    print()

    launch_sm_acquisition(args)

if __name__ == '__main__':
    main()
