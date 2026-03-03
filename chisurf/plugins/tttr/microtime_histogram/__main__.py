#!/usr/bin/env python

import argparse
import os
import sys
import json
import pathlib
from pathlib import Path
import numpy as np

from qtpy import QtWidgets, QtCore
from chisurf.gui import get_app

from .wizard import MicrotimeHistogram


def main():
    """
    Command-line entry point for the microtime_histogram plugin.
    
    This function parses command-line arguments and launches the microtime_histogram plugin
    with the specified BID folder and setup.
    
    Usage:
        csc_microtime_histogram --bid-folder /path/to/bid/folder [--setup-name SETUP_NAME]
    """
    parser = argparse.ArgumentParser(
        description='Process BID files and create microtime histograms.'
    )
    
    parser.add_argument(
        '--bid-folder',
        type=str,
        help='Path to the folder containing BID/BUR/BST files'
    )
    
    parser.add_argument(
        '--setup-name',
        type=str,
        help='Name of the setup to use (if not provided, will try to read from photon_selection_parameters.json)'
    )
    
    parser.add_argument(
        '--auto-transfer',
        action='store_true',
        help='Automatically transfer the histogram to ChiSurf'
    )
    
    args = parser.parse_args()
    
    # Create Qt application
    app = get_app()
    
    # Get or create the microtime histogram widget using the singleton pattern
    widget = MicrotimeHistogram.get_instance()
    widget.show()
    widget.raise_()  # Bring window to front
    
    # If a BID folder is specified, process it
    if args.bid_folder:
        bid_folder = Path(args.bid_folder)
        if not bid_folder.exists():
            print(f"Error: BID folder '{bid_folder}' does not exist")
            sys.exit(1)
        
        # Try to read setup name from photon_selection_parameters.json if not provided
        setup_name = args.setup_name
        if not setup_name:
            # Look for photon_selection_parameters.json in the Info folder
            # First check if there's an Info folder in the parent directory
            info_folder = bid_folder.parent / "Info"
            if not info_folder.exists():
                # Try looking for Info folder in the grandparent directory
                info_folder = bid_folder.parent.parent / "Info"
            
            if info_folder.exists():
                params_file = info_folder / "photon_selection_parameters.json"
                if params_file.exists():
                    try:
                        with open(params_file, 'r') as f:
                            params = json.load(f)
                            setup_name = params.get("selected_setup")
                            if setup_name:
                                print(f"Found setup name '{setup_name}' in photon_selection_parameters.json")
                    except Exception as e:
                        print(f"Error reading photon_selection_parameters.json: {e}")
        
        # Process the BID folder
        # We need to use QtCore.QTimer.singleShot to ensure the widget is fully initialized
        def process_bid_folder():
            # If setup name is provided, select it
            if setup_name:
                # Find the index of the setup in the combobox
                index = widget.setup_selection_combobox.findText(setup_name)
                if index >= 0:
                    widget.setup_selection_combobox.setCurrentIndex(index)
                    print(f"Selected setup: {setup_name}")
                else:
                    print(f"Warning: Setup '{setup_name}' not found in available setups")
            
            # Find all .bst files in the BID folder
            bst_files = list(bid_folder.glob("*.bst"))
            if not bst_files:
                print(f"Warning: No .bst files found in {bid_folder}")
            
            # Add the files to the widget
            for bst_file in bst_files:
                widget.listWidget_BID.add_file(str(bst_file))
            
            # Compute the microtime histogram
            widget.compute_microtime_histogram()
            
            # If auto-transfer is enabled, transfer the histogram to ChiSurf
            if args.auto_transfer:
                widget.add_to_chisurf()
        
        # Use a timer to ensure the widget is fully initialized before processing
        QtCore.QTimer.singleShot(500, process_bid_folder)
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
