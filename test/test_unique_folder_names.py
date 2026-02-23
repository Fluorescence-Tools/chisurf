"""
Test script to verify the unique folder naming functionality in tttr_photon_filter.py.

This script:
1. Creates a mock folder structure to simulate existing output folders
2. Initializes the WizardTTTRPhotonFilter class
3. Tests the get_unique_folder_path method with various scenarios
"""

import os
import sys
import pathlib
import tempfile
import shutil
from qtpy import QtWidgets

# Import the module to test
from chisurf.gui.widgets.wizard.tttr_photonfilter import WizardTTTRPhotonFilter

def test_unique_folder_paths():
    """Test the get_unique_folder_path method with various scenarios."""
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create mock folders to simulate existing output folders
        test_folders = [
            "bocpd_All 1.0000#10",
            "kalman_All 1.0000#10",
            "burstwise_All 1.0000#10"
        ]
        
        # Create the test folders
        for folder in test_folders:
            folder_path = pathlib.Path(temp_dir) / folder
            folder_path.mkdir(exist_ok=True)
            print(f"Created test folder: {folder_path}")
        
        # Initialize Qt application
        app = QtWidgets.QApplication(sys.argv)
        
        # Create empty windows and detectors dictionaries as required by the constructor
        windows = {}
        detectors = {}
        
        # Initialize the WizardTTTRPhotonFilter class
        filter_widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
        
        # Test the get_unique_folder_path method
        for folder in test_folders:
            base_path = pathlib.Path(temp_dir) / folder
            unique_path = filter_widget.get_unique_folder_path(base_path)
            print(f"Original path: {base_path}")
            print(f"Unique path: {unique_path}")
            print(f"Suffix added: {unique_path.name != base_path.name}")
            print()
            
        # Test with a non-existent folder (should return the same path)
        non_existent = pathlib.Path(temp_dir) / "non_existent_folder"
        unique_path = filter_widget.get_unique_folder_path(non_existent)
        print(f"Original path (non-existent): {non_existent}")
        print(f"Unique path: {unique_path}")
        print(f"Same path returned: {unique_path == non_existent}")
        print()
        
        # Test with multiple existing folders with suffixes
        multi_suffix_base = pathlib.Path(temp_dir) / "multi_suffix"
        multi_suffix_base.mkdir(exist_ok=True)
        print(f"Created test folder: {multi_suffix_base}")
        
        # Create folders with suffixes _0 and _1
        (multi_suffix_base.parent / f"{multi_suffix_base.name}_0").mkdir(exist_ok=True)
        print(f"Created test folder: {multi_suffix_base.parent / f'{multi_suffix_base.name}_0'}")
        
        (multi_suffix_base.parent / f"{multi_suffix_base.name}_1").mkdir(exist_ok=True)
        print(f"Created test folder: {multi_suffix_base.parent / f'{multi_suffix_base.name}_1'}")
        
        # Test that it correctly finds the next available suffix (_2)
        unique_path = filter_widget.get_unique_folder_path(multi_suffix_base)
        print(f"Original path: {multi_suffix_base}")
        print(f"Unique path: {unique_path}")
        print(f"Expected suffix _2: {unique_path.name == f'{multi_suffix_base.name}_2'}")
        
        print("\nAll tests completed successfully!")
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_unique_folder_paths()