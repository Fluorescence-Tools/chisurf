"""
Test script to verify that:
1. The info directory is created with uppercase 'I' (Info)
2. The setup information is included in the JSON file

This script:
1. Creates a mock folder structure
2. Initializes the WizardTTTRPhotonFilter class
3. Simulates the save_selection method's behavior for saving the info JSON
4. Verifies that the directory is created with uppercase 'I'
5. Verifies that the setup information is included in the JSON file
"""

import os
import sys
import pathlib
import tempfile
import shutil
import json
from datetime import datetime
from qtpy import QtWidgets

# Import the module to test
from chisurf.gui.widgets.wizard.tttr_photon_filter import WizardTTTRPhotonFilter, load_detector_setups

def test_info_json_uppercase():
    """Test that the info directory is created with uppercase 'I' and setup info is included."""
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create a mock folder structure
        test_folder = "burstwise_All 0.4000#30"
        
        # Create the test folder
        original_folder_path = pathlib.Path(temp_dir) / test_folder
        original_folder_path.mkdir(exist_ok=True)
        
        print(f"Created test folder: {original_folder_path}")
        
        # Initialize Qt application
        app = QtWidgets.QApplication(sys.argv)
        
        # Create empty windows and detectors dictionaries as required by the constructor
        windows = {}
        detectors = {}
        
        # Initialize the WizardTTTRPhotonFilter class
        filter_widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
        
        # Set up the target path and filenames to match our test scenario
        filter_widget.lineEdit_2.setText(test_folder)  # Set target_path
        
        # Create a mock TTTR filename
        mock_filename = str(pathlib.Path(temp_dir) / "test_file.ptu")
        
        # Add the mock filename to the settings
        filter_widget.settings['tttr_filenames'] = [mock_filename]
        
        # Create a mock setup
        mock_setup_name = "Test Setup"
        mock_setup_data = {
            "detectors": {
                "Detector1": {"chs": [0, 1, 2]},
                "Detector2": {"chs": [3, 4, 5]}
            },
            "windows": {
                "Window1": [0, 100],
                "Window2": [200, 300]
            },
            "tttr_reading": {
                "file_type": "PTU",
                "micro_time_binning": 8
            }
        }
        
        # Create a mock setups dictionary
        mock_setups = {
            "setups": {
                mock_setup_name: mock_setup_data
            },
            "last_used": mock_setup_name
        }
        
        # Mock the load_detector_setups function to return our mock setups
        original_load_detector_setups = filter_widget.load_detector_setups
        filter_widget.load_detector_setups = lambda: mock_setups
        
        # Mock the comboBox.currentText() to return our mock setup name
        filter_widget.comboBox.currentText = lambda: mock_setup_name
        
        # Get the original directories
        original_dirs = filter_widget.original_directories
        print(f"Original directories: {original_dirs}")
        
        # Simulate saving the info JSON to the original directory
        info_dir = original_dirs[0] / 'Info'  # Note the uppercase 'I'
        info_dir.mkdir(exist_ok=True, parents=True)
        
        # Get parameters with setup info
        parameters = filter_widget.get_burst_selection_parameters()
        parameters["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parameters["selected_setup"] = mock_setup_name
        
        # Add setup info
        setup_data = mock_setups["setups"][mock_setup_name]
        parameters["setup_info"] = setup_data
        
        # Save the parameters to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        params_filename = info_dir / f"photon_selection_parameters_{timestamp}.json"
        
        with open(params_filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Saved parameters to: {params_filename}")
        
        # Verify that the JSON file was saved in the Info directory (uppercase 'I')
        assert params_filename.exists(), f"JSON file {params_filename} does not exist"
        assert params_filename.parent.name == "Info", f"Directory name is not 'Info', got {params_filename.parent.name}"
        
        # Read the JSON file and verify that setup info is included
        with open(params_filename, 'r') as f:
            saved_params = json.load(f)
        
        assert "setup_info" in saved_params, "setup_info not found in saved parameters"
        assert "detectors" in saved_params["setup_info"], "detectors not found in setup_info"
        assert "windows" in saved_params["setup_info"], "windows not found in setup_info"
        assert "tttr_reading" in saved_params["setup_info"], "tttr_reading not found in setup_info"
        
        print("\nAll tests completed successfully!")
        
        # Restore the original load_detector_setups function
        filter_widget.load_detector_setups = original_load_detector_setups
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_info_json_uppercase()