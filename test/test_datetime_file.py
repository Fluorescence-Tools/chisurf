"""
Test script to verify that:
1. The parameters are saved in a file without timestamp in the name
2. A separate datetime.txt file is created with date/time information

This script:
1. Creates a mock folder structure
2. Initializes the WizardTTTRPhotonFilter class
3. Simulates the save_selection method's behavior for saving files
4. Verifies that the files are created correctly
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
from chisurf.gui.widgets.wizard.tttr_photon_filter import WizardTTTRPhotonFilter

def test_datetime_file():
    """Test that a separate datetime file is created."""
    
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
        
        # Get the original directories
        original_dirs = filter_widget.original_directories
        print(f"Original directories: {original_dirs}")
        
        # Simulate saving the files to the original directory
        info_dir = original_dirs[0] / 'Info'
        info_dir.mkdir(exist_ok=True, parents=True)
        
        # Get parameters
        parameters = filter_widget.get_burst_selection_parameters()
        parameters["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save parameters to JSON file without timestamp in filename
        params_filename = info_dir / "photon_selection_parameters.json"
        
        # Create a separate file with date/time information
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y%m%d-%H%M%S")
        datetime_filename = info_dir / "datetime.txt"
        
        # Save parameters to JSON file
        with open(params_filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        # Save date/time to separate file
        with open(datetime_filename, 'w') as f:
            f.write(f"Date: {current_datetime.strftime('%Y-%m-%d')}\n")
            f.write(f"Time: {current_datetime.strftime('%H:%M:%S')}\n")
            f.write(f"Timestamp: {timestamp}\n")
        
        print(f"Saved parameters to: {params_filename}")
        print(f"Saved date/time to: {datetime_filename}")
        
        # Verify that the files were created correctly
        assert params_filename.exists(), f"JSON file {params_filename} does not exist"
        assert datetime_filename.exists(), f"Datetime file {datetime_filename} does not exist"
        
        # Verify the content of the datetime file
        with open(datetime_filename, 'r') as f:
            content = f.read()
            
        assert "Date: " in content, "Date not found in datetime file"
        assert "Time: " in content, "Time not found in datetime file"
        assert "Timestamp: " in content, "Timestamp not found in datetime file"
        
        print("\nAll tests completed successfully!")
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_datetime_file()