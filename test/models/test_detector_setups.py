import json
import os
import pathlib
import tempfile
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.gui.widgets.wizard.tttr_channeldefinition import save_detector_setups, load_detector_setups

def test_save_detector_setups():
    """Test that save_detector_setups updates the file instead of overwriting it."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        temp_path = temp_file.name
    
    try:
        # Initial data
        initial_data = {
            "setups": {
                "setup1": {
                    "windows": {"prompt": [0, 2048]},
                    "detectors": {"green": {"chs": [0, 1]}}
                }
            }
        }
        
        # Save initial data
        save_detector_setups(initial_data, temp_path)
        
        # Verify initial data was saved
        loaded_data = load_detector_setups(temp_path)
        print("Initial data saved:")
        print(json.dumps(loaded_data, indent=2))
        
        # New data to add
        new_data = {
            "setups": {
                "setup2": {
                    "windows": {"delayed": [2048, 4095]},
                    "detectors": {"red": {"chs": [2, 3]}}
                }
            }
        }
        
        # Save new data (should update, not overwrite)
        save_detector_setups(new_data, temp_path)
        
        # Verify both setups are in the file
        updated_data = load_detector_setups(temp_path)
        print("\nUpdated data (should contain both setup1 and setup2):")
        print(json.dumps(updated_data, indent=2))
        
        # Check if both setups exist
        assert "setup1" in updated_data["setups"], "setup1 was overwritten!"
        assert "setup2" in updated_data["setups"], "setup2 was not added!"
        
        print("\nTest passed! The file was updated correctly.")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_save_detector_setups()