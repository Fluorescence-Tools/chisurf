import json
import os
import sys
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.gui.widgets.wizard.tttr_channeldefinition import (
    load_detector_setups,
    save_detector_setups,
)
from chisurf.gui.widgets.wizard.tttr_channeldefinition import (
    tttr_detector_setups as detector_setups_module,
)


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


def test_default_detector_setups_store_in_mfdb(tmp_path):
    """Default detector setup storage should persist in MFDB, not JSON.

    Uses dependency injection (``db_path`` / ``user_id`` / ``skip_migration``)
    so the test never touches the real database — no monkeypatching.
    """
    db_path = str(tmp_path / "mfdb.sqlite")

    setup_data = {
        "setups": {
            "BH SPC-130": {
                "windows": {"prompt": [0, 2048], "delayed": [2048, 4095]},
                "detectors": {"green": {"chs": [0, 8]}, "red": {"chs": [1, 9]}},
                "tttr_reading": {"file_type": "SPC-130"},
            }
        },
        "last_used": "BH SPC-130",
    }

    assert save_detector_setups(setup_data, db_path=db_path, user_id="")
    loaded = load_detector_setups(db_path=db_path, user_id="", skip_migration=True)
    setup_id = detector_setups_module.setup_id_for_name("BH SPC-130")
    db = MFDatabase(db_path)
    row = db.get_setup(setup_id)

    assert row is not None
    assert loaded["last_used"] == "BH SPC-130"
    assert loaded["setups"]["BH SPC-130"]["detectors"]["green"]["chs"] == [0, 8]
    assert loaded["setups"]["BH SPC-130"]["detectors"]["red"]["chs"] == [1, 9]
    assert not (tmp_path / "detector_setups.json").exists()

if __name__ == "__main__":
    test_save_detector_setups()
