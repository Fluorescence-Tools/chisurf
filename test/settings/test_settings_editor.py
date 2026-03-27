import os
import tempfile
import yaml
from qtpy import QtWidgets, QtCore
from chisurf.gui.widgets.settings_editor import SettingsEditor, SettingsTreeModel

def test_type_preservation():
    """Test that types are preserved when saving and loading settings."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Create a test settings dictionary with various types
        test_settings = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "list_of_ints": [1, 2, 3],
            "list_of_floats": [1.1, 2.2, 3.3],
            "list_of_strings": ["a", "b", "c"],
            "nested": {
                "nested_int": 100,
                "nested_float": 99.9,
                "nested_bool": False
            }
        }

        # Save the test settings to the temporary file
        with open(temp_filename, 'w') as f:
            yaml.dump(test_settings, f)

        # Create a QApplication instance (required for Qt widgets)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        # Create a settings editor and load the test settings
        editor = SettingsEditor(filename=temp_filename)

        # Save the settings back to the file
        editor.save_settings()

        # Load the settings from the file and check if types are preserved
        with open(temp_filename, 'r') as f:
            loaded_settings = yaml.safe_load(f)

        # Check that all values have the correct types
        assert isinstance(loaded_settings["string_value"], str)
        assert loaded_settings["string_value"] == "test"

        assert isinstance(loaded_settings["int_value"], int)
        assert loaded_settings["int_value"] == 42

        assert isinstance(loaded_settings["float_value"], float)
        assert loaded_settings["float_value"] == 3.14

        assert isinstance(loaded_settings["bool_value"], bool)
        assert loaded_settings["bool_value"] is True

        assert isinstance(loaded_settings["list_of_ints"], list)
        assert all(isinstance(x, int) for x in loaded_settings["list_of_ints"])
        assert loaded_settings["list_of_ints"] == [1, 2, 3]

        assert isinstance(loaded_settings["list_of_floats"], list)
        assert all(isinstance(x, float) for x in loaded_settings["list_of_floats"])
        assert loaded_settings["list_of_floats"] == [1.1, 2.2, 3.3]

        assert isinstance(loaded_settings["list_of_strings"], list)
        assert all(isinstance(x, str) for x in loaded_settings["list_of_strings"])
        assert loaded_settings["list_of_strings"] == ["a", "b", "c"]

        assert isinstance(loaded_settings["nested"], dict)
        assert isinstance(loaded_settings["nested"]["nested_int"], int)
        assert loaded_settings["nested"]["nested_int"] == 100
        assert isinstance(loaded_settings["nested"]["nested_float"], float)
        assert loaded_settings["nested"]["nested_float"] == 99.9
        assert isinstance(loaded_settings["nested"]["nested_bool"], bool)
        assert loaded_settings["nested"]["nested_bool"] is False

        print("All types are correctly preserved!")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    test_type_preservation()