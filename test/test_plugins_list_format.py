"""
Test script to reproduce the issue with plugins lists being converted to comma-separated strings.

This script:
1. Creates a test settings dictionary with the problematic plugins lists
2. Writes it to a YAML file using our custom YAML dumper
3. Reads it back to verify the format
"""

import sys
import os
import yaml
import tempfile
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml

def test_plugins_list_format():
    """Test YAML formatting specifically for the plugins lists."""
    
    # Create a test settings dictionary with the problematic plugins lists
    test_data = {
        'plugins': {
            'disabled_models': [
                'Et-Model free',
                'Dye-diffusion'
            ],
            'disabled_plugins': [
                'TTTR:Splitter',
                'TTTR:Correlate',
                'TTTR:Generate Decay',
                'Tools:Bayesian FRET Analysis',
                'Single-Molecule:FIDA-2D',
                'Single-Molecule:FIDA'
            ],
            'hide_disabled_models': True,
            'hide_disabled_plugins': True,
            'icons_enabled': True,
            'plugin_order': {},
            'toolbar_plugins': [
                'Tools:Histogram-Microtime',
                'FCS:Correlator',
                'Single-Molecule:Burst-Selection',
                'Single-Molecule:Burst MLE Lifetime Analysis',
                'Tools:ndXplorer'
            ]
        }
    }
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Write the test data to the YAML file using our custom dumper
        with open(temp_filename, 'w', encoding='utf-8') as file:
            dump_yaml(test_data, file)
        
        print(f"Test data written to temporary file: {temp_filename}")
        
        # Read the YAML file back
        with open(temp_filename, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nYAML content:")
        print(yaml_content)
        
        # Parse the YAML content back to Python
        parsed_data = yaml.safe_load(yaml_content)
        
        print("\nParsed data:")
        pprint.pprint(parsed_data)
        
        # Verify the data is preserved correctly
        verify_plugins_lists(test_data, parsed_data)
        
        # Now simulate what happens in the settings editor
        # The settings editor displays lists as comma-separated strings in the UI
        # Let's convert the lists to comma-separated strings and back
        ui_representation = convert_to_ui_representation(test_data)
        print("\nUI representation (as shown in the settings editor):")
        pprint.pprint(ui_representation)
        
        # Now convert back as if the user edited the values
        edited_data = convert_from_ui_representation(ui_representation)
        print("\nData after editing in UI:")
        pprint.pprint(edited_data)
        
        # Write the edited data to a new YAML file
        edited_filename = temp_filename + '.edited'
        with open(edited_filename, 'w', encoding='utf-8') as file:
            dump_yaml(edited_data, file)
        
        print(f"\nEdited data written to: {edited_filename}")
        
        # Read the edited YAML file back
        with open(edited_filename, 'r', encoding='utf-8') as file:
            edited_yaml_content = file.read()
        
        print("\nEdited YAML content:")
        print(edited_yaml_content)
        
        # Check if the lists are still preserved as lists
        if "toolbar_plugins:" in edited_yaml_content and "- Tools:" in edited_yaml_content:
            print("\nSUCCESS: Lists are preserved as lists in the edited YAML")
        else:
            print("\nFAILURE: Lists are converted to comma-separated strings in the edited YAML")
        
    finally:
        # Clean up the temporary files
        for filename in [temp_filename, temp_filename + '.edited']:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")

def verify_plugins_lists(original, parsed):
    """Verify that the parsed plugins lists match the original."""
    
    # Check that all plugins lists are preserved correctly
    assert parsed['plugins']['disabled_models'] == original['plugins']['disabled_models'], \
        "disabled_models list not preserved"
    assert parsed['plugins']['disabled_plugins'] == original['plugins']['disabled_plugins'], \
        "disabled_plugins list not preserved"
    assert parsed['plugins']['toolbar_plugins'] == original['plugins']['toolbar_plugins'], \
        "toolbar_plugins list not preserved"
    
    print("All plugins lists are preserved correctly in the YAML")

def convert_to_ui_representation(data):
    """
    Convert the data to how it would be represented in the UI.
    This simulates what happens in the settings editor's _populate_model method.
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = convert_to_ui_representation(value)
        elif isinstance(value, list):
            # Convert lists to comma-separated strings, as done in the UI
            result[key] = ", ".join(str(item) for item in value)
        else:
            result[key] = value
    
    return result

def convert_from_ui_representation(data):
    """
    Convert the UI representation back to the data structure.
    This simulates what happens in the settings editor's setModelData method.
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = convert_from_ui_representation(value)
        elif isinstance(value, str) and "," in value:
            # Convert comma-separated strings back to lists
            result[key] = [item.strip() for item in value.split(",")]
        else:
            result[key] = value
    
    return result

if __name__ == "__main__":
    test_plugins_list_format()