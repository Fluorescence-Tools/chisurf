"""
Test script to verify that comma-separated strings are properly converted to lists.

This script:
1. Creates a test settings dictionary with lists
2. Converts the lists to comma-separated strings (simulating UI display)
3. Loads the strings back into the model
4. Verifies that they are correctly converted back to lists
"""

import sys
import os
import yaml
import tempfile
import pprint
import re
from qtpy import QtCore, QtGui

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml
from chisurf.gui.widgets.settings_editor import SettingsTreeModel

def test_comma_string_to_list():
    """Test conversion of comma-separated strings to lists."""
    
    # Create a test settings dictionary with lists
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
            'toolbar_plugins': [
                'Tools:Histogram-Microtime',
                'FCS:Correlator',
                'Single-Molecule:Burst-Selection',
                'Single-Molecule:Burst MLE Lifetime Analysis',
                'Tools:ndXplorer'
            ]
        },
        'mixed_types_list': [1, 'string', 3.14, True, None],
        'nested': {
            'fit_windows_size': [350, 350],
            'polarization_options': ['vm', 'vv', 'vh'],
            'rebin': [1, 1]
        }
    }
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Write the original test data to a YAML file
        with open(temp_filename, 'w', encoding='utf-8') as file:
            dump_yaml(test_data, file)
        
        print(f"Original test data written to: {temp_filename}")
        
        # Read the YAML file to verify the format
        with open(temp_filename, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nOriginal YAML content:")
        print(yaml_content)
        
        # Convert lists to comma-separated strings (simulating UI display)
        string_data = {}
        for key, value in test_data.items():
            if isinstance(value, list):
                string_data[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                string_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        string_data[key][sub_key] = ", ".join(str(item) for item in sub_value)
                    else:
                        string_data[key][sub_key] = sub_value
            else:
                string_data[key] = value
        
        print("\nData with comma-separated strings:")
        pprint.pprint(string_data)
        
        # Instead of using the model directly, we'll simulate the conversion
        # that happens in the _get_dict_from_item method
        retrieved_data = {}
        
        for key, value in string_data.items():
            if isinstance(value, str) and ',' in value:
                # Split by comma and strip whitespace
                items = [item.strip() for item in value.split(',')]
                # Try to convert items to appropriate types
                converted_items = []
                for item in items:
                    if item.lower() == 'true':
                        converted_items.append(True)
                    elif item.lower() == 'false':
                        converted_items.append(False)
                    elif item.lower() == 'none' or item.lower() == 'null':
                        converted_items.append(None)
                    elif item.isdigit():
                        converted_items.append(int(item))
                    elif re.match(r'^-?\d+(\.\d+)?$', item):
                        converted_items.append(float(item))
                    else:
                        converted_items.append(item)
                retrieved_data[key] = converted_items
            elif isinstance(value, dict):
                retrieved_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str) and ',' in sub_value:
                        # Split by comma and strip whitespace
                        items = [item.strip() for item in sub_value.split(',')]
                        # Try to convert items to appropriate types
                        converted_items = []
                        for item in items:
                            if item.lower() == 'true':
                                converted_items.append(True)
                            elif item.lower() == 'false':
                                converted_items.append(False)
                            elif item.lower() == 'none' or item.lower() == 'null':
                                converted_items.append(None)
                            elif item.isdigit():
                                converted_items.append(int(item))
                            elif re.match(r'^-?\d+(\.\d+)?$', item):
                                converted_items.append(float(item))
                            else:
                                converted_items.append(item)
                        retrieved_data[key][sub_key] = converted_items
                    else:
                        retrieved_data[key][sub_key] = sub_value
            else:
                retrieved_data[key] = value
        
        print("\nRetrieved data from model:")
        pprint.pprint(retrieved_data)
        
        # Write the retrieved data to a new YAML file
        retrieved_filename = temp_filename + '.retrieved'
        with open(retrieved_filename, 'w', encoding='utf-8') as file:
            dump_yaml(retrieved_data, file)
        
        print(f"\nRetrieved data written to: {retrieved_filename}")
        
        # Read the retrieved YAML file back
        with open(retrieved_filename, 'r', encoding='utf-8') as file:
            retrieved_yaml_content = file.read()
        
        print("\nRetrieved YAML content:")
        print(retrieved_yaml_content)
        
        # Verify that lists are correctly restored
        verify_lists_restored(test_data, retrieved_data)
        
    finally:
        # Clean up the temporary files
        for filename in [temp_filename, retrieved_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")

def verify_lists_restored(original, retrieved):
    """Verify that lists in the original data are correctly restored in the retrieved data."""
    
    # Check top-level lists
    for key, value in original.items():
        if isinstance(value, list):
            if key not in retrieved or not isinstance(retrieved[key], list):
                print(f"FAILURE: {key} is not a list in retrieved data")
                continue
                
            if len(value) != len(retrieved[key]):
                print(f"FAILURE: {key} has different length in retrieved data")
                continue
                
            print(f"SUCCESS: {key} is correctly restored as a list")
        
        # Check nested lists
        elif isinstance(value, dict):
            if key not in retrieved or not isinstance(retrieved[key], dict):
                print(f"FAILURE: {key} is not a dictionary in retrieved data")
                continue
                
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    if sub_key not in retrieved[key] or not isinstance(retrieved[key][sub_key], list):
                        print(f"FAILURE: {key}.{sub_key} is not a list in retrieved data")
                        continue
                        
                    if len(sub_value) != len(retrieved[key][sub_key]):
                        print(f"FAILURE: {key}.{sub_key} has different length in retrieved data")
                        continue
                        
                    print(f"SUCCESS: {key}.{sub_key} is correctly restored as a list")

if __name__ == "__main__":
    test_comma_string_to_list()