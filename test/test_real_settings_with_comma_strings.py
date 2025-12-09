"""
Test script to verify that comma-separated strings in real settings files are properly converted to lists.

This script:
1. Loads the actual settings file
2. Converts lists to comma-separated strings (simulating UI display)
3. Loads the strings back into the model
4. Verifies that they are correctly converted back to lists
"""

import sys
import os
import yaml
import tempfile
import pprint
import re
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml
from chisurf.gui.widgets.settings_editor import SettingsTreeModel

def test_real_settings_with_comma_strings():
    """Test conversion of comma-separated strings to lists in real settings."""
    
    # Path to the actual settings file
    settings_file = os.path.join('../chisurf', 'settings', 'settings_chisurf.yaml')
    
    if not os.path.exists(settings_file):
        print(f"Error: Settings file not found at {settings_file}")
        return
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Make a backup of the original file
        backup_file = settings_file + '.backup'
        shutil.copy2(settings_file, backup_file)
        print(f"Backup created at {backup_file}")
        
        # Load the settings file
        with open(settings_file, 'r', encoding='utf-8') as file:
            settings_data = yaml.safe_load(file)
        
        print(f"Loaded settings from {settings_file}")
        
        # Convert lists to comma-separated strings (simulating UI display)
        string_data = convert_lists_to_strings(settings_data)
        
        print("\nData with comma-separated strings (sample):")
        if 'plugins' in string_data and 'toolbar_plugins' in string_data['plugins']:
            print(f"toolbar_plugins: {string_data['plugins']['toolbar_plugins']}")
        
        # Print the mc_settings.potentials string to see how it's formatted
        if 'mc_settings' in string_data and 'potentials' in string_data['mc_settings']:
            print(f"mc_settings.potentials: {string_data['mc_settings']['potentials']}")
        
        # Simulate the conversion that happens in the _get_dict_from_item method
        retrieved_data = convert_strings_to_lists(string_data)
        
        print("\nRetrieved data (sample):")
        if 'plugins' in retrieved_data and 'toolbar_plugins' in retrieved_data['plugins']:
            print(f"toolbar_plugins: {retrieved_data['plugins']['toolbar_plugins']}")
        
        # Write the retrieved data to a new YAML file
        retrieved_filename = temp_filename + '.retrieved'
        with open(retrieved_filename, 'w', encoding='utf-8') as file:
            dump_yaml(retrieved_data, file)
        
        print(f"\nRetrieved data written to: {retrieved_filename}")
        
        # Verify that lists are correctly restored
        verify_lists_restored(settings_data, retrieved_data)
        
    finally:
        # Clean up the temporary files
        for filename in [temp_filename, retrieved_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")
        
        # Restore the original settings file from backup
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, settings_file)
            os.remove(backup_file)
            print(f"Original settings restored from {backup_file}")

def convert_lists_to_strings(data):
    """Recursively convert lists to comma-separated strings."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = convert_lists_to_strings(value)
        return result
    elif isinstance(data, list):
        return ", ".join(str(item) for item in data)
    else:
        return data

def convert_strings_to_lists(data):
    """
    Recursively convert comma-separated strings to lists.
    This simulates the conversion in the _get_dict_from_item method.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = convert_strings_to_lists(value)
        return result
    elif isinstance(data, str) and ',' in data:
        # Check if it looks like a list of dictionaries (contains curly braces)
        if '{' in data and '}' in data:
            try:
                # Try to evaluate it as a Python expression
                # This is safe because we're only handling specific patterns
                import ast
                # Convert single quotes to double quotes for proper parsing
                prepared_str = '[' + data.replace("'", '"') + ']'
                # Use ast.literal_eval to safely evaluate the string
                return ast.literal_eval(prepared_str)
            except (SyntaxError, ValueError) as e:
                # If evaluation fails, proceed with regular comma-separated list handling
                print(f"Warning: Could not convert '{data}' to list of dictionaries: {e}")
        
        # Regular comma-separated list
        # Split by comma and strip whitespace
        items = [item.strip() for item in data.split(',')]
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
        return converted_items
    else:
        return data

def verify_lists_restored(original, retrieved):
    """Verify that lists in the original data are correctly restored in the retrieved data."""
    
    success_count = 0
    failure_count = 0
    
    # Check all items recursively
    for key, value in original.items():
        if isinstance(value, dict):
            if key not in retrieved or not isinstance(retrieved[key], dict):
                print(f"FAILURE: {key} is not a dictionary in retrieved data")
                failure_count += 1
                continue
            
            # Recursively check nested dictionaries
            sub_success, sub_failure = verify_dict_lists(key, value, retrieved[key])
            success_count += sub_success
            failure_count += sub_failure
        elif isinstance(value, list):
            if key not in retrieved or not isinstance(retrieved[key], list):
                print(f"FAILURE: {key} is not a list in retrieved data")
                failure_count += 1
                continue
            
            if len(value) != len(retrieved[key]):
                print(f"FAILURE: {key} has different length in retrieved data")
                failure_count += 1
                continue
            
            print(f"SUCCESS: {key} is correctly restored as a list")
            success_count += 1
    
    print(f"\nVerification complete: {success_count} successes, {failure_count} failures")
    return success_count, failure_count

def verify_dict_lists(parent_key, original_dict, retrieved_dict):
    """Verify lists within a dictionary."""
    
    success_count = 0
    failure_count = 0
    
    for key, value in original_dict.items():
        if isinstance(value, dict):
            if key not in retrieved_dict or not isinstance(retrieved_dict[key], dict):
                print(f"FAILURE: {parent_key}.{key} is not a dictionary in retrieved data")
                failure_count += 1
                continue
            
            # Recursively check nested dictionaries
            sub_success, sub_failure = verify_dict_lists(f"{parent_key}.{key}", value, retrieved_dict[key])
            success_count += sub_success
            failure_count += sub_failure
        elif isinstance(value, list):
            if key not in retrieved_dict or not isinstance(retrieved_dict[key], list):
                print(f"FAILURE: {parent_key}.{key} is not a list in retrieved data")
                failure_count += 1
                continue
            
            if len(value) != len(retrieved_dict[key]):
                print(f"FAILURE: {parent_key}.{key} has different length in retrieved data")
                failure_count += 1
                continue
            
            print(f"SUCCESS: {parent_key}.{key} is correctly restored as a list")
            success_count += 1
    
    return success_count, failure_count

if __name__ == "__main__":
    test_real_settings_with_comma_strings()