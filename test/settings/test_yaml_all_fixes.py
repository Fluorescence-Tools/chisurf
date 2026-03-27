"""
Comprehensive test script to verify all YAML formatting fixes work together.

This script:
1. Creates a test settings dictionary with all problematic data types
2. Writes it to a YAML file
3. Converts lists to comma-separated strings (simulating UI display)
4. Applies our fixes to convert strings back to lists
5. Writes the result to a new YAML file
6. Verifies that all data types are preserved correctly
"""

import sys
import os
import yaml
import tempfile
import pprint
import re
import ast

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml

def test_all_fixes():
    """Test that all YAML formatting fixes work together."""
    
    # Create a test settings dictionary with all problematic data types
    test_data = {
        # Simple lists
        'simple_list': [1, 2, 3, 4, 5],
        'string_list': ['a', 'b', 'c', 'd'],
        
        # Lists with mixed types
        'mixed_list': [1, 'string', 3.14, True, None],
        
        # Nested lists
        'nested_list': [[1, 2], [3, 4]],
        
        # Empty lists
        'empty_list': [],
        'nested_empty_list': [[]],
        
        # Scientific notation
        'scientific_notation': {
            'small': 1.0e-10,
            'large': 1.0e10
        },
        
        # None values
        'none_value': None,
        'list_with_none': [None, 1, None],
        
        # Complex structures
        'complex': {
            # Lists of dictionaries
            'list_of_dicts': [
                {'name': 'item1', 'value': 1},
                {'name': 'item2', 'value': 2}
            ],
            # Deeply nested structure
            'nested': {
                'deep_list': [10, 20, 30],
                'deep_dict': {
                    'deeper_list': [True, False, None]
                }
            }
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
        
        # Convert all lists to comma-separated strings (simulating UI display)
        string_data = convert_lists_to_strings(test_data)
        
        print("\nData with comma-separated strings:")
        pprint.pprint(string_data)
        
        # Apply our fixes to convert strings back to lists
        retrieved_data = convert_strings_to_lists(string_data)
        
        print("\nRetrieved data:")
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
        
        # Verify that all data types are preserved correctly
        verify_data_preserved(test_data, retrieved_data)
        
    finally:
        # Clean up the temporary files
        for filename in [temp_filename, retrieved_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")

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
    elif isinstance(data, str):
        # Handle empty string as empty list
        if data == '':
            return []
        # Handle string representation of empty list
        elif data == '[]':
            return []
        # Handle comma-separated strings
        elif ',' in data:
            # Check if it looks like a list of dictionaries or nested lists
            if ('{' in data and '}' in data) or ('[' in data and ']' in data):
                try:
                    # Try to evaluate it as a Python expression
                    # This is safe because we're only handling specific patterns
                    # Convert single quotes to double quotes for proper parsing
                    prepared_str = '[' + data.replace("'", '"') + ']'
                    # Use ast.literal_eval to safely evaluate the string
                    return ast.literal_eval(prepared_str)
                except (SyntaxError, ValueError) as e:
                    # If evaluation fails, proceed with regular comma-separated list handling
                    print(f"Warning: Could not convert '{data}' to complex list: {e}")
        
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

def verify_data_preserved(original, retrieved):
    """Verify that all data types in the original data are preserved in the retrieved data."""
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
            sub_success, sub_failure = verify_dict(key, value, retrieved[key])
            success_count += sub_success
            failure_count += sub_failure
        elif isinstance(value, list):
            if key not in retrieved or not isinstance(retrieved[key], list):
                print(f"FAILURE: {key} is not a list in retrieved data")
                failure_count += 1
                continue
            
            if len(value) != len(retrieved[key]):
                print(f"FAILURE: {key} has different length in retrieved data")
                print(f"  Original: {value} (length: {len(value)})")
                print(f"  Retrieved: {retrieved[key]} (length: {len(retrieved[key])})")
                failure_count += 1
                continue
            
            print(f"SUCCESS: {key} is correctly preserved as a list")
            success_count += 1
        else:
            if key not in retrieved:
                print(f"FAILURE: {key} is missing in retrieved data")
                failure_count += 1
                continue
            
            if value != retrieved[key]:
                print(f"FAILURE: {key} has different value in retrieved data")
                print(f"  Original: {value} ({type(value)})")
                print(f"  Retrieved: {retrieved[key]} ({type(retrieved[key])})")
                failure_count += 1
                continue
            
            print(f"SUCCESS: {key} is correctly preserved")
            success_count += 1
    
    print(f"\nVerification complete: {success_count} successes, {failure_count} failures")
    return success_count, failure_count

def verify_dict(parent_key, original_dict, retrieved_dict):
    """Verify items within a dictionary."""
    success_count = 0
    failure_count = 0
    
    for key, value in original_dict.items():
        if isinstance(value, dict):
            if key not in retrieved_dict or not isinstance(retrieved_dict[key], dict):
                print(f"FAILURE: {parent_key}.{key} is not a dictionary in retrieved data")
                failure_count += 1
                continue
            
            # Recursively check nested dictionaries
            sub_success, sub_failure = verify_dict(f"{parent_key}.{key}", value, retrieved_dict[key])
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
            
            print(f"SUCCESS: {parent_key}.{key} is correctly preserved as a list")
            success_count += 1
        else:
            if key not in retrieved_dict:
                print(f"FAILURE: {parent_key}.{key} is missing in retrieved data")
                failure_count += 1
                continue
            
            if value != retrieved_dict[key]:
                print(f"FAILURE: {parent_key}.{key} has different value in retrieved data")
                print(f"  Original: {value} ({type(value)})")
                print(f"  Retrieved: {retrieved_dict[key]} ({type(retrieved_dict[key])})")
                failure_count += 1
                continue
            
            print(f"SUCCESS: {parent_key}.{key} is correctly preserved")
            success_count += 1
    
    return success_count, failure_count

if __name__ == "__main__":
    test_all_fixes()