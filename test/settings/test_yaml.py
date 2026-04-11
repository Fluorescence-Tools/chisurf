# Consolidated test file: test_yaml.py


# --- FROM test_yaml_saving.py ---
import sys
import os
import yaml
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml, prepare_for_yaml

def test_yaml_saving():
    """Test YAML saving with problematic data types."""
    
    # Path to the real settings file
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
        
        # Add some additional test cases with problematic data types
        settings_data['test_edge_cases'] = {
            'empty_list': [],
            'list_with_empty_string': [''],
            'list_with_special_chars': ['a,b', 'c:d', 'e-f'],
            'nested_empty_list': [[]],
            'list_with_mixed_types': [1, 'string', 3.14, None, True],
            'scientific_notation_small': 1.0e-15,
            'scientific_notation_large': 1.0e15,
            'list_of_lists': [[1, 2], [3, 4]],
            'complex_nested_structure': {
                'level1': {
                    'level2': {
                        'level3': [
                            {'name': 'item1', 'value': 1},
                            {'name': 'item2', 'value': 2}
                        ]
                    }
                }
            }
        }
        
        # Debug: Print the type of empty list before preparation
        print("\nBefore preparation:")
        print(f"Empty list type: {type(settings_data['test_edge_cases']['empty_list'])}")
        print(f"Empty list value: {settings_data['test_edge_cases']['empty_list']}")
        
        # Write the settings to the temporary file using our custom YAML dumper
        with open(temp_filename, 'w', encoding='utf-8') as file:
            dump_yaml(settings_data, file)
        
        print(f"Settings written to temporary file: {temp_filename}")
        
        # Read the temporary file back
        with open(temp_filename, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nYAML content (first 500 characters):")
        print(yaml_content[:500] + "...")
        
        # Check for specific problematic patterns
        check_yaml_format(yaml_content)
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} removed")
        
        # Restore the original settings file from backup
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, settings_file)
            os.remove(backup_file)
            print(f"Original settings restored from {backup_file}")

def check_yaml_format(yaml_content):
    """Check for specific formatting issues in the YAML content."""
    
    issues_found = False
    
    # Check for lists converted to comma-separated strings
    if "toolbar_plugins: " in yaml_content and "- Tools:" not in yaml_content:
        print("ISSUE: Lists of strings are being converted to comma-separated strings")
        issues_found = True
    
    # Check for scientific notation preservation
    if "discriminate_amplitude: 1e-10" in yaml_content and "discriminate_amplitude: 1.0e-10" not in yaml_content:
        print("ISSUE: Scientific notation for floats is not being preserved correctly")
        issues_found = True
    
    # Check for nested lists
    if "fit_windows_size: " in yaml_content and "- 350" not in yaml_content:
        print("ISSUE: Nested lists are being converted to comma-separated values")
        issues_found = True
    
    # Check for None values
    if "move_map: None" in yaml_content and "move_map: null" not in yaml_content:
        print("ISSUE: None values are being converted to string representations")
        issues_found = True
    
    # Check for lists of dictionaries
    if "potentials: " in yaml_content and "- name: H-Potential" not in yaml_content:
        print("ISSUE: Lists of dictionaries are being converted to string representations")
        issues_found = True
    
    # Check edge cases
    
    # Empty list - look for the exact pattern in the YAML
    empty_list_pattern = "empty_list: []"
    empty_list_block_pattern = "empty_list:\n"
    if empty_list_pattern in yaml_content:
        print(f"ISSUE: Empty lists are using flow style: '{empty_list_pattern}'")
        issues_found = True
    elif empty_list_block_pattern in yaml_content:
        print(f"Empty lists are correctly formatted as block style: '{empty_list_block_pattern}'")
    
    # List with special characters
    if "list_with_special_chars: " in yaml_content and "- 'a,b'" not in yaml_content:
        print("ISSUE: Lists with special characters are not being formatted correctly")
        issues_found = True
    
    # Nested empty list - look for the exact pattern
    nested_empty_list_pattern = "nested_empty_list:\n- []"
    nested_empty_list_block_pattern = "nested_empty_list:\n-\n"
    if nested_empty_list_pattern in yaml_content:
        print(f"ISSUE: Nested empty lists are using flow style: '{nested_empty_list_pattern}'")
        issues_found = True
    elif nested_empty_list_block_pattern in yaml_content:
        print(f"Nested empty lists are correctly formatted as block style: '{nested_empty_list_block_pattern}'")
    
    # List of lists
    if "list_of_lists:" in yaml_content:
        if "- - 1" not in yaml_content or "  - 2" not in yaml_content:
            print("ISSUE: Lists of lists are not being formatted correctly")
            issues_found = True
    
    # Scientific notation for extreme values
    if "scientific_notation_small:" in yaml_content and "1e-15" in yaml_content and "1.0e-15" not in yaml_content:
        print("ISSUE: Scientific notation for small values is not being preserved correctly")
        issues_found = True
    
    if "scientific_notation_large:" in yaml_content and "1e+15" in yaml_content and "1.0e+15" not in yaml_content:
        print("ISSUE: Scientific notation for large values is not being preserved correctly")
        issues_found = True
    
    # Complex nested structure
    if "complex_nested_structure:" in yaml_content:
        if "level3:" in yaml_content and "- name: item1" not in yaml_content:
            print("ISSUE: Complex nested structures with lists of dictionaries are not being formatted correctly")
            issues_found = True
    
    if not issues_found:
        print("No formatting issues detected in the YAML content")
    else:
        print("\nFormatting issues were found in the YAML content")
    
    # Print the raw YAML content for the test edge cases section for detailed inspection
    print("\nRaw YAML content for test edge cases section:")
    start_idx = yaml_content.find("test_edge_cases:")
    if start_idx != -1:
        end_idx = yaml_content.find("verbose:", start_idx)
        if end_idx == -1:  # If not found, go to the end
            end_idx = len(yaml_content)
        test_section = yaml_content[start_idx:end_idx].strip()
        # Print with line numbers for easier analysis
        for i, line in enumerate(test_section.split('\n')):
            print(f"{i+1:3d}: {line}")


# --- FROM test_yaml_all_fixes.py ---
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


# --- FROM test_yaml_comprehensive.py ---
"""
Comprehensive test for YAML formatting fixes.

This script tests all the YAML formatting fixes implemented in yaml_utils.py:
- Empty lists formatting
- Nested empty lists formatting
- Scientific notation preservation
- None values representation
- Lists of dictionaries formatting
"""

import sys
import os
import yaml
import tempfile
import shutil
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml

def test_yaml_formatting():
    """Test all YAML formatting fixes."""
    
    # Create a comprehensive test data structure with all problematic types
    test_data = {
        # Empty lists
        'empty_list': [],
        
        # Nested empty lists
        'nested_empty_list': [[]],
        'double_nested_empty_list': [[[]]],
        
        # Lists with various content
        'list_with_empty_string': [''],
        'list_with_special_chars': ['a,b', 'c:d', 'e-f'],
        'list_with_mixed_types': [1, 'string', 3.14, None, True],
        
        # Scientific notation for floats
        'scientific_notation': {
            'very_small': 1.0e-15,
            'small': 1.0e-10,
            'medium_small': 1.0e-5,
            'medium_large': 1.0e5,
            'large': 1.0e10,
            'very_large': 1.0e15
        },
        
        # None values
        'none_value': None,
        'list_with_none': [None, 1, None],
        
        # Lists of lists
        'list_of_lists': [[1, 2], [3, 4]],
        
        # Lists of dictionaries
        'list_of_dicts': [
            {'name': 'item1', 'value': 1},
            {'name': 'item2', 'value': 2}
        ],
        
        # Complex nested structure
        'complex_nested': {
            'level1': {
                'level2': {
                    'level3': [
                        {'name': 'item1', 'value': 1, 'tags': ['a', 'b', []]},
                        {'name': 'item2', 'value': 2, 'tags': None}
                    ]
                }
            }
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
        verify_data(test_data, parsed_data)
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} removed")

def verify_data(original, parsed):
    """Verify that the parsed data matches the original data."""
    
    # Check empty list
    assert isinstance(parsed['empty_list'], list), "Empty list not preserved as list"
    assert len(parsed['empty_list']) == 0, "Empty list not empty"
    
    # Check nested empty list
    assert isinstance(parsed['nested_empty_list'], list), "Nested empty list not preserved as list"
    assert len(parsed['nested_empty_list']) == 1, "Nested empty list not preserved"
    assert isinstance(parsed['nested_empty_list'][0], list), "Nested empty list item not a list"
    assert len(parsed['nested_empty_list'][0]) == 0, "Nested empty list item not empty"
    
    # Check double nested empty list
    assert isinstance(parsed['double_nested_empty_list'], list), "Double nested empty list not preserved as list"
    assert len(parsed['double_nested_empty_list']) == 1, "Double nested empty list not preserved"
    assert isinstance(parsed['double_nested_empty_list'][0], list), "Double nested empty list item not a list"
    assert len(parsed['double_nested_empty_list'][0]) == 1, "Double nested empty list item not preserved"
    assert isinstance(parsed['double_nested_empty_list'][0][0], list), "Double nested empty list inner item not a list"
    assert len(parsed['double_nested_empty_list'][0][0]) == 0, "Double nested empty list inner item not empty"
    
    # Check list with empty string
    assert parsed['list_with_empty_string'] == [''], "List with empty string not preserved"
    
    # Check list with special characters
    assert parsed['list_with_special_chars'] == ['a,b', 'c:d', 'e-f'], "List with special characters not preserved"
    
    # Check list with mixed types
    assert parsed['list_with_mixed_types'] == [1, 'string', 3.14, None, True], "List with mixed types not preserved"
    
    # Check scientific notation
    for key, value in original['scientific_notation'].items():
        assert abs(parsed['scientific_notation'][key] - value) < 1e-10, f"Scientific notation for {key} not preserved"
    
    # Check None value
    assert parsed['none_value'] is None, "None value not preserved"
    
    # Check list with None
    assert parsed['list_with_none'] == [None, 1, None], "List with None not preserved"
    
    # Check list of lists
    assert parsed['list_of_lists'] == [[1, 2], [3, 4]], "List of lists not preserved"
    
    # Check list of dictionaries
    assert parsed['list_of_dicts'] == [
        {'name': 'item1', 'value': 1},
        {'name': 'item2', 'value': 2}
    ], "List of dictionaries not preserved"
    
    # Check complex nested structure
    assert parsed['complex_nested']['level1']['level2']['level3'][0]['tags'] == ['a', 'b', []], \
        "Complex nested structure with empty list not preserved"
    assert parsed['complex_nested']['level1']['level2']['level3'][1]['tags'] is None, \
        "Complex nested structure with None not preserved"
    
    print("All verifications passed! The YAML formatting is working correctly.")


# --- FROM test_yaml_format_fixes.py ---
import sys
import os
import yaml
import tempfile
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import the settings editor module to ensure our custom representers are registered
from chisurf.gui.widgets.settings_editor import (
    float_representer, 
    list_representer, 
    none_representer,
    dict_representer
)

def test_yaml_formatting():
    """Test YAML formatting with various data types and structures."""
    
    # Test data with various problematic types
    test_data = {
        # Lists of strings
        'toolbar_plugins': [
            'Tools:Histogram-Microtime',
            'FCS:Correlator',
            'Single-Molecule:Burst-Selection',
            'Single-Molecule:Burst MLE Lifetime Analysis',
            'Tools:ndXplorer'
        ],
        
        # Lists of simple values
        'simple_list': [1, 2, 3, 4, 5],
        
        # Lists with None values
        'list_with_none': [1, None, 3, None, 5],
        
        # Nested lists
        'nested_list': [
            [1, 2, 3],
            [4, 5, 6]
        ],
        
        # Scientific notation floats
        'scientific_floats': {
            'small': 1.0e-10,
            'large': 1.0e10
        },
        
        # None values
        'none_value': None,
        
        # Lists of dictionaries
        'potentials': [
            {'name': 'H-Potential', 'weight': 2},
            {'name': 'Iso-UNRES', 'weight': 1}
        ],
        
        # Nested structures with lists
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
        # Write the test data to the YAML file
        with open(temp_filename, 'w', encoding='utf-8') as file:
            yaml.dump(test_data, file, default_flow_style=False)
        
        print(f"Test data written to temporary file: {temp_filename}")
        print("\nOriginal data:")
        pprint.pprint(test_data)
        
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
        assert parsed_data['toolbar_plugins'] == test_data['toolbar_plugins'], "List of strings not preserved"
        assert parsed_data['simple_list'] == test_data['simple_list'], "Simple list not preserved"
        assert parsed_data['list_with_none'] == test_data['list_with_none'], "List with None values not preserved"
        assert parsed_data['nested_list'] == test_data['nested_list'], "Nested list not preserved"
        assert parsed_data['scientific_floats']['small'] == test_data['scientific_floats']['small'], "Small scientific float not preserved"
        assert parsed_data['scientific_floats']['large'] == test_data['scientific_floats']['large'], "Large scientific float not preserved"
        assert parsed_data['none_value'] is None, "None value not preserved"
        assert parsed_data['potentials'] == test_data['potentials'], "List of dictionaries not preserved"
        assert parsed_data['nested']['fit_windows_size'] == test_data['nested']['fit_windows_size'], "Nested list not preserved"
        
        print("\nAll assertions passed! The YAML formatting is working correctly.")
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"\nTemporary file {temp_filename} removed")


# --- FROM test_yaml_list_format.py ---
import sys
import os
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import the settings editor module to ensure our custom representers are registered
from chisurf.gui.widgets.settings_editor import list_representer, float_representer

# Test data with a list
test_data = {
    'toolbar_plugins': [
        'Tools:Histogram-Microtime',
        'FCS:Correlator',
        'Single-Molecule:Burst-Selection',
        'Single-Molecule:Burst MLE Lifetime Analysis',
        'Tools:ndXplorer'
    ],
    'other_setting': 'value',
    'nested': {
        'another_list': [1, 2, 3, 4]
    }
}

# Output file path
output_file = 'test_yaml_output.yaml'

# Write the test data to a YAML file
with open(output_file, 'w', encoding='utf-8') as f:
    yaml.dump(test_data, f, default_flow_style=False)

print(f"Test data written to {output_file}")
print("Contents of the YAML file:")
with open(output_file, 'r', encoding='utf-8') as f:
    print(f.read())

# Clean up
os.remove(output_file)
print(f"Test file {output_file} removed")