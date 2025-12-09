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

if __name__ == "__main__":
    test_yaml_formatting()