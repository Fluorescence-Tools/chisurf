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

if __name__ == "__main__":
    test_yaml_formatting()