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

if __name__ == "__main__":
    test_yaml_saving()