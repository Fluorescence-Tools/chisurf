import yaml
import tempfile
import os

def test_yaml_type_preservation():
    """Test that YAML preserves types when saving and loading."""
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
            "scientific_notation_small": 1.23e-6,
            "scientific_notation_large": 4.56e8,
            "nested": {
                "nested_int": 100,
                "nested_float": 99.9,
                "nested_bool": False,
                "nested_scientific": 7.89e-10
            }
        }

        # Save the test settings to the temporary file
        with open(temp_filename, 'w') as f:
            yaml.dump(test_settings, f)

        # Load the settings from the file and check if types are preserved
        with open(temp_filename, 'r') as f:
            loaded_settings = yaml.safe_load(f)

        # Check that all values have the correct types
        print(f"string_value: {type(loaded_settings['string_value']).__name__} = {loaded_settings['string_value']}")
        print(f"int_value: {type(loaded_settings['int_value']).__name__} = {loaded_settings['int_value']}")
        print(f"float_value: {type(loaded_settings['float_value']).__name__} = {loaded_settings['float_value']}")
        print(f"bool_value: {type(loaded_settings['bool_value']).__name__} = {loaded_settings['bool_value']}")

        print(f"scientific_notation_small: {type(loaded_settings['scientific_notation_small']).__name__} = {loaded_settings['scientific_notation_small']}")
        print(f"scientific_notation_large: {type(loaded_settings['scientific_notation_large']).__name__} = {loaded_settings['scientific_notation_large']}")

        print(f"list_of_ints: {type(loaded_settings['list_of_ints']).__name__} = {loaded_settings['list_of_ints']}")
        print(f"  item types: {[type(x).__name__ for x in loaded_settings['list_of_ints']]}")

        print(f"list_of_floats: {type(loaded_settings['list_of_floats']).__name__} = {loaded_settings['list_of_floats']}")
        print(f"  item types: {[type(x).__name__ for x in loaded_settings['list_of_floats']]}")

        print(f"list_of_strings: {type(loaded_settings['list_of_strings']).__name__} = {loaded_settings['list_of_strings']}")
        print(f"  item types: {[type(x).__name__ for x in loaded_settings['list_of_strings']]}")

        print(f"nested: {type(loaded_settings['nested']).__name__}")
        print(f"  nested_int: {type(loaded_settings['nested']['nested_int']).__name__} = {loaded_settings['nested']['nested_int']}")
        print(f"  nested_float: {type(loaded_settings['nested']['nested_float']).__name__} = {loaded_settings['nested']['nested_float']}")
        print(f"  nested_bool: {type(loaded_settings['nested']['nested_bool']).__name__} = {loaded_settings['nested']['nested_bool']}")
        print(f"  nested_scientific: {type(loaded_settings['nested']['nested_scientific']).__name__} = {loaded_settings['nested']['nested_scientific']}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    test_yaml_type_preservation()
