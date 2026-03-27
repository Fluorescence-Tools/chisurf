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