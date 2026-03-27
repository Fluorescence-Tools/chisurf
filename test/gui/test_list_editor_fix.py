"""
Test script to verify that the list editor fix works correctly.

This script:
1. Creates a test settings dictionary with the problematic plugins lists
2. Simulates editing the lists in the settings editor
3. Verifies that the lists are preserved correctly
"""

import sys
import os
import yaml
import tempfile
import pprint
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QApplication

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities and list editor
from chisurf.gui.widgets.yaml_utils import dump_yaml
from chisurf.gui.widgets.list_editor import ListEditorButton

def test_list_editor_fix():
    """Test that the list editor fix works correctly."""
    
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
        # Write the original test data to a YAML file
        with open(temp_filename, 'w', encoding='utf-8') as file:
            dump_yaml(test_data, file)
        
        print(f"Original test data written to: {temp_filename}")
        
        # Read the YAML file to verify the format
        with open(temp_filename, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nOriginal YAML content:")
        print(yaml_content)
        
        # Create a QApplication instance for the list editor
        app = QApplication.instance() or QApplication(sys.argv)
        
        # Simulate editing the lists in the settings editor
        # Create a list editor button for each list
        toolbar_plugins = test_data['plugins']['toolbar_plugins']
        disabled_models = test_data['plugins']['disabled_models']
        disabled_plugins = test_data['plugins']['disabled_plugins']
        
        # Create list editor buttons
        toolbar_editor = ListEditorButton(None, toolbar_plugins, str)
        disabled_models_editor = ListEditorButton(None, disabled_models, str)
        disabled_plugins_editor = ListEditorButton(None, disabled_plugins, str)
        
        # Simulate getting the edited lists
        edited_toolbar_plugins = toolbar_editor.get_items()
        edited_disabled_models = disabled_models_editor.get_items()
        edited_disabled_plugins = disabled_plugins_editor.get_items()
        
        # Update the test data with the edited lists
        test_data['plugins']['toolbar_plugins'] = edited_toolbar_plugins
        test_data['plugins']['disabled_models'] = edited_disabled_models
        test_data['plugins']['disabled_plugins'] = edited_disabled_plugins
        
        # Write the edited data to a new YAML file
        edited_filename = temp_filename + '.edited'
        with open(edited_filename, 'w', encoding='utf-8') as file:
            dump_yaml(test_data, file)
        
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
        for filename in [temp_filename, edited_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")

if __name__ == "__main__":
    test_list_editor_fix()