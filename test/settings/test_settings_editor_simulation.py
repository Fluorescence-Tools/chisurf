"""
Test script to simulate the actual behavior of the settings editor.

This script:
1. Creates a test settings dictionary with the problematic plugins lists
2. Simulates the process of loading settings into the model and retrieving them
3. Writes the retrieved settings to a YAML file to check if lists are preserved
"""

import sys
import os
import yaml
import tempfile
import pprint
from qtpy import QtCore, QtGui

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml

class MockQStandardItem:
    """Mock implementation of QStandardItem for testing."""
    
    def __init__(self, text=""):
        self.text_value = text
        self.data_value = None
        self.children = []
        self.editable_value = True
        self.tooltip_value = ""
    
    def text(self):
        return self.text_value
    
    def setText(self, text):
        self.text_value = text
    
    def setData(self, value, role):
        if role == QtCore.Qt.EditRole:
            self.data_value = value
    
    def data(self, role):
        if role == QtCore.Qt.EditRole:
            return self.data_value
        return self.text_value
    
    def setEditable(self, editable):
        self.editable_value = editable
    
    def setToolTip(self, tooltip):
        self.tooltip_value = tooltip
    
    def appendRow(self, row):
        self.children.append(row)
    
    def child(self, row, column):
        if row < len(self.children):
            return self.children[row][column]
        return None
    
    def rowCount(self):
        return len(self.children)
    
    def hasChildren(self):
        return len(self.children) > 0

class MockSettingsTreeModel:
    """Mock implementation of SettingsTreeModel for testing."""
    
    def __init__(self):
        self.root = MockQStandardItem()
    
    def invisibleRootItem(self):
        return self.root
    
    def load_settings(self, settings_dict):
        """Simulate loading settings into the model."""
        self.root = MockQStandardItem()
        self._populate_model(settings_dict)
    
    def _populate_model(self, settings_dict, parent=None, path=""):
        """Simulate populating the model with settings."""
        if parent is None:
            parent = self.root
        
        for key, value in sorted(settings_dict.items()):
            # Create key item
            key_item = MockQStandardItem(key)
            key_item.setEditable(False)
            
            # Create value item
            value_item = MockQStandardItem()
            value_item.setData(value, QtCore.Qt.EditRole)
            
            # Set display text based on data type
            if isinstance(value, dict):
                # For dictionaries, don't set display text
                pass
            elif isinstance(value, (list, tuple)):
                # For lists, show comma-separated values
                if not value:
                    # Empty list
                    value_item.setText("")
                elif any(isinstance(item, dict) for item in value):
                    # List contains dictionaries - show a placeholder
                    value_item.setText("[complex list - edit with caution]")
                else:
                    # Regular list - convert None to 'None' for display
                    items_str = []
                    for item in value:
                        if item is None:
                            items_str.append("None")
                        else:
                            items_str.append(str(item))
                    value_item.setText(", ".join(items_str))
            else:
                # For other types, show string representation
                value_item.setText(str(value))
            
            # Add items to model
            row = [key_item, value_item]
            
            if isinstance(value, dict):
                # For dictionaries, add as parent and recurse
                parent.appendRow(row)
                self._populate_model(value, key_item, path + "." + key if path else key)
            else:
                # For other types, add as leaf
                parent.appendRow(row)
    
    def get_settings_dict(self):
        """Simulate getting settings from the model."""
        return self._get_dict_from_item(self.root)
    
    def _get_dict_from_item(self, item):
        """Simulate recursively building a dictionary from a model item."""
        result_dict = {}
        
        for row in range(item.rowCount()):
            key_item = item.child(row, 0)
            value_item = item.child(row, 1)
            
            key = key_item.text()
            
            if key_item.hasChildren():
                # If the key item has children, it's a dictionary
                value = self._get_dict_from_item(key_item)
            else:
                # Otherwise, get the value from the value item
                value = value_item.data(QtCore.Qt.EditRole)
                
                # Convert string values to appropriate types if possible
                if isinstance(value, str):
                    value_str = value
                    
                    # Try to convert to appropriate type
                    try:
                        # Check for boolean values
                        if value_str.lower() == "true":
                            value = True
                        elif value_str.lower() == "false":
                            value = False
                        # Check for integer values
                        elif value_str.isdigit():
                            value = int(value_str)
                        # Check for float values
                        elif re.match(r'^-?\d+(\.\d+)?$', value_str):
                            value = float(value_str)
                    except (ValueError, NameError):
                        # If conversion fails, keep as string
                        pass
            
            result_dict[key] = value
        
        return result_dict

def test_settings_editor_simulation():
    """Test simulating the settings editor behavior."""
    
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
        
        # Now simulate the settings editor behavior
        model = MockSettingsTreeModel()
        model.load_settings(test_data)
        
        # Get the settings back from the model
        retrieved_data = model.get_settings_dict()
        
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
        
        # Check if the lists are still preserved as lists
        if "toolbar_plugins:" in retrieved_yaml_content and "- Tools:" in retrieved_yaml_content:
            print("\nSUCCESS: Lists are preserved as lists in the retrieved YAML")
        else:
            print("\nFAILURE: Lists are converted to comma-separated strings in the retrieved YAML")
            
            # Print the type of the toolbar_plugins value in the retrieved data
            print(f"\nType of toolbar_plugins in retrieved data: {type(retrieved_data['plugins']['toolbar_plugins'])}")
            print(f"Value: {retrieved_data['plugins']['toolbar_plugins']}")
        
    finally:
        # Clean up the temporary files
        for filename in [temp_filename, retrieved_filename]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file {filename} removed")

if __name__ == "__main__":
    # Import re here to avoid NameError in the mock model
    import re
    test_settings_editor_simulation()