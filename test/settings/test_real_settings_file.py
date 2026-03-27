"""
Test script to directly load and save the actual settings file.

This script:
1. Loads the actual settings file
2. Saves it back to a temporary file
3. Checks if the lists are preserved correctly
"""

import sys
import os
import yaml
import tempfile
import pprint
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import our custom YAML utilities
from chisurf.gui.widgets.yaml_utils import dump_yaml

def test_real_settings_file():
    """Test loading and saving the actual settings file."""
    
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
        
        # Check the type of the plugins lists
        if 'plugins' in settings_data:
            plugins = settings_data['plugins']
            
            if 'toolbar_plugins' in plugins:
                print(f"Type of toolbar_plugins: {type(plugins['toolbar_plugins'])}")
                print(f"Value of toolbar_plugins: {plugins['toolbar_plugins']}")
            
            if 'disabled_models' in plugins:
                print(f"Type of disabled_models: {type(plugins['disabled_models'])}")
                print(f"Value of disabled_models: {plugins['disabled_models']}")
            
            if 'disabled_plugins' in plugins:
                print(f"Type of disabled_plugins: {type(plugins['disabled_plugins'])}")
                print(f"Value of disabled_plugins: {plugins['disabled_plugins']}")
        
        # Write the settings to the temporary file
        with open(temp_filename, 'w', encoding='utf-8') as file:
            dump_yaml(settings_data, file)
        
        print(f"Settings written to temporary file: {temp_filename}")
        
        # Read the temporary file back
        with open(temp_filename, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nYAML content (excerpt):")
        # Find the plugins section in the YAML content
        plugins_start = yaml_content.find("plugins:")
        if plugins_start != -1:
            # Extract a portion of the YAML content around the plugins section
            excerpt_end = yaml_content.find("\n\n", plugins_start)
            if excerpt_end == -1:
                excerpt_end = len(yaml_content)
            excerpt = yaml_content[plugins_start:excerpt_end]
            print(excerpt)
        else:
            print("Plugins section not found in YAML content")
        
        # Load the temporary file back to verify
        with open(temp_filename, 'r', encoding='utf-8') as file:
            temp_settings_data = yaml.safe_load(file)
        
        # Check if the lists are preserved
        if 'plugins' in temp_settings_data:
            plugins = temp_settings_data['plugins']
            
            if 'toolbar_plugins' in plugins:
                print(f"\nType of toolbar_plugins after reload: {type(plugins['toolbar_plugins'])}")
                print(f"Value of toolbar_plugins after reload: {plugins['toolbar_plugins']}")
                
                # Check if it's still a list
                if isinstance(plugins['toolbar_plugins'], list):
                    print("SUCCESS: toolbar_plugins is still a list after reload")
                else:
                    print("FAILURE: toolbar_plugins is not a list after reload")
            
            if 'disabled_models' in plugins:
                print(f"\nType of disabled_models after reload: {type(plugins['disabled_models'])}")
                print(f"Value of disabled_models after reload: {plugins['disabled_models']}")
                
                # Check if it's still a list
                if isinstance(plugins['disabled_models'], list):
                    print("SUCCESS: disabled_models is still a list after reload")
                else:
                    print("FAILURE: disabled_models is not a list after reload")
            
            if 'disabled_plugins' in plugins:
                print(f"\nType of disabled_plugins after reload: {type(plugins['disabled_plugins'])}")
                print(f"Value of disabled_plugins after reload: {plugins['disabled_plugins']}")
                
                # Check if it's still a list
                if isinstance(plugins['disabled_plugins'], list):
                    print("SUCCESS: disabled_plugins is still a list after reload")
                else:
                    print("FAILURE: disabled_plugins is not a list after reload")
        
        # Now try to write the settings back to the original file
        # This simulates what happens when the user saves settings in the editor
        with open(settings_file, 'w', encoding='utf-8') as file:
            dump_yaml(settings_data, file)
        
        print(f"\nSettings written back to {settings_file}")
        
        # Read the original file again
        with open(settings_file, 'r', encoding='utf-8') as file:
            yaml_content = file.read()
        
        print("\nOriginal file content after write (excerpt):")
        # Find the plugins section in the YAML content
        plugins_start = yaml_content.find("plugins:")
        if plugins_start != -1:
            # Extract a portion of the YAML content around the plugins section
            excerpt_end = yaml_content.find("\n\n", plugins_start)
            if excerpt_end == -1:
                excerpt_end = len(yaml_content)
            excerpt = yaml_content[plugins_start:excerpt_end]
            print(excerpt)
        else:
            print("Plugins section not found in YAML content")
        
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

if __name__ == "__main__":
    test_real_settings_file()