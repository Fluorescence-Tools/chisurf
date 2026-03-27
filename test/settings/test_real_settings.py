import sys
import os
import yaml
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import the settings editor module to ensure our custom representers are registered
from chisurf.gui.widgets.settings_editor import (
    float_representer, 
    list_representer, 
    none_representer,
    dict_representer
)

def test_real_settings_file():
    """Test YAML formatting with a real settings file."""
    
    # Path to the real settings file
    settings_file = os.path.join('../chisurf', 'settings', 'settings_chisurf.yaml')
    
    if not os.path.exists(settings_file):
        print(f"Error: Settings file not found at {settings_file}")
        return
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # First, make a backup of the original file
        backup_file = settings_file + '.backup'
        shutil.copy2(settings_file, backup_file)
        print(f"Backup created at {backup_file}")
        
        # Load the settings file
        with open(settings_file, 'r', encoding='utf-8') as file:
            settings_data = yaml.safe_load(file)
        
        print(f"Loaded settings from {settings_file}")
        
        # Write the settings to the temporary file
        with open(temp_filename, 'w', encoding='utf-8') as file:
            yaml.dump(settings_data, file, default_flow_style=False)
        
        print(f"Settings written to temporary file: {temp_filename}")
        
        # Read the temporary file back
        with open(temp_filename, 'r', encoding='utf-8') as file:
            temp_settings_data = yaml.safe_load(file)
        
        # Verify the data is preserved correctly
        verify_settings(settings_data, temp_settings_data)
        
        # Now test writing back to the original file
        with open(settings_file, 'w', encoding='utf-8') as file:
            yaml.dump(settings_data, file, default_flow_style=False)
        
        print(f"Settings written back to {settings_file}")
        
        # Read the original file again
        with open(settings_file, 'r', encoding='utf-8') as file:
            new_settings_data = yaml.safe_load(file)
        
        # Verify the data is still preserved correctly
        verify_settings(settings_data, new_settings_data)
        
        print("All tests passed! The YAML formatting is working correctly with real settings.")
        
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

def verify_settings(original, loaded):
    """Verify that the loaded settings match the original settings."""
    
    # Check that all keys are present
    assert set(original.keys()) == set(loaded.keys()), "Keys don't match"
    
    # Check specific problematic fields
    if 'plugins' in original and 'toolbar_plugins' in original['plugins']:
        assert original['plugins']['toolbar_plugins'] == loaded['plugins']['toolbar_plugins'], \
            "toolbar_plugins list not preserved"
    
    if 'gui' in original and 'fit_windows_size' in original['gui']:
        assert original['gui']['fit_windows_size'] == loaded['gui']['fit_windows_size'], \
            "fit_windows_size list not preserved"
    
    if 'tcspc' in original and 'polarization_options' in original['tcspc']:
        assert original['tcspc']['polarization_options'] == loaded['tcspc']['polarization_options'], \
            "polarization_options list not preserved"
    
    if 'tcspc' in original and 'rebin' in original['tcspc']:
        assert original['tcspc']['rebin'] == loaded['tcspc']['rebin'], \
            "rebin list not preserved"
    
    if 'mc_settings' in original and 'potentials' in original['mc_settings']:
        assert original['mc_settings']['potentials'] == loaded['mc_settings']['potentials'], \
            "potentials list not preserved"
    
    if 'mc_settings' in original and 'move_map' in original['mc_settings']:
        assert original['mc_settings']['move_map'] == loaded['mc_settings']['move_map'], \
            "move_map None value not preserved"
    
    print("Settings verification passed!")

if __name__ == "__main__":
    test_real_settings_file()