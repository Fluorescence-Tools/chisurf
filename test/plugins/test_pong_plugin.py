import sys
import importlib
import chisurf.plugins

# Try to import the pong_game plugin
module_path = "chisurf.plugins.pong_game"
print(f"Attempting to import {module_path}...")

try:
    # Import the module
    module = importlib.import_module(module_path)
    print(f"Successfully imported {module_path}")
    
    # Set __name__ to "plugin" to trigger the plugin's initialization code
    print("Setting __name__ to 'plugin' to initialize the plugin...")
    module.__name__ = "plugin"
    
    print("Plugin initialization successful!")
    
except Exception as e:
    print(f"Error loading plugin {module_path}: {e}")
    import traceback
    traceback.print_exc()