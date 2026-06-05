import sys
import os
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import chisurf.core.settings
from chisurf.plugins.updater.updater import ChiSurfUpdater

# Debug: Print the contents of the cs_settings dictionary
print("Contents of cs_settings:")
for key, value in chisurf.core.settings.cs_settings.items():
    print(f"  {key}: {value}")

# Debug: Print all attributes of the settings module
print("\nAttributes of chisurf.core.settings:")
for attr in dir(chisurf.core.settings):
    if not attr.startswith('__'):
        try:
            value = getattr(chisurf.core.settings, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: Error - {e}")

# Create an updater instance and print its update URL
updater = ChiSurfUpdater()
print(f"\nUpdate URL from updater: {updater.update_url}")

print("\nTest completed")
