import sys
import os
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import chisurf.core.settings
from chisurf.plugins.updater.updater import ChiSurfUpdater

# Delete the cached update info file to force a fresh scan
from chisurf.core.settings.path_utils import get_path
update_info_file = get_path('settings') / 'update_info.json'
if update_info_file.exists():
    print(f"Deleting cached update info file: {update_info_file}")
    os.remove(update_info_file)

# Create an updater instance with a non-existent URL to test fallback
non_existent_url = "https://non-existent-url.example.com/chisurf"
updater = ChiSurfUpdater(update_url=non_existent_url)
print(f"\nInitial Update URL: {updater.update_url}")

# Check if the update URL is a local folder
is_local = updater._is_local_folder()
print(f"Is local folder: {is_local}")

# Try to list available versions from the non-existent URL (should fail)
print("\nTrying to fetch versions from non-existent URL...")
versions = updater._list_remote_versions()
print(f"Available versions from non-existent URL: {len(versions)}")

# Get update info (should fall back to the hardcoded URL)
print("\nGetting update info (should use fallback URL)...")
update_info = updater._get_update_info()
if update_info:
    print(f"Latest version: {update_info.get('latest_version', 'N/A')}")
    print(f"Available versions: {len(update_info.get('available_versions', []))}")
    for version in update_info.get('available_versions', []):
        print(f"  {version['version']} - {version['file_name']}")
    print("\nFallback mechanism worked successfully!")
else:
    print("No update info available - fallback mechanism failed")

print("\nTest completed")