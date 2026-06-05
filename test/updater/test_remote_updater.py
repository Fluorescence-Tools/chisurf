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

# Create an updater instance with a remote HTTP URL
remote_url = "https://www.peulen.xyz/downloads/chisurf/conda/"
updater = ChiSurfUpdater(update_url=remote_url)
print(f"\nUpdate URL from updater: {updater.update_url}")

# Check if the update URL is a local folder
is_local = updater._is_local_folder()
print(f"Is local folder: {is_local}")

# List available versions from the remote URL
print("\nFetching versions from remote URL...")
versions = updater._list_remote_versions()
print(f"Available versions: {len(versions)}")
for version in versions:
    print(f"  {version['version']} - {version['file_name']}")

# Get update info
print("\nGetting update info...")
update_info = updater._get_update_info()
if update_info:
    print(f"Latest version: {update_info.get('latest_version', 'N/A')}")
    print(f"Available versions: {len(update_info.get('available_versions', []))}")
    for version in update_info.get('available_versions', []):
        print(f"  {version['version']} - {version['file_name']}")
else:
    print("No update info available")

print("\nTest completed")