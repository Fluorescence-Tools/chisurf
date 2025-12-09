import sys
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

# Create an updater instance
updater = ChiSurfUpdater()

# Print the update URL
print(f"Update URL: {updater.update_url}")

# Check if the update_info_file attribute exists
has_update_info_file = hasattr(updater, 'update_info_file')
print(f"Has update_info_file attribute: {has_update_info_file}")

# Check if the _save_update_info method exists
has_save_update_info = hasattr(updater, '_save_update_info')
print(f"Has _save_update_info method: {has_save_update_info}")

# Try to get update info
print("\nGetting update info...")
update_info = updater._get_update_info()

if update_info:
    print(f"Latest version: {update_info.get('latest_version', 'N/A')}")
    print(f"Available versions: {len(update_info.get('available_versions', []))}")
    for version in update_info.get('available_versions', [])[:3]:  # Show only first 3 versions
        print(f"  {version['version']} - {version['file_name']}")
else:
    print("No update info available")

# Check for updates
print("\nChecking for updates...")
update_available, latest_version, error = updater.check_for_updates()
print(f"Update available: {update_available}")
print(f"Latest version: {latest_version}")
print(f"Error: {error}")

print("\nTest completed")