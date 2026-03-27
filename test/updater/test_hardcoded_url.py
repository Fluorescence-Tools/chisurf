import sys
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

# Create an updater instance without specifying an update_url
updater = ChiSurfUpdater()

# Print the update_url to verify it's using the hardcoded URL
print(f"Update URL: {updater.update_url}")

# Check if the update URL is the hardcoded URL (with '/conda' appended)
hardcoded_url = "https://www.peulen.xyz/downloads/chisurf/conda"
if updater.update_url == hardcoded_url:
    print("SUCCESS: The updater is using the hardcoded URL as expected.")
else:
    print(f"ERROR: The updater is not using the hardcoded URL. Expected '{hardcoded_url}', got '{updater.update_url}'")

# Test with a custom URL
custom_url = "https://example.com/custom"
custom_updater = ChiSurfUpdater(update_url=custom_url)
if custom_updater.update_url == custom_url + "/conda":
    print("SUCCESS: The updater respects custom URLs when provided.")
else:
    print(f"ERROR: The updater is not respecting custom URLs. Expected '{custom_url}/conda', got '{custom_updater.update_url}'")

print("\nTest completed")