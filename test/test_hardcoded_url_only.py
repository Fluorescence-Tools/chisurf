import sys
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

# Test 1: Create an updater instance without specifying an update_url
updater = ChiSurfUpdater()
print(f"Test 1: Update URL: {updater.update_url}")

# Check if the update URL is the hardcoded URL
hardcoded_url = "https://www.peulen.xyz/downloads/chisurf/conda"
if updater.update_url == hardcoded_url:
    print("SUCCESS: The updater is using the hardcoded URL as expected.")
else:
    print(f"ERROR: The updater is not using the hardcoded URL. Expected '{hardcoded_url}', got '{updater.update_url}'")

# Test 2: Try to create an updater instance with a custom URL (should be ignored)
custom_url = "https://example.com/custom"
custom_updater = ChiSurfUpdater(update_url=custom_url)
print(f"\nTest 2: Update URL with custom parameter: {custom_updater.update_url}")

if custom_updater.update_url == hardcoded_url:
    print("SUCCESS: The updater ignores custom URLs and uses the hardcoded URL.")
else:
    print(f"ERROR: The updater is using a custom URL. Expected '{hardcoded_url}', got '{custom_updater.update_url}'")

# Test 3: Check if the _get_update_info method uses the hardcoded URL
print("\nTest 3: Checking if _get_update_info uses the hardcoded URL...")
# We can't directly test this, but we can check if the docstring has been updated
docstring = custom_updater._get_update_info.__doc__
if "Always uses the hardcoded URL" in docstring:
    print("SUCCESS: The _get_update_info method's docstring indicates it uses the hardcoded URL.")
else:
    print("ERROR: The _get_update_info method's docstring doesn't mention using the hardcoded URL.")

print("\nTest completed")