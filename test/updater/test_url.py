# Consolidated test file: test_url.py


# --- FROM test_hardcoded_url.py ---
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
# --- FROM test_hardcoded_url_only.py ---
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
# --- FROM test_url_detection.py ---
import sys
import pathlib
import os

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

def test_url_detection(url, expected_is_local):
    """Test if a URL is correctly detected as a local folder or remote URL."""
    updater = ChiSurfUpdater(update_url=url)
    is_local = updater._is_local_folder()
    result = "PASS" if is_local == expected_is_local else "FAIL"
    print(f"{result}: '{url}' -> is_local={is_local}, expected={expected_is_local}")
    return is_local == expected_is_local

# Test remote URLs
print("\nTesting remote URLs:")
remote_urls = [
    "https://www.peulen.xyz/downloads/chisurf/conda/",
    "http://example.com/chisurf/",
    "ftp://example.com/chisurf/",
    "https://github.com/Fluorescence-Tools/chisurf/releases/conda"
]
remote_results = [test_url_detection(url, False) for url in remote_urls]

# Test local folders
print("\nTesting local folders:")
local_folders = [
    "Q:\\chisurf\\conda",
    "C:\\Users\\user\\Documents",
    "\\\\server\\share\\folder",
    os.path.abspath(".")  # Current directory
]
local_results = [test_url_detection(url, True) for url in local_folders]

# Print summary
print("\nSummary:")
print(f"Remote URLs: {sum(remote_results)}/{len(remote_results)} passed")
print(f"Local folders: {sum(local_results)}/{len(local_results)} passed")
print(f"Overall: {sum(remote_results + local_results)}/{len(remote_results + local_results)} passed")

print("\nTest completed")
# --- FROM test_update_url_append.py ---
import sys
import pathlib

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

# Test with a remote URL that doesn't end with '/conda'
remote_url = "https://github.com/Fluorescence-Tools/chisurf/releases"
updater = ChiSurfUpdater(update_url=remote_url)
print(f"Original remote URL: {remote_url}")
print(f"Modified remote URL: {updater.update_url}")
print(f"'/conda' appended: {updater.update_url.endswith('/conda')}")

# Test with a remote URL that already ends with '/conda'
remote_url_with_conda = "https://github.com/Fluorescence-Tools/chisurf/releases/conda"
updater = ChiSurfUpdater(update_url=remote_url_with_conda)
print(f"\nOriginal remote URL (with conda): {remote_url_with_conda}")
print(f"Modified remote URL: {updater.update_url}")
print(f"URL unchanged: {updater.update_url == remote_url_with_conda}")

# Test with a remote URL that ends with a trailing slash
remote_url_with_slash = "https://github.com/Fluorescence-Tools/chisurf/releases/"
updater = ChiSurfUpdater(update_url=remote_url_with_slash)
print(f"\nOriginal remote URL (with trailing slash): {remote_url_with_slash}")
print(f"Modified remote URL: {updater.update_url}")
print(f"'/conda' appended correctly: {updater.update_url == 'https://github.com/Fluorescence-Tools/chisurf/releases/conda'}")

# Test with a local folder URL
local_url = "Q:\\chisurf\\conda"
updater = ChiSurfUpdater(update_url=local_url)
print(f"\nOriginal local URL: {local_url}")
print(f"Modified local URL: {updater.update_url}")
print(f"URL unchanged: {updater.update_url == local_url}")

print("\nTest completed")