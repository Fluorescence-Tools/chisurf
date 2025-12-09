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