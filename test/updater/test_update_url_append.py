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