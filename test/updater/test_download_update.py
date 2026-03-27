import sys
import os
import pathlib
import tempfile

# Add the parent directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chisurf.plugins.updater.updater import ChiSurfUpdater

# Create a temporary directory to store logs
temp_dir = tempfile.mkdtemp(prefix="chisurf_test_")
log_file = os.path.join(temp_dir, "update_log.txt")

# Create a callback function to log progress
def log_callback(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Create an updater instance with a remote URL
remote_url = "https://www.peulen.xyz/downloads/chisurf/conda/"
updater = ChiSurfUpdater(update_url=remote_url)
print(f"Update URL: {updater.update_url}")

# Get update info to find the latest version
update_info = updater._get_update_info()
if not update_info or "available_versions" not in update_info or not update_info["available_versions"]:
    print("No available versions found")
    sys.exit(1)

# Get the latest version
latest_version = update_info["available_versions"][0]
print(f"Latest version: {latest_version['version']}")
print(f"File path: {latest_version['file_path']}")

# Print information about the test
print("\nThis test will attempt to download the update file but will not install it.")

# Create a mock update_to_version method that only downloads the file
def mock_update_to_version(self, file_path, callback=None, auto_restart=False):
    """
    Mock version of update_to_version that only downloads the file.
    """
    try:
        # Report progress
        if callback:
            callback(f"Preparing to update from file: {file_path}")

        # Check if it's a remote URL
        is_remote_url = bool(re.match(r'^(https?|ftp)://', file_path))
        local_file_path = file_path

        # If it's a remote URL, download it to a temporary file first
        if is_remote_url:
            if callback:
                callback(f"Downloading update file from: {file_path}")

            try:
                # Create a temporary directory to store the downloaded file
                temp_dir = tempfile.mkdtemp(prefix="chisurf_update_")

                # Extract the filename from the URL
                file_name = os.path.basename(file_path)
                if not file_name:
                    file_name = "chisurf_update.tar.bz2"

                # Create the local file path
                local_file_path = os.path.join(temp_dir, file_name)

                # Download the file
                import urllib.request
                urllib.request.urlretrieve(
                    file_path, 
                    local_file_path,
                    reporthook=lambda count, block_size, total_size: callback(
                        f"Downloading: {count * block_size / (1024 * 1024):.1f} MB of {total_size / (1024 * 1024):.1f} MB"
                    ) if callback and total_size > 0 else None
                )

                if callback:
                    callback(f"Download complete. Saved to: {local_file_path}")

                # Verify the file exists
                if os.path.exists(local_file_path):
                    callback(f"File exists at: {local_file_path}")
                    callback(f"File size: {os.path.getsize(local_file_path) / (1024 * 1024):.1f} MB")
                    return True, None
                else:
                    return False, f"Downloaded file not found at: {local_file_path}"
            except Exception as e:
                return False, f"Failed to download update file: {str(e)}"
        else:
            callback(f"Not a remote URL, skipping download: {file_path}")
            return True, None

    except Exception as e:
        return False, f"Error during update: {str(e)}"

# Replace the update_to_version method with our mock version
import re
import types
updater.update_to_version = types.MethodType(mock_update_to_version, updater)

# Call the mock update_to_version method
print("\nTesting download functionality...")
success, error = updater.update_to_version(latest_version['file_path'], callback=log_callback)

if success:
    print("\nDownload test successful!")
else:
    print(f"\nDownload test failed: {error}")

print(f"Log file: {log_file}")
print("\nTest completed")
