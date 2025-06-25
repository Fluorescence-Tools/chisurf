"""
ChiSurf Update Mechanism
========================

This module provides functionality to update ChiSurf using conda packages.
It supports updating on Windows, macOS, and Linux, and handles elevated
privileges when needed.

Usage:
------
To check for updates:
```python
from chisurf.plugins.updater import check_for_updates
update_available, latest_version, error = check_for_updates()
```

To perform an update:
```python
from chisurf.plugins.updater import update_chisurf
success, error = update_chisurf(callback=lambda msg: print(msg))
```

The update mechanism works by:
1. Checking for updates from a specified URL
2. Downloading and installing updates using conda
3. Handling platform-specific update logic (Windows, macOS, Linux)
4. Handling elevated privileges when needed
5. Informing the user to restart the application manually after updating

The update process will close all ChiSurf windows and continue in a separate window.
After the update completes, the user will need to restart ChiSurf manually.
"""

from __future__ import annotations

import os
import sys
import json
import platform
import subprocess
import pathlib
import tempfile
import time
import urllib.request
import urllib.error
import re
import shutil
from typing import Optional, Dict, Any, Tuple, List, Callable

from chisurf.settings.file_utils import safe_open_file
from chisurf.settings.path_utils import get_path
from chisurf import info

class ChiSurfUpdater:
    """
    A class to handle the updating of ChiSurf via conda packages.

    This updater can:
    1. Check for updates from a specified URL
    2. Download and install updates using conda
    3. Handle platform-specific update logic (Windows, macOS, Linux)
    4. Restart the application after updating
    """

    def __init__(self, update_url: Optional[str] = None, channel: str = "main"):
        """
        Initialize the updater.

        Args:
            update_url: URL to check for updates. If None, uses the default from settings or info.py
            channel: Conda channel to use for updates
        """
        # Import here to avoid circular imports
        from chisurf.settings import cs_settings

        # Initialize system attribute first
        self.system = platform.system().lower()

        # Get update_url from settings if available
        settings_update_url = cs_settings.get('update_url', None)

        # Use the provided URL, or the one from settings, or fall back to info.py
        url = update_url or settings_update_url or info.update_url

        # Check if the URL is a local folder using the improved logic
        is_local_folder = False
        if url:
            # Check if it's a URL (starts with http://, https://, ftp://, etc.)
            if re.match(r'^(https?|ftp)://', url):
                is_local_folder = False
            else:
                # Try to convert to a Path object and check if it exists
                try:
                    path = pathlib.Path(url)
                    # If the path exists and is a directory, it's a local folder
                    if path.exists() and path.is_dir():
                        is_local_folder = True
                    else:
                        # If it doesn't exist yet, check if it looks like a local path
                        if self.system == "windows":
                            # Windows path patterns: drive letter, UNC path, or absolute path
                            is_local_folder = bool(re.match(r'^[a-zA-Z]:\\', url) or  # Drive letter
                                                 re.match(r'^\\\\', url) or         # UNC path
                                                 os.path.isabs(url))                # Absolute path
                        else:
                            # Unix path pattern: starts with / or ~
                            is_local_folder = url.startswith('/') or url.startswith('~')
                except:
                    # If there's an error, assume it's not a local folder
                    is_local_folder = False

        # If it's not a local folder and doesn't already end with '/conda', append '/conda'
        if not is_local_folder and not url.endswith('/conda'):
            # Remove trailing slash if present
            if url.endswith('/'):
                url = url[:-1]
            # Append '/conda'
            url += '/conda'

        self.update_url = url
        self.channel = channel
        self.current_version = info.__version__
        self.settings_path = get_path('settings')
        self.update_info_file = self.settings_path / 'update_info.json'

    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if updates are available.

        Returns:
            Tuple containing:
            - Boolean indicating if an update is available
            - Latest version string if update is available, None otherwise
            - Error message if an error occurred, None otherwise
        """
        try:
            # In a real implementation, this would fetch version info from a server
            # For now, we'll simulate by checking a local file or creating one if it doesn't exist
            update_info = self._get_update_info()

            if not update_info:
                # If no update info exists, create a default one
                update_info = {
                    "latest_version": self.current_version,
                    "package_url": f"https://example.com/chisurf/{self.current_version}/chisurf-{self.current_version}.tar.bz2",
                    "release_notes": "Initial version",
                    "channels": ["conda-forge", "defaults"]
                }
                self._save_update_info(update_info)
                return False, None, None

            # Compare versions (in a real implementation, this would be more sophisticated)
            # For now, we'll just compare the version strings
            if update_info["latest_version"] > self.current_version:
                return True, update_info["latest_version"], None

            return False, None, None

        except Exception as e:
            return False, None, f"Error checking for updates: {str(e)}"

    def update_to_version(self, file_path: str, callback=None, auto_restart=True) -> Tuple[bool, Optional[str]]:
        """
        Update ChiSurf to a specific version using the provided file.

        Args:
            file_path: Path to the update file
            callback: Optional callback function to report progress
            auto_restart: This parameter is ignored. The user must restart manually after the update.

        Returns:
            Tuple containing:
            - Boolean indicating if the update was successful
            - Error message if an error occurred, None otherwise

        Note:
            The update process will close all ChiSurf windows and continue in a separate window.
            After the update completes, the user will need to restart ChiSurf manually.
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
                    urllib.request.urlretrieve(
                        file_path, 
                        local_file_path,
                        reporthook=lambda count, block_size, total_size: callback(
                            f"Downloading: {count * block_size / (1024 * 1024):.1f} MB of {total_size / (1024 * 1024):.1f} MB"
                        ) if callback and total_size > 0 else None
                    )

                    if callback:
                        callback(f"Download complete. Saved to: {local_file_path}")
                except Exception as e:
                    return False, f"Failed to download update file: {str(e)}"

            # Check if the file exists
            if not os.path.exists(local_file_path):
                return False, f"Update file not found: {local_file_path}"

            # Determine if we need elevated privileges
            needs_elevation = self._needs_elevation()

            # Prepare the update command based on the file type
            file_ext = os.path.splitext(local_file_path)[1].lower()

            if file_ext == '.exe':
                # For Windows installers
                cmd = [local_file_path, '/S']  # Silent install
            elif file_ext in ['.msi', '.msix']:
                # For Windows MSI packages
                cmd = ['msiexec', '/i', local_file_path, '/quiet', '/norestart']
            elif file_ext in ['.tar.bz2', '.tar.gz', '.bz2', '.gz', '.conda']:
                # For conda packages
                conda_exe = self._get_conda_executable()
                # Get the environment path (where chisurf is installed)
                env_path = sys.prefix
                cmd = [conda_exe, 'install', '--yes', '--force-reinstall', '--prefix', env_path, local_file_path]
            else:
                # Unknown file type
                return False, f"Unsupported update file type: {file_ext}"

            # Create a string representation of the command for display
            cmd_str = " ".join(cmd)

            # Show the command to the user
            if callback:
                callback(f"Command: {cmd_str}")

            # Display a warning that all ChiSurf windows will be closed
            if callback:
                callback("WARNING: All ChiSurf windows will be closed before starting the update.")
                callback("The update will continue in a separate window.")

            # Run the update in a separate process
            return self._run_update_in_separate_process(cmd, callback)

        except Exception as e:
            return False, f"Error during update: {str(e)}"

    def update(self, callback=None, auto_restart=True) -> Tuple[bool, Optional[str]]:
        """
        Update ChiSurf to the latest version.

        Args:
            callback: Optional callback function to report progress
            auto_restart: This parameter is ignored. The user must restart manually after the update.

        Returns:
            Tuple containing:
            - Boolean indicating if the update was successful
            - Error message if an error occurred, None otherwise

        Note:
            The update process will close all ChiSurf windows and continue in a separate window.
            After the update completes, the user will need to restart ChiSurf manually.
        """
        try:
            # Check if an update is available
            update_available, latest_version, error = self.check_for_updates()

            if error:
                return False, error

            if not update_available:
                return True, "Already up to date"

            # Get update information
            update_info = self._get_update_info()
            if not update_info:
                return False, "Update information not available"

            # Report progress
            if callback:
                callback("Preparing to update...")

            # If the update URL is a local folder and we have available versions,
            # use the first (latest) version
            if self._is_local_folder() and "available_versions" in update_info and update_info["available_versions"]:
                latest_version_info = update_info["available_versions"][0]
                return self.update_to_version(latest_version_info["file_path"], callback)

            # Otherwise, use the standard update mechanism
            # Determine if we need elevated privileges
            needs_elevation = self._needs_elevation()

            # Prepare the update command
            cmd, cmd_str = self._prepare_update_command(update_info)

            # Show the command to the user
            if callback:
                callback(f"Command: {cmd_str}")

            # Display a warning that all ChiSurf windows will be closed
            if callback:
                callback("WARNING: All ChiSurf windows will be closed before starting the update.")
                callback("The update will continue in a separate window.")

            # Run the update in a separate process
            return self._run_update_in_separate_process(cmd, callback)

        except Exception as e:
            return False, f"Error during update: {str(e)}"

    def _is_local_folder(self) -> bool:
        """
        Check if the update URL is a local folder.

        Returns:
            Boolean indicating if the update URL is a local folder
        """
        # Check if it's a URL (starts with http://, https://, ftp://, etc.)
        if re.match(r'^(https?|ftp)://', self.update_url):
            return False

        # Try to convert to a Path object and check if it exists
        try:
            path = pathlib.Path(self.update_url)
            # If the path exists and is a directory, it's a local folder
            if path.exists() and path.is_dir():
                return True
            else:
                # If it doesn't exist yet, check if it looks like a local path
                if self.system == "windows":
                    # Windows path patterns: drive letter, UNC path, or absolute path
                    return bool(re.match(r'^[a-zA-Z]:\\', self.update_url) or  # Drive letter
                               re.match(r'^\\\\', self.update_url) or         # UNC path
                               os.path.isabs(self.update_url))                # Absolute path
                else:
                    # Unix path pattern: starts with / or ~
                    return self.update_url.startswith('/') or self.update_url.startswith('~')
        except:
            # If there's an error, assume it's not a local folder
            return False

    def _process_update_file(self, file: pathlib.Path) -> List[Dict[str, Any]]:
        """
        Process an update file to extract version information.

        Args:
            file: Path to the update file

        Returns:
            List of dictionaries containing version information
        """
        file_name = file.name.lower()

        # Try to extract version from filename
        # First try the standard format (X.Y.Z)
        version_match = re.search(r'(\d+\.\d+\.\d+)', file_name)

        # If that doesn't work, try the date format (YY.MM.DD)
        if not version_match:
            version_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', file_name)

        # If that doesn't work, try just finding any sequence of digits
        if not version_match:
            version_match = re.search(r'(\d+)', file_name)

        if not version_match:
            return []

        version = version_match.group(1)

        # Return version information
        return [{
            "version": version,
            "file_path": str(file),
            "file_name": file.name
        }]

    def _list_available_versions(self) -> List[Dict[str, Any]]:
        """
        List available versions from the update folder.

        Returns:
            List of dictionaries containing version information
        """
        if not self._is_local_folder():
            return []

        try:
            # Convert the update URL to a Path object
            update_path = pathlib.Path(self.update_url)

            # Check if the path exists
            if not update_path.exists():
                return []

            if not update_path.is_dir():
                return []

            # Get the current OS
            current_os = self.system

            # List of supported OS names to look for in filenames
            os_names = {
                "windows": ["win", "windows"],
                "darwin": ["mac", "macos", "darwin"],
                "linux": ["linux"]
            }

            # Get the OS-specific names to look for
            os_specific_names = os_names.get(current_os, [current_os])

            # List all files in the directory and subdirectories
            versions = []

            # First, check if there are OS-specific subdirectories
            os_subdirs = []
            for subdir in update_path.glob("*"):
                if subdir.is_dir():
                    subdir_name = subdir.name.lower()
                    if any(os_name in subdir_name for os_name in os_specific_names):
                        os_subdirs.append(subdir)

            # If we found OS-specific subdirectories, scan them
            if os_subdirs:
                for subdir in os_subdirs:
                    for file in subdir.glob("*"):
                        if file.is_file():
                            # Skip files that don't contain "chisurf"
                            file_name = file.name.lower()
                            if "chisurf" not in file_name:
                                continue
                            versions.extend(self._process_update_file(file))

            # Also scan the main directory
            for file in update_path.glob("*"):
                # Skip directories
                if file.is_dir():
                    continue

                # Skip files that don't match the current OS
                file_name = file.name.lower()
                if not any(os_name in file_name for os_name in os_specific_names):
                    continue

                # Skip files that don't contain "chisurf"
                if "chisurf" not in file_name:
                    continue

                # Process the file to extract version information
                file_versions = self._process_update_file(file)
                versions.extend(file_versions)

            # Sort versions by version number (newest first)
            versions.sort(key=lambda x: x["version"], reverse=True)
            return versions

        except Exception as e:
            print(f"Error listing available versions: {str(e)}")
            return []

    def _list_remote_versions(self) -> List[Dict[str, Any]]:
        """
        List available versions from a remote HTTP source.

        Returns:
            List of dictionaries containing version information
        """
        if self._is_local_folder():
            return []

        try:
            # Get the current OS
            current_os = self.system

            # List of supported OS names to look for in filenames
            os_names = {
                "windows": ["win", "windows"],
                "darwin": ["mac", "macos", "darwin"],
                "linux": ["linux"]
            }

            # Get the OS-specific names to look for
            os_specific_names = os_names.get(current_os, [current_os])

            # Ensure the URL ends with a slash
            url = self.update_url
            if not url.endswith('/'):
                url += '/'

            # Function to extract version from filename
            def extract_version(filename):
                # First try the standard format (X.Y.Z)
                version_match = re.search(r'(\d+\.\d+\.\d+)', filename)

                # If that doesn't work, try the date format (YY.MM.DD)
                if not version_match:
                    version_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', filename)

                # If that doesn't work, try just finding any sequence of digits
                if not version_match:
                    version_match = re.search(r'(\d+)', filename)

                if not version_match:
                    return None

                return version_match.group(1)

            # Function to process a file link
            def process_file_link(link, base_url, skip_os_check=False):
                # Skip if it's a directory link (ends with /)
                if link.endswith('/'):
                    return None

                # Skip if it doesn't match the current OS (unless skip_os_check is True)
                link_lower = link.lower()
                if not skip_os_check and not any(os_name in link_lower for os_name in os_specific_names):
                    return None

                # Skip if it doesn't contain "chisurf"
                if "chisurf" not in link_lower:
                    return None

                # Try to extract version from filename
                version = extract_version(link_lower)
                if not version:
                    return None

                # Return version information
                return {
                    "version": version,
                    "file_path": base_url + link,
                    "file_name": link
                }

            # Function to fetch and parse HTML from a URL
            def fetch_and_parse_html(url):
                try:
                    with urllib.request.urlopen(url) as response:
                        html = response.read().decode('utf-8')

                    # Look for href attributes in the HTML
                    links = re.findall(r'href=[\'"]?([^\'" >]+)', html)
                    return links
                except urllib.error.URLError as e:
                    print(f"Error fetching directory listing from {url}: {e}")
                    return []

            # List to store all versions
            versions = []

            # First, fetch the main directory
            main_links = fetch_and_parse_html(url)

            # Check for OS-specific subdirectories
            os_subdirs = []
            for link in main_links:
                if link.endswith('/') and not link.startswith('..'):
                    link_lower = link.lower()
                    if any(os_name in link_lower for os_name in os_specific_names):
                        os_subdirs.append(link)

            # Process files in the main directory
            for link in main_links:
                version_info = process_file_link(link, url)
                if version_info:
                    versions.append(version_info)

            # Process files in OS-specific subdirectories
            for subdir in os_subdirs:
                subdir_url = url + subdir
                subdir_links = fetch_and_parse_html(subdir_url)

                for link in subdir_links:
                    # Skip OS check for files in OS-specific subdirectories
                    version_info = process_file_link(link, subdir_url, skip_os_check=True)
                    if version_info:
                        versions.append(version_info)

            # Sort versions by version number (newest first)
            versions.sort(key=lambda x: x["version"], reverse=True)
            return versions

        except Exception as e:
            print(f"Error listing remote versions: {str(e)}")
            return []

    def _get_update_info(self) -> Optional[Dict[str, Any]]:
        """
        Get update information from the local cache or remote server.

        Returns:
            Dictionary containing update information, or None if not available
        """
        # First try to get from local cache
        if self.update_info_file.exists():
            update_info = safe_open_file(
                self.update_info_file,
                processor=json.load,
                default_value=None
            )
            if update_info:
                return update_info

        # Check if the update URL is a local folder
        if self._is_local_folder():
            # Get available versions from the folder
            versions = self._list_available_versions()

            if not versions:
                return None

            # Use the newest version
            latest_version = versions[0]

            # Create update info
            update_info = {
                "latest_version": latest_version["version"],
                "package_url": latest_version["file_path"],
                "release_notes": f"Version {latest_version['version']} from local folder",
                "channels": ["conda-forge", "defaults"],
                "available_versions": versions
            }

            # Save to cache
            self._save_update_info(update_info)

            return update_info
        else:
            # For remote URLs, try to fetch available versions
            versions = self._list_remote_versions()

            if versions:
                # Use the newest version
                latest_version = versions[0]

                # Create update info
                update_info = {
                    "latest_version": latest_version["version"],
                    "package_url": latest_version["file_path"],
                    "release_notes": f"Version {latest_version['version']} from remote server",
                    "channels": ["conda-forge", "defaults"],
                    "available_versions": versions
                }

                # Save to cache
                self._save_update_info(update_info)

                return update_info

            # If we couldn't get versions from the remote URL, fall back to a dummy update info
            try:
                # In a real implementation, this would fetch from a server
                # For demonstration, we'll create a dummy update info
                # This should be replaced with actual server communication
                update_info = {
                    "latest_version": "99.99.99",  # A future version
                    "package_url": "https://example.com/chisurf/latest/chisurf-latest.tar.bz2",
                    "release_notes": "This is a newer version with many improvements",
                    "channels": ["conda-forge", "defaults"]
                }

                # Save to cache
                self._save_update_info(update_info)

                return update_info
            except Exception as e:
                print(f"Error fetching update information: {str(e)}")
                return None

    def _save_update_info(self, update_info: Dict[str, Any]) -> None:
        """
        Save update information to the local cache.

        Args:
            update_info: Dictionary containing update information
        """
        try:
            with open(self.update_info_file, 'w') as f:
                json.dump(update_info, f, indent=2)
        except Exception as e:
            print(f"Error saving update information: {str(e)}")

    def _needs_elevation(self) -> bool:
        """
        Determine if elevated privileges are needed for the update.

        Returns:
            Boolean indicating if elevated privileges are needed
        """
        if self.system == "windows":
            # Check if the installation directory is in Program Files
            chisurf_path = get_path('chisurf')
            program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
            program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')

            return (str(chisurf_path).startswith(program_files) or 
                    str(chisurf_path).startswith(program_files_x86))

        # On Unix-like systems, check if the conda environment is in a system directory
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        return conda_prefix.startswith('/usr') and not conda_prefix.startswith('/usr/local')

    def _prepare_update_command(self, update_info: Dict[str, Any]) -> Tuple[List[str], str]:
        """
        Prepare the command to update ChiSurf.

        Args:
            update_info: Dictionary containing update information

        Returns:
            Tuple containing:
            - List of command arguments
            - String representation of the command for display
        """
        # Get the conda executable
        conda_exe = self._get_conda_executable()

        # Prepare channels
        channels = update_info.get("channels", ["conda-forge", "defaults"])
        channel_args = []
        for channel in channels:
            channel_args.extend(["-c", channel])

        # Get the environment path (where chisurf is installed)
        env_path = sys.prefix

        # Prepare the command with the environment path
        cmd = [
            conda_exe, "update", "-y", "--prefix", env_path, "chisurf",
            *channel_args
        ]

        # Create a string representation for display
        cmd_str = " ".join(cmd)

        return cmd, cmd_str

    def _get_conda_executable(self) -> str:
        """
        Get the path to the conda executable.

        Returns:
            Path to the conda executable
        """
        # Try to get from environment
        conda_exe = os.environ.get('CONDA_EXE', '')
        if conda_exe and os.path.exists(conda_exe):
            return conda_exe

        # Try common locations
        if self.system == "windows":
            # Get the application directory (where ChiSurf is installed)
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            conda_locations = [
                # First check the pip-installed conda executable
                os.path.join(app_dir, "Scripts", "conda.exe"),
                # Then check standard locations
                os.path.join(sys.prefix, "Scripts", "conda.exe"),
                os.path.join(sys.prefix, "condabin", "conda.bat"),
                os.path.join(os.environ.get('CONDA_PREFIX', ''), "Scripts", "conda.exe"),
                os.path.join(os.environ.get('CONDA_PREFIX', ''), "condabin", "conda.bat")
            ]
        else:
            # Get the application directory (where ChiSurf is installed)
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            conda_locations = [
                # First check the pip-installed conda executable
                os.path.join(app_dir, "bin", "conda"),
                # Then check standard locations
                os.path.join(sys.prefix, "bin", "conda"),
                os.path.join(os.environ.get('CONDA_PREFIX', ''), "bin", "conda")
            ]

        for location in conda_locations:
            if os.path.exists(location):
                # Set the CONDA_EXE environment variable to ensure conda commands use this executable
                os.environ['CONDA_EXE'] = location
                return location

        # If we can't find conda, try to use the system PATH
        return "conda"

    def _run_command(self, cmd: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Run a command and return the result.

        Args:
            cmd: Command to run as a list of arguments

        Returns:
            Tuple containing:
            - Boolean indicating if the command was successful
            - Error message if the command failed, None otherwise
        """
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                return False, f"Command failed with exit code {process.returncode}: {stderr}"

            return True, None
        except Exception as e:
            return False, str(e)

    def _run_with_elevation(self, cmd: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Run a command with elevated privileges on Windows.

        Args:
            cmd: Command to run as a list of arguments

        Returns:
            Tuple containing:
            - Boolean indicating if the command was successful
            - Error message if the command failed, None otherwise
        """
        if self.system != "windows":
            return self._run_command(cmd)

        try:
            # Create a temporary batch file to run the command
            with tempfile.NamedTemporaryFile(suffix='.bat', delete=False, mode='w') as f:
                f.write('@echo off\n')
                f.write(' '.join(cmd) + '\n')
                f.write('if %ERRORLEVEL% NEQ 0 (\n')
                f.write('  echo Update failed with error code %ERRORLEVEL%\n')
                f.write('  exit /b %ERRORLEVEL%\n')
                f.write(')\n')
                f.write('echo Update successful\n')
                batch_file = f.name

            # Run the batch file with elevated privileges using PowerShell
            powershell_cmd = [
                'powershell.exe', '-Command',
                f'Start-Process -FilePath "{batch_file}" -Verb RunAs -Wait'
            ]

            process = subprocess.Popen(
                powershell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            # Clean up the temporary file
            try:
                os.unlink(batch_file)
            except:
                pass

            if process.returncode != 0:
                return False, f"Elevation failed with exit code {process.returncode}: {stderr}"

            return True, None
        except Exception as e:
            return False, str(e)

    def _run_update_in_separate_process(self, cmd: List[str], callback=None) -> Tuple[bool, Optional[str]]:
        """
        Run the update in a separate process after closing ChiSurf.

        Args:
            cmd: Command to run as a list of arguments
            callback: Optional callback function to report progress

        Returns:
            Tuple containing:
            - Boolean indicating if the update was started successfully
            - Error message if an error occurred, None otherwise
        """
        try:
            # Report progress
            if callback:
                callback("Preparing to run update in a separate process...")
                callback("WARNING: All ChiSurf windows will be closed before starting the update.")

            # Check if we need elevated privileges
            needs_elevation = self._needs_elevation()
            if needs_elevation and callback:
                callback("Administrator privileges will be required for this update.")

            # Create a temporary directory to store the update script
            temp_dir = tempfile.mkdtemp(prefix="chisurf_update_")

            # Create a string representation of the command for display
            cmd_str = " ".join(cmd)

            # Create the update script
            if self.system == "windows":
                # Create a properly quoted command string for Windows batch file execution
                # This ensures paths with spaces are handled correctly
                win_cmd_str = " ".join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])

                # On Windows, use a batch file
                update_script_path = os.path.join(temp_dir, "update_chisurf.bat")
                with open(update_script_path, 'w') as f:
                    f.write('@echo off\n')
                    f.write('title ChiSurf Update\n')
                    f.write('echo ChiSurf Update Process\n')
                    f.write('echo =====================\n')
                    f.write('echo.\n')
                    f.write('echo Starting update process...\n')
                    f.write('echo Script: ' + update_script_path + '\n')
                    f.write('echo Command: ' + cmd_str + '\n')
                    f.write('echo.\n')
                    f.write('echo Update in progress. Please wait...\n')
                    f.write('echo.\n')
                    f.write(win_cmd_str + '\n')
                    f.write('if %ERRORLEVEL% NEQ 0 (\n')
                    f.write('  echo.\n')
                    f.write('  echo Update failed with error code %ERRORLEVEL%\n')
                    f.write('  echo.\n')
                    f.write('  echo Press any key to close this window...\n')
                    f.write('  pause > nul\n')
                    f.write('  exit /b %ERRORLEVEL%\n')
                    f.write(')\n')
                    f.write('echo.\n')
                    f.write('echo Update successful!\n')
                    f.write('echo.\n')
                    f.write('echo Please restart ChiSurf manually to complete the update.\n')
                    f.write('echo.\n')
                    f.write('echo Press any key to close this window...\n')
                    f.write('pause > nul\n')

                # Create a launcher script that will be executed to start the update process
                launcher_script_path = os.path.join(temp_dir, "start_update.bat")
                with open(launcher_script_path, 'w') as f:
                    f.write('@echo off\n')
                    f.write('timeout /t 1 /nobreak >nul\n')  # Wait a bit for the current process to exit

                    # If we need elevated privileges, use PowerShell to run the script as administrator
                    if needs_elevation:
                        f.write('echo Administrator privileges are required for this update.\n')
                        f.write('echo The User Account Control (UAC) dialog may appear.\n')
                        f.write('echo Please click "Yes" to allow the update to proceed.\n')
                        f.write('echo.\n')
                        f.write('powershell.exe -Command "Start-Process -FilePath \\"' + update_script_path + '\\" -Verb RunAs"\n')
                    else:
                        f.write(f'start "" "{update_script_path}"\n')

                # Make the script executable
                os.chmod(update_script_path, 0o755)
                os.chmod(launcher_script_path, 0o755)

                # Run the launcher script
                subprocess.Popen(['cmd', '/c', launcher_script_path], shell=True)
            else:
                # On Unix-like systems, use a shell script
                update_script_path = os.path.join(temp_dir, "update_chisurf.sh")

                # Create a properly quoted command string for shell script execution
                # This ensures paths with spaces are handled correctly
                shell_cmd_str = " ".join([f"'{arg}'" if ' ' in arg else arg for arg in cmd])

                with open(update_script_path, 'w') as f:
                    f.write('#!/bin/sh\n')
                    f.write('echo "ChiSurf Update Process"\n')
                    f.write('echo "====================="\n')
                    f.write('echo\n')
                    f.write('echo "Starting update process..."\n')
                    f.write('echo "Script: ' + update_script_path + '"\n')
                    f.write('echo "Command: ' + cmd_str + '"\n')
                    f.write('echo\n')
                    f.write('echo "Update in progress. Please wait..."\n')
                    f.write('echo\n')
                    f.write(shell_cmd_str + '\n')
                    f.write('if [ $? -ne 0 ]; then\n')
                    f.write('  echo\n')
                    f.write('  echo "Update failed with error code $?"\n')
                    f.write('  echo\n')
                    f.write('  echo "Press Enter to close this window..."\n')
                    f.write('  read\n')
                    f.write('  exit $?\n')
                    f.write('fi\n')
                    f.write('echo\n')
                    f.write('echo "Update successful!"\n')
                    f.write('echo\n')
                    f.write('echo "Please restart ChiSurf manually to complete the update."\n')
                    f.write('echo\n')
                    f.write('echo "Press Enter to close this window..."\n')
                    f.write('read\n')

                # Create a launcher script that will be executed to start the update process
                launcher_script_path = os.path.join(temp_dir, "start_update.sh")
                with open(launcher_script_path, 'w') as f:
                    f.write('#!/bin/sh\n')
                    f.write('sleep 1\n')  # Wait a bit for the current process to exit

                    # If we need elevated privileges, use sudo or pkexec to run the script as administrator
                    if needs_elevation:
                        f.write('echo "Administrator privileges are required for this update."\n')
                        f.write('echo "You may be prompted for your password."\n')
                        f.write('echo\n')

                        # Try pkexec first (for desktop environments), then sudo
                        f.write('if command -v pkexec >/dev/null 2>&1; then\n')
                        f.write(f'  xterm -e "pkexec {update_script_path}" &\n')
                        f.write('elif command -v sudo >/dev/null 2>&1; then\n')
                        f.write(f'  xterm -e "sudo {update_script_path}" &\n')
                        f.write('else\n')
                        f.write('  echo "Error: Neither pkexec nor sudo is available. Cannot elevate privileges."\n')
                        f.write(f'  xterm -e "{update_script_path}" &\n')
                        f.write('fi\n')
                    else:
                        f.write(f'xterm -e "{update_script_path}" &\n')

                # Make the scripts executable
                os.chmod(update_script_path, 0o755)
                os.chmod(launcher_script_path, 0o755)

                # Run the launcher script
                subprocess.Popen(['/bin/sh', launcher_script_path])

            # Report progress
            if callback:
                callback("Update process started in a separate window.")
                callback("ChiSurf will now close.")

            # Exit the current process
            sys.exit(0)

            return True, None
        except Exception as e:
            return False, f"Error starting update process: {str(e)}"

    def _schedule_restart(self) -> None:
        """
        Schedule the application to restart after the update.
        """
        # Get the current executable
        executable = sys.executable

        # Get the script that was run
        script = sys.argv[0]

        # Get the arguments
        args = sys.argv[1:]

        # Prepare the restart command
        if self.system == "windows":
            # On Windows, use a batch file to restart
            with tempfile.NamedTemporaryFile(suffix='.bat', delete=False, mode='w') as f:
                f.write('@echo off\n')
                f.write('timeout /t 1 /nobreak >nul\n')  # Wait a bit for the current process to exit
                f.write(f'start "" "{executable}" "{script}" {" ".join(args)}\n')
                restart_script = f.name

            # Run the restart script
            subprocess.Popen(['cmd', '/c', restart_script], shell=True)
        else:
            # On Unix-like systems, use a shell script
            with tempfile.NamedTemporaryFile(suffix='.sh', delete=False, mode='w') as f:
                f.write('#!/bin/sh\n')
                f.write('sleep 1\n')  # Wait a bit for the current process to exit
                f.write(f'"{executable}" "{script}" {" ".join(args)} &\n')
                restart_script = f.name

            # Make the script executable
            os.chmod(restart_script, 0o755)

            # Run the restart script
            subprocess.Popen(['/bin/sh', restart_script])

        # Exit the current process
        sys.exit(0)

def check_for_updates() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if updates are available for ChiSurf.

    Returns:
        Tuple containing:
        - Boolean indicating if an update is available
        - Latest version string if update is available, None otherwise
        - Error message if an error occurred, None otherwise
    """
    updater = ChiSurfUpdater()
    return updater.check_for_updates()

def update_chisurf(callback=None, auto_restart=True) -> Tuple[bool, Optional[str]]:
    """
    Update ChiSurf to the latest version.

    Args:
        callback: Optional callback function to report progress
        auto_restart: This parameter is ignored. The user must restart manually after the update.

    Returns:
        Tuple containing:
        - Boolean indicating if the update was successful
        - Error message if an error occurred, None otherwise

    Note:
        The update process will close all ChiSurf windows and continue in a separate window.
        After the update completes, the user will need to restart ChiSurf manually.
    """
    updater = ChiSurfUpdater()
    return updater.update(callback, auto_restart)
