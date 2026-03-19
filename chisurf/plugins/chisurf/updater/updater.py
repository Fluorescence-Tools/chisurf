"""
ChiSurf Update Mechanism
========================

This module provides functionality to update ChiSurf.
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
2. Downloading and installing updates using the package manager
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
import urllib.parse
import re
import shutil
import logging
from typing import Optional, Dict, Any, Tuple, List, Callable
from datetime import datetime, timedelta, timezone

from chisurf.settings.file_utils import safe_open_file
from chisurf.settings.path_utils import get_path
from chisurf import info

class ChiSurfUpdater:
    """
    A class to handle the updating of ChiSurf.

    This updater can:
    1. Check for updates from a specified URL
    2. Download and install updates using the package manager
    3. Handle platform-specific update logic (Windows, macOS, Linux)
    4. Restart the application after updating
    
    Note: For broader package management (listing/searching/installing arbitrary
    packages, managing environments and channels), see the `PackageManager` class defined
    in this module. The updater will instantiate and share configuration with it.
    """

    def __init__(self, update_url: Optional[str] = None, channel: str = "main"):
        """
        Initialize the updater.

        Args:
            update_url: This parameter is ignored. The updater always uses the hardcoded URL
                        "https://www.peulen.xyz/downloads/chisurf/"
            channel: Update channel to use
        """
        # Initialize system attribute first
        self.system = platform.system().lower()

        # Define the hardcoded URL - this is the only URL that will be used
        url = "https://www.peulen.xyz/downloads/chisurf/"

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
            # Append '/conda' (kept for server directory structure compatibility)
            url += '/conda'

        logging.info(f"Update URL: {url}")
        self.update_url = url
        self.channel = channel
        self.current_version = info.__version__
        self.settings_path = get_path('settings')
        # Expose a shared PackageManager for general package management
        try:
            self.pkg_manager = PackageManager(self)
        except Exception:
            self.pkg_manager = None

    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if updates are available.

        Returns:
            Tuple containing:
            - Boolean indicating if an update is available
            - Latest version string if update is available, None otherwise
            - Error message if an error occurred, None otherwise
        """
        logging.info(f"Checking for updates (current version: {self.current_version})")

        def _parse_version(v: str) -> Tuple:
            """Parse version string into a tuple for robust comparison.
            Supports semantic versions like '1.10.2', date-like '25.08.14', and dev versions like '26.dev123'.
            Falls back to extracting integers; non-numeric parts are ignored.
            """
            try:
                # Handle dev versions like '26.dev123' - treat as lower than any release
                dev_match = re.match(r'^(\d+)\.dev(\d+)$', v)
                if dev_match:
                    year = int(dev_match.group(1))
                    dev_num = int(dev_match.group(2))
                    # Return as (year, 0, dev_num, 0) so it sorts below releases
                    return (year, 0, dev_num, 0)
                
                # Normalize separators to dots and split for regular versions
                parts = re.split(r"[^0-9]+", v)
                nums = [int(p) for p in parts if p != ""]
                # Pad to 3 for common semver comparisons
                while len(nums) < 3:
                    nums.append(0)
                return tuple(nums[:4])
            except Exception:
                return (0,)

        try:
            logging.debug("Getting update information")
            update_info = self._get_update_info()

            if not update_info:
                logging.info("No update information available")
                return False, None, None

            latest_version = update_info["latest_version"]
            logging.info(f"Latest version available: {latest_version}")

            if _parse_version(latest_version) > _parse_version(self.current_version):
                logging.info(f"Update available: {self.current_version} -> {latest_version}")
                return True, latest_version, None

            logging.info(f"Already up to date (version {self.current_version})")
            return False, None, None

        except Exception as e:
            error_msg = f"Error checking for updates: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg

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
        logging.info(f"Starting update to version from file: {file_path}")
        try:
            # Report progress
            message = f"Preparing to update from file: {file_path}"
            logging.info(message)
            if callback:
                callback(message)

            # Check if it's a remote URL
            is_remote_url = bool(re.match(r'^(https?|ftp)://', file_path))
            local_file_path = file_path

            # If it's a remote URL, download it to a temporary file first
            if is_remote_url:
                message = f"Downloading update file from: {file_path}"
                logging.info(message)
                if callback:
                    callback(message)

                try:
                    # Create a temporary directory to store the downloaded file
                    temp_dir = tempfile.mkdtemp(prefix="chisurf_update_")
                    logging.debug(f"Created temporary directory: {temp_dir}")

                    # Extract the filename from the URL
                    file_name = os.path.basename(file_path)
                    if not file_name:
                        file_name = "chisurf_update.tar.bz2"
                    logging.debug(f"Using filename: {file_name}")

                    # Create the local file path
                    local_file_path = os.path.join(temp_dir, file_name)
                    logging.debug(f"Local file path: {local_file_path}")

                    # Define a download progress hook that logs and calls the callback
                    def download_progress_hook(count, block_size, total_size):
                        if total_size > 0:
                            downloaded = count * block_size
                            percent = 100.0 * downloaded / total_size
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)

                            progress_msg = f"Downloading: {mb_downloaded:.1f} MB of {mb_total:.1f} MB ({percent:.1f}%)"
                            logging.debug(progress_msg)

                            if callback:
                                callback(progress_msg)

                    # Download the file
                    logging.debug(f"Starting download from {file_path} to {local_file_path}")
                    urllib.request.urlretrieve(
                        file_path, 
                        local_file_path,
                        reporthook=download_progress_hook
                    )

                    message = f"Download complete. Saved to: {local_file_path}"
                    logging.info(message)
                    if callback:
                        callback(message)
                except Exception as e:
                    error_msg = f"Failed to download update file: {str(e)}"
                    logging.error(error_msg)
                    return False, error_msg

            # Check if the file exists
            if not os.path.exists(local_file_path):
                error_msg = f"Update file not found: {local_file_path}"
                logging.error(error_msg)
                return False, error_msg

            # Determine if we need elevated privileges
            needs_elevation = self._needs_elevation()
            logging.debug(f"Needs elevation: {needs_elevation}")

            # Prepare the update command based on the file type
            file_ext = os.path.splitext(local_file_path)[1].lower()
            logging.debug(f"File extension: {file_ext}")

            if file_ext == '.exe':
                # For Windows installers
                cmd = [local_file_path, '/S']  # Silent install
                logging.info("Using Windows installer (.exe)")
            elif file_ext in ['.msi', '.msix']:
                # For Windows MSI packages
                cmd = ['msiexec', '/i', local_file_path, '/quiet', '/norestart']
                logging.info("Using Windows MSI package")
            elif file_ext in ['.tar.bz2', '.tar.gz', '.bz2', '.gz', '.conda']:
                # For environment packages
                pkg_exe = self._get_pkg_executable()
                logging.info(f"Using update package with executable: {pkg_exe}")

                # Get the environment path (where chisurf is installed)
                env_path = sys.prefix
                logging.debug(f"Environment path: {env_path}")

                # Use install command with --update-deps to handle dependencies automatically
                cmd = [pkg_exe, 'install', '--yes', '--update-deps', '--force-reinstall', '--prefix', env_path, local_file_path]
                logging.info("Using package manager install with --update-deps")
            else:
                # Unknown file type
                error_msg = f"Unsupported update file type: {file_ext}"
                logging.error(error_msg)
                return False, error_msg

            # Create a string representation of the command for display
            cmd_str = " ".join(cmd)
            logging.info(f"Update command: {cmd_str}")

            # Show the command to the user
            if callback:
                callback(f"Command: {cmd_str}")

            # Display a warning that all ChiSurf windows will be closed
            warning_msg = "WARNING: All ChiSurf windows will be closed before starting the update."
            logging.warning(warning_msg)
            if callback:
                callback(warning_msg)
                callback("The update will continue in a separate window.")

            # Run the update in a separate process
            logging.info("Starting update in separate process")
            return self._run_update_in_separate_process(cmd, callback)

        except Exception as e:
            error_msg = f"Error during update: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

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
        logging.info("Starting update to latest version")
        try:
            # Check if an update is available
            logging.debug("Checking if an update is available")
            update_available, latest_version, error = self.check_for_updates()

            if error:
                logging.error(f"Error checking for updates: {error}")
                return False, error

            if not update_available:
                message = "Already up to date"
                logging.info(message)
                return True, message

            # Get update information
            logging.debug("Getting update information")
            update_info = self._get_update_info()
            if not update_info:
                error_msg = "Update information not available"
                logging.error(error_msg)
                return False, error_msg

            # Report progress
            message = "Preparing to update..."
            logging.info(message)
            if callback:
                callback(message)

            # If the update URL is a local folder and we have available versions,
            # use the first (latest) version
            if self._is_local_folder() and "available_versions" in update_info and update_info["available_versions"]:
                latest_version_info = update_info["available_versions"][0]
                logging.info(f"Using local version: {latest_version_info['version']} from {latest_version_info['file_path']}")
                return self.update_to_version(latest_version_info["file_path"], callback)

            # Otherwise, use the standard update mechanism
            logging.info("Using standard update mechanism")

            # Determine if we need elevated privileges
            needs_elevation = self._needs_elevation()
            logging.debug(f"Needs elevation: {needs_elevation}")

            # Prepare the update command
            logging.debug("Preparing update command")
            cmd, cmd_str = self._prepare_update_command(update_info)
            logging.info(f"Update command: {cmd_str}")

            # Show the command to the user
            if callback:
                callback(f"Command: {cmd_str}")

            # Display a warning that all ChiSurf windows will be closed
            warning_msg = "WARNING: All ChiSurf windows will be closed before starting the update."
            logging.warning(warning_msg)
            if callback:
                callback(warning_msg)
                callback("The update will continue in a separate window.")

            # Run the update in a separate process
            logging.info("Starting update in separate process")
            return self._run_update_in_separate_process(cmd, callback)

        except Exception as e:
            error_msg = f"Error during update: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

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

        version_match = re.search(r'(\d+\.\d+\.\d+)', file_name)
        if not version_match:
            version_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', file_name)
        if not version_match:
            version_match = re.search(r'(\d+\.dev\d+)', file_name)
        if not version_match:
            version_match = re.search(r'(\d+)', file_name)
        if not version_match:
            return []

        version = version_match.group(1)
        return [{
            "version": version,
            "file_path": str(file),
            "file_name": file.name,
        }]

    def _list_available_versions(self) -> List[Dict[str, Any]]:
        """
        List available versions from the update folder.

        Returns:
            List of dictionaries containing version information
        """
        if not self._is_local_folder():
            return []

        def _parse_version(v: str) -> Tuple:
            try:
                dev_match = re.match(r'^(\d+)\.dev(\d+)$', v)
                if dev_match:
                    return (int(dev_match.group(1)), 0, int(dev_match.group(2)), 0)
                parts = re.split(r"[^0-9]+", v)
                nums = [int(p) for p in parts if p != ""]
                while len(nums) < 3:
                    nums.append(0)
                return tuple(nums[:4])
            except Exception:
                return (0,)

        try:
            update_path = pathlib.Path(self.update_url)
            if not update_path.exists() or not update_path.is_dir():
                return []

            current_os = self.system
            os_names = {
                "windows": ["win", "windows"],
                "darwin": ["mac", "macos", "darwin"],
                "linux": ["linux"],
            }
            os_specific_names = os_names.get(current_os, [current_os])

            versions = []
            os_subdirs = []
            for subdir in update_path.glob("*"):
                if subdir.is_dir():
                    subdir_name = subdir.name.lower()
                    if any(os_name in subdir_name for os_name in os_specific_names):
                        os_subdirs.append(subdir)

            if os_subdirs:
                for subdir in os_subdirs:
                    for file in subdir.glob("*"):
                        if file.is_file():
                            file_name = file.name.lower()
                            if "chisurf" not in file_name:
                                continue
                            versions.extend(self._process_update_file(file))

            for file in update_path.glob("*"):
                if file.is_dir():
                    continue
                file_name = file.name.lower()
                if not any(os_name in file_name for os_name in os_specific_names):
                    continue
                if "chisurf" not in file_name:
                    continue
                versions.extend(self._process_update_file(file))

            versions.sort(key=lambda x: _parse_version(x["version"]), reverse=True)
            return versions
        except Exception as e:
            logging.error(f"Error listing available versions: {str(e)}")
            return []

    def _list_remote_versions(self) -> List[Dict[str, Any]]:
        """
        List available versions from a remote HTTP source.

        Returns:
            List of dictionaries containing version information
        """
        if self._is_local_folder():
            return []

        def _parse_version(v: str) -> Tuple:
            try:
                dev_match = re.match(r'^(\d+)\.dev(\d+)$', v)
                if dev_match:
                    return (int(dev_match.group(1)), 0, int(dev_match.group(2)), 0)
                parts = re.split(r"[^0-9]+", v)
                nums = [int(p) for p in parts if p != ""]
                while len(nums) < 3:
                    nums.append(0)
                return tuple(nums[:4])
            except Exception:
                return (0,)

        try:
            current_os = self.system
            os_names = {
                "windows": ["win", "windows"],
                "darwin": ["mac", "macos", "darwin"],
                "linux": ["linux"],
            }
            os_specific_names = os_names.get(current_os, [current_os])

            url = self.update_url
            if not url.endswith('/'):
                url += '/'

            def extract_version(filename):
                version_match = re.search(r'(\d+\.\d+\.\d+)', filename)
                if not version_match:
                    version_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', filename)
                if not version_match:
                    version_match = re.search(r'(\d+\.dev\d+)', filename)
                if not version_match:
                    version_match = re.search(r'(\d+)', filename)
                if not version_match:
                    return None
                return version_match.group(1)

            def process_file_link(link, base_url, skip_os_check=False):
                if link.endswith('/'):
                    return None
                link_lower = link.lower()
                if not skip_os_check and not any(os_name in link_lower for os_name in os_specific_names):
                    return None
                if "chisurf" not in link_lower:
                    return None
                version = extract_version(link_lower)
                if not version:
                    return None
                return {
                    "version": version,
                    "file_path": base_url + link,
                    "file_name": link,
                }

            def fetch_and_parse_html(fetch_url):
                try:
                    with urllib.request.urlopen(fetch_url) as response:
                        html = response.read().decode('utf-8')
                    return re.findall(r'href=[\'\"]?([^\'\" >]+)', html)
                except urllib.error.URLError as e:
                    logging.error(f"Error fetching directory listing from {fetch_url}: {e}")
                    return []

            versions = []
            main_links = fetch_and_parse_html(url)

            os_subdirs = []
            for link in main_links:
                if link.endswith('/') and not link.startswith('..'):
                    link_lower = link.lower()
                    if any(os_name in link_lower for os_name in os_specific_names):
                        os_subdirs.append(link)

            for link in main_links:
                version_info = process_file_link(link, url)
                if version_info:
                    versions.append(version_info)

            for subdir in os_subdirs:
                subdir_url = url + subdir
                subdir_links = fetch_and_parse_html(subdir_url)
                for link in subdir_links:
                    version_info = process_file_link(link, subdir_url, skip_os_check=True)
                    if version_info:
                        versions.append(version_info)

            versions.sort(key=lambda x: _parse_version(x["version"]), reverse=True)
            return versions
        except Exception as e:
            logging.error(f"Error listing remote versions: {str(e)}")
            return []

    def _get_update_info(self) -> Optional[Dict[str, Any]]:
        """
        Get update information from the remote server.

        Returns:
            Dictionary containing update information, or None if not available
        """

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
                "available_versions": versions,
                "changelog": self._build_changelog(self.current_version, latest_version["version"])
            }

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
                    "available_versions": versions,
                    "changelog": self._build_changelog(self.current_version, latest_version["version"])
                }

                return update_info

            # If we couldn't get versions from the remote URL, return None
            logging.warning("No versions found at the update URL")
            return None


    def _parse_version_date(self, v: str) -> Optional[datetime]:
        """Parse version strings like 'yy.mm.dd' or 'yyyy.mm.dd' to a UTC datetime at start of day."""
        try:
            parts = re.split(r"[^0-9]+", v)
            nums = [int(p) for p in parts if p != ""]
            if len(nums) >= 3:
                y, m, d = nums[0], nums[1], nums[2]
                if y < 100:
                    y += 2000
                return datetime(y, m, d, tzinfo=timezone.utc)
        except Exception:
            pass
        return None

    def _extract_repo_slug(self) -> Optional[Tuple[str, str]]:
        """Extract (owner, repo) from info.update_url or help_url if possible."""
        url = getattr(info, 'update_url', '') or getattr(info, '__url__', '') or getattr(info, 'help_url', '')
        if not url:
            return None
        try:
            m = re.search(r"github\.com/([^/]+)/([^/]+)", url)
            if m:
                owner = m.group(1)
                repo = m.group(2)
                # Strip trailing anchors like 'releases' from repo if present
                repo = repo.replace('.git', '')
                if repo.endswith('?'):
                    repo = repo.split('?')[0]
                return owner, repo
        except Exception:
            return None
        return None

    def _build_changelog(self, from_version: str, to_version: str, limit: int = 50) -> str:
        """Build a simple changelog by querying GitHub commits between version dates.
        Falls back to a helpful message if not available.
        """
        try:
            owner_repo = self._extract_repo_slug()
            since_dt = self._parse_version_date(from_version) if from_version else None
            until_dt = self._parse_version_date(to_version) if to_version else None

            if not owner_repo or not until_dt:
                return (
                    f"Changes since {from_version} -> {to_version} could not be determined automatically.\n"
                    f"Visit the repository for details: https://github.com/Fluorescence-Tools/chisurf"
                )

            owner, repo = owner_repo

            # If since date missing or invalid, assume 14 days prior to 'until'
            if not since_dt:
                since_dt = until_dt - timedelta(days=14)

            # Ensure since < until; if equal or after, step back a day
            if since_dt >= until_dt:
                since_dt = until_dt - timedelta(days=1)

            # Use end-of-day for until by adding one day
            until_plus = until_dt + timedelta(days=1)

            # Determine branch to query (default to development)
            branch = getattr(self, 'channel', None)
            if isinstance(branch, str) and branch:
                bl = branch.lower()
                if bl.startswith('dev'):
                    branch = 'development'
                elif bl in ('main', 'master'):
                    branch = bl
                else:
                    branch = branch
            else:
                branch = 'development'

            branch_q = urllib.parse.quote(str(branch))

            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/commits?"
                f"since={since_dt.isoformat()}&until={until_plus.isoformat()}&per_page=100&sha={branch_q}"
            )

            headers = {
                'User-Agent': 'ChiSurf-Updater',
                'Accept': 'application/vnd.github+json'
            }

            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read().decode('utf-8')
            commits = json.loads(data)
            if not isinstance(commits, list):
                commits = []

            lines = []
            for c in commits[:limit]:
                try:
                    commit = c.get('commit', {})
                    msg = commit.get('message', '').split('\n')[0].strip()
                    if msg.lower().startswith('merge'):
                        continue
                    author = commit.get('author', {}).get('name') or c.get('author', {}).get('login') or 'unknown'
                    date_str = commit.get('author', {}).get('date', '')
                    # Normalize date short
                    date_short = date_str[:10] if date_str else ''
                    if msg:
                        lines.append(f"- {date_short} {msg} (by {author})")
                except Exception:
                    continue

            compare_hint = f"https://github.com/{owner}/{repo}/commits"

            if not lines:
                return (
                    f"Changes between {from_version} and {to_version} on branch '{branch}':\n"
                    f"(No commits found in the requested date range.)\n\n"
                    f"See commit history: {compare_hint}"
                )

            if len(commits) > limit:
                lines.append(f"... and {len(commits) - limit} more commits")

            header = f"Changes between {from_version} and {to_version}:\n"
            footer = f"\nMore details: {compare_hint}"
            return header + "\n".join(lines) + footer
        except Exception as e:
            # Fallback message on any failure
            return (
                f"Changes since {from_version} -> {to_version} could not be retrieved ({e}).\n"
                f"Visit: https://github.com/Fluorescence-Tools/chisurf/commits"
            )

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

        # On Unix-like systems, check if the environment is in a system directory
        pkg_prefix = os.environ.get('CONDA_PREFIX', '')
        return pkg_prefix.startswith('/usr') and not pkg_prefix.startswith('/usr/local')

    def _check_missing_dependencies(self) -> List[str]:
        """
        This method is kept for backward compatibility but now returns an empty list.
        Dependency resolution is handled automatically when installing or updating packages.

        Returns:
            Empty list (no missing dependencies to manually install)
        """
        logging.info("Dependency checking is handled by the package manager automatically")
        return []

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
        # Get the package manager executable
        pkg_exe = self._get_pkg_executable()

        # Prepare channels
        channels = update_info.get("channels", ["conda-forge", "defaults"])
        channel_args = []
        for channel in channels:
            channel_args.extend(["-c", channel])

        # Get the environment path (where chisurf is installed)
        env_path = sys.prefix

        # Use install command with --update-deps to ensure all dependencies are installed/updated
        cmd = [
            pkg_exe, "install", "-y", "--update-deps", "--prefix", env_path, "chisurf",
            *channel_args
        ]
        logging.info("Using package manager install with --update-deps")

        # Create a string representation for display
        cmd_str = " ".join(cmd)

        return cmd, cmd_str

    def _get_pkg_executable(self) -> str:
        """
        Get the path to the package manager executable.

        Returns:
            Path to the executable
        """
        # Try to get from package manager class
        if self.pkg_manager:
            return self.pkg_manager.pkg_exe()
        
        return "micromamba"

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
            # Log the command being executed
            logging.debug(f"Executing command: {' '.join(cmd)}")

            popen_cmd = cmd
            use_shell = False

            # On Windows, calling a .bat/.cmd directly without shell may fail.
            if self.system == 'windows':
                exe = (cmd[0] if cmd else '').lower()
                if exe.endswith('.bat') or exe.endswith('.cmd'):
                    # Wrap with cmd.exe /C
                    popen_cmd = ['cmd.exe', '/C', *cmd]
                    logging.debug("Wrapping batch file execution with cmd.exe /C for Windows")

            process = subprocess.Popen(
                popen_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=use_shell
            )
            stdout, stderr = process.communicate()

            # Log command output at debug level instead of displaying it
            if stdout:
                logging.debug(f"Command stdout:\n{stdout}")
            if stderr:
                logging.debug(f"Command stderr:\n{stderr}")

            if process.returncode != 0:
                error_msg = f"Command failed with exit code {process.returncode}: {stderr}"
                logging.error(error_msg)
                return False, error_msg

            return True, None
        except Exception as e:
            error_msg = f"Exception while executing command: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

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
            # Create a temporary batch file to run the command with logging
            temp_dir = tempfile.mkdtemp(prefix="chisurf_elev_")
            log_file = os.path.join(temp_dir, "elevated_command.log")

            # Properly quote arguments that contain spaces
            quoted_cmd = [f'"{arg}"' if ' ' in str(arg) and not str(arg).startswith('"') else str(arg) for arg in cmd]
            win_cmd_str = " ".join(quoted_cmd)

            batch_file = os.path.join(temp_dir, "run_elevated.bat")
            with open(batch_file, 'w') as f:
                f.write('@echo off\n')
                f.write(f'echo Running elevated command at %DATE% %TIME% > "{log_file}"\n')
                f.write(f'echo Command: {win_cmd_str} >> "{log_file}"\n')
                # Execute the command and capture all output to the log
                f.write(f'{win_cmd_str} >> "{log_file}" 2>&1\n')
                f.write('set EXITCODE=%ERRORLEVEL%\n')
                f.write('echo. >> "' + log_file + '"\n')
                f.write('echo Exit code: %EXITCODE% >> "' + log_file + '"\n')
                f.write('if %EXITCODE% NEQ 0 (\n')
                f.write('  echo Elevated command failed with error code %EXITCODE% >> "' + log_file + '"\n')
                f.write('  exit /b %EXITCODE%\n')
                f.write(')\n')
                f.write('echo Elevated command completed successfully >> "' + log_file + '"\n')
                f.write('exit /b 0\n')

            # Run the batch file with elevated privileges using PowerShell and wait for completion
            powershell_cmd = [
                'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
                f"$p = Start-Process -FilePath '{batch_file}' -Verb RunAs -Wait -PassThru; exit $p.ExitCode"
            ]

            logging.debug(f"Running elevated batch: {batch_file}")
            logging.debug(f"Elevated log will be written to: {log_file}")

            process = subprocess.Popen(
                powershell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if stdout:
                logging.debug(f"Elevation launcher stdout:\n{stdout}")
            if stderr:
                logging.debug(f"Elevation launcher stderr:\n{stderr}")

            # Read last lines of the elevated log if present for quick context
            tail_hint = ""
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r', errors='ignore') as lf:
                        lines = lf.readlines()
                        tail = "".join(lines[-25:]) if lines else ""
                        tail_hint = tail.strip()
            except Exception:
                pass

            if process.returncode != 0:
                err_msg = f"Elevation failed with exit code {process.returncode}. See log: {log_file}"
                if tail_hint:
                    err_msg += f"\n--- Log tail ---\n{tail_hint}"
                return False, err_msg

            # The batch itself exits with the wrapped command's exit code; inspect the log tail for visibility
            logging.info(f"Elevated command finished. Log: {log_file}")
            if tail_hint:
                logging.debug(f"Elevated command log tail:\n{tail_hint}")

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
        logging.info("Preparing to run update in a separate process")
        try:
            # Report progress
            message = "Preparing to run update in a separate process..."
            logging.info(message)
            if callback:
                callback(message)
                callback("WARNING: All ChiSurf windows will be closed before starting the update.")

            # Check if we need elevated privileges
            needs_elevation = self._needs_elevation()
            logging.debug(f"Needs elevation: {needs_elevation}")
            if needs_elevation:
                message = "Administrator privileges will be required for this update."
                logging.info(message)
                if callback:
                    callback(message)

            # Create a temporary directory to store the update script
            temp_dir = tempfile.mkdtemp(prefix="chisurf_update_")
            logging.debug(f"Created temporary directory: {temp_dir}")

            # Create a string representation of the command for display
            cmd_str = " ".join(cmd)
            logging.debug(f"Command string: {cmd_str}")

            # Create the update script
            if self.system == "windows":
                logging.info("Creating Windows update scripts")
                # Create a properly quoted command string for Windows batch file execution
                # This ensures paths with spaces are handled correctly
                win_cmd_str = " ".join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
                logging.debug(f"Windows command string: {win_cmd_str}")

                # On Windows, use a batch file
                update_script_path = os.path.join(temp_dir, "update_chisurf.bat")
                logging.debug(f"Update script path: {update_script_path}")

                # Create the update script with logging redirected to a file
                log_file = os.path.join(temp_dir, "update_log.txt")
                logging.debug(f"Log file path: {log_file}")

                with open(update_script_path, 'w') as f:
                    f.write('@echo off\n')
                    f.write('title ChiSurf Update\n')
                    f.write('echo ChiSurf Update Process > "' + log_file + '"\n')
                    f.write('echo ===================== >> "' + log_file + '"\n')
                    f.write('echo. >> "' + log_file + '"\n')
                    f.write('echo Starting update process... >> "' + log_file + '"\n')
                    f.write('echo Script: ' + update_script_path + ' >> "' + log_file + '"\n')
                    f.write('echo Command: ' + cmd_str + ' >> "' + log_file + '"\n')
                    f.write('echo. >> "' + log_file + '"\n')
                    f.write('echo Update in progress. Please wait...\n')
                    f.write('echo Update in progress. Please wait... >> "' + log_file + '"\n')
                    f.write('echo. >> "' + log_file + '"\n')

                    # Execute the command and redirect output to log file
                    f.write(win_cmd_str + ' >> "' + log_file + '" 2>&1\n')

                    # Dependencies are handled automatically by the package manager

                    f.write('if %ERRORLEVEL% NEQ 0 (\n')
                    f.write('  echo. >> "' + log_file + '"\n')
                    f.write('  echo Update failed with error code %ERRORLEVEL% >> "' + log_file + '"\n')
                    f.write('  echo.\n')
                    f.write('  echo Update failed with error code %ERRORLEVEL%\n')
                    f.write('  echo See log file for details: ' + log_file + '\n')
                    f.write('  echo.\n')
                    f.write('  echo Press any key to close this window...\n')
                    f.write('  pause > nul\n')
                    f.write('  exit /b %ERRORLEVEL%\n')
                    f.write(')\n')
                    f.write('echo. >> "' + log_file + '"\n')
                    f.write('echo Update successful! >> "' + log_file + '"\n')
                    f.write('echo.\n')
                    f.write('echo Update successful!\n')
                    f.write('echo.\n')
                    f.write('echo Please restart ChiSurf manually to complete the update.\n')
                    f.write('echo Please restart ChiSurf manually to complete the update. >> "' + log_file + '"\n')
                    f.write('echo.\n')
                    f.write('echo Log file: ' + log_file + '\n')
                    f.write('echo.\n')
                    f.write('echo Press any key to close this window...\n')
                    f.write('pause > nul\n')

                # Create a launcher script that will be executed to start the update process
                launcher_script_path = os.path.join(temp_dir, "start_update.bat")
                logging.debug(f"Launcher script path: {launcher_script_path}")

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
                logging.debug("Made scripts executable")

                # Run the launcher script
                logging.info("Starting launcher script")
                subprocess.Popen(['cmd', '/c', launcher_script_path], shell=True)
            else:
                logging.info("Creating Unix update scripts")
                # On Unix-like systems, use a shell script
                update_script_path = os.path.join(temp_dir, "update_chisurf.sh")
                logging.debug(f"Update script path: {update_script_path}")

                # Create a properly quoted command string for shell script execution
                # This ensures paths with spaces are handled correctly
                shell_cmd_str = " ".join([f"'{arg}'" if ' ' in arg else arg for arg in cmd])
                logging.debug(f"Shell command string: {shell_cmd_str}")

                # Create the update script with logging redirected to a file
                log_file = os.path.join(temp_dir, "update_log.txt")
                logging.debug(f"Log file path: {log_file}")

                with open(update_script_path, 'w') as f:
                    f.write('#!/bin/sh\n')
                    f.write('echo "ChiSurf Update Process" > "' + log_file + '"\n')
                    f.write('echo "=====================" >> "' + log_file + '"\n')
                    f.write('echo >> "' + log_file + '"\n')
                    f.write('echo "Starting update process..." >> "' + log_file + '"\n')
                    f.write('echo "Script: ' + update_script_path + '" >> "' + log_file + '"\n')
                    f.write('echo "Command: ' + cmd_str + '" >> "' + log_file + '"\n')
                    f.write('echo >> "' + log_file + '"\n')
                    f.write('echo "Update in progress. Please wait..."\n')
                    f.write('echo "Update in progress. Please wait..." >> "' + log_file + '"\n')
                    f.write('echo >> "' + log_file + '"\n')

                    # Execute the command and redirect output to log file
                    f.write(shell_cmd_str + ' >> "' + log_file + '" 2>&1\n')

                    # Store the exit code
                    f.write('UPDATE_EXIT_CODE=$?\n')

                    # Dependencies are handled automatically by the package manager

                    f.write('if [ $UPDATE_EXIT_CODE -ne 0 ]; then\n')
                    f.write('  echo >> "' + log_file + '"\n')
                    f.write('  echo "Update failed with error code $?" >> "' + log_file + '"\n')
                    f.write('  echo\n')
                    f.write('  echo "Update failed with error code $?"\n')
                    f.write('  echo "See log file for details: ' + log_file + '"\n')
                    f.write('  echo\n')
                    f.write('  echo "Press Enter to close this window..."\n')
                    f.write('  read\n')
                    f.write('  exit $?\n')
                    f.write('fi\n')
                    f.write('echo >> "' + log_file + '"\n')
                    f.write('echo "Update successful!" >> "' + log_file + '"\n')
                    f.write('echo\n')
                    f.write('echo "Update successful!"\n')
                    f.write('echo\n')
                    f.write('echo "Please restart ChiSurf manually to complete the update."\n')
                    f.write('echo "Please restart ChiSurf manually to complete the update." >> "' + log_file + '"\n')
                    f.write('echo\n')
                    f.write('echo "Log file: ' + log_file + '"\n')
                    f.write('echo\n')
                    f.write('echo "Press Enter to close this window..."\n')
                    f.write('read\n')

                # Create a launcher script that will be executed to start the update process
                launcher_script_path = os.path.join(temp_dir, "start_update.sh")
                logging.debug(f"Launcher script path: {launcher_script_path}")

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
                        f.write('  xterm -e "{update_script_path}" &\n')
                        f.write('fi\n')
                    else:
                        f.write(f'xterm -e "{update_script_path}" &\n')

                # Make the scripts executable
                os.chmod(update_script_path, 0o755)
                os.chmod(launcher_script_path, 0o755)
                logging.debug("Made scripts executable")

                # Run the launcher script
                logging.info("Starting launcher script")
                subprocess.Popen(['/bin/sh', launcher_script_path])

            # Report progress
            message = "Update process started in a separate window."
            logging.info(message)
            if callback:
                callback(message)
                callback("ChiSurf will now close.")

            # Log that we're exiting
            logging.info("Exiting ChiSurf to complete the update")

            # Exit the current process
            sys.exit(0)

            return True, None
        except Exception as e:
            error_msg = f"Error starting update process: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

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


class PackageManager:
    """
    Lightweight package/environment/channels manager used by ChiSurf.

    It prefers an existing micromamba/mamba in the current environment, and falls back to
    system PATH. Most commands support JSON output for structured results.
    """
    def __init__(self, updater: Optional[ChiSurfUpdater] = None):
        self.updater = updater
        self.system = (updater.system if updater else platform.system().lower())
        self._pkg_exe_cache: Optional[str] = None
        self._preferred: List[str] = []  # execution preference order
        # Prefer micromamba/mamba when found
        # We'll detect lazily.

    # ---------- Detection ----------
    def pkg_exe(self) -> str:
        if self._pkg_exe_cache:
            return self._pkg_exe_cache
        candidates: List[str] = []
        sys_prefix = sys.prefix
        pkg_prefix = os.environ.get('CONDA_PREFIX', '')
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        if self.system == 'windows':
            # micromamba/mamba common locations
            candidates += [
                os.path.join(app_dir, 'Scripts', 'micromamba.exe'),
                os.path.join(sys_prefix, 'Scripts', 'micromamba.exe'),
                os.path.join(pkg_prefix, 'Scripts', 'micromamba.exe'),
                os.path.join(app_dir, 'Scripts', 'mamba.exe'),
                os.path.join(sys_prefix, 'Scripts', 'mamba.exe'),
                os.path.join(pkg_prefix, 'Scripts', 'mamba.exe'),
                os.path.join(app_dir, 'Scripts', 'conda.exe'),
                os.path.join(sys_prefix, 'Scripts', 'conda.exe'),
                os.path.join(pkg_prefix, 'Scripts', 'conda.exe'),
            ]
        else:
            candidates += [
                os.path.join(app_dir, 'bin', 'micromamba'),
                os.path.join(sys_prefix, 'bin', 'micromamba'),
                os.path.join(pkg_prefix, 'bin', 'micromamba'),
                os.path.join(app_dir, 'bin', 'mamba'),
                os.path.join(sys_prefix, 'bin', 'mamba'),
                os.path.join(pkg_prefix, 'bin', 'mamba'),
                os.path.join(app_dir, 'bin', 'conda'),
                os.path.join(sys_prefix, 'bin', 'conda'),
                os.path.join(pkg_prefix, 'bin', 'conda'),
            ]
        # Finally, rely on PATH
        candidates += ['micromamba', 'mamba', 'conda']
        for c in candidates:
            if os.path.exists(c) or c in ['micromamba', 'mamba', 'conda']:
                self._pkg_exe_cache = c
                # Remember preference order by tool name
                name = os.path.basename(c).lower()
                if 'micro' in name:
                    self._preferred = ['micromamba', 'mamba', 'conda']
                elif 'mamba' in name:
                    self._preferred = ['mamba', 'conda']
                else:
                    self._preferred = ['conda']
                break
        return self._pkg_exe_cache or 'micromamba'

    def preferred_solver(self) -> str:
        """Return the name of the preferred solver (micromamba, mamba, or conda)."""
        try:
            if self._preferred:
                return self._preferred[0]
            # Fallback: determine from cached executable
            if self._pkg_exe_cache:
                name = os.path.basename(self._pkg_exe_cache).lower()
                if 'micro' in name:
                    return 'micromamba'
                elif 'mamba' in name:
                    return 'mamba'
            return 'micromamba'
        except Exception:
            return 'micromamba'

    # ---------- Running helpers ----------
    def _popen(self, cmd: List[str]) -> Tuple[bool, str, str, int]:
        try:
            popen_cmd = cmd
            if self.system == 'windows':
                exe = (cmd[0] if cmd else '').lower()
                if exe.endswith('.bat') or exe.endswith('.cmd'):
                    popen_cmd = ['cmd.exe', '/C', *cmd]
            logging.debug(f"PackageManager executing: {' '.join(popen_cmd)}")
            p = subprocess.Popen(popen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = p.communicate()
            if out:
                logging.debug(f"stdout:\n{out}")
            if err:
                logging.debug(f"stderr:\n{err}")
            return (p.returncode == 0), out, err, p.returncode
        except Exception as e:
            return False, '', str(e), -1

    def _with_prefix(self, args: List[str], prefix: Optional[str]) -> List[str]:
        if not prefix:
            prefix = sys.prefix
        return args + ['-p', prefix]

    def _channels_args(self, channels: Optional[List[str]]) -> List[str]:
        ch: List[str] = []
        if channels:
            for c in channels:
                ch += ['-c', c]
        return ch

    # ---------- Public operations ----------
    def info(self) -> Tuple[bool, Any, str]:
        cmd = [self.pkg_exe(), 'info', '--json']
        ok, out, err, _ = self._popen(cmd)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                ok = False
                err = err or 'Failed to parse info JSON'
        return ok, data, err

    def list_installed(self, prefix: Optional[str] = None) -> Tuple[bool, Any, str]:
        cmd = self._with_prefix([self.pkg_exe(), 'list', '--json'], prefix)
        ok, out, err, _ = self._popen(cmd)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                ok = False
                err = err or 'Failed to parse list JSON'
        return ok, data, err

    def search(self, query: str, channels: Optional[List[str]] = None) -> Tuple[bool, Any, str]:
        cmd = [self.pkg_exe(), 'search', query, '--json'] + self._channels_args(channels)
        ok, out, err, _ = self._popen(cmd)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                ok = False
                err = err or 'Failed to parse search JSON'
        return ok, data, err

    def install(self, packages: List[str], prefix: Optional[str] = None, channels: Optional[List[str]] = None, update_deps: bool = True) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'install', '-y']
        if update_deps:
            args += ['--update-deps']
        args = self._with_prefix(args, prefix) + packages + self._channels_args(channels)
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def remove(self, packages: List[str], prefix: Optional[str] = None) -> Tuple[bool, str]:
        args = self._with_prefix([self.pkg_exe(), 'remove', '-y'], prefix) + packages
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def update(self, packages: Optional[List[str]] = None, prefix: Optional[str] = None) -> Tuple[bool, str]:
        args = self._with_prefix([self.pkg_exe(), 'update', '-y'], prefix)
        if packages and len(packages) > 0:
            args += packages
        else:
            args += ['--all']
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def dry_run_update_all(self, prefix: Optional[str] = None) -> Tuple[bool, Any, str]:
        args = self._with_prefix([self.pkg_exe(), 'update', '--dry-run', '--json', '--all'], prefix)
        ok, out, err, _ = self._popen(args)
        data = None
        if ok:
            try:
                data = json.loads(out)
            except Exception:
                ok = False
                err = err or 'Failed to parse dry-run update JSON'
        return ok, data, err

    def clean_all(self) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'clean', '-y', '--all']
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    # ----- Environments -----
    def list_envs(self) -> Tuple[bool, List[str], str]:
        args = [self.pkg_exe(), 'env', 'list', '--json']
        ok, out, err, _ = self._popen(args)
        envs: List[str] = []
        if ok:
            try:
                data = json.loads(out)
                envs = data.get('envs', [])
            except Exception:
                ok = False
                err = err or 'Failed to parse env list JSON'
        return ok, envs, err

    def current_prefix(self) -> str:
        return sys.prefix

    def create_env(self, name: Optional[str] = None, prefix: Optional[str] = None, python: Optional[str] = None, packages: Optional[List[str]] = None) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'create', '-y']
        if prefix and not name:
            args += ['-p', prefix]
        elif name and not prefix:
            args += ['-n', name]
        else:
            if not name:
                name = 'chisurf-env'
            args += ['-n', name]
        if python:
            args += [f'python={python}']
        if packages:
            args += packages
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def remove_env(self, name: Optional[str] = None, prefix: Optional[str] = None) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'env', 'remove', '-y']
        if prefix and not name:
            args += ['-p', prefix]
        elif name and not prefix:
            args += ['-n', name]
        else:
            return False, 'Specify either name or prefix'
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def clone_env(self, name_src: Optional[str] = None, prefix_src: Optional[str] = None, name_dst: Optional[str] = None, prefix_dst: Optional[str] = None) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'create', '-y']
        if name_dst and not prefix_dst:
            args += ['-n', name_dst]
        elif prefix_dst and not name_dst:
            args += ['-p', prefix_dst]
        else:
            return False, 'Specify destination name or prefix'
        if name_src and not prefix_src:
            args += ['--clone', name_src]
        elif prefix_src and not name_src:
            args += ['--clone', prefix_src]
        else:
            return False, 'Specify source name or prefix'
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def export_env(self, prefix: Optional[str] = None) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'env', 'export']
        if prefix:
            args += ['-p', prefix]
        else:
            args += ['-p', sys.prefix]
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def import_env(self, file_path: str, name: Optional[str] = None) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'env', 'create', '-f', file_path]
        if name:
            args += ['-n', name]
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    # ----- Channels -----
    def get_channels(self) -> Tuple[bool, List[str], str]:
        args = [self.pkg_exe(), 'config', '--show', '--json']
        ok, out, err, _ = self._popen(args)
        channels: List[str] = []
        if ok:
            try:
                data = json.loads(out)
                channels = data.get('channels', []) or data.get('channel_aliases', [])
            except Exception:
                ok = False
                err = err or 'Failed to parse config JSON'
        return ok, channels, err

    def add_channel(self, channel: str) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'config', '--add', 'channels', channel]
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def remove_channel(self, channel: str) -> Tuple[bool, str]:
        args = [self.pkg_exe(), 'config', '--remove', 'channels', channel]
        ok, out, err, _ = self._popen(args)
        return ok, (out if ok else err)

    def set_channels(self, channels: List[str]) -> Tuple[bool, str]:
        ok, out = self._popen([self.pkg_exe(), 'config', '--remove-key', 'channels'])[:2]
        last_msg = ''
        for ch in channels:
            ok2, msg = self.add_channel(ch)
            last_msg = msg
            if not ok2:
                return False, msg
        return True, (last_msg or 'Channels updated')
