# ChiSurf Updater Plugin

This plugin provides functionality to check for and install updates for ChiSurf.

## Features

- Check for available updates
- Download and install updates using conda
- Handle platform-specific update logic (Windows, macOS, Linux)
- Handle elevated privileges when needed
- Run updates in a separate process
- Inform the user to restart the application manually after updating

## Overview

The ChiSurf Updater plugin enables users to check for and install updates to the ChiSurf application. It provides a 
simple interface for checking if updates are available and installing them with a single click.

The updater uses conda to perform the actual updates and has platform-specific logic for Windows, macOS, and Linux. 
It can handle elevated privileges when needed, such as when ChiSurf is installed in a system directory.

When an update is started, the plugin closes all ChiSurf windows and continues the update process in a separate window.
After the update completes, the user needs to restart ChiSurf manually to apply the changes.

## Requirements

- Python packages:
  - PyQt5
  - conda (for package management)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Help > Check for Updates
2. The plugin will display the current version of ChiSurf
3. Click "Check for Updates" to check if updates are available
4. If updates are available, the "Update Now" button will be enabled
5. Click "Update Now" to download and install the updates
6. A warning message will appear informing you that all ChiSurf windows will be closed
7. After confirming, all ChiSurf windows will close and the update will continue in a separate window
8. Once the update completes, restart ChiSurf manually to apply the changes

## Applications

The ChiSurf Updater plugin can be used for:
- Keeping ChiSurf up to date with the latest features and bug fixes
- Ensuring compatibility with new operating system versions
- Getting security updates
- Accessing new plugins and functionality

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
