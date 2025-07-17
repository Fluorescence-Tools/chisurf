# Channel Definition Plugin

This plugin provides a graphical interface for defining and configuring detector channels and time windows for Time-Tagged Time-Resolved (TTTR) fluorescence spectroscopy data.

## Features

- Definition of detector channels with specific routing channel numbers
- Configuration of Pulsed Interleaved Excitation (PIE) time windows
- Association of microtime ranges with specific detectors
- Import and export of channel definitions via JSON files
- Interactive wizard interface for intuitive setup

## Overview

The Channel Definition plugin is essential for preprocessing TTTR data before analysis, allowing researchers to properly map physical detector channels to their experimental setup and define time windows for techniques like PIE. These definitions are used by other analysis plugins to correctly interpret the raw photon data.

The plugin is particularly useful for multi-detector setups and advanced fluorescence techniques such as single-molecule FRET, fluorescence lifetime imaging, and fluorescence correlation spectroscopy, where proper channel assignment is critical for accurate data analysis.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - json

## Usage

1. Launch the plugin from the ChiSurf menu: Setup > Channel Definition
2. Define detector channels:
   - Assign routing channel numbers to physical detectors
   - Set detector names and properties
3. Configure time windows:
   - Define microtime ranges for different excitation pulses
   - Set up Pulsed Interleaved Excitation (PIE) windows
4. Associate microtime ranges with specific detectors
5. Save the channel definition to a JSON file for use with other plugins
6. Load existing channel definitions for modification or reuse

## Applications

- Setting up multi-detector fluorescence experiments
- Configuring Pulsed Interleaved Excitation (PIE) experiments
- Preparing TTTR data for single-molecule FRET analysis
- Defining channels for fluorescence lifetime imaging
- Configuring detector settings for fluorescence correlation spectroscopy
- Standardizing channel definitions across multiple experiments

## Benefits

- Ensures consistent interpretation of detector channels across different analysis plugins
- Simplifies the setup of complex multi-detector experiments
- Provides a standardized format for storing and sharing channel configurations
- Reduces errors in data analysis due to incorrect channel assignments
- Streamlines the workflow for advanced fluorescence spectroscopy techniques

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.