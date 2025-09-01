# Microtime Histogram Plugin

This plugin provides tools for creating, visualizing, and analyzing microtime histograms from Time-Tagged Time-Resolved 
(TTTR) fluorescence data.

## Features

- Loading and processing of TTTR files from various formats with progress bars
- Creation of microtime histograms for parallel and perpendicular detection channels with interactive legend
- Support for interleaved channel format (matching MLE burst plugin implementation)
- Support for burst selection using BID/BUR/BST files for single-molecule analysis
- Cumulative histogram generation from multiple files
- Export of histogram data for further analysis
- Integration with ChiSurf for advanced data processing
- Dedicated detector setup tab using DetectorWizardPage for advanced channel configuration
- Support for PIE-window definitions and microtime ranges
- Progress indicators for file loading and processing operations

## Overview

Microtime histograms represent the distribution of photon arrival times relative to the excitation pulse, providing 
valuable information about fluorescence lifetimes and molecular dynamics. In polarization-resolved measurements, 
separate histograms for parallel and perpendicular detection channels enable fluorescence anisotropy analysis.

The plugin is particularly useful for time-resolved fluorescence spectroscopy, fluorescence lifetime imaging (FLIM), 
and single-molecule experiments where temporal information about photon arrival is critical for understanding molecular 
properties and dynamics.

## Requirements

- Python packages:
  - numpy
  - matplotlib
  - tttrlib (for TTTR file handling)
  - PyQt5 (for GUI components)

## Usage

### GUI Usage

1. Launch the plugin from the ChiSurf menu: Tools > Histogram-Microtime
2. The plugin now has two tabs:
   - **Detector Setup**: Configure detector channels and PIE-windows
   - **Histogram**: Generate and visualize microtime histograms

### Detector Setup Tab
1. Load or create a detector setup:
   - Select an existing setup from the dropdown
   - Define PIE-windows and detector channels
   - Save your configuration for future use
2. Use the TTTR Reading Routine to read parameters from a file
3. After configuring detectors, the plugin will automatically switch to the Histogram tab

### Histogram Tab
1. Load TTTR data files
   - Progress bars show loading status when dropping multiple files
   - Cancel buttons allow interrupting long loading operations
2. View the currently selected setup displayed at the top of the tab
   - Use the setup selection dropdown to quickly switch between different setups without going back to the Detector Setup tab
3. Use the detector selection dropdown to choose between detectors (green, red)
   - Selecting a detector automatically updates the parallel/perpendicular channels and g-factor
4. Configure histogram parameters:
   - File type, binning, and time resolution (dt) are automatically set based on the detector setup
     - Note: For SPC files, microtime resolution is not automatically updated to preserve file-specific settings
   - Channel information is automatically populated from the Detector Setup tab
   - Channels are handled in interleaved format (parallel/perpendicular alternating) for compatibility with MLE burst plugin
   - Adjust timeshift values for parallel and perpendicular channels if needed
5. For single-molecule analysis, drop BID/BUR/BST files or folders containing these files
   - You can drop entire folders containing .bur or .bst files for batch processing
   - The plugin will automatically search for corresponding TTTR files with progress indicators
   - Progress bars show both the search progress and file loading progress
6. Generate and visualize microtime histograms
   - Progress dialog shows which file is being processed and allows cancellation
   - The plot shows only cumulative data (not individual files) with a concise legend showing Cumulative Parallel, Cumulative Perpendicular, and Combined channels
7. Export data for further analysis or fitting

### Command-Line Usage

The plugin can also be used from the command line, which is especially useful for automation and integration with other tools:

```
csc_microtime_histogram --bid-folder /path/to/bid/folder [--setup-name SETUP_NAME] [--auto-transfer]
```

Parameters:
- `--bid-folder`: Path to the folder containing BID/BUR/BST files
- `--setup-name`: (Optional) Name of the setup to use. If not provided, the plugin will try to read it from photon_selection_parameters.json in the Info folder
- `--auto-transfer`: (Optional) If specified, automatically transfer the histogram to ChiSurf

### Direct Export from NDXplorer

The plugin now supports direct export from NDXplorer:

1. When saving burst IDs in NDXplorer, the microtime histogram plugin is automatically launched
2. The plugin reads the selected setup from photon_selection_parameters.json in the Info folder
3. The microtime histogram is automatically generated and transferred to ChiSurf
4. This streamlines the workflow from burst selection to decay analysis in ChiSurf

## Applications

- Fluorescence lifetime analysis
- Time-resolved fluorescence spectroscopy
- Fluorescence anisotropy measurements
- Single-molecule FRET experiments
- Fluorescence lifetime imaging (FLIM)
- Molecular dynamics studies

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.