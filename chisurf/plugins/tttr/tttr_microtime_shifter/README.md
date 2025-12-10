# PTU Micro Time Shifter Plugin

This plugin provides tools for adjusting the micro-time channels in Time-Tagged Time-Resolved (TTTR) files, 
particularly those in the PicoQuant PTU format.

## Features

- Loading and visualizing TTTR files with drag-and-drop support
- Applying global or channel-specific micro-time shifts
- Real-time visualization of micro-time histograms
- Color-coded display of different detection channels
- Interactive controls for precise timing adjustments
- Saving modified TTTR files with preserved event data

## Overview

The PTU Micro Time Shifter plugin enables users to adjust the micro-time channels in TTTR files, which is essential for 
precise temporal alignment in fluorescence experiments. Micro-time shifting is particularly valuable for multi-detector 
setups where electronic delays can cause timing offsets between different detection channels.

The plugin provides an intuitive graphical interface that displays micro-time histograms for each detection channel 
with color coding for easy identification. Users can apply global shifts that affect all channels or channel-specific 
shifts to align particular detectors. The changes are visualized in real-time, allowing for precise adjustments.

This tool is particularly valuable for multi-color FRET experiments where precise temporal alignment of detection 
channels is critical for accurate analysis. By correcting timing offsets, users can ensure that fluorescence decay 
curves from different detectors are properly aligned for subsequent analysis.

## Requirements

- Python packages:
  - PyQt5
  - pyqtgraph
  - numpy
  - tttrlib
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > PTU Microtime Shifter
2. Load a TTTR file:
   - Click "Load..." to browse for a file
   - Or drag and drop a file into the input field
3. Adjust micro-time shifts:
   - Use the "All" control to apply a global shift to all channels
   - Use the channel-specific controls (Ch0, Ch1, etc.) to adjust individual channels
   - The spin boxes allow precise numerical input
   - The "↻" buttons can be used to quickly apply changes
4. Monitor the effects in real-time:
   - The plot displays color-coded micro-time histograms for each channel
   - Observe how the shifts affect the alignment of peaks across channels
5. Save the modified file:
   - Click "Save..." to choose a save location
   - The modified file will preserve all event data with the adjusted micro-times

## Applications

The PTU Micro Time Shifter plugin can be used for:
- Correcting timing offsets between different detection channels
- Aligning fluorescence decay curves from different detectors
- Compensating for electronic delays in the detection system
- Preparing data for multi-detector analysis
- Improving the accuracy of lifetime measurements
- Enhancing the precision of FRET efficiency calculations
- Standardizing data from different experimental setups

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.