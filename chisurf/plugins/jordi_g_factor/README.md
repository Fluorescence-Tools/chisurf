# Jordi G-Factor Calculator

A standalone plugin for calculating the g-factor based on tail matching for Jordi files.

## Overview

The g-factor is an important correction factor in fluorescence anisotropy measurements, accounting for the different detection efficiencies of the parallel and perpendicular emission components. This plugin provides a simple interface for:

- Loading and displaying Jordi files
- Selecting a tail matching region using interactive controls
- Calculating the g-factor based on the selected region
- Displaying the g-factor and its standard deviation in copyable text fields
- Visualizing the tail-matched decays in a separate plot
- Displaying decay curves in semilog plots (logarithmic y-axis) for better visualization of exponential decays
- Updating calculations when the region selection changes
- Applying background correction with a separate background region selector
- Displaying both corrected and uncorrected g-factors

## Installation

This plugin can be used in two ways:

1. **As a ChiSurf Plugin**: It's included with ChiSurf. No additional installation is required.
2. **As a Standalone Application**: The plugin can run independently of ChiSurf. Simply run the `__init__.py` file directly with Python:
   ```
   python __init__.py
   ```

## File Format

Jordi files are simple text files containing intensity values with the following structure:
- The file contains a single column of numbers
- The first half of the values represent the VV (parallel) channel
- The second half of the values represent the VH (perpendicular) channel

The plugin loads these files directly using numpy.loadtxt and automatically splits the data into the two channels.

## Usage

### As a ChiSurf Plugin
1. Launch ChiSurf
2. Go to the "Plugins" menu
3. Select "Analysis" > "Jordi G-Factor Calculator"
4. Follow the steps below

### As a Standalone Application
1. Run the plugin directly: `python __init__.py`

### Using the Interface
1. Click "Load Jordi File" to select a Jordi file
2. The file will be displayed with parallel (blue) and perpendicular (red) components in the left plot
3. A blue shaded region will appear, which can be adjusted by dragging the edges
4. The g-factor and its standard deviation will be calculated based on the selected region
5. The calculated values appear in text fields that can be easily copied
6. The right plot shows the tail-matched decays for the entire range
7. As you adjust the region, both the g-factor values and the tail-matched plot update automatically

### Background Correction
1. Check the "Background Correction" checkbox to enable background subtraction
2. A red shaded region will appear, which can be adjusted to select the background region
3. The background level is calculated as the average intensity in the selected region
4. Both corrected and uncorrected g-factors are displayed when background correction is enabled
5. The tail-matched plot will show the background-corrected data
6. The background correction is useful for removing constant offsets in the data

## Theory

The g-factor is calculated as the ratio of the parallel to perpendicular intensities in the tail region of the decay, where the anisotropy should approach zero:

g = I_parallel / I_perpendicular

In the tail region of a fluorescence decay, where rotational diffusion has randomized the orientation of the fluorophores, the theoretical anisotropy should be zero. Any deviation from zero is due to the different detection efficiencies of the parallel and perpendicular channels, which is what the g-factor corrects for.

## Requirements

When running as a standalone application:
- Python 3.6+
- PyQt5
- PyQtGraph
- NumPy

For icon generation (optional):
- PIL/Pillow

When running as a ChiSurf plugin:
- ChiSurf (which includes all the above dependencies)

## License

This plugin is licensed under the same license as ChiSurf.