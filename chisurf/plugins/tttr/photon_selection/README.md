# Photon/Burst Selection Plugin

This plugin provides a graphical interface for selecting and filtering photon data from Time-Tagged Time-Resolved 
(TTTR) measurements.

## Features

- Selection of detector channels for analysis
- Filtering of photon events based on detector parameters
- Interactive wizard interface for intuitive data selection
- Seamless integration with TTTR data processing workflows
- Two-step wizard process for efficient workflow

## Overview

The Photon/Burst Selection plugin is essential for preprocessing single-molecule fluorescence data, allowing users to 
isolate specific photon populations before further analysis such as correlation, burst analysis, or intensity trace 
examination.

The plugin implements a user-friendly wizard interface that guides researchers through the process of selecting and 
filtering photon data from TTTR measurements. This preprocessing step is crucial for ensuring that subsequent analyses 
focus on the relevant photon events, improving the quality and reliability of results.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - tttrlib (for TTTR file handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > Photon/Burst Selection
2. Follow the two-step wizard process:
   - First page: Select detector channels of interest
     - Choose which detector channels to include in the analysis
     - Configure channel properties as needed
   - Second page: Apply filters to the photon data
     - Set filtering parameters based on selected detectors
     - Preview the effects of filtering on the data
3. Click Finish to save the photon selection
4. Use the filtered photon data in subsequent analysis steps within ChiSurf

## Applications

- Preprocessing for single-molecule FRET analysis
- Data preparation for fluorescence correlation spectroscopy (FCS)
- Filtering photon events for burst analysis
- Isolating specific detector channels for specialized analyses
- Removing background or unwanted photon events
- Preparing data for intensity trace examination

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.