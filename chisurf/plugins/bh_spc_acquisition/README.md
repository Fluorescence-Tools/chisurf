# BH SPC Acquisition Plugin

This plugin provides tools for acquiring single molecule data, e.g., using a Becker & Hickl SPC 830 TCSPC board. 
It displays fluorescence decays of user-defined channels (up to 4) and correlation curves. 
Data is acquired into RAM and saved at the end of data acquisition.

## Features

- Acquisition of time-tagged time-resolved (TTTR) data from BH SPC 830
- Display of fluorescence decays for up to 4 user-defined channels
- Display of correlation curves
- Configurable acquisition time
- RAM usage monitoring
- Data saving at the end of acquisition
- Manufacturer-agnostic wrapper for future extension to other hardware (e.g., Picoquant)

## Requirements

- Becker & Hickl SPC-830 TCSPC board
- Becker & Hickl SPCM-DLL (part of the TCSPC Package or SPCM Data Acquisition Software)
- Python packages:
  - PyQt5
  - pyqtgraph
  - numpy
  - psutil
  - tttrlib

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > BH SPC Acquisition
2. Initialize the device (in simulation mode or hardware mode)
3. Configure the acquisition parameters:
   - Set the acquisition time
   - Select the channels to display (up to 4)
4. Start the acquisition
5. Monitor the fluorescence decays and correlation curves in real-time
6. Save the data when the acquisition is complete:
   - Click the "Save Data" button
   - Choose a location and filename for the data
   - All data formats (binary, NPZ, CSV, and Kristine files) will be saved automatically

## Data Format

The plugin saves data in the following formats:

1. Raw data: Binary file (.bin) containing the 32-bit records from the TCSPC board
2. Decay data: NPZ file (.decay.npz) containing the fluorescence decay histograms for each channel
3. Decay data: CSV files (.ch{channel}.csv) containing the time and counts data for each channel
4. FCS curves: Kristine files (.cor) containing the correlation times, amplitudes, mean countrate, and the actual acquisition time (measured during data collection)

## Extending to Other Hardware

The plugin uses a manufacturer-agnostic wrapper (`TCSPCDevice` class) from the `chisurf.plugins.bh_spc_wrapper` module 
that can be extended to support other TCSPC hardware, such as Picoquant. To add support for a new hardware type:

1. Add a new device type to the `TCSPCDevice` class in the `chisurf.plugins.bh_spc_wrapper.wrapper` module
2. Implement the device-specific methods for initialization, measurement control, and data acquisition
3. Update the UI to include the new device type in the device selection dropdown

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
