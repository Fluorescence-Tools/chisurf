# BH SPC Wrapper Module

This module provides a wrapper for the Becker & Hickl SPC hardware. It includes classes and functions for initializing and controlling BH SPC devices, as well as for acquiring data from them.

## Features

- Direct wrapper for the BH SPC DLL
- Manufacturer-agnostic interface for TCSPC devices
- Support for simulation mode
- Helper functions for INI file creation

## Classes

### DLLOperationMode

Enum for the operation mode of SPCM-DLL. Values include:
- `HARDWARE`: Hardware mode
- `SIMULATE_SPC_830`: Simulation mode for SPC-830
- And many other simulation modes for different SPC models

### InitStatus

Enum for the initialization status of an SPC module. Values include:
- `INIT_OK`: Initialization successful
- `INIT_NOT_DONE`: Initialization not requested
- And many other status codes for different error conditions

### ParID

Enum of SPC parameter IDs. Values include:
- `MODE`: Operation mode
- `STOP_ON_TIME`: Stop on time flag
- `STOP_ON_OVFL`: Stop on overflow flag

### SPCMError

Exception raised for errors in the SPCM DLL.

### BHSPC

Direct wrapper for the BH SPC DLL. Methods include:
- `init`: Initialize the SPCM DLL
- `close`: Close the SPCM DLL
- `get_init_status`: Get the initialization status of a module
- `set_parameter`: Set a parameter for a module
- `get_parameter`: Get a parameter from a module
- `start_measurement`: Start a measurement
- `stop_measurement`: Stop a measurement
- `read_fifo`: Read data from the FIFO buffer
- `get_fifo_usage`: Get the FIFO usage

### TCSPCDevice

A manufacturer-agnostic wrapper for TCSPC devices. Methods include:
- `initialize`: Initialize the TCSPC device
- `start_measurement`: Start a measurement
- `stop_measurement`: Stop a measurement
- `read_fifo`: Read data from the FIFO buffer
- `get_fifo_usage`: Get the FIFO usage
- `close`: Close the TCSPC device

## Functions

### minimal_spcm_ini

Return the text for a minimal .ini file for use with SPCM DLL.

### ini_file

Context manager providing a temporary .ini file with the given text.

## Usage

```python
from chisurf.plugins.core.acq.tcspc_devices.bh_spc.bh_spc_wrapper import TCSPCDevice, DLLOperationMode
```
# Create a TCSPC device
device = TCSPCDevice()

# Initialize the device in simulation mode
device.initialize(simulation=True)

# Start a measurement
device.start_measurement()

# Read data from the FIFO buffer
data = device.read_fifo()

# Stop the measurement
device.stop_measurement()

# Close the device
device.close()
```

## Extending to Other Hardware

The module is designed to be manufacturer-agnostic, allowing for future extension to other hardware (e.g., Picoquant). To add support for a new hardware type:

1. Add a new device type to the `TCSPCDevice` class
2. Implement the device-specific methods for initialization, measurement control, and data acquisition

## License

This module is part of the ChiSurf package and is distributed under the same license.