# Becker & Hickl File Readers

## Overview

The `bhfiles` module provides unified access to various Becker & Hickl file formats used in time-resolved spectroscopy. It combines readers for:

- `.set` files: Setup parameter files containing timing and hardware configuration
- `.sdt` files: Data files containing time-correlated single photon counting measurements

This module replaces the separate `becker_hickl_set` and `sdtfile` modules, providing a more cohesive API while maintaining backward compatibility.

## Usage

```python
from chisurf.fio.fluorescence.bhfiles import BeckerHicklSetReader, SdtFile

# Reading .set files
set_reader = BeckerHicklSetReader("path/to/file.set")
macro_time_res = set_reader.macro_time_resolution  # in seconds
micro_time_res = set_reader.micro_time_resolution  # in nanoseconds
set_reader.summary()  # Print a summary of key timing parameters

# Reading .sdt files
sdt_file = SdtFile("path/to/file.sdt")
data = sdt_file.data  # List of data arrays
times = sdt_file.times  # List of time axes
```

## Key Classes

### BeckerHicklSetReader

Reader for Becker & Hickl SPC `.set` files, which contain setup parameters for time-resolved measurements.

**Key Properties:**
- `macro_time_resolution`: Coarse (macro) time resolution in seconds
- `micro_time_resolution`: Fine (micro) time resolution in nanoseconds
- `tac_range`: Full TAC span in seconds

**Methods:**
- `get_param(name, default=None)`: Get a raw SPC parameter
- `summary()`: Print a summary of key timing parameters

### SdtFile

Reader for Becker & Hickl `.sdt` files, which contain time-correlated single photon counting data.

**Key Attributes:**
- `data`: List of data arrays containing photon counts
- `times`: List of time axes for each data set
- `measure_info`: Measurement description blocks
- `header`: File header information
- `info`: General file information in ASCII format
- `setup`: Setup block containing system parameters

## Migration from Previous Modules

If you were previously using `becker_hickl_set` or `sdtfile` modules directly, you can continue to use the same classes as before. The imports in `__init__.py` have been updated to maintain backward compatibility:

```python
# Old code still works
from chisurf.fio.fluorescence import BeckerHicklSetReader
from chisurf.fio.fluorescence import SdtFile
```

For new code, it's recommended to import directly from the unified module:

```python
# Recommended for new code
from chisurf.fio.fluorescence.bhfiles import BeckerHicklSetReader, SdtFile
```

## Implementation Details

The module combines the functionality of the previous separate modules:

- `BeckerHicklSetReader`: Parses `.set` files using regular expressions to extract parameters
- `SdtFile`: Reads binary `.sdt` files with complex header structures and data blocks

Both readers provide access to timing parameters and other metadata from Becker & Hickl SPC hardware, making them complementary tools for working with time-resolved spectroscopy data.