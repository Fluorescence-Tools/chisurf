# Becker & Hickl .set File Reader

## Overview

The `BeckerHicklSetReader` class provides functionality to read and parse Becker & Hickl SPC .set files. These files contain important timing parameters for time-resolved spectroscopy data acquired with Becker & Hickl SPC hardware.

## Usage

```python
from chisurf.fio.fluorescence import BeckerHicklSetReader

# Create a reader for a .set file
reader = BeckerHicklSetReader("path/to/file.set")

# Access raw parameters
sync_frequency = reader.get_param('SYN_FQ')  # in MHz
tac_tc = reader.get_param('TAC_TC')  # in seconds

# Access derived properties
macro_time_res = reader.macro_time_resolution  # in seconds
micro_time_res = reader.micro_time_resolution  # in seconds
tac_range = reader.tac_range  # in seconds

# Print a summary of key timing parameters
reader.summary()
```

## Key Properties

- `macro_time_resolution`: Coarse (macro) time resolution in seconds, calculated as 1 / |SP_SYN_FQ| (where SP_SYN_FQ is in MHz)
- `micro_time_resolution`: Fine (micro) time resolution in seconds, equal to SP_TAC_TC
- `tac_range`: Full TAC span in seconds, equal to SP_TAC_R

## Implementation Details

The reader uses a regular expression to parse parameters from the .set file. Parameters are stored in a dictionary with the parameter name (without the "SP_" prefix) as the key. The values are converted to the appropriate type (float or int) based on the type indicator in the file.

## Common Parameters

- `SYN_FQ`: Sync frequency in MHz
- `TAC_TC`: TAC conversion factor in seconds (micro-time channel width)
- `TAC_R`: Full TAC span in seconds
- `ADC_RE`: ADC resolution (number of channels)

## Example

```python
reader = BeckerHicklSetReader("m_003.set")
reader.summary()
```

Output:
```
File: m_003.set
  SP_SYN_FQ (MHz):          -50.98
  Macro-time resolution:    19.616 ns
  SP_TAC_TC (s):            1.830e-11 s
  Micro-time resolution:    18.300 ps
  SP_TAC_R (s):             5.000e-08 s
  TAC full-scale range:     50.000 ns
```