# Count Rate Analysis Plugin

This plugin allows analysis of count rates in TTTR files, both through a graphical user interface (GUI) and a command-line interface (CLI).

## Features

- Load and analyze TTTR files
- Compute count rates for all combinations of windows and detectors
- Calculate mean and standard deviation of count rates
- Report count rates for individual channels
- Export results to a text file
- Command-line interface for batch processing

## GUI Usage

The GUI provides an interactive interface for:
- Drag and drop TTTR files
- Define detector channels using the DetectorWizardPage
- View count rates in a table and plot
- Export results to a text file

## CLI Usage

The CLI allows for batch processing of TTTR files from the command line:

```bash
# Show help
csc_count_rate --help

# List available detector setups in a file
csc_count_rate list-setups SETUP_FILE

# Analyze TTTR files
csc_count_rate analyze FILE1 FILE2... --setup-file SETUP_FILE [OPTIONS]

# Options:
#   --setup-name NAME       Name of the setup to use
#   --output OUTPUT_FILE    Save results to file
#   --verbose               Enable verbose output
```

### Example

```bash
# List available setups
csc_count_rate list-setups detector_setups.json

# Analyze files with verbose output and save results
csc_count_rate analyze data/*.ptu --setup-file detector_setups.json --output results.txt --verbose
```

## Programmatic Usage

The CLI can also be used programmatically:

```python
from chisurf.plugins.tttr.tttr_count_rate_analysis.cli import cli
from click.testing import CliRunner
```
# Create a runner
runner = CliRunner()

# Run a command
result = runner.invoke(cli, ['list-setups', 'detector_setups.json'])
print(result.output)

# Analyze files
result = runner.invoke(cli, [
    'analyze', 'file1.ptu', 'file2.ptu',
    '--setup-file', 'detector_setups.json',
    '--output', 'results.txt',
    '--verbose'
])
```

## Installation

The CLI is automatically installed as part of ChiSurf and is available as the `csc_count_rate` command.