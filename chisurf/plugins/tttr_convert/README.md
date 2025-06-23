# TTTR Convert Plugin

This plugin provides a graphical interface for converting Time-Tagged Time-Resolved (TTTR) data files between 
different formats.

## Features

- Support for various TTTR file formats
- Batch conversion capabilities
- Configuration of conversion parameters
- Preview of file contents
- Preservation of essential metadata during conversion
- Progress tracking for large file conversions

## Overview

The TTTR Convert plugin enables users to convert Time-Tagged Time-Resolved (TTTR) data files between different formats. 
TTTR data is commonly used in fluorescence spectroscopy and single-molecule experiments, and different analysis 
software packages often require specific file formats.

This plugin bridges the gap between different TTTR file formats, allowing researchers to prepare their data for 
analysis in various software environments. The conversion process preserves essential metadata and timing information 
while transforming the file structure to meet the requirements of the target format.

The intuitive graphical interface makes it easy to configure conversion parameters and process multiple files in batch 
mode, saving time when working with large datasets.

## Requirements

- Python packages:
  - PyQt5
  - tttrlib
  - numpy
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > Convert
2. Select input files:
   - Click "Add Files" to select individual TTTR files
   - Or click "Add Directory" to add all TTTR files from a directory
   - The selected files will appear in the file list
3. Configure conversion settings:
   - Select the target format from the dropdown menu
   - Adjust conversion parameters as needed
   - Set output directory for converted files
4. Preview file contents (optional):
   - Select a file from the list
   - Click "Preview" to see basic information about the file
5. Start conversion:
   - Click "Convert" to begin the conversion process
   - A progress bar will show the status of each file
   - Converted files will be saved to the specified output directory

## Applications

The TTTR Convert plugin can be used for:
- Preparing data for analysis in different software packages
- Converting proprietary formats to more accessible formats
- Standardizing data formats across multiple experiments
- Archiving data in preferred formats for long-term storage
- Enabling collaboration by sharing data in formats compatible with collaborators' tools
- Batch processing of multiple files to save time

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.