# Batch Analysis Plugin

This plugin provides a wizard interface for processing multiple files with the same fit model.

## Features

- Select multiple files for batch processing
- Choose a fit method to apply to all files
- Run the fits and save results to a CSV file
- View the results in a table
- Consistent parameter application across multiple datasets

## Overview

The Batch Analysis plugin streamlines the process of applying the same analysis method to multiple data files. 
This is particularly useful for experiments that generate numerous similar datasets that need to be analyzed with the 
same model and parameters.

The initial parameter values are taken from the template fit. Before batch processing, you should manually optimize the 
parameters of this template fit using data similar to the files you plan to process to ensure reliable and meaningful 
results.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - pandas (for CSV export)
  - matplotlib (for plotting)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Batch-Analysis
2. Create and optimize a template fit with one representative dataset
3. Select multiple files for batch processing
4. Choose the fit method to apply to all files
5. Configure processing options:
   - Parameter constraints
   - Output format
   - Result handling
6. Run the batch process
7. View results in the table and export to CSV

## Applications

- Processing large datasets from high-throughput experiments
- Analyzing time series data with consistent parameters
- Comparing results across multiple samples or conditions
- Automating routine analysis tasks
- Ensuring consistent analysis methodology across datasets

## Benefits

- Saves time by automating repetitive analysis tasks
- Ensures consistent analysis methodology across datasets
- Provides organized output of results for easy comparison
- Reduces human error in applying analysis parameters
- Facilitates high-throughput data processing workflows

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.