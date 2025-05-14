"""
Batch Analysis Plugin

This plugin provides a wizard interface for processing multiple files with the same fit model.
It allows users to:
1. Select multiple files for batch processing
2. Choose a fit method to apply to all files
3. Run the fits and save results to a CSV file
4. View the results in a table

The initial parameter values are taken from the template fit. Before batch processing,
you should manually optimize the parameters of this template fit using data similar to
the files you plan to process to ensure reliable and meaningful results.
"""

name = "Tools:Batch-Analysis"

