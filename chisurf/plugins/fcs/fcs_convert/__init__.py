"""FCS conversion plugin.

Convert fluorescence correlation spectroscopy files between supported
formats directly from the ChiSurf CLI.
"""

name = "Fluorescence Correlation Spectroscopy:FCS Converter"

# Expose the Click CLI through the top-level ``csc`` entry point.
cli_entrypoint = "fcs-convert=chisurf.plugins.fcs.fcs_convert.cli:cli"
cli_only = True

__all__ = ["name", "cli_entrypoint", "cli_only"]
