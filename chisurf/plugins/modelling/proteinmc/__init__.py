"""Protein Monte Carlo CLI plugin.

Expose the ProteinMC cmd tool via the unified ChiSurf CLI.
"""

name = "Structure:Computation:Protein Monte Carlo"

# Register CLI entrypoint for chisurf.core.cli discovery
cli_entrypoint = "proteinmc=chisurf.plugins.modelling.proteinmc.cli:cli"
cli_only = True

__all__ = ["name", "cli_entrypoint", "cli_only"]
