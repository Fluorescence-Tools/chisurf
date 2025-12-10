"""Protein Monte Carlo CLI plugin.

Expose the ProteinMC cmd tool via the unified ChiSurf CLI.
"""

name = "Modelling:Protein Monte Carlo"

# Register CLI entrypoint for chisurf.cli discovery
cli_entrypoint = "proteinmc=chisurf.plugins.modelling.proteinmc.cli:cli"

__all__ = ["name", "cli_entrypoint"]
