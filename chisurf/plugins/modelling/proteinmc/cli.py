"""Protein Monte Carlo command-line interface.

This CLI wraps :mod:`chisurf.plugins.modelling.proteinmc.core` so ProteinMC
simulations can be launched via ``csc proteinmc`` without scripting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from chisurf.plugins.modelling.proteinmc.core import run_protein_mc


def _validate_output_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    target = Path(path)
    if target.exists() and target.is_dir():
        raise click.ClickException(f"Output path '{path}' is a directory.")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise click.ClickException(f"Unable to create output directory for '{path}': {exc}") from exc
    return str(target)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("pdb_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "-s",
    "--settings-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Optional JSON configuration that overrides mc_settings.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Trajectory HDF5 file to write. Defaults to a temporary file.",
)
@click.option("--scale", type=float, help="Override torsion angle scale factor.")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed progress output.")
def cli(
    pdb_file: str,
    settings_file: Optional[str],
    output: Optional[str],
    scale: Optional[float],
    verbose: bool,
) -> None:
    """Run a Protein Monte Carlo simulation for the given PDB/PQR structure."""

    output_path = _validate_output_path(output)

    try:
        traj_file = run_protein_mc(
            pdb_file=pdb_file,
            settings_file=settings_file,
            output_file=output_path,
            verbose=verbose,
            scale=scale,
        )
    except Exception as exc:  # pragma: no cover - delegated logic
        raise click.ClickException(f"ProteinMC simulation failed: {exc}") from exc

    click.echo(traj_file)
