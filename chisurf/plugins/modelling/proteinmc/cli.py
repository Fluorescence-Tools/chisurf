"""Protein Monte Carlo command-line interface.

This CLI wraps :mod:`chisurf.plugins.modelling.proteinmc.model` so ProteinMC
simulations can be launched via ``csc proteinmc`` without scripting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from chisurf.plugins.modelling.proteinmc.model import run_protein_mc


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
@click.argument("structure", type=str)
@click.option(
    "-s",
    "--settings-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Optional JSON configuration that overrides mc_settings.",
)
@click.option(
    "-l",
    "--labeling-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="FPS JSON labeling file used as the ProteinMC optimization target.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Trajectory RMF/RMF3 file to write. Defaults to a temporary .rmf3 file.",
)
@click.option("--scale", type=float, help="Override torsion angle scale factor.")
@click.option("--n-iter", type=int, help="Override total MC trial count.")
@click.option("--n-out", type=int, help="Override accepted moves between RMF writes.")
@click.option("--n-written", type=int, help="Override number of accepted frames to write.")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed progress output.")
def cli(
    structure: str,
    settings_file: Optional[str],
    labeling_file: Optional[str],
    output: Optional[str],
    scale: Optional[float],
    n_iter: Optional[int],
    n_out: Optional[int],
    n_written: Optional[int],
    verbose: bool,
) -> None:
    """Run ProteinMC for a local PDB/PQR file or four-character PDB ID."""

    output_path = _validate_output_path(output)

    try:
        traj_file = run_protein_mc(
            pdb_file=structure,
            settings_file=settings_file,
            labeling_file=labeling_file,
            output_file=output_path,
            verbose=verbose,
            scale=scale,
            n_iter=n_iter,
            n_out=n_out,
            n_written=n_written,
        )
    except Exception as exc:  # pragma: no cover - delegated logic
        raise click.ClickException(f"ProteinMC simulation failed: {exc}") from exc

    click.echo(traj_file)
