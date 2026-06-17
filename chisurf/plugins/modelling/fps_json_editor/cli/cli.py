"""Click-based CLI for FPS JSON Editor support tools."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..api.contract import contract_descriptor, normalize_pdb_id
from ..backend.services import download_pdb_file


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """FPS JSON Editor support tools."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("fetch-pdb")
@click.argument("pdb_id")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Write PDB to this file instead of the plugin cache.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    help="Directory for the downloaded PDB file.",
)
def fetch_pdb_cmd(pdb_id: str, output: str | None, output_dir: str | None) -> None:
    """Download a PDB file from RCSB by four-character ID."""
    normalized_id = normalize_pdb_id(pdb_id)
    if output:
        target = Path(output)
        if target.exists() and target.is_dir():
            raise click.ClickException(f"Output path is a directory: {output}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            path = download_pdb_file(normalized_id, output_dir=str(target.parent))
            if path.name != target.name:
                path.replace(target)
                path = target
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        try:
            path = download_pdb_file(normalized_id, output_dir=output_dir)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(str(path))


@cli.command("contract")
def contract_cmd() -> None:
    """Print the JSON workflow contract for node/RPC integrations."""
    click.echo(json.dumps(contract_descriptor(), indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
