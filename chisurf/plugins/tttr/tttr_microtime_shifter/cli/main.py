"""Command Line Interface for Micro-time Shifter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from ..api.contract import (
    contract_descriptor,
    shift_request_from_payload,
    shift_result_to_payload,
)
from ..api.models import ShiftResult
from ..api.shift import shift_file


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Micro-time Shifter CLI.

    Apply micro-time shifts to TTTR files.
    """
    if version:
        click.echo("Micro-time Shifter CLI v1.0.0")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--global-shift", default=0, type=int, help="Global micro-time shift.")
@click.option(
    "--channel-shift", "channel_shifts", multiple=True,
    type=(int, int),
    help="Per-channel shift as CHANNEL VALUE (repeatable).",
)
@click.option("--filetype", default=None, help="Explicit TTTR file type.")
@click.option("--output-dir", type=click.Path(file_okay=False), help="Output directory.")
def apply(
    files: list[str],
    global_shift: int,
    channel_shifts: list[tuple[int, int]],
    filetype: str | None,
    output_dir: str | None,
) -> None:
    """Apply micro-time shifts to TTTR FILES."""
    if not files:
        click.echo("No files specified. Use --help for usage information.", err=True)
        return

    ch_map: dict[int, int] = dict(channel_shifts)
    result = ShiftResult()

    for path in files:
        out_path, applied = shift_file(
            path,
            global_shift=global_shift,
            channel_shifts=ch_map,
            filetype=filetype,
            output_dir=output_dir,
        )
        result.output_paths_by_file[str(path)] = out_path
        result.applied_shifts_by_file[str(path)] = {
            "global_shift": global_shift,
            "channel_shifts": {int(k): int(v) for k, v in applied.items()},
        }

    click.echo(json.dumps(shift_result_to_payload(result), indent=2, default=str))


@cli.command("contract")
def contract_cmd() -> None:
    """Print the JSON workflow contract."""
    click.echo(json.dumps(contract_descriptor(), indent=2, default=str))


@cli.command("metadata")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def metadata_cmd(path: str) -> None:
    """Print metadata for a TTTR file."""
    from ..api.shift import load_file_metadata

    meta = load_file_metadata(path)
    click.echo(json.dumps(meta, indent=2))


if __name__ == "__main__":
    cli()
