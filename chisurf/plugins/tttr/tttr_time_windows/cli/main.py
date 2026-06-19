"""Command Line Interface for Time Window Bins."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..api.contract import request_from_payload, result_to_payload, contract_descriptor
from ..api.io import compute_and_save
from ..api.models import TimeWindowResult


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Time Window Bins CLI.

    Split TTTR files into fixed-duration time-window BID files (.bst).
    """
    if version:
        click.echo("Time Window Bins CLI v1.0.0")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--time-window-ms",
    default=10.0,
    show_default=True,
    type=float,
    help="Time window duration in milliseconds.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    help="Directory for generated .bst files.",
)
def analyze(
    files: list[str],
    time_window_ms: float,
    output_dir: str | None,
) -> None:
    """Split TTTR FILES into time-window BIDs."""
    if not files:
        click.echo("No files specified. Use --help for usage information.", err=True)
        return

    time_window_s = time_window_ms / 1000.0
    if output_dir:
        out_dir = Path(output_dir)
    elif files:
        first = Path(files[0])
        folder_name = f"{first.stem}_TW_{time_window_ms:.0f}ms"
        out_dir = first.parent / folder_name
    else:
        out_dir = Path("time_window_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows: dict[str, int] = {}
    output_paths: dict[str, str] = {}

    for fp in files:
        try:
            cnt, out_path = compute_and_save(fp, time_window_s, out_dir)
            n_windows[str(fp)] = cnt
            output_paths[str(fp)] = out_path
            click.echo(f"{Path(fp).name}: {cnt} windows -> {Path(out_path).name}")
        except Exception as exc:
            click.echo(f"Error processing {fp}: {exc}", err=True)

    result = TimeWindowResult(
        files=list(files),
        n_windows=n_windows,
        output_paths=output_paths,
        metadata={
            "n_files": len(files),
            "total_windows": sum(n_windows.values()),
            "output_dir": str(out_dir),
            "time_window_ms": time_window_ms,
        },
    )
    click.echo(json.dumps(result_to_payload(result), indent=2, default=str))


@cli.command("contract")
def contract_cmd() -> None:
    """Print the JSON workflow contract for integrations."""
    click.echo(json.dumps(contract_descriptor(), indent=2, default=str))


if __name__ == "__main__":
    cli()
