"""Command-line interface for ALEX Creator.

Batch-convert or merge ALEX ``.sm`` (and other TTTR) files into micro-time,
delegating to the Qt-free :mod:`..core` via the :mod:`..api` layer.
"""

from __future__ import annotations

import click

from ..api.contract import request_from_payload, run


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """ALEX Creator CLI — convert / merge ALEX TTTR files to micro-time."""
    if version:
        click.echo("ALEX Creator CLI v1.0.0")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _common_options(fn):
    fn = click.option("--period", "alex_period", default=8000, type=int, help="ALEX period.")(fn)
    fn = click.option("--shift", "period_shift", default=0, type=int, help="Period shift.")(fn)
    fn = click.option("--output-format", default="PTU", help="Output container (PTU, HT3, SM, …).")(
        fn
    )
    fn = click.option("--input-format", default="Auto", help="Input container, or Auto to detect.")(
        fn
    )
    return fn


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@_common_options
@click.option("--output-dir", type=click.Path(file_okay=False), help="Output directory.")
def convert(files, alex_period, period_shift, output_format, input_format, output_dir):
    """Convert each ALEX FILE to a micro-time file."""
    if not files:
        click.echo("No files specified. Use --help for usage.", err=True)
        return
    result = run(
        request_from_payload(
            {
                "files": list(files),
                "alex_period": alex_period,
                "period_shift": period_shift,
                "output_format": output_format,
                "input_format": input_format,
                "mode": "convert",
                "output_dir": output_dir or "",
            }
        )
    )
    for path in result.output_paths:
        click.echo(f"wrote {path}")


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@_common_options
@click.option(
    "--output", "output_path", type=click.Path(dir_okay=False), help="Merged output path."
)
def merge(files, alex_period, period_shift, output_format, input_format, output_path):
    """Merge several ALEX FILES into a single micro-time file."""
    if not files:
        click.echo("No files specified. Use --help for usage.", err=True)
        return
    result = run(
        request_from_payload(
            {
                "files": list(files),
                "alex_period": alex_period,
                "period_shift": period_shift,
                "output_format": output_format,
                "input_format": input_format,
                "mode": "merge",
                "output_path": output_path or "",
            }
        )
    )
    for path in result.output_paths:
        click.echo(f"wrote {path}")


if __name__ == "__main__":
    cli()
