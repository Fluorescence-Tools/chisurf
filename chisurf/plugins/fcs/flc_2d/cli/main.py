"""Command-line interface for 2D fluorescence lifetime correlation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .. import api


@click.group()
def cli() -> None:
    """Run 2D-FLCS analysis tasks."""


@cli.command("metadata")
@click.argument("tttr_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--routing", type=int, multiple=True, help="Routing channel to keep. Repeatable.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Optional JSON output path.")
def metadata(tttr_file: str, routing: tuple[int, ...], output: str | None) -> None:
    """Load TTTR metadata used by 2D-FLCS."""
    try:
        data = api.load_tttr(tttr_file, routing_channels=routing or None)
        payload = {
            "path": str(Path(tttr_file)),
            "n_photons": int(data.n_photons),
            "macro_time_resolution_s": float(data.macro_time_resolution_s),
            "micro_time_resolution_ns": float(data.micro_time_resolution_ns),
            "n_microtime_channels": int(data.n_microtime_channels),
        }
        _emit_json(payload, output)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("lifetime")
@click.argument("tttr_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--routing", type=int, multiple=True, help="Routing channel to keep. Repeatable.")
@click.option("--tau-min", default=0.3, show_default=True, type=float, help="Minimum lifetime in ns.")
@click.option("--tau-max", default=8.0, show_default=True, type=float, help="Maximum lifetime in ns.")
@click.option("--components", default=40, show_default=True, type=int, help="Lifetime grid size.")
@click.option(
    "--method",
    default="nnls",
    show_default=True,
    type=click.Choice(["nnls", "tikhonov"]),
    help="1D inverse-Laplace solver.",
)
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Optional JSON output path.")
def lifetime(
    tttr_file: str,
    routing: tuple[int, ...],
    tau_min: float,
    tau_max: float,
    components: int,
    method: str,
    output: str | None,
) -> None:
    """Resolve a 1D lifetime distribution from a TTTR file."""
    try:
        data = api.load_tttr(tttr_file, routing_channels=routing or None)
        result = api.lifetime_spectrum(
            data.micro_times,
            data.n_microtime_channels,
            data.micro_time_resolution_ns,
            tau_range=(tau_min, tau_max),
            n_components=components,
            method=method,
        )
        payload = {
            "path": str(Path(tttr_file)),
            "tau_grid": result.tau_grid.tolist(),
            "amplitudes": result.amplitudes.tolist(),
            "peak_lifetimes": result.peak_lifetimes(2).tolist(),
            "chi2": float(result.chi2),
        }
        _emit_json(payload, output)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


def _emit_json(payload: dict, output: str | None) -> None:
    """Print or write JSON payload."""
    text = json.dumps(payload, indent=2)
    if output:
        Path(output).write_text(text + "\n")
        click.echo(f"Wrote {output}")
    else:
        click.echo(text)


if __name__ == "__main__":
    cli()
