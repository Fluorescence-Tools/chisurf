"""Click-based CLI for MaxEnt TCSPC lifetime/FRET MEM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from ..api.helpers import run_fret_mem_from_arrays, run_lifetime_mem_from_arrays
from ..api.serialization import to_jsonable


def _load_two_column(path: str) -> list[float]:
    """Load the second column from a text file."""
    data = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    data.append(float(parts[1]))
                except ValueError:
                    continue
    if not data:
        raise click.ClickException(f"no two-column data found in {path}")
    return data


@click.group()
def cli() -> None:
    """MaxEnt TCSPC lifetime/FRET tools."""


@cli.command("lifetime")
@click.option("--decay", "decay_path", required=True, type=click.Path(exists=True))
@click.option("--irf", "irf_path", required=True, type=click.Path(exists=True))
@click.option("--dt", type=float, default=1.0, show_default=True)
@click.option("--tau-min", type=float, default=0.01, show_default=True)
@click.option("--tau-max", type=float, default=6.0, show_default=True)
@click.option("--tau-bins", type=int, default=192, show_default=True)
@click.option("--nu", type=float, default=1e-3, show_default=True)
def lifetime(decay_path: str, irf_path: str, dt: float, tau_min: float, tau_max: float, tau_bins: int, nu: float) -> None:
    """Run lifetime MEM from two-column text files."""
    result = run_lifetime_mem_from_arrays(
        decay=_load_two_column(decay_path),
        irf=_load_two_column(irf_path),
        dt=dt,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_bins=tau_bins,
        nu=nu,
    )
    click.echo(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


@cli.command("fret")
@click.option("--decay", "decay_path", required=True, type=click.Path(exists=True))
@click.option("--irf", "irf_path", required=True, type=click.Path(exists=True))
@click.option("--dt", type=float, default=1.0, show_default=True)
@click.option("--r0", type=float, default=52.0, show_default=True)
@click.option("--tau0", type=float, default=4.1, show_default=True)
@click.option("--r-bins", type=int, default=96, show_default=True)
@click.option("--nu", type=float, default=5e-2, show_default=True)
def fret(decay_path: str, irf_path: str, dt: float, r0: float, tau0: float, r_bins: int, nu: float) -> None:
    """Run FRET distance MEM from two-column text files."""
    result = run_fret_mem_from_arrays(
        decay=_load_two_column(decay_path),
        irf=_load_two_column(irf_path),
        dt=dt,
        R0=r0,
        tau0=tau0,
        r_bins=r_bins,
        nu=nu,
    )
    click.echo(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    cli.main(args=argv, prog_name="maxent-decay")
