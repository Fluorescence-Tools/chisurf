"""Command Line Interface for Jordi G-Factor Calculator."""

from __future__ import annotations

import json
import click
import numpy as np

from ..core.calculations import calculate_g_factor_core


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Jordi G-Factor CLI.

    Run tail matching G-factor calculations headlessly.
    """
    pass


@cli.command("calculate")
@click.argument("jordi_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--region-min", type=float, required=True, help="Tail match region min channel.")
@click.option("--region-max", type=float, required=True, help="Tail match region max channel.")
@click.option("--shift", type=float, default=0.0, show_default=True, help="Perpendicular decay shift.")
@click.option("--use-bg", is_flag=True, help="Enable background subtraction.")
@click.option("--bg-min", type=float, default=None, help="Background region min channel.")
@click.option("--bg-max", type=float, default=None, help="Background region max channel.")
@click.option("--flip", is_flag=True, help="Swap parallel and perpendicular decays.")
def calculate(
    jordi_file: str,
    region_min: float,
    region_max: float,
    shift: float,
    use_bg: bool,
    bg_min: float | None,
    bg_max: float | None,
    flip: bool,
) -> None:
    """Calculate G-factor for JORDI_FILE using tail matching."""
    try:
        from chisurf.core.fio import read_jordi as _read_jordi
    except Exception:
        _read_jordi = None

    try:
        if _read_jordi is not None:
            vv, vh = _read_jordi(jordi_file, split=True)
        else:
            vec = np.loadtxt(jordi_file)
            half = len(vec) // 2
            vv, vh = vec[:half], vec[half:]
    except Exception as e:
        click.echo(f"Error loading Jordi file: {e}", err=True)
        sys.exit(1)

    bg_bounds = None
    if use_bg:
        if bg_min is None or bg_max is None:
            click.echo("Error: --bg-min and --bg-max must be provided if --use-bg is set.", err=True)
            sys.exit(1)
        bg_bounds = [bg_min, bg_max]

    res = calculate_g_factor_core(
        parallel_data=vv,
        perpendicular_data=vh,
        region_bounds=[region_min, region_max],
        decay_shift=shift,
        use_bg=use_bg,
        bg_region_bounds=bg_bounds,
        flip=flip,
    )
    click.echo(json.dumps(res, indent=2))
